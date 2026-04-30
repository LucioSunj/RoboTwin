#!/usr/bin/env python3
"""Generate block-level subtask boundaries for RoboTwin stack_blocks_three."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_DATASET_ROOT = Path(
    "/root/autodl-tmp/Robotwin-dataset/Stack_Three_Blocks/aloha-agilex_clean_50"
)
DEFAULT_OUTPUT_NAME = "subtask_segments.json"
GRIPPER_FIELDS = {
    "left": "/joint_action/left_gripper",
    "right": "/joint_action/right_gripper",
}
VECTOR_FIELD = "/joint_action/vector"
CAMERA_FIELDS = (
    "/observation/head_camera/rgb",
    "/observation/front_camera/rgb",
    "/observation/left_camera/rgb",
    "/observation/right_camera/rgb",
)
SUBTASK_NAMES = (
    ("place_red_block_as_base", "red block"),
    ("place_green_block_on_red", "green block"),
    ("place_blue_block_on_green", "blue block"),
)


@dataclass(frozen=True)
class GripperEvent:
    frame_index: int
    arm: str
    event_type: str
    before_open: bool
    after_open: bool
    before_value: float
    after_value: float

    def to_json(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "step_boundary_index": self.frame_index,
            "arm": self.arm,
            "type": self.event_type,
            "before_open": self.before_open,
            "after_open": self.after_open,
            "before_value": self.before_value,
            "after_value": self.after_value,
        }


def parse_episode_id(path: Path) -> int:
    name = path.stem
    if not name.startswith("episode"):
        raise ValueError(f"Unexpected episode filename: {path.name}")
    return int(name.removeprefix("episode"))


def load_scene_info(dataset_root: Path) -> dict[str, Any]:
    scene_info_path = dataset_root / "scene_info.json"
    if not scene_info_path.exists():
        return {}
    with scene_info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require_dataset(h5: h5py.File, key: str, path: Path, errors: list[str]) -> bool:
    if key not in h5:
        errors.append(f"missing dataset {key} in {path.name}")
        return False
    return True


def detect_gripper_events(
    left_gripper: np.ndarray,
    right_gripper: np.ndarray,
    *,
    threshold: float,
) -> tuple[list[GripperEvent], list[str]]:
    errors: list[str] = []
    values = np.stack([left_gripper, right_gripper], axis=1)
    is_open = values >= threshold
    changed_rows = np.where(np.any(np.diff(is_open.astype(np.int8), axis=0) != 0, axis=1))[0] + 1

    events: list[GripperEvent] = []
    arms = ("left", "right")
    for frame_index in changed_rows.tolist():
        changed = np.where(is_open[frame_index] != is_open[frame_index - 1])[0]
        if changed.size != 1:
            errors.append(
                "expected exactly one gripper to change at frame "
                f"{frame_index}, got {changed.size}"
            )
            continue

        arm_index = int(changed[0])
        after_open = bool(is_open[frame_index, arm_index])
        before_open = bool(is_open[frame_index - 1, arm_index])
        event_type = "open" if after_open else "close"
        events.append(
            GripperEvent(
                frame_index=int(frame_index),
                arm=arms[arm_index],
                event_type=event_type,
                before_open=before_open,
                after_open=after_open,
                before_value=float(values[frame_index - 1, arm_index]),
                after_value=float(values[frame_index, arm_index]),
            )
        )

    return events, errors


def validate_event_pairs(events: list[GripperEvent]) -> list[str]:
    errors: list[str] = []
    if len(events) != 6:
        errors.append(f"expected 6 gripper events, got {len(events)}")
        return errors

    for pair_id in range(3):
        close_event = events[2 * pair_id]
        open_event = events[2 * pair_id + 1]
        if close_event.event_type != "close":
            errors.append(
                f"event {2 * pair_id} should be close, got {close_event.event_type}"
            )
        if open_event.event_type != "open":
            errors.append(
                f"event {2 * pair_id + 1} should be open, got {open_event.event_type}"
            )
        if close_event.arm != open_event.arm:
            errors.append(
                "close/open arm mismatch for subtask "
                f"{pair_id}: {close_event.arm} vs {open_event.arm}"
            )
        if close_event.frame_index >= open_event.frame_index:
            errors.append(
                "close event must be before open event for subtask "
                f"{pair_id}: {close_event.frame_index} >= {open_event.frame_index}"
            )
    return errors


def scene_expected_arms(scene_info: dict[str, Any], episode_id: int) -> list[str] | None:
    entry = scene_info.get(f"episode_{episode_id}")
    if not isinstance(entry, dict):
        return None
    info = entry.get("info")
    if not isinstance(info, dict):
        return None

    arms = [info.get("{a}"), info.get("{b}"), info.get("{c}")]
    if all(isinstance(arm, str) for arm in arms):
        return [str(arm) for arm in arms]
    return None


def build_subtasks(events: list[GripperEvent], num_frames: int) -> list[dict[str, Any]]:
    num_steps = num_frames - 1
    start_steps = [0, events[1].frame_index, events[3].frame_index]
    end_steps = [events[1].frame_index, events[3].frame_index, num_steps]

    subtasks: list[dict[str, Any]] = []
    for subtask_id, ((name, obj), start_step, end_step) in enumerate(
        zip(SUBTASK_NAMES, start_steps, end_steps)
    ):
        close_event = events[2 * subtask_id]
        open_event = events[2 * subtask_id + 1]
        subtasks.append(
            {
                "subtask_id": subtask_id,
                "name": name,
                "object": obj,
                "arm": close_event.arm,
                "start_step": int(start_step),
                "end_step": int(end_step),
                "observation_slice": [int(start_step), int(end_step)],
                "state_slice": [int(start_step), int(end_step)],
                "action_slice": [int(start_step + 1), int(end_step + 1)],
                "num_steps": int(end_step - start_step),
                "close_event_frame": close_event.frame_index,
                "open_event_frame": open_event.frame_index,
            }
        )
    return subtasks


def validate_subtasks(subtasks: list[dict[str, Any]], num_frames: int) -> list[str]:
    errors: list[str] = []
    num_steps = num_frames - 1
    if len(subtasks) != 3:
        return [f"expected 3 subtasks, got {len(subtasks)}"]

    if subtasks[0]["start_step"] != 0:
        errors.append("first subtask must start at step 0")
    if subtasks[-1]["end_step"] != num_steps:
        errors.append(f"last subtask must end at num_steps {num_steps}")

    for idx, subtask in enumerate(subtasks):
        start_step = subtask["start_step"]
        end_step = subtask["end_step"]
        if not 0 <= start_step < end_step <= num_steps:
            errors.append(
                f"invalid subtask bounds for subtask {idx}: [{start_step}, {end_step})"
            )
        if subtask["observation_slice"][1] - subtask["observation_slice"][0] != subtask["num_steps"]:
            errors.append(f"observation slice length mismatch for subtask {idx}")
        if subtask["action_slice"][1] - subtask["action_slice"][0] != subtask["num_steps"]:
            errors.append(f"action slice length mismatch for subtask {idx}")
        if idx > 0 and subtasks[idx - 1]["end_step"] != start_step:
            errors.append(f"subtask {idx - 1} and {idx} are not contiguous")
    return errors


def process_episode(path: Path, dataset_root: Path, threshold: float, scene_info: dict[str, Any]) -> dict[str, Any]:
    episode_id = parse_episode_id(path)
    errors: list[str] = []

    with h5py.File(path, "r") as h5:
        required_keys = [*GRIPPER_FIELDS.values(), VECTOR_FIELD, *CAMERA_FIELDS]
        for key in required_keys:
            require_dataset(h5, key, path, errors)

        if errors:
            return {
                "file": str(path.relative_to(dataset_root)),
                "valid": False,
                "errors": errors,
            }

        left_gripper = np.asarray(h5[GRIPPER_FIELDS["left"]][()])
        right_gripper = np.asarray(h5[GRIPPER_FIELDS["right"]][()])
        vector = h5[VECTOR_FIELD]
        num_frames = int(vector.shape[0])

        length_by_field = {
            GRIPPER_FIELDS["left"]: int(left_gripper.shape[0]),
            GRIPPER_FIELDS["right"]: int(right_gripper.shape[0]),
            VECTOR_FIELD: num_frames,
        }
        for camera_field in CAMERA_FIELDS:
            length_by_field[camera_field] = int(h5[camera_field].shape[0])

        mismatched = {
            field: length
            for field, length in length_by_field.items()
            if length != num_frames
        }
        if mismatched:
            errors.append(f"length mismatch against {VECTOR_FIELD}: {mismatched}")

        if num_frames < 2:
            errors.append(f"episode must have at least 2 frames, got {num_frames}")

        events, event_errors = detect_gripper_events(
            left_gripper,
            right_gripper,
            threshold=threshold,
        )
        errors.extend(event_errors)
        errors.extend(validate_event_pairs(events))

        expected_arms = scene_expected_arms(scene_info, episode_id)
        detected_arms = [events[idx].arm for idx in (0, 2, 4)] if len(events) >= 5 else []
        if expected_arms is not None and detected_arms and detected_arms != expected_arms:
            errors.append(
                f"scene_info arm sequence mismatch: expected {expected_arms}, got {detected_arms}"
            )

        subtasks: list[dict[str, Any]] = []
        if not errors:
            subtasks = build_subtasks(events, num_frames)
            errors.extend(validate_subtasks(subtasks, num_frames))

    result = {
        "file": str(path.relative_to(dataset_root)),
        "num_frames": num_frames,
        "num_steps": max(num_frames - 1, 0),
        "valid": not errors,
        "expected_arms_from_scene_info": expected_arms,
        "detected_arms": detected_arms,
        "gripper_events": [event.to_json() for event in events],
        "subtasks": subtasks if not errors else [],
    }
    if errors:
        result["errors"] = errors
    return result


def discover_episode_paths(dataset_root: Path) -> list[Path]:
    data_dir = dataset_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing raw HDF5 data directory: {data_dir}")
    paths = sorted(data_dir.glob("episode*.hdf5"), key=parse_episode_id)
    if not paths:
        raise FileNotFoundError(f"No episode*.hdf5 files found in {data_dir}")
    return paths


def build_manifest(dataset_root: Path, threshold: float) -> dict[str, Any]:
    scene_info = load_scene_info(dataset_root)
    paths = discover_episode_paths(dataset_root)
    episodes: dict[str, Any] = {}

    for path in paths:
        episode_id = parse_episode_id(path)
        episodes[f"episode{episode_id}"] = process_episode(
            path=path,
            dataset_root=dataset_root,
            threshold=threshold,
            scene_info=scene_info,
        )

    valid_episode_count = sum(1 for episode in episodes.values() if episode["valid"])
    invalid_episode_count = len(episodes) - valid_episode_count
    return {
        "schema_version": "robotwin_stack_three_subtasks_v1",
        "dataset_root": str(dataset_root),
        "source": "raw_hdf5",
        "task": "stack_blocks_three",
        "gripper_threshold": threshold,
        "gripper_state_fields": GRIPPER_FIELDS,
        "state_field": VECTOR_FIELD,
        "camera_fields": list(CAMERA_FIELDS),
        "alignment": {
            "num_steps": "num_frames - 1",
            "observation_slice": "[start_step:end_step]",
            "state_slice": "[start_step:end_step]",
            "action_slice": "[start_step+1:end_step+1]",
            "slice_end_is_exclusive": True,
        },
        "summary": {
            "episode_count": len(episodes),
            "valid_episode_count": valid_episode_count,
            "invalid_episode_count": invalid_episode_count,
        },
        "episodes": episodes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate stack_blocks_three block-level subtask annotations from raw "
            "RoboTwin HDF5 gripper transitions."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root containing data/episode*.hdf5 and optional scene_info.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <dataset-root>/subtask_segments.json.",
    )
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=0.5,
        help="Values >= threshold are treated as open; lower values are closed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print summary without writing the JSON file.",
    )
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="Write the manifest even if some episodes fail validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output = args.output.resolve() if args.output else dataset_root / DEFAULT_OUTPUT_NAME

    manifest = build_manifest(dataset_root, args.gripper_threshold)
    summary = manifest["summary"]
    print(
        "Processed {episode_count} episodes: {valid_episode_count} valid, "
        "{invalid_episode_count} invalid".format(**summary)
    )

    if summary["invalid_episode_count"] and not args.allow_invalid:
        invalid = [
            (episode_name, episode["errors"])
            for episode_name, episode in manifest["episodes"].items()
            if not episode["valid"]
        ]
        for episode_name, errors in invalid[:10]:
            print(f"[INVALID] {episode_name}: {'; '.join(errors)}")
        raise SystemExit(
            "Refusing to write manifest because invalid episodes were found. "
            "Use --allow-invalid to write anyway."
        )

    if args.dry_run:
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
