from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

sys.path.append("./")
sys.path.append("./policy")

import h5py
import numpy as np
import torch
import yaml

from envs import CONFIGS_PATH


DEFAULT_FULL_PROMPT = (
    "Move red block, green block, and blue block to the center. "
    "Stack green block on red block and blue block on green block."
)

SUBTASK_PROMPTS = {
    0: "Place the red block at the stack base location.",
    1: "Place the green block on top of the red block.",
    2: "Place the blue block on top of the green block.",
}


@dataclass(frozen=True)
class SubtaskSegment:
    episode_idx: int
    subtask_id: int
    name: str
    arm: str
    start_step: int
    end_step: int
    action_slice: tuple[int, int]
    num_steps: int
    hdf5_file: str


def class_decorator(task_name: str):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
    except AttributeError as exc:
        raise SystemExit(f"No such task: {task_name}") from exc
    return env_class()


def parse_index_spec(values: list[str] | None) -> list[int] | None:
    if values is None:
        return None
    result: list[int] = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                start, end = token.split("-", 1)
                result.extend(range(int(start), int(end) + 1))
            else:
                result.append(int(token))
    return sorted(set(result))


def episode_sort_key(episode_key: str) -> int:
    match = re.search(r"(\d+)$", str(episode_key))
    if match is None:
        raise ValueError(f"Cannot parse episode index from {episode_key!r}")
    return int(match.group(1))


def load_seed_list(seed_path: Path) -> list[int]:
    seeds = [int(token) for token in seed_path.read_text().split()]
    if not seeds:
        raise ValueError(f"No seeds found in {seed_path}")
    return seeds


def load_subtask_segments(
    manifest_path: Path,
    episode_ids: list[int] | None,
    subtask_ids: list[int] | None,
) -> list[SubtaskSegment]:
    manifest = json.loads(manifest_path.read_text())
    episodes = manifest.get("episodes", {})
    selected_episodes = set(episode_ids) if episode_ids is not None else None
    selected_subtasks = set(subtask_ids) if subtask_ids is not None else None
    segments: list[SubtaskSegment] = []

    for episode_key in sorted(episodes.keys(), key=episode_sort_key):
        episode_idx = episode_sort_key(episode_key)
        if selected_episodes is not None and episode_idx not in selected_episodes:
            continue
        episode = episodes[episode_key]
        if not episode.get("valid", False):
            continue
        for subtask in episode.get("subtasks", []):
            subtask_id = int(subtask["subtask_id"])
            if selected_subtasks is not None and subtask_id not in selected_subtasks:
                continue
            action_slice = tuple(int(x) for x in subtask["action_slice"])
            segments.append(
                SubtaskSegment(
                    episode_idx=episode_idx,
                    subtask_id=subtask_id,
                    name=str(subtask["name"]),
                    arm=str(subtask.get("arm", "")),
                    start_step=int(subtask["start_step"]),
                    end_step=int(subtask["end_step"]),
                    action_slice=(action_slice[0], action_slice[1]),
                    num_steps=int(subtask["num_steps"]),
                    hdf5_file=str(episode["file"]),
                )
            )
    return segments


def get_embodiment_config(robot_file: str):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def prepare_task_args(task_name: str, task_config_name: str) -> dict[str, Any]:
    with open(f"./task_config/{task_config_name}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config_name
    args["eval_mode"] = True
    args["save_data"] = False
    args["render_freq"] = 0
    args["eval_video_log"] = False

    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type: str) -> str:
        robot_file = embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise ValueError(f"No embodiment file configured for {embodiment_type}")
        return robot_file

    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), "r", encoding="utf-8") as f:
        camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_config[head_camera_type]["h"]
    args["head_camera_w"] = camera_config[head_camera_type]["w"]

    embodiment_type = args.get("embodiment")
    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
        args["embodiment_name"] = str(embodiment_type[0])
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
        args["embodiment_name"] = f"{embodiment_type[0]}+{embodiment_type[1]}"
    else:
        raise ValueError("embodiment config must contain either 1 or 3 items")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    return args


def load_episode_prompt(
    dataset_root: Path,
    episode_idx: int,
    prompt_split: str,
    prompt_index: int,
) -> str:
    instruction_path = dataset_root / "instructions" / f"episode{episode_idx}.json"
    if not instruction_path.exists():
        return DEFAULT_FULL_PROMPT
    instructions = json.loads(instruction_path.read_text())
    prompts = instructions.get(prompt_split) or instructions.get("seen") or instructions.get("unseen")
    if not prompts:
        return DEFAULT_FULL_PROMPT
    return str(prompts[prompt_index % len(prompts)])


def prompt_for_segment(
    segment: SubtaskSegment,
    *,
    dataset_root: Path,
    prompt_mode: str,
    prompt_split: str,
    prompt_index: int,
) -> str:
    if prompt_mode == "subtask":
        return SUBTASK_PROMPTS[segment.subtask_id]
    return load_episode_prompt(dataset_root, segment.episode_idx, prompt_split, prompt_index)


def load_expert_vectors(dataset_root: Path, segment: SubtaskSegment) -> np.ndarray:
    hdf5_path = dataset_root / segment.hdf5_file
    with h5py.File(hdf5_path, "r") as f:
        return np.asarray(f["joint_action/vector"], dtype=np.float64)


def execute_qpos_actions(task, actions: np.ndarray, *, stop_fn=None) -> int:
    executed = 0
    for action in np.asarray(actions):
        before_count = int(getattr(task, "take_action_cnt", 0))
        task.take_action(action, action_type="qpos")
        after_count = int(getattr(task, "take_action_cnt", before_count))
        if after_count <= before_count:
            break
        executed += 1
        if stop_fn is not None and stop_fn():
            break
    return executed


def encode_pi05_obs(observation: dict[str, Any]):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]
    return input_rgb_arr, input_state


class SubtaskSuccessChecker:
    def __init__(self, task, segment: SubtaskSegment):
        self.task = task
        self.segment = segment

    def success(self) -> bool:
        subtask_id = self.segment.subtask_id
        both_grippers_open_half = (
            self.task.is_left_gripper_open_half()
            and self.task.is_right_gripper_open_half()
        )
        if not both_grippers_open_half:
            return False

        if subtask_id == 0:
            target = self.target_for_subtask0()
            err = np.abs(self.task.block1.get_pose().p - target)
            return bool(np.all(err < np.array([0.04, 0.04, 0.03])))
        if subtask_id == 1:
            target = self.target_for_subtask1()
            err = np.abs(self.task.block2.get_pose().p - target)
            return bool(np.all(err < np.array([0.035, 0.035, 0.025])))
        if subtask_id == 2:
            block1_pose = self.task.block1.get_pose().p
            block2_pose = self.task.block2.get_pose().p
            block3_pose = self.task.block3.get_pose().p
            eps = [0.025, 0.025, 0.012]
            return bool(
                np.all(
                    abs(block2_pose - np.array(block1_pose[:2].tolist() + [block1_pose[2] + 0.05]))
                    < eps
                )
                and np.all(
                    abs(block3_pose - np.array(block2_pose[:2].tolist() + [block2_pose[2] + 0.05]))
                    < eps
                )
            )
        raise ValueError(f"Unsupported subtask_id={subtask_id}")

    def target_for_current_subtask(self) -> np.ndarray:
        if self.segment.subtask_id == 0:
            return self.target_for_subtask0()
        if self.segment.subtask_id == 1:
            return self.target_for_subtask1()
        block2 = self.task.block2.get_pose().p
        return np.array([block2[0], block2[1], block2[2] + 0.05])

    def target_for_subtask0(self) -> np.ndarray:
        if hasattr(self.task, "block1_target_pose"):
            return np.asarray(self.task.block1_target_pose[:3], dtype=np.float64)
        return np.asarray([0.0, -0.13, 0.75 + self.task.table_z_bias], dtype=np.float64)

    def target_for_subtask1(self) -> np.ndarray:
        block1 = self.task.block1.get_pose().p
        return np.asarray([block1[0], block1[1], block1[2] + 0.05], dtype=np.float64)

    def diagnostics(self) -> dict[str, Any]:
        block1 = self.task.block1.get_pose().p
        block2 = self.task.block2.get_pose().p
        block3 = self.task.block3.get_pose().p
        target = self.target_for_current_subtask()
        current = [block1, block2, block3][self.segment.subtask_id]
        return {
            "block1_pose": block1.tolist(),
            "block2_pose": block2.tolist(),
            "block3_pose": block3.tolist(),
            "target_pose": target.tolist(),
            "target_l2": float(np.linalg.norm(current - target)),
            "target_linf": float(np.max(np.abs(current - target))),
            "left_gripper_value": float(self.task.robot.get_left_gripper_val()),
            "right_gripper_value": float(self.task.robot.get_right_gripper_val()),
            "left_gripper_open_half": bool(self.task.is_left_gripper_open_half()),
            "right_gripper_open_half": bool(self.task.is_right_gripper_open_half()),
            "left_gripper_open_strict": bool(self.task.is_left_gripper_open()),
            "right_gripper_open_strict": bool(self.task.is_right_gripper_open()),
            "full_task_success_strict": bool(self.task.check_success()),
        }


def reset_to_subtask_start(
    *,
    segment: SubtaskSegment,
    seed: int,
    task_args: dict[str, Any],
    prompt: str,
    expert_vectors: np.ndarray,
    pi0_step: int,
):
    task = class_decorator(task_args["task_name"])
    setup_args = dict(task_args)
    setup_args["step_lim"] = max(
        int(task_args.get("step_lim", 1000)),
        segment.start_step + segment.num_steps + int(pi0_step),
    )
    setup_args["instruction"] = prompt
    task.setup_demo(now_ep_num=segment.episode_idx, seed=seed, is_test=True, **setup_args)
    task.set_instruction(prompt)

    if segment.start_step > 0:
        prefix_actions = expert_vectors[1 : segment.start_step + 1]
        executed = execute_qpos_actions(task, prefix_actions)
        if executed < len(prefix_actions):
            task.close_env(clear_cache=True)
            raise RuntimeError(
                f"Failed to replay prefix for episode {segment.episode_idx} "
                f"subtask {segment.subtask_id}: {executed}/{len(prefix_actions)}"
            )

    task.take_action_cnt = 0
    task.eval_success = False
    task.step_lim = int(segment.num_steps)
    task.run_steps = 0
    task.reward_step = 0
    return task


def load_pi05_model(args):
    pi05_dir = Path("policy/pi05").resolve()
    if str(pi05_dir) not in sys.path:
        sys.path.insert(0, str(pi05_dir))
    from pi_model import PI0

    return PI0(
        args.train_config_name,
        args.model_name,
        args.checkpoint_id,
        args.pi0_step,
    )


def reset_pi05_model(model):
    model.reset_obsrvationwindows()


def run_model_subtask(task, model, checker: SubtaskSuccessChecker, max_steps: int) -> bool:
    if model.observation_window is None:
        model.set_language(task.get_instruction())

    while int(getattr(task, "take_action_cnt", 0)) < max_steps:
        observation = task.get_obs()
        input_rgb_arr, input_state = encode_pi05_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)
        actions = model.get_action()[: model.pi0_step]

        for action in actions:
            task.take_action(action, action_type="qpos")
            if checker.success():
                task.eval_success = True
                return True
            if int(getattr(task, "take_action_cnt", 0)) >= max_steps:
                break
            observation = task.get_obs()
            input_rgb_arr, input_state = encode_pi05_obs(observation)
            model.update_observation_window(input_rgb_arr, input_state)

    return checker.success()


def run_expert_subtask(task, expert_vectors: np.ndarray, segment: SubtaskSegment, checker: SubtaskSuccessChecker) -> bool:
    start, end = segment.action_slice
    actions = expert_vectors[start:end]
    execute_qpos_actions(task, actions, stop_fn=checker.success)
    return checker.success()


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"total": len(records)}
    by_subtask: dict[str, Any] = {}
    for subtask_id in sorted({int(r["subtask_id"]) for r in records}):
        rows = [r for r in records if int(r["subtask_id"]) == subtask_id]
        successes = sum(bool(r["success"]) for r in rows)
        by_subtask[str(subtask_id)] = {
            "subtask_name": rows[0]["subtask_name"],
            "success": successes,
            "total": len(rows),
            "success_rate": successes / len(rows) if rows else 0.0,
            "avg_elapsed_steps": float(np.mean([r["elapsed_steps"] for r in rows])) if rows else math.nan,
            "avg_target_l2": float(np.mean([r["target_l2"] for r in rows])) if rows else math.nan,
        }
    successes = sum(bool(r["success"]) for r in records)
    summary["success"] = successes
    summary["success_rate"] = successes / len(records) if records else 0.0
    summary["by_subtask"] = by_subtask
    return summary


def print_summary(summary: dict[str, Any]):
    print("\nSubtask evaluation summary")
    print(f"overall: {summary['success']}/{summary['total']} = {summary['success_rate'] * 100:.1f}%")
    for subtask_id, row in summary["by_subtask"].items():
        print(
            f"subtask {subtask_id} ({row['subtask_name']}): "
            f"{row['success']}/{row['total']} = {row['success_rate'] * 100:.1f}% "
            f"| avg_steps={row['avg_elapsed_steps']:.1f} "
            f"| avg_target_l2={row['avg_target_l2']:.4f}"
        )


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="stack_blocks_three")
    parser.add_argument("--task_config", default="demo_clean")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--subtask_manifest", default=None)
    parser.add_argument("--seed_path", default=None)
    parser.add_argument("--episode_ids", nargs="*", default=None)
    parser.add_argument("--subtask_ids", nargs="*", default=["0", "1", "2"])
    parser.add_argument("--max_rollouts", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--prompt_mode", choices=["full", "subtask"], default="full")
    parser.add_argument("--prompt_split", choices=["seen", "unseen"], default="seen")
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--pi0_step", type=int, default=50)
    parser.add_argument("--dry_run_expert", action="store_true")
    parser.add_argument("--clear_cache_freq", type=int, default=1)
    parser.add_argument("--train_config_name", default="pi05_aloha_stack_three_blocks_full")
    parser.add_argument("--model_name", default="stack_three_blocks_full")
    parser.add_argument("--checkpoint_id", default="19000")
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def main():
    args = build_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_root = Path(args.dataset_root).resolve()
    manifest_path = Path(args.subtask_manifest).resolve() if args.subtask_manifest else dataset_root / "subtask_segments.json"
    seed_path = Path(args.seed_path).resolve() if args.seed_path else dataset_root / "seed.txt"
    episode_ids = parse_index_spec(args.episode_ids)
    subtask_ids = parse_index_spec(args.subtask_ids)

    segments = load_subtask_segments(manifest_path, episode_ids, subtask_ids)
    if args.max_rollouts is not None:
        segments = segments[: args.max_rollouts]
    if not segments:
        raise ValueError("No selected subtask segments")

    output_dir = Path(args.output_dir or f"eval_result/{args.task_name}/pi05_subtask/{time.strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "subtask_eval_records.jsonl"
    summary_path = output_dir / "subtask_eval_summary.json"
    if records_path.exists():
        records_path.unlink()

    seed_list = load_seed_list(seed_path)
    task_args = prepare_task_args(args.task_name, args.task_config)
    model = None if args.dry_run_expert else load_pi05_model(args)

    records: list[dict[str, Any]] = []
    for rollout_idx, segment in enumerate(segments):
        if segment.episode_idx >= len(seed_list):
            raise IndexError(f"Episode {segment.episode_idx} has no seed in {seed_path}")

        expert_vectors = load_expert_vectors(dataset_root, segment)
        prompt = prompt_for_segment(
            segment,
            dataset_root=dataset_root,
            prompt_mode=args.prompt_mode,
            prompt_split=args.prompt_split,
            prompt_index=args.prompt_index,
        )
        task = reset_to_subtask_start(
            segment=segment,
            seed=seed_list[segment.episode_idx],
            task_args=task_args,
            prompt=prompt,
            expert_vectors=expert_vectors,
            pi0_step=args.pi0_step,
        )
        checker = SubtaskSuccessChecker(task, segment)
        try:
            if args.dry_run_expert:
                success = run_expert_subtask(task, expert_vectors, segment, checker)
            else:
                reset_pi05_model(model)
                success = run_model_subtask(task, model, checker, int(segment.num_steps))

            record = {
                "rollout_index": rollout_idx,
                "episode_idx": segment.episode_idx,
                "sim_seed": seed_list[segment.episode_idx],
                "subtask_id": segment.subtask_id,
                "subtask_name": segment.name,
                "subtask_arm": segment.arm,
                "start_step": segment.start_step,
                "end_step": segment.end_step,
                "max_steps": segment.num_steps,
                "elapsed_steps": int(getattr(task, "take_action_cnt", 0)),
                "success": bool(success),
                "prompt_mode": args.prompt_mode,
                "prompt_split": args.prompt_split,
                "prompt": prompt,
                **checker.diagnostics(),
            }
            records.append(record)
            with records_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps(record, sort_keys=True))
        finally:
            task.close_env(clear_cache=((rollout_idx + 1) % int(args.clear_cache_freq) == 0))

    summary = summarize(records)
    summary_path.write_text(json.dumps(summary, indent=2))
    print_summary(summary)
    print(f"records: {records_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
