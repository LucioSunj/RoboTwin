#!/usr/bin/env python3
"""Visualize wrist-camera observations from a RoboTwin HDF5 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATASET_ROOT = Path(
    "/root/autodl-tmp/Robotwin-dataset/Stack_Three_Blocks/aloha-agilex_clean_50"
)
WRIST_CAMERA_ALIASES = {
    "left": "left_camera",
    "left_wrist": "left_camera",
    "left_camera": "left_camera",
    "right": "right_camera",
    "right_wrist": "right_camera",
    "right_camera": "right_camera",
    "wrist": "both",
    "both": "both",
}


def parse_episode_ids(value: str) -> list[int]:
    """Parse comma-separated episode ids or ranges, e.g. '0,3,5-8'."""
    episode_ids: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise argparse.ArgumentTypeError(f"Invalid episode range: {part}")
            episode_ids.extend(range(start, end + 1))
        else:
            episode_ids.append(int(part))
    if not episode_ids:
        raise argparse.ArgumentTypeError("No episode ids were provided.")
    return sorted(set(episode_ids))


def parse_frame_ids(value: str | None, num_frames: int, sample_count: int) -> list[int]:
    if value:
        frame_ids = [int(part.strip()) for part in value.split(",") if part.strip()]
        if not frame_ids:
            raise ValueError("--frames was provided but no valid frame index was parsed.")
    else:
        sample_count = min(sample_count, num_frames)
        frame_ids = np.linspace(0, num_frames - 1, sample_count, dtype=int).tolist()

    bad = [idx for idx in frame_ids if idx < 0 or idx >= num_frames]
    if bad:
        raise IndexError(f"Frame indices out of range for {num_frames} frames: {bad}")
    return frame_ids


def normalize_camera_names(values: Iterable[str]) -> list[str]:
    cameras: list[str] = []
    for value in values:
        key = value.strip().lower()
        if key not in WRIST_CAMERA_ALIASES:
            choices = ", ".join(sorted(WRIST_CAMERA_ALIASES))
            raise argparse.ArgumentTypeError(f"Unknown camera '{value}'. Choices: {choices}")
        camera = WRIST_CAMERA_ALIASES[key]
        if camera == "both":
            cameras.extend(["left_camera", "right_camera"])
        else:
            cameras.append(camera)
    return list(dict.fromkeys(cameras))


def episode_path(dataset_root: Path, episode_id: int) -> Path:
    path = dataset_root / "data" / f"episode{episode_id}.hdf5"
    if not path.exists():
        raise FileNotFoundError(f"Episode file does not exist: {path}")
    return path


def decode_jpeg_from_hdf5(value: np.bytes_ | bytes) -> np.ndarray:
    """Decode one padded JPEG byte string from the RoboTwin HDF5 files as RGB."""
    if isinstance(value, np.bytes_):
        data = bytes(value)
    else:
        data = value
    data = data.rstrip(b"\0")
    encoded = np.frombuffer(data, dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Failed to decode JPEG image from HDF5 byte string.")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_camera_frames(h5: h5py.File, camera_name: str, frame_ids: list[int]) -> list[np.ndarray]:
    dataset_key = f"observation/{camera_name}/rgb"
    if dataset_key not in h5:
        available = sorted(
            name for name in h5["observation"].keys() if f"observation/{name}/rgb" in h5
        )
        raise KeyError(f"Camera '{camera_name}' not found. Available image cameras: {available}")
    rgb_data = h5[dataset_key]
    return [decode_jpeg_from_hdf5(rgb_data[idx]) for idx in frame_ids]


def save_grid(
    *,
    images_by_camera: dict[str, list[np.ndarray]],
    frame_ids: list[int],
    episode_id: int,
    output_path: Path,
) -> None:
    cameras = list(images_by_camera)
    rows = len(cameras)
    cols = len(frame_ids)
    fig_width = max(2.2 * cols, 8)
    fig_height = max(2.2 * rows, 3)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)

    for row, camera_name in enumerate(cameras):
        for col, frame_id in enumerate(frame_ids):
            ax = axes[row][col]
            ax.imshow(images_by_camera[camera_name][col])
            ax.set_axis_off()
            if row == 0:
                ax.set_title(f"t={frame_id}", fontsize=9)
            if col == 0:
                ax.set_ylabel(camera_name, fontsize=10)

    fig.suptitle(f"Episode {episode_id}: wrist-camera observations", fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_video(
    *,
    h5: h5py.File,
    camera_names: list[str],
    output_path: Path,
    fps: int,
    max_frames: int | None,
) -> None:
    rgb_datasets = [h5[f"observation/{camera}/rgb"] for camera in camera_names]
    num_frames = min(len(dataset) for dataset in rgb_datasets)
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)
    if num_frames <= 0:
        raise ValueError("No frames available for video export.")

    first_images = [decode_jpeg_from_hdf5(dataset[0]) for dataset in rgb_datasets]
    height = max(image.shape[0] for image in first_images)
    resized_first = [
        cv2.resize(image, (round(image.shape[1] * height / image.shape[0]), height))
        if image.shape[0] != height
        else image
        for image in first_images
    ]
    width = sum(image.shape[1] for image in resized_first)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    for frame_id in range(num_frames):
        images = [decode_jpeg_from_hdf5(dataset[frame_id]) for dataset in rgb_datasets]
        resized = [
            cv2.resize(image, (round(image.shape[1] * height / image.shape[0]), height))
            if image.shape[0] != height
            else image
            for image in images
        ]
        combined_rgb = np.concatenate(resized, axis=1)
        writer.write(cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR))

    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize left/right wrist-camera RGB observations from RoboTwin HDF5 episodes."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--episodes", type=parse_episode_ids, default=parse_episode_ids("0"))
    parser.add_argument("--cameras", nargs="+", default=["both"], help="left, right, both, left_camera, right_camera")
    parser.add_argument("--num-frames", type=int, default=12, help="Uniform samples per episode when --frames is unset.")
    parser.add_argument("--frames", type=str, default=None, help="Comma-separated frame ids, e.g. '0,50,100'.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/autodl-tmp/robotwin-wrist-visualizations"),
    )
    parser.add_argument("--make-video", action="store_true", help="Also export a side-by-side wrist-camera MP4.")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-video-frames", type=int, default=None)
    args = parser.parse_args()

    camera_names = normalize_camera_names(args.cameras)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for episode_id in args.episodes:
        path = episode_path(args.dataset_root, episode_id)
        with h5py.File(path, "r") as h5:
            first_camera_key = f"observation/{camera_names[0]}/rgb"
            if first_camera_key not in h5:
                raise KeyError(f"Camera '{camera_names[0]}' not found in {path}")
            num_frames = len(h5[first_camera_key])
            frame_ids = parse_frame_ids(args.frames, num_frames, args.num_frames)

            images_by_camera = {
                camera_name: load_camera_frames(h5, camera_name, frame_ids)
                for camera_name in camera_names
            }

            camera_label = "-".join(camera_names)
            grid_path = args.output_dir / f"episode{episode_id}_{camera_label}_grid.png"
            save_grid(
                images_by_camera=images_by_camera,
                frame_ids=frame_ids,
                episode_id=episode_id,
                output_path=grid_path,
            )
            print(f"Saved image grid: {grid_path}")

            if args.make_video:
                video_path = args.output_dir / f"episode{episode_id}_{camera_label}.mp4"
                save_video(
                    h5=h5,
                    camera_names=camera_names,
                    output_path=video_path,
                    fps=args.fps,
                    max_frames=args.max_video_frames,
                )
                print(f"Saved video: {video_path}")


if __name__ == "__main__":
    main()
