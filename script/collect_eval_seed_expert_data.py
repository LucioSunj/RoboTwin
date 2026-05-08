#!/usr/bin/env python3
"""Collect programmatic expert trajectories on the same seed distribution as eval.

This intentionally does not use a learned policy. It mirrors RoboTwin eval's
expert-check seed filtering, then records expert trajectories for the accepted
seeds so downstream subtask evaluation can start from eval-distribution states.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

sys.path.append("./")
sys.path.append("./description/utils")

from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError
from generate_episode_instructions import generate_episode_descriptions


def class_decorator(task_name: str):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
    except AttributeError as exc:
        raise SystemExit(f"No such task: {task_name}") from exc
    return env_class()


def get_embodiment_config(robot_file: str) -> dict[str, Any]:
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def prepare_args(task_name: str, task_config: str, output_root: Path) -> dict[str, Any]:
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["save_path"] = str(output_root)
    args["eval_mode"] = True
    args["eval_video_log"] = False
    args["render_freq"] = 0

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment: str) -> str:
        robot_file = embodiment_types[embodiment]["file_path"]
        if robot_file is None:
            raise ValueError(f"Missing embodiment file for {embodiment}")
        return robot_file

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


def read_seed_file(seed_path: Path) -> list[int]:
    if not seed_path.exists():
        return []
    return [int(token) for token in seed_path.read_text().split()]


def write_seed_file(seed_path: Path, seeds: list[int]) -> None:
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    seed_path.write_text(" ".join(str(seed) for seed in seeds) + "\n")


def close_quietly(task, *, clear_cache: bool = False) -> None:
    try:
        task.close_env(clear_cache=clear_cache)
    except Exception:
        pass


def collect_accepted_seed_plans(
    *,
    task_name: str,
    args: dict[str, Any],
    output_root: Path,
    start_seed: int,
    episode_num: int,
    max_seed_tries: int,
    resume: bool,
    clear_cache_freq: int,
) -> list[int]:
    seed_path = output_root / "seed.txt"
    seeds = read_seed_file(seed_path) if resume else []
    if seeds and len(seeds) > episode_num:
        seeds = seeds[:episode_num]
        write_seed_file(seed_path, seeds)

    now_seed = max(start_seed, seeds[-1] + 1) if seeds else start_seed
    tries = 0
    plan_args = dict(args)
    plan_args["need_plan"] = True
    plan_args["save_data"] = False

    while len(seeds) < episode_num:
        if tries >= max_seed_tries:
            raise RuntimeError(
                f"Only collected {len(seeds)}/{episode_num} accepted seeds after "
                f"{max_seed_tries} tries from start_seed={start_seed}."
            )
        tries += 1
        episode_idx = len(seeds)
        task = class_decorator(task_name)
        accepted = False
        try:
            task.setup_demo(
                now_ep_num=episode_idx,
                seed=now_seed,
                is_test=True,
                **plan_args,
            )
            task.play_once()
            accepted = bool(task.plan_success and task.check_success())
            if accepted:
                task.save_traj_data(episode_idx)
                seeds.append(now_seed)
                write_seed_file(seed_path, seeds)
                print(f"[ACCEPT] episode={episode_idx} seed={now_seed}")
            else:
                print(f"[REJECT] seed={now_seed} expert plan/check failed")
        except UnStableError as exc:
            print(f"[REJECT] seed={now_seed} unstable: {exc}")
        except Exception as exc:
            print(f"[REJECT] seed={now_seed} error: {exc}")
        finally:
            close_quietly(
                task,
                clear_cache=bool(accepted and len(seeds) % clear_cache_freq == 0),
            )
        now_seed += 1

    return seeds


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_hdf5_data(
    *,
    task_name: str,
    args: dict[str, Any],
    output_root: Path,
    seeds: list[int],
    resume: bool,
    clear_cache_freq: int,
) -> None:
    scene_info_path = output_root / "scene_info.json"
    scene_info = load_json(scene_info_path, {})
    data_args = dict(args)
    data_args["need_plan"] = False
    data_args["save_data"] = True
    data_args["render_freq"] = 0

    for episode_idx, seed in enumerate(seeds):
        hdf5_path = output_root / "data" / f"episode{episode_idx}.hdf5"
        if resume and hdf5_path.exists() and f"episode_{episode_idx}" in scene_info:
            print(f"[SKIP] episode={episode_idx} seed={seed} already collected")
            continue

        task = class_decorator(task_name)
        try:
            task.setup_demo(
                now_ep_num=episode_idx,
                seed=seed,
                is_test=True,
                **data_args,
            )
            traj_data = task.load_tran_data(episode_idx)
            replay_args = dict(data_args)
            replay_args["left_joint_path"] = traj_data["left_joint_path"]
            replay_args["right_joint_path"] = traj_data["right_joint_path"]
            task.set_path_lst(replay_args)

            info = task.play_once()
            success = bool(task.check_success())
            if not success:
                raise RuntimeError(f"Expert replay failed for episode={episode_idx} seed={seed}")

            scene_info[f"episode_{episode_idx}"] = info
            scene_info_path.parent.mkdir(parents=True, exist_ok=True)
            with scene_info_path.open("w", encoding="utf-8") as f:
                json.dump(scene_info, f, ensure_ascii=False, indent=2)

            task.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
            task.merge_pkl_to_hdf5_video()
            task.remove_data_cache()
            print(f"[DATA] episode={episode_idx} seed={seed} wrote {hdf5_path}")
        except Exception:
            close_quietly(task, clear_cache=True)
            raise


def write_instructions(
    *,
    task_name: str,
    output_root: Path,
    language_num: int,
) -> None:
    scene_info = load_json(output_root / "scene_info.json", {})
    episodes = []
    for idx in range(len(scene_info)):
        entry = scene_info.get(f"episode_{idx}", {})
        episodes.append(entry.get("info", {}))

    results = generate_episode_descriptions(task_name, episodes, language_num)
    instructions_dir = output_root / "instructions"
    instructions_dir.mkdir(parents=True, exist_ok=True)
    for episode_desc in results:
        episode_index = episode_desc["episode_index"]
        output_file = instructions_dir / f"episode{episode_index}.json"
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "seen": episode_desc.get("seen", []),
                    "unseen": episode_desc.get("unseen", []),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    print(f"[INSTRUCTIONS] wrote {instructions_dir}")


def write_metadata(
    *,
    output_root: Path,
    task_name: str,
    task_config: str,
    eval_seed_arg: int,
    start_seed: int,
    episode_num: int,
    seeds: list[int],
) -> None:
    metadata = {
        "task_name": task_name,
        "task_config": task_config,
        "collection_mode": "programmatic_expert_eval_seed_distribution",
        "eval_seed_arg": eval_seed_arg,
        "start_seed": start_seed,
        "episode_num": episode_num,
        "accepted_seeds": seeds,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with (output_root / "eval_seed_expert_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="stack_blocks_three")
    parser.add_argument("--task_config", default="demo_clean")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--seed", type=int, default=0, help="Same seed argument as eval_policy.py.")
    parser.add_argument("--episode_num", type=int, default=100)
    parser.add_argument("--max_seed_tries", type=int, default=1000)
    parser.add_argument("--language_num", type=int, default=None)
    parser.add_argument("--clear_cache_freq", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_instructions", action="store_true")
    return parser


def main() -> None:
    parsed = build_parser().parse_args()

    from test_render import Sapien_TEST

    Sapien_TEST()
    output_root = Path(parsed.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    args = prepare_args(parsed.task_name, parsed.task_config, output_root)
    language_num = parsed.language_num if parsed.language_num is not None else int(args.get("language_num", 100))
    start_seed = 100000 * (1 + parsed.seed)

    seeds = collect_accepted_seed_plans(
        task_name=parsed.task_name,
        args=args,
        output_root=output_root,
        start_seed=start_seed,
        episode_num=parsed.episode_num,
        max_seed_tries=parsed.max_seed_tries,
        resume=parsed.resume,
        clear_cache_freq=parsed.clear_cache_freq,
    )
    collect_hdf5_data(
        task_name=parsed.task_name,
        args=args,
        output_root=output_root,
        seeds=seeds,
        resume=parsed.resume,
        clear_cache_freq=parsed.clear_cache_freq,
    )
    if not parsed.no_instructions:
        write_instructions(
            task_name=parsed.task_name,
            output_root=output_root,
            language_num=language_num,
        )
    write_metadata(
        output_root=output_root,
        task_name=parsed.task_name,
        task_config=parsed.task_config,
        eval_seed_arg=parsed.seed,
        start_seed=start_seed,
        episode_num=parsed.episode_num,
        seeds=seeds,
    )
    print(f"[DONE] eval-seed expert dataset: {output_root}")


if __name__ == "__main__":
    main()
