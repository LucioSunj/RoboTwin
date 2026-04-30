import numpy as np
import torch
import dill
import os, sys
import math
from pathlib import Path

from PIL import Image

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)

from pi_model import *


# Encode observation for the model
def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]

    return input_rgb_arr, input_state


def _as_uint8_hwc(image):
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        image = image[..., None]
    if np.issubdtype(image.dtype, np.floating):
        if image.size and image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] > 3:
        image = image[..., :3]
    return image


def _save_vla_observation(TASK_ENV, model, input_rgb_arr):
    save_root = getattr(TASK_ENV, "eval_obs_save_dir", None) or getattr(TASK_ENV, "eval_video_path", None)
    if save_root is None:
        return

    episode_id = int(getattr(TASK_ENV, "test_num", 0))
    if getattr(TASK_ENV, "_pi05_obs_save_episode", None) != episode_id:
        step_lim = max(1, int(getattr(TASK_ENV, "step_lim", 1) or 1))
        pi0_step = max(1, int(getattr(model, "pi0_step", 1) or 1))
        total_vla_obs = max(1, math.ceil(step_lim / pi0_step))
        if total_vla_obs >= 10:
            save_indices = set(np.linspace(0, total_vla_obs - 1, 10, dtype=int).tolist())
        else:
            save_indices = set(range(total_vla_obs))

        episode_dir = Path(save_root) / "vla_observations" / f"episode{episode_id:04d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        TASK_ENV._pi05_obs_save_episode = episode_id
        TASK_ENV._pi05_obs_save_seen = 0
        TASK_ENV._pi05_obs_save_count = 0
        TASK_ENV._pi05_obs_save_indices = save_indices
        TASK_ENV._pi05_obs_save_dir = episode_dir
        print(f"[pi05] saving VLA observations to {episode_dir}")

    obs_index = TASK_ENV._pi05_obs_save_seen
    if obs_index in TASK_ENV._pi05_obs_save_indices:
        # Save the same visual views that are packed into the pi05 observation:
        # cam_high, cam_left_wrist, cam_right_wrist.
        views = [
            _as_uint8_hwc(input_rgb_arr[0]),
            _as_uint8_hwc(input_rgb_arr[2]),
            _as_uint8_hwc(input_rgb_arr[1]),
        ]
        height = max(view.shape[0] for view in views)
        width = sum(view.shape[1] for view in views)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        x_offset = 0
        for view in views:
            h, w = view.shape[:2]
            canvas[:h, x_offset:x_offset + w] = view
            x_offset += w

        save_count = TASK_ENV._pi05_obs_save_count
        step = int(getattr(TASK_ENV, "take_action_cnt", 0))
        out_path = TASK_ENV._pi05_obs_save_dir / f"obs_{save_count:02d}_vlaidx_{obs_index:03d}_step_{step:04d}.png"
        Image.fromarray(canvas).save(out_path)
        TASK_ENV._pi05_obs_save_count = save_count + 1

    TASK_ENV._pi05_obs_save_seen = obs_index + 1


def get_model(usr_args):
    train_config_name, model_name, checkpoint_id, pi0_step = (usr_args["train_config_name"], usr_args["model_name"],
                                                              usr_args["checkpoint_id"], usr_args["pi0_step"])
    return PI0(train_config_name, model_name, checkpoint_id, pi0_step)


def eval(TASK_ENV, model, observation):

    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)
    _save_vla_observation(TASK_ENV, model, input_rgb_arr)

    # ======== Get Action ========

    actions = model.get_action()[:model.pi0_step]

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)

    # ============================


def reset_model(model):
    model.reset_obsrvationwindows()
