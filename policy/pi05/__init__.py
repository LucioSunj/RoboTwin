"""pi05 policy entrypoint for RoboTwin eval."""

import os


def _patch_sapien_rt_denoiser():
    denoiser = os.environ.get("ROBOTWIN_RT_DENOISER")
    if not denoiser:
        return

    allowed = {"none", "oidn", "optix"}
    if denoiser not in allowed:
        raise ValueError(
            f"ROBOTWIN_RT_DENOISER must be one of {sorted(allowed)}, got {denoiser!r}"
        )

    try:
        import sapien.render as sapien_render
    except Exception:
        return

    original_setter = sapien_render.set_ray_tracing_denoiser

    def set_ray_tracing_denoiser(name):
        if name == "oidn":
            name = denoiser
        return original_setter(name)

    sapien_render.set_ray_tracing_denoiser = set_ray_tracing_denoiser


_patch_sapien_rt_denoiser()

from .deploy_policy import *
