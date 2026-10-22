"""The roll-and-stand expert the policy learns from (roll12): a scripted backward roll, then
Pollen's alpha_stand.

- The script: flip_feasibility.py's open-loop keyframes, symmetric joint targets held as steps.
  Per keyframe: duration (s), then offsets from the default pose for the left hip_pitch, knee,
  ankle (mirrored onto the right leg), neck_pitch and head_pitch. ``roll_variants.json`` holds
  the variants roll_search.py found in the training physics, each with the time it hands over.
- alpha_stand (pollen-robotics/microduck-policies): the deployed stand policy, same 61-number
  observation and 14 targets. It lands the roll, stands the duck up and holds it between rolls.
  Its ONNX file (normalizer baked in, then a 512-256-128 ELU MLP) is rebuilt here as a torch
  module so it labels a whole batch at once. Its twist input is zeroed: it never saw a trigger.

In the training physics the script alone came round 0-2 times in 16; script for 0.9 s then
alpha_stand came round and stood up 12-13 times in 16 (ROLL_RUNS.md).
"""

import json
from pathlib import Path

import torch

from arena.constants import DEFAULT_POSE, JOINT_NAMES

PER_KEYFRAME = 6
KEYFRAME_LO = (0.04, -1.6, -1.6, -1.6, -1.6, -1.6)
KEYFRAME_HI = (0.60, 1.6, 1.6, 1.6, 1.2, 1.6)
HANDOVER_RANGE = (0.6, 1.3)
VARIANTS = Path(__file__).with_name("roll_variants.json")
STAND_REPO, STAND_FILE = "pollen-robotics/microduck-policies", "alpha_stand.onnx"
TWIST = slice(48, 51)  # the actor observation's twist command block

_J = {n: JOINT_NAMES.index(n) for n in JOINT_NAMES}
# (left joint, right joint or None, sign on the right) for each keyframe offset after the duration.
_MAP = (("left_hip_pitch", "right_hip_pitch"), ("left_knee", "right_knee"),
        ("left_ankle", "right_ankle"), ("neck_pitch", None), ("head_pitch", None))


def keyframes(params: torch.Tensor, n_keyframes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(end times [V, K], target offsets from the default pose [V, K, 14]) of V scripts."""
    lo = torch.tensor(KEYFRAME_LO * n_keyframes, device=params.device)
    hi = torch.tensor(KEYFRAME_HI * n_keyframes, device=params.device)
    p = torch.clamp(params, lo, hi).view(-1, n_keyframes, PER_KEYFRAME)
    ends = torch.cumsum(p[..., 0], dim=-1)
    offsets = torch.zeros(*p.shape[:2], len(JOINT_NAMES), device=params.device)
    for k, (left, right) in enumerate(_MAP):
        offsets[..., _J[left]] = p[..., k + 1]
        if right is not None:
            offsets[..., _J[right]] = -p[..., k + 1]
    return ends, offsets


def script_action(ends: torch.Tensor, offsets: torch.Tensor, which: torch.Tensor,
                  t: torch.Tensor) -> torch.Tensor:
    """Each env's action at time t (s) into script ``which``: the keyframe that is on, held."""
    e = ends[which]  # [N, K]
    k = torch.clamp((t.unsqueeze(-1) >= e).sum(dim=-1), max=e.shape[-1] - 1)
    return offsets[which, k]


def load_variants(device, path: Path = VARIANTS) -> dict:
    """roll_variants.json as tensors: params [V, K*6], handover [V], n_keyframes."""
    data = json.loads(path.read_text())
    return {"params": torch.tensor([v["params"] for v in data["variants"]], device=device),
            "handover": torch.tensor([v["handover"] for v in data["variants"]], device=device),
            "n_keyframes": data["n_keyframes"]}


class AlphaStand(torch.nn.Module):
    """alpha_stand.onnx as torch: normalize, then the MLP. Input is the 61-number actor obs."""

    def __init__(self, onnx_path: Path):
        super().__init__()
        import onnx
        from onnx import numpy_helper

        init = {i.name: torch.tensor(numpy_helper.to_array(i)) for i in onnx.load(str(onnx_path)).graph.initializer}
        self.register_buffer("mean", init.pop("obs_normalizer._mean").view(-1))
        self.register_buffer("std", next(v for k, v in init.items() if "Div" in k).view(-1))
        layers = []
        for i in (0, 2, 4, 6):
            lin = torch.nn.Linear(init[f"mlp.{i}.weight"].shape[1], init[f"mlp.{i}.weight"].shape[0])
            lin.weight.data.copy_(init[f"mlp.{i}.weight"])
            lin.bias.data.copy_(init[f"mlp.{i}.bias"])
            layers += [lin, torch.nn.ELU()] if i < 6 else [lin]
        self.mlp = torch.nn.Sequential(*layers)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.clone()
        x[:, TWIST] = 0.0
        return self.mlp((x - self.mean) / self.std)


def load_alpha_stand(device) -> AlphaStand:
    from huggingface_hub import hf_hub_download

    return AlphaStand(Path(hf_hub_download(STAND_REPO, STAND_FILE))).to(device).eval()

