"""PPO plus behavior cloning from the roll-and-stand expert (roll_expert.py), on the student's own
frames (roll12).

After every PPO update, a cloning pass regresses the student's deterministic action onto the
expert's over every frame the student just collected (DAgger-flavoured, after
mjlab_microduck tasks/distill.py: states from the student, labels from the expert, its own Adam
so PPO's adaptive learning rate does not throttle it). The expert needs the roll clock, which
the actor does not see: the "expert" observation group (roll.expert_state_obs, stored with every
frame like any group) carries [scripted, time into the script, variant]. Scripted frames are
labelled with the variant's keyframe; every other frame, standing, landing, getting up, with
alpha_stand on the actor obs (its twist zeroed).

Why: in roll1-11 the policy never found "come round onto the feet and stand" on its own, while
script-then-alpha_stand does it in 12-13 of 16 tries in the training physics.
"""

from dataclasses import dataclass, field

import torch
from rsl_rl.algorithms import PPO

from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

from microduck_challenges.got_a_talent import roll_expert as rx


def default_bc_cfg() -> dict:
    return {
        "coef": 1.0,               # MSE weight (actions in rad)
        "learning_rate": 3e-4,
        "epochs": 2,
        "mini_batches": 4,
        "expert_group": "expert",  # obs group with [scripted, script time, variant]
        "actor_group": "actor",
        # Weight the scripted frames up: they are ~10% of the data and the roll lives there.
        "script_weight": 3.0,
        # Linear fade of the cloning weight to fade_to by fade_end_iter (PPO keeps refining).
        "fade_start_iter": 1500,
        "fade_end_iter": 3000,
        "fade_to": 0.2,
    }


@dataclass
class PpoWithRollExpertCfg(PpoWithSymmetryCfg):
    bc_cfg: dict | None = field(default_factory=default_bc_cfg)
    class_name: str = "microduck_challenges.got_a_talent.roll_distill.PpoWithRollExpert"


class PpoWithRollExpert(PPO):
    def __init__(self, actor, critic, storage, *args, bc_cfg: dict | None = None, **kwargs) -> None:
        kwargs.pop("symmetry_cfg", None)
        super().__init__(actor, critic, storage, *args, **kwargs)
        self.bc_cfg = bc_cfg
        self.bc_iter = 0
        if bc_cfg:
            self.stand = rx.load_alpha_stand(self.device)
            v = rx.load_variants(self.device)
            self.ends, self.offsets = rx.keyframes(v["params"], v["n_keyframes"])
            self.bc_optimizer = torch.optim.Adam(self.actor.parameters(), lr=bc_cfg["learning_rate"])
            print(f"[roll_distill] expert BC ON: {len(v['handover'])} roll variants + alpha_stand, coef {bc_cfg['coef']}")

    def _coef(self) -> float:
        c = self.bc_cfg
        f = min(max((self.bc_iter - c["fade_start_iter"]) / max(c["fade_end_iter"] - c["fade_start_iter"], 1), 0.0), 1.0)
        return c["coef"] * (1.0 - f * (1.0 - c["fade_to"]))

    def expert_actions(self, actor_obs: torch.Tensor, expert_obs: torch.Tensor) -> torch.Tensor:
        scripted = expert_obs[:, 0] > 0.5
        target = self.stand(actor_obs)
        if scripted.any():
            which = expert_obs[scripted, 2].round().long().clamp(0, self.ends.shape[0] - 1)
            target[scripted] = rx.script_action(self.ends, self.offsets, which, expert_obs[scripted, 1])
        return target, scripted

    def _bc_update(self) -> dict[str, float]:
        c = self.bc_cfg
        obs = self.storage.observations.flatten(0, 1)
        actor_obs, expert_obs = obs[c["actor_group"]], obs[c["expert_group"]]
        with torch.no_grad():
            target, scripted = self.expert_actions(actor_obs, expert_obs)
        weight = torch.where(scripted, c["script_weight"], 1.0)
        coef = self._coef()
        n = target.shape[0]
        mb = max(1, n // c["mini_batches"])
        total, steps = 0.0, 0
        for _ in range(c["epochs"]):
            perm = torch.randperm(n, device=self.device)
            for s in range(0, n, mb):
                sel = perm[s:s + mb]
                pred = self.actor(obs[sel])
                loss = coef * (weight[sel] * (pred - target[sel]).pow(2).mean(dim=1)).mean()
                self.bc_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.bc_optimizer.step()
                total += loss.item()
                steps += 1
        self.bc_iter += 1
        return {"expert_bc": total / max(steps, 1), "expert_bc_coef": coef,
                "expert_bc_scripted_frac": scripted.float().mean().item()}

    def update(self) -> dict[str, float]:
        loss_dict = super().update()
        if self.bc_cfg:
            loss_dict.update(self._bc_update())
        return loss_dict
