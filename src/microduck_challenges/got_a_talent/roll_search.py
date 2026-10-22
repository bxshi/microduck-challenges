"""Search the scripted backward roll in the TRAINING physics (domain randomization on), for the
expert the policy learns from (roll_expert.py, roll12).

Each candidate is a script (keyframes) plus the time it hands over to alpha_stand. Every env
plays the cycle the timeline asks for: alpha_stand stands for STAND_S, the script rolls, then
alpha_stand lands it and stands up; then the same again. A roll scores
min(turn, 360)/360, plus 2 if it came round (>= 330 deg backward) and the duck then stood
(trunk above 0.10 m and within 20 deg of upright, arena.stage.standing) through the cycle's last
0.5 s, minus a little for turning past 400 deg. A simple evolution strategy (rank-weighted
mean, decaying step) runs from roll_variants.json's first variant; the best distinct candidates
that succeed most often are written back as the variants.

    uv run python -m microduck_challenges.got_a_talent.roll_search --device cuda:0 \
        --population 256 --seeds 16 --generations 60 --out roll_variants.json
"""

import argparse
import importlib
import json
import math
import time
from pathlib import Path

import torch

from microduck_challenges.got_a_talent import roll_expert as rx

STAND_S = 1.0      # alpha_stand before the first roll (a settled, realistic start)
CYCLE_S = 2.5      # each roll's window: script, then alpha_stand standing it up
N_ROLLS = 2
ROLLED = math.radians(330.0)
OVERSHOOT = math.radians(400.0)


def _backward_increment(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    """Backward rotation (rad) about the trunk's own lateral axis from q0 to q1 (roll_check's)."""
    w0, v0 = q0[:, 0], -q0[:, 1:]
    w1, v1 = q1[:, 0], q1[:, 1:]
    w = w0 * w1 - (v0 * v1).sum(-1)
    v = w0[:, None] * v1 + w1[:, None] * v0 + torch.cross(v0, v1, dim=-1)
    flip = w < 0
    w = torch.where(flip, -w, w)
    v = torch.where(flip[:, None], -v, v)
    norm = v.norm(dim=-1)
    angle = 2 * torch.atan2(norm, w)
    axis_y = torch.where(norm > 1e-12, v[:, 1] / norm.clamp_min(1e-12), torch.zeros_like(norm))
    return -angle * axis_y


def make_env(num_envs: int, device: str):
    from mjlab.envs import ManagerBasedRlEnv

    env_mod = importlib.import_module("microduck_challenges.got_a_talent.env")
    cfg = env_mod.make_env_cfg()
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = STAND_S + N_ROLLS * CYCLE_S + 5.0
    cfg.curriculum["roll_start"].params["start_step"] = 10**12  # no triggers, no mid-roll spawns
    cfg.terminations = {k: v for k, v in cfg.terminations.items() if k == "time_out"}
    return ManagerBasedRlEnv(cfg=cfg, device=device)


def evaluate(env, stand, params: torch.Tensor, handover: torch.Tensor, n_keyframes: int,
             seeds: int) -> dict:
    """Play every candidate on ``seeds`` envs; per candidate: score, success rate, mean turn."""
    n = params.shape[0]
    which = torch.arange(n, device=env.device).repeat_interleave(seeds)
    ends, offsets = rx.keyframes(params, n_keyframes)
    obs, _ = env.reset()
    robot = env.scene["robot"]
    dt = env.step_dt
    steps = round((STAND_S + N_ROLLS * CYCLE_S) / dt)
    starts = [STAND_S + k * CYCLE_S for k in range(N_ROLLS)]
    turn = torch.zeros(N_ROLLS, env.num_envs, device=env.device)
    stood = torch.ones(N_ROLLS, env.num_envs, dtype=torch.bool, device=env.device)
    q_prev = robot.data.root_link_quat_w.clone()
    for i in range(steps):
        t = i * dt
        k = sum(t >= s for s in starts) - 1  # current roll window, -1 before the first
        into = torch.full((env.num_envs,), t - starts[k] if k >= 0 else 1e9, device=env.device)
        scripted = into < handover[which]
        with torch.no_grad():
            action = stand(obs["actor"])
            if scripted.any():
                action[scripted] = script_action_for(ends, offsets, which, into)[scripted]
        obs, *_ = env.step(action)
        q = robot.data.root_link_quat_w
        if k >= 0:
            turn[k] += _backward_increment(q_prev, q)
            if into[0] >= CYCLE_S - 0.5:
                z = robot.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2]
                up = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
                stood[k] &= (z > 0.10) & (up > math.cos(math.radians(20.0)))
        q_prev = q.clone()
    turn = torch.nan_to_num(turn, nan=0.0)
    ok = (turn >= ROLLED) & stood
    score = (torch.clamp(turn, max=2 * math.pi) / (2 * math.pi) + 2.0 * ok.float()
             - torch.clamp(turn - OVERSHOOT, min=0.0) / (2 * math.pi))
    per = lambda x: x.float().mean(dim=0).view(n, seeds).mean(dim=1)
    return {"score": per(score), "success": per(ok), "turn_deg": torch.rad2deg(per(turn))}


def script_action_for(ends, offsets, which, into):
    return rx.script_action(ends, offsets, which, torch.clamp(into, min=0.0))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--population", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--generations", type=int, default=1)
    ap.add_argument("--sigma", type=float, default=0.06, help="initial step, in units of each parameter's range")
    ap.add_argument("--keep", type=int, default=8, help="variants to write")
    ap.add_argument("--out", default=str(rx.VARIANTS))
    args = ap.parse_args(argv)

    seed = rx.load_variants(args.device)
    nk = seed["n_keyframes"]
    lo = torch.tensor(rx.KEYFRAME_LO * nk + (rx.HANDOVER_RANGE[0],), device=args.device)
    hi = torch.tensor(rx.KEYFRAME_HI * nk + (rx.HANDOVER_RANGE[1],), device=args.device)
    mean = (torch.cat([seed["params"][0], seed["handover"][:1]]) - lo) / (hi - lo)
    env = make_env(args.population * args.seeds, args.device)
    stand = rx.load_alpha_stand(args.device)
    archive = []
    top = max(2, args.population // 4)
    w = torch.log(torch.tensor(top + 0.5)) - torch.log(torch.arange(1, top + 1).float())
    w = (w / w.sum()).to(args.device)
    for g in range(args.generations):
        sigma = args.sigma * (0.97 ** g)
        z = mean + sigma * torch.randn(args.population, mean.numel(), device=args.device)
        z[0] = mean
        z = z.clamp(0.0, 1.0)
        x = lo + z * (hi - lo)
        t0 = time.time()
        r = evaluate(env, stand, x[:, :-1], x[:, -1], nk, args.seeds)
        order = torch.argsort(r["score"], descending=True)
        mean = (w[:, None] * z[order[:top]]).sum(dim=0)
        for i in range(args.population):
            archive.append((r["score"][i].item(), r["success"][i].item(), r["turn_deg"][i].item(), z[i].cpu()))
        b = order[0]
        print(f"gen {g:3d}: best score {r['score'][b]:.2f} success {r['success'][b]:.2f} turn {r['turn_deg'][b]:.0f} deg, "
              f"mean-candidate success {r['success'][0]:.2f}, sigma {sigma:.3f}, {time.time() - t0:.0f} s", flush=True)

    # Re-judge the best distinct candidates on 4x the seeds, then keep the most reliable.
    archive.sort(key=lambda a: (a[1], a[0]), reverse=True)
    pool = []
    for a in archive:
        if all((a[3] - c[3]).norm() / math.sqrt(a[3].numel()) > 0.08 for c in pool):
            pool.append(a)
        if len(pool) == args.population // 4:
            break
    pool += [pool[0]] * (args.population // 4 - len(pool))
    z = torch.stack([a[3] for a in pool]).to(args.device)
    x = lo + z * (hi - lo)
    r = evaluate(env, stand, x[:, :-1], x[:, -1], nk, args.seeds * 4)
    final = sorted({i: (r["score"][i].item(), r["success"][i].item(), r["turn_deg"][i].item(), z[i].cpu())
                    for i in range(len(pool))}.values(), key=lambda a: (a[1], a[0]), reverse=True)
    chosen = []
    for a in final:
        if all((a[3] - c[3]).norm() > 1e-6 for c in chosen):
            chosen.append(a)
        if len(chosen) == args.keep:
            break
    variants = []
    for score, success, turn, z in chosen:
        x = (lo.cpu() + z * (hi - lo).cpu()).tolist()
        variants.append({"params": [round(v, 4) for v in x[:-1]], "handover": round(x[-1], 3),
                         "success": round(success, 3), "turn_deg": round(turn, 1), "score": round(score, 3)})
        print(f"variant: success {success:.2f} turn {turn:.0f} handover {x[-1]:.2f}")
    Path(args.out).write_text(json.dumps({
        "source": f"roll_search.py: population {args.population} x {args.seeds} seeds, {args.generations} generations, "
                  "training physics with domain randomization, alpha_stand handover",
        "n_keyframes": nk, "variants": variants}, indent=1) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
