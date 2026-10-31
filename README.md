# Microduck RL on Genesis: learning to walk on an AMD GPU

**English** · [Français](README.fr.md)

A port to **[Genesis](https://github.com/Genesis-Embodied-AI/Genesis)** of the
walking task from **[pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)**,
so it runs on an **AMD GPU (ROCm)**.

The **Microduck** is an ~800 g, ~25 cm biped with 14 Dynamixel XL330 servos,
designed by [Pollen Robotics](https://github.com/pollen-robotics/microduck). The
upstream repository trains its policies with **mjlab (MuJoCo Warp)**, which
**requires an NVIDIA card**. This repository rebuilds the environment on Genesis,
which does run on ROCm, **while preserving the upstream sim-to-real recipe**,
because the recipe is what has value, not the code.

---

## Quick start

Three commands, nothing installed on the host system:

```bash
./install.sh                                        # 1. the environment (isolated container)
distrobox enter genesis-box                         #    then, inside the container:
source ~/venvs/genesis/bin/activate                 #    (or the venv printed by ./install.sh)
python train.py -e smoke -B 64 --max-iterations 5   # 2. the smoke test (one minute)
python train.py -e microduck-velocity -B 4096       # 3. the real training run
```

Follow the learning: `tensorboard --logdir logs`.
Remove everything: `./uninstall.sh --tout` (see `--help`).

---

## Table of contents

1. [The demo](#1-the-demo)
2. [Installation](#2-installation)
3. [Usage](#3-usage)
4. [The problem, in one measurement](#4-the-problem-in-one-measurement)
5. [What carries over, what gets rewritten](#5-what-carries-over-what-gets-rewritten)
6. [The heart of it: the actuator](#6-the-heart-of-it-the-actuator)
7. [Numerical validation of the port](#7-numerical-validation-of-the-port)
8. [Sim-to-real: what is guaranteed, what is not](#8-sim-to-real-what-is-guaranteed-what-is-not)
9. [What is ported, what is not](#9-what-is-ported-what-is-not)
10. [Training results](#10-training-results)
11. [Repository layout](#11-repository-layout)
12. [Invariants you must not break](#12-invariants-you-must-not-break)
13. [Deliberate deviations from upstream](#13-deliberate-deviations-from-upstream)
14. [Credits and licences](#14-credits-and-licences)

---

## 1. The demo

![The Microduck walking a marked course, then dancing in front of the AMD logo](demo/apercu.gif)

▶ **[Full 30-second clip](demo/microduck-parcours-30s.mp4)**: arrival at the AMD
logo and the closing choreography, with the learning curve and the banner.

Robot on the left, learning curve on the right filling in at the same pace, and
a banner converting compute time into robot experience: **2.9 GPU-hours ≈ 3.1
months of robot experience**, because 4096 robots train in parallel.

The robot walks a totem-marked course over the training terrain (bumps, a
slope, steps), then performs a short choreography in front of the AMD logo.

**What it shows, and what it does not.** The policy is blind: it sees neither
the totems nor the terrain. The path is **commanded** from outside by waypoint
following that writes the velocity command, exactly like an operator on a
gamepad. The video demonstrates **locomotion**, not navigation. The closing
choreography is not a separately learned skill either: it is scripted velocity
and head-pose commands executed by the same walking policy, which works without
retraining because `head_pose` is part of the 61-D observation contract and the
policy is trained to track it. If the robot falls, it is stood back up at the
last waypoint and **the number of falls is displayed in the banner**.

The course reuses the training terrain features: 1 cm bumps, a 3.4° slope and
0.8 cm steps, all inside the envelope the policy was trained on (training
randomises steps up to 1.5 cm and slopes up to 5.7°), and the BAM actuator
with its friction and saturation stays active. Only the training perturbations
are switched off for rendering (pushes, randomisation, sensor noise) so the
progress stays readable.

## 2. Installation

A Linux box with `distrobox` and `podman` (or docker). Nothing is installed on
the host system: everything lives in a container.

```bash
sudo dnf install distrobox podman     # Fedora
sudo apt install distrobox podman     # Debian / Ubuntu
sudo pacman -S distrobox podman       # Arch
```

Then a single command:

```bash
./install.sh              # detects AMD or NVIDIA on its own
./install.sh --amd        # force ROCm     (TESTED configuration)
./install.sh --nvidia     # force CUDA     (UNTESTED, see below)
```

The script creates a container from the vendor's official image (GPU PyTorch is
already compiled inside), creates a Python environment that **inherits** that
PyTorch (do not reinstall it with pip, you would get a CPU build), installs
Genesis and the pinned dependencies, then checks that the GPU is visible from
inside.

> **NVIDIA: untested.** The code depends on no AMD-specific API, so it should
> work, but nobody has checked. You need the NVIDIA Container Toolkit on the
> host. If it breaks, the output of the script's verification step is the right
> place to start.

Uninstalling:

```bash
./uninstall.sh              # only what belongs to this repository
./uninstall.sh --partage    # + THE CONTAINER and the shared venv (shared!)
./uninstall.sh --image      # + the base image (~15-20 GB)
./uninstall.sh --tout       # everything: container, venv, image
./uninstall.sh --tout -y    # same, without confirmation
```

The container and the venv have generic names: other projects installed the same
way may be using them. That is why removing them takes an explicit flag instead
of being swept along by a single "Continue?" prompt. The script **lists exactly
what it is about to erase** before asking for confirmation, and never touches
your code or `logs/`.

Reference configuration: Radeon **RX 9070**, ROCm 7.2.4, PyTorch 2.10,
Genesis 1.2.2, rsl-rl-lib 5.4.2, Python 3.12.

## 3. Usage

```bash
distrobox enter genesis-box
source ~/venvs/genesis/bin/activate   # or the venv printed at the end of ./install.sh
cd microduck_rl_genesis

# SMOKE TEST, always first. Five iterations at 64 envs catch most
# configuration mistakes for a few cents of compute.
python train.py -e smoke -B 64 --max-iterations 5

# Walking
python train.py -e microduck-velocity -B 4096

# Gear-backlash variant: ±1° of play in series with each servo, encoder read
# through the play. Closer to the real servo, so harder, and more transferable.
python train.py -e microduck-backlash -B 4096 --backlash

# Rough terrain (resumed from walking: terrain is fine-tuned, it is not learned
# from scratch)
python train.py -e microduck-rough -B 4096 --rough \
    --resume logs/microduck-velocity/model_2999.pt

# Follow the learning
tensorboard --logdir logs

# Replay a policy in the viewer
python play.py -e microduck-velocity

# Export for the robot (normaliser baked into the graph, mandatory path)
python export_onnx.py -e microduck-velocity -o walk.onnx
```

## 4. The problem, in one measurement

`microduck_rl` is built on mjlab, itself built on MuJoCo Warp. The upstream
README is explicit: *"Requires a CUDA GPU (training runs through MuJoCo Warp)"*.
On a Radeon RX 9070 the check takes three lines:

```
warp 1.12.0  →  "Could not find or load the NVIDIA CUDA driver"
devices      :  ['cpu']
cuda devices :  []
```

`mujoco_warp` is CUDA-only with no fallback path: it would run on CPU, hence
useless for 4096 parallel environments. The choice is binary: rent NVIDIA, or
port the environment onto an engine that speaks ROCm.

Genesis speaks ROCm. And it loads the Microduck MJCF files untouched:

| Model | Genesis load |
|---|---|
| `robot_walk.xml` | 20 DoF (6 free + 14 servos), 15 links |
| `robot_allcollisions_rollers.xml` | 24 DoF (14 servos + 4 passive wheels), 19 links |

More importantly, **link-by-link masses are identical** between MuJoCo and
Genesis (0.73724 kg total, zero difference): the inertias from the Onshape
export are preserved. That is the first critical sim-to-real point, and it comes
for free.

## 5. What carries over, what gets rewritten

| Block | Status |
|---|---|
| MJCF + meshes (26 MB) | **taken as is** |
| PPO (`rsl_rl`) | **unchanged**: same library, same hyperparameters |
| Reward recipe, DR ranges, curricula | **transplanted value by value**, upstream comments included |
| 61-D observation contract | **preserved identically** |
| BAM M6 actuator | **rewritten** (upstream depends on `mujoco_warp`) |
| Environment (mjlab's Isaac-Lab-like managers) | **rewritten** as a Genesis class |
| Sensors (contact, air time, terrain height) | **rewritten** (different API) |
| Rough terrain | **rewritten** (mjlab boxes → Genesis height field) |
| ONNX export | **rewritten**, normaliser baked into the graph |

[`microduck/velocity_cfg.py`](microduck/velocity_cfg.py) deserves a word: it is
**the recipe**, and every value keeps the upstream comment that explains it.
Those comments are not documentation, they are **scars**: each one is the
result of a failed run. For example:

> `foot_slip` deliberately low (−0.1 and not −1.0): −1.0 over-constrained the
> pivot turn, which is how this robot turns.

Do not change a value without reading the comment next to it.

## 6. The heart of it: the actuator

> *"At this scale, tiny servos driving a ~800 g biped, actuator fidelity is
> most of the sim2real gap, which is why the actuator is modeled down to its
> voltage control law instead of an ideal PD."*
> ([upstream README](https://github.com/pollen-robotics/microduck_rl#readme))

The model is Rhoban's **[BAM](https://github.com/Rhoban/bam) M6**, bench-fitted
for the Dynamixel XL330. Three stages:

1. **Firmware control law**: position P controller → PWM duty cycle → voltage,
   with the current limiter modelled as a constraint on the duty cycle (the
   firmware can only act on PWM, not synthesise an arbitrary voltage: at high
   speed back-EMF makes the limit unreachable, exactly as on the real servo).
2. **DC motor torque**: `τ = kt·V/R − kt²·q̇/R`.
3. **Friction budget**: Coulomb + Stribeck + load-dependent term, directional
   and quadratic.

The third stage is what decides the fidelity of the port. Under MuJoCo, BAM does
**not** inject a passive friction torque: it writes its budget into
`dof_frictionloss`, and the **solver** performs the static clipping (BAM's
algorithm 1). And **Genesis implements `frictionloss` exactly the same way**: a
constraint with an identity Jacobian, stored in `efc`. So the port can be
*identical* rather than *approximate*: the budget is written into the model at
every physics step, and Genesis does the rest.

This imposes one thing, and it is a silent trap:
`RigidOptions(batch_dofs_info=True)`. Without it, `frictionloss` and `armature`
are shared across all environments, and any per-env randomisation becomes an
invisible no-op.

## 7. Numerical validation of the port

"It runs" proves nothing. Every link is compared against its reference.

| Test | What it proves | Measured result |
|---|---|---|
| [`test_bam_formulas.py`](tests/test_bam_formulas.py) | firmware law and motor torque ported identically | **bit-exact** (0.000e+00 difference over 2000 samples) |
| [`test_external_torque.py`](tests/test_external_torque.py) | sign conventions and indexing of the external torque read from Genesis | **3·10⁻⁹ N·m** difference against MuJoCo |
| [`test_bam_vs_mujoco.py`](tests/test_bam_vs_mujoco.py) | full actuator loop, both feet loaded on the ground | **0.2°** difference at 0.3 s, on joints moving 6.7°; **0.2 mm** on trunk height |
| [`test_dr.py`](tests/test_dr.py) | randomisation is per-env and does not accumulate | **exact** ratios (2.0000 / 0.5000) |
| [`test_obs_contract.py`](tests/test_obs_contract.py) | the 61-D deployment contract, slice by slice, servo order included | matches [`SIM2REAL.md`](SIM2REAL.md) |
| [`test_backlash.py`](tests/test_backlash.py) | gear backlash is bounded and the observation reads through it | **1.015°** of play for a ±1° spec; obs 61 / action 14 unchanged |
| [`test_onnx_deploy.py`](tests/test_onnx_deploy.py) | **the entire deployment chain**: the ONNX that will ship to the robot replays the policy on real observations, not on test vectors | **4.8·10⁻⁷ rad** difference over 60 steps of an episode |
| [`smoke_env.py`](tests/smoke_env.py) | the env builds, steps, observations and rewards finite | OK, flat and rough |

The third is the most telling: it runs **the same robot, the same actuator, the
same initial state** under Genesis and under MuJoCo, and compares joint
trajectories while the robot is still standing, and that is where the full loop
(firmware law → torque → friction budget → solver clipping → contacts) is
genuinely tested.

> Note: holding the HOME pose open-loop is an **unstable equilibrium**. Both
> engines eventually topple, and the moment of toppling is chaotic. That is why
> the comparison stops before it, not because it flattered the result.

Run the whole suite:

```bash
python tests/run_all.py
# The two tests that compare against BAM need the reference repository:
git clone -b mjlab_frictionloss https://github.com/Rhoban/bam.git /tmp/bam
BAM_REPO=/tmp/bam python tests/run_all.py
```

## 8. Sim-to-real: what is guaranteed, what is not

**The goal is explicit: what is trained here must be deployable on the real
robot by the [pollen-robotics/microduck](https://github.com/pollen-robotics/microduck)
runtime.** Here is where it stands, without rounding.

### What is identical to upstream

- **The deployment contract.** 61-D actor observation in the exact order:
  48 of proprioception (`base_ang_vel(3)`, `projected_gravity(3)`,
  `joint_pos(14)`, `joint_vel(14)`, `actions(14)`) + 13 of commands
  (`twist(3)`, `head_pose(4)`, `body_pose(6)`). Action = delta in radians around
  the HOME pose, 14 dimensions, **at 50 Hz**. The ONNX produced here goes into
  the upstream runtime unmodified.
- **Actuator physics**: BAM M6 XL330, numerically validated (§7).
- **The robot model**: same MJCF, same masses, same inertias, same joint limits.
- **Domain randomisation**: 9 sources with the upstream ranges. Battery voltage
  (6.5–8.2 V) and its drop under load, trunk and head CoM, mass and inertia,
  armature, joint friction, sole friction, pushes, encoder bias (±0.86°), IMU
  mounting misalignment (≤6°), bus delays (15–30 ms) and sensor delays.
- **Observation normalisation baked into the ONNX**: an upstream invariant, and
  the bug is invisible in simulation.

### The limits, plainly

1. **The contact engine is not the same.** Genesis is not MuJoCo. The
   difference measured over one second of walking is small (§7), but it is not
   zero and it has not been measured over tens of seconds of trained gait. A
   policy trained here will be *close* to one trained under mjlab, not
   identical.
2. **No real-robot validation was done here.** Nobody has yet deployed an ONNX
   produced by this repository on a physical Microduck. Everything above says
   the contract is respected and the physics faithful; that does not replace a
   trial.
3. **The policy is blind.** As upstream: proprioception only, no camera, no
   lidar. It cannot avoid an obstacle it cannot see.
4. **Only one of the 13 upstream tasks is ported** (see §9), in its four
   variants: flat / rough terrain × with or without gear backlash.

### What remains for a complete sim-to-real

- Deploy an ONNX on the robot and measure the actual gap.
- Replay a log from the real robot in both simulators and compare (upstream
  provides `scripts/testbench_sim2real.py` for this).

## 9. What is ported, what is not

| Upstream task | Here |
|---|---|
| `Mjlab-Velocity-Flat-MicroDuck` (**the main task**) | ✅ ported |
| `Mjlab-Velocity-Rough-MicroDuck` | ✅ ported |
| `Mjlab-Velocity-{Flat,Rough}-Backlash-MicroDuck` (gear backlash) | ✅ ported (`--backlash`) |
| `VelStand`, `StandUp`, `SitStand`, `GroundPick`, `BallKick`, `Roulade` | ❌ |
| Roller tasks (`Velocity-Rollers`, `Swizzle`, `RollerCrouch`, `RollerSlope`, `RollerStandUp`, `Spin`) | ❌ |
| `-Backlash-` variants of the other tasks | ❌ (the mechanism is there, it follows the task) |

This is not an arbitrary choice: the Velocity task **is** the foundation every
other one inherits from upstream (DR, observations, noise, delays, NaN guards).
The infrastructure ported here (BAM actuator, randomisation, observation
contract, terrain, export) is what the other tasks would reuse.

### What "the whole upstream repository" would actually cost

Rather than a vague "it's incremental", here are the numbers, read from the
upstream configurations and scaled to the throughput **measured here** (2.17 s
per iteration at 4096 environments on an RX 9070):

| Upstream task | Upstream iterations | ≈ GPU at our throughput | What blocks it |
|---|---:|---:|---|
| `velocity` (ported) | 50,000 | 30 h | nothing, it is done |
| `velstand` | 20,000 | 12 h | all-collision XML + 9 `mdp` functions + 4 curricula |
| `standup` | 15,000 | 9 h | same, + a prone init |
| `sitstand` | 15,000 | 9 h | same |
| `ground_pick` | 20,000 | 12 h | dedicated XML + grasping |
| `ball_kick` | 10,000 | 6 h | all-collision XML + ball |
| `roulade` | 10,000 | 6 h | all-collision XML |
| `velocity_rollers`, `spin`, `roller_crouch`, `roller_slope`, `roller_standup` | 8,000 – 50,000 | 5 – 30 h | roller XML (passive wheels) |

That is on the order of **120 GPU-hours** for the whole set, not counting the
tuning iterations, and the upstream comments show each task took several failed
campaigns before converging (`velstand` alone documents seven successive "runs").
All the necessary XML files are already in the upstream repository: what is
missing is wiring, not assets.

### And on the ported task itself

> **Worth knowing.** Upstream trains Velocity for **50,000 iterations**. The
> campaign run here does **3000** (flat) then 1200 resumed on rough terrain,
> roughly **2 GPU-hours instead of 30**. The resulting policy walks and holds
> 944 steps out of 1000, but it does not have the maturity of a full run: on the
> real robot, expect a smaller robustness margin than the upstream budget would
> produce. The port is not at fault, this is a compute-time choice, and it is
> recovered by re-running `train.py` with `--max-iterations 50000` and
> `--resume`.

## 10. Training results

Three campaigns on a single RX 9070. Rough terrain and backlash are **resumed**
from the flat walking policy: terrain and play are fine-tuned, they are not
learned from scratch.

| Run | Task | Resumed from | Iterations | Final reward | Episode length | GPU time |
|---|---|---|---:|---:|---|---:|
| `microduck-velocity` | walking, flat | from scratch | 3000 | 108.8 | **944 / 1000 steps** | 1.81 h |
| `microduck-rough` | rough terrain | `microduck-velocity` | 1200 | 125.2 | **962 / 1000 steps** | 1.05 h |
| `microduck-backlash` | ±1° gear backlash | `microduck-velocity` | 1200 | 121.9 | **961 / 1000 steps** | 1.10 h |

### Reading the reward curve

Two things surprise people, and neither is a regression.

**The drop at a resume is not one.** `Train/mean_reward` is a **sum over the
episode**, not a per-step average. Right after a resume the episodes are only a
few steps old: the sum is small because it covers few steps, not because the
policy got worse. The numbers say it plainly: reward per step is constant from
the first logged point to the last:

| Iteration | Reward | Episode length | Reward / step |
|---:|---:|---:|---:|
| 2999 (resume) | 1.66 | 12.7 | 0.131 |
| 3017 | 64.92 | 439 | 0.148 |
| 4198 (end) | 124.1 | 963 | 0.129 |

**The dips along the way are expected.** The terrain curriculum hardens the
ground over time, the randomisation curricula widen their ranges, and PPO keeps
exploring. A reward that plateaus while conditions get harder is progress.

The signal to watch for "does it work" is not reward but **episode length**:
944, 962 and 961 steps out of 1000. That is what says the robot stays upright.

## 11. Repository layout

```
microduck/
├── assets/microduck/     MJCF + meshes, copied from upstream (CC BY-SA-NC)
├── assets/xl330_m6.json  BAM M6 parameters bench-fitted for the XL330
├── constants.py          servo order, HOME pose, observation layout, BAM config
├── bam_actuator.py       vectorised BAM actuator + delay buffer
├── terrain.py            rough height field + height lookup
├── velocity_cfg.py       THE RECIPE: weights, DR ranges, curricula
└── velocity_env.py       the Genesis environment

train.py  play.py  export_onnx.py    train, replay, export
tests/                              the port's validation suite
policies/                           deployment-ready ONNX policies
demo/                               the 30 s clip and its animated preview
install.sh  uninstall.sh            containerised environment, AMD and NVIDIA
```

The repository is deliberately limited to what serves the port, its validation
and deployment on the robot. The tooling used to produce the videos and to chain
the overnight campaign is kept out of it.

## 12. Invariants you must not break

Taken from the upstream `AGENTS.md`; they survive the change of engine.

- **The actor observation is 61 dimensions**, in the order of
  `constants.OBS_LAYOUT`. This is what lets the embedded runtime hot-swap
  policies. An unused command slot is set to **zero**, it is never removed.
- **Order of the 14 servos**: 0-4 left leg, 5-8 neck/head, 9-13 right leg.
- **Observation normalisation is on** → it must be baked into the ONNX.
  `export_onnx.py` does it; in simulation, forgetting is invisible.
- **Policies are not filtered** (no low-pass on the action during training).
  Adding one without the matching flag on the runtime side breaks transfer, in
  either direction.
- **Randomisation must not accumulate across resets**
  (`tests/test_dr.py` checks this; an accumulating CoM randomiser degraded
  months of upstream runs).
- **`batch_dofs_info=True`** in the Genesis options, otherwise all per-env
  randomisation is a silent no-op.

## 13. Deliberate deviations from upstream

**Quadratic friction.** The reference BAM model (`bam/model.py`) enables its
quadratic term only when motor torque and external torque have **opposite**
signs. The mjlab port (`bam/mjlab.py`), the one that actually trained the
policies deployed on the real robot, lost that guard. Measured difference:
**1.5 % at worst** on the friction budget. This repository reproduces **mjlab**
by default, to stay on the recipe that transfers;
`BamActuator(..., quadratic_sign_gate=True)` restores the paper version.

**Foot sites.** Genesis does not expose MJCF `<site>` elements. Foot site poses
are recomposed from the ankle link frame and the offset read from the XML
(`constants.FOOT_SITE_OFFSETS`).

**Foot height above ground.** mjlab casts a ray (`TerrainHeightSensor`). Here the
height field is kept on the Python side and bilinearly interpolated: same
quantity, without a raycast.

**Rough terrain.** mjlab stacks boxes, Genesis takes a height field. **Vertical
amplitudes are preserved**; the patches are smaller (3 m instead of 8 m, since the
robot tops out at 0.4 m/s) and the grid smaller (10×10 instead of 10×20),
because Genesis builds an SDF over the whole terrain and 2.2 M cells will not
compile. This is not a loss: Genesis environments are **independent** worlds
sharing the same static geometry, so variety comes from origin placement.

**Angular momentum.** No `subtreeangmom` sensor in Genesis: it is computed from
link velocities and inertias.

**Performance.** `set_dofs_armature` costs **1.7 s per call** under Genesis
1.2.2, with or without `envs_idx`: changing the armature invalidates the mass
matrix, which Genesis refactorises. Called at every reset it multiplied the
iteration time by 6. The neighbouring setters (`set_dofs_frictionloss`,
`set_dofs_damping`) sit at 0.02 ms; the problem is specific to armature.
Armature randomisation is therefore drawn once per environment at startup and
held for the whole run, which is also the physically honest choice: a servo's
rotor inertia does not change between episodes.

## 14. Credits and licences

- **[pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)**
  provides the robot, the 3D models, and above all **the sim-to-real recipe**
  that this repository merely transports. Code under Apache 2.0.

  > **This repository is dual-licensed.** The MJCF files and meshes in
  > `microduck/assets/` keep their upstream licence, **Creative Commons
  > BY-SA-NC**, and are not covered by Apache 2.0. If you reuse them: credit
  > Pollen Robotics, keep derivatives under the same licence, and note that
  > commercial use is not granted. The code is Apache 2.0 and carries none of
  > those conditions.
- **[Rhoban/bam](https://github.com/Rhoban/bam)**: the BAM actuator model
  (Marc Duclusaud & Grégoire Passault). Apache 2.0.
- **[mjlab](https://github.com/mujocolab/mjlab)**: the upstream training
  framework, whose semantics served as the reference.
- **[Genesis](https://github.com/Genesis-Embodied-AI/Genesis)**: the simulation
  engine used here.
- **[rsl_rl](https://github.com/leggedrobotics/rsl_rl)**: the PPO
  implementation.

The AMD logo that appears in the video simply indicates that training runs on an
AMD GPU. It is a registered trademark of Advanced Micro Devices, Inc.; this
project is neither affiliated with nor endorsed by AMD.

**This repository's licence, in two lines:** the code is under **Apache 2.0**,
like the upstream it derives from; the 3D models in `microduck/assets/` remain
under **CC BY-SA-NC** and are not covered by Apache 2.0. Details in
[`LICENSE`](LICENSE), [`NOTICE`](NOTICE) and
[`microduck/assets/LICENSE-ASSETS.md`](microduck/assets/LICENSE-ASSETS.md).
