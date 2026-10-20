# SO-101 Robot Arm: Simulation, IK, and Imitation Learning

A MuJoCo simulation of the low-cost SO-101 (LeRobot) arm picking up a cube from a random position and placing it on a target. A scripted inverse-kinematics controller performs the task first; its recorded demonstrations then train a behavior-cloning policy that performs the same task from camera images and joint angles alone. The two are compared on 50 unseen episodes.

![Scene](videos/scene.png)

## Features

- **Simulation**: the MuJoCo Menagerie SO-101 model in a custom scene (floor, 3 cm cube, target disc, front camera, wrist camera), controlled at 20 Hz with absolute joint-position actions.
- **Inverse kinematics**: a damped-least-squares solver written from scratch on MuJoCo's Jacobian, with joint-limit handling and random restarts.
- **Scripted expert**: Cartesian waypoints converted to joint targets with IK and connected by minimum-jerk motions.
- **Demonstration recording**: joint angles, both camera images, and expert actions at every 50 ms step, with injected action noise.
- **Behavior cloning**: a CNN with spatial softmax that predicts 10-step action chunks from two camera images and joint angles, trained on Apple Silicon (MPS).
- **Evaluation**: success rates with 95% confidence intervals, failure analysis, and rendered videos for both policies.

## Results

50 evaluation episodes with cube poses not seen in training (seeds 0–49):

| Policy | Observations | Success | 95% CI |
|---|---|---|---|
| Scripted IK expert | Exact cube pose from the simulator | **50/50 (100%)** | 93–100% |
| Behavior cloning | 2 cameras + joint angles | **46/50 (92%)** | 81–97% |

- **Checkpoint**: `checkpoints/bc_policy.pt`, saved at step 25,000 of 30,000. It had the best validation success (100% on 20 validation episodes, tied with steps 15k and 20k; ties go to the later step). The final step-30k model scored 90% on validation.
- **Training**: 200 demonstrations, 30k steps, 30 minutes on an M4 Pro GPU (MPS).
- **Failures**: all 4 follow the same pattern. The gripper descends to the cube with a slightly misaligned grasp, nudges the cube instead of closing on it, and hovers without retrying (the demonstrations contain no retries). Three of the four cubes were at the inner edge of the spawn area.
- The comparison is not like-for-like: the expert receives the true cube pose, while the learned policy must locate the cube in pixels.

Videos (front camera, with the wrist camera inset top right):
[scripted expert](videos/eval_scripted.mp4) ·
[learned policy](videos/eval_bc_policy.mp4) ·
[learned policy failure, seed 23](videos/eval_bc_policy_failure_seed23.mp4)

![Training curves](results/training_curves.png)

Per-episode results: [results/eval_results.json](results/eval_results.json).

## Architecture

```
scene_pick_place.xml ──► env.py (PickPlaceEnv: reset, step, cameras, success check)
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
scripted.py ── ik.py                         policy.py (CNN + spatial softmax)
(waypoints → IK → minimum-jerk)               images + joints → 10-step action chunk
        │                                           ▲
        ▼                                           │
record.py ──► data/demos/ ──────────────► train.py (MPS, best checkpoint by rollout)
                                                    │
                              evaluate.py ◄─────────┘  (50 unseen seeds, both policies)
```

### Inverse kinematics

Each iteration measures the position and orientation error of the `gripperframe` site, computes the Jacobian `J`, and updates the joints by

```
dq = Jᵀ (J Jᵀ + λ² I)⁻¹ · error
```

The damping term `λ` limits step size near singularities, where `J` loses rank. Joints are clipped to their limits, joints pinned at a limit are dropped from the solve, and the solver restarts from random configurations if the first attempt fails.

The SO-101 has 5 arm joints and cannot reach arbitrary 6-D poses. Top-down grasps remain feasible because the arm is planar: the base selects the plane, three pitch joints set height, reach, and tilt, and wrist roll aligns the jaw with the cube.

### Scripted expert

Eight waypoints (above → down → close → lift → carry → lower → open → back off) are converted to joint angles with IK and connected by minimum-jerk trajectories. It succeeded on 100 of 100 development episodes.

### Behavior cloning

Behavior cloning is trained as supervised learning from observations to expert actions. To limit compounding error from distribution shift, it uses:

- **Noise injection during recording** (DART), so demonstrations include recoveries from small deviations.
- **Action chunking with temporal ensembling** (ACT): the policy predicts the next 10 actions, and overlapping predictions are averaged at run time.
- **Image augmentation.**

Spatial softmax converts each CNN feature map into an (x, y) keypoint, a useful inductive bias for manipulation. Checkpoints are selected by simulator rollouts on validation seeds rather than by validation loss.

### Tech stack

| Component | Role |
|---|---|
| MuJoCo 3 | Contact-rich physics with headless rendering on macOS |
| MuJoCo Menagerie SO-101 | Curated model of the real arm: meshes, joint limits, servo gains, wrist camera |
| NumPy | IK math, demonstration storage, data processing |
| PyTorch (MPS) | Policy training on Apple Silicon, with CPU fallback |
| imageio, imageio-ffmpeg | MP4 output without a system ffmpeg |
| matplotlib | Training curves |
| pytest | Tests for IK, the environment, and the policy |

## Quickstart

```bash
conda create -n robot-arm-sim python=3.11 -y
conda activate robot-arm-sim
pip install -r requirements.txt
```

Tested on macOS with an M4 Pro (24 GB). No GPU drivers are required; PyTorch uses MPS when available and falls back to CPU.

Run each stage in order:

```bash
python -m robot_arm_sim.render                  # scene image and joint-sweep video
python -m robot_arm_sim.scripted --episodes 5   # scripted pick-and-place → videos/scripted.mp4
python -m robot_arm_sim.record --episodes 200   # record demos → data/demos/ (~15 min)
python -m robot_arm_sim.train --steps 30000     # train policy → checkpoints/ (~30 min on M4 Pro)
python -m robot_arm_sim.evaluate --episodes 50  # compare policies → results/, videos/
```

## Running tests

```bash
pytest
```

18 tests:

- `test_ik.py`: FK→IK round trips, joint limits, reachability, rotation math.
- `test_env.py`: environment resets and the scripted expert.
- `test_policy.py`: network output shapes, spatial softmax, augmentation, and the runtime agent.

## Project structure

```
robot-arm-sim/
├── DECISIONS.md               Design decision log
├── requirements.txt           Pinned dependencies
├── pyproject.toml             pytest configuration
├── assets/so101/
│   ├── so101.xml              SO-101 model (unmodified, from MuJoCo Menagerie)
│   ├── assets/*.stl           Robot meshes (from Menagerie)
│   ├── LICENSE, README.md     Menagerie Apache-2.0 license and model notes
│   └── scene_pick_place.xml   Scene: floor, lights, cube, target disc, front camera
├── robot_arm_sim/
│   ├── env.py                 PickPlaceEnv: random cube reset, step, cameras, success check
│   ├── ik.py                  Damped-least-squares IK solver
│   ├── render.py              Loads the arm; saves an image and a joint-sweep video
│   ├── scripted.py            IK-based scripted pick-and-place expert
│   ├── record.py              Records expert demonstrations
│   ├── policy.py              Behavior-cloning network and runtime agent
│   ├── train.py               Training on MPS with rollout-based checkpoint selection
│   └── evaluate.py            Scripted vs learned policy on 50 unseen episodes
├── tests/                     pytest suite
├── results/                   Evaluation tables, JSON, training curves
├── videos/                    Rendered videos
├── data/demos/                Recorded demonstrations (not tracked)
└── checkpoints/               Trained weights (not tracked)
```

## Design decisions

See [DECISIONS.md](DECISIONS.md) for every choice and the measurements behind it, including cube size, camera placement, IK weighting, restart counts, and training settings.

## Limitations and future work

- The policy does not retry failed grasps, since the demonstrations contain no retry behavior.
- Results are simulation-only. Transfer to a physical SO-101 would require:
  - **Calibration**: servo ticks (0–4095) mapped to angles, with zero pose and joint signs matched to the MJCF. Menagerie follows LeRobot's `so101_new_calib` convention, but each joint should be verified.
  - **Actuator matching**: STS3215 gains, torque limits, backlash (~0.5°), and latency differ from the simulated position actuators, giving errors of a few millimeters relative to a 3 cm cube. Options are tuning `kp`/`forcerange` against recorded step responses or randomizing them.
  - **Perception**: the expert uses the true cube pose. Hardware requires a detector with calibrated cameras, or human teleoperation with the SO-101 leader arm (the standard LeRobot workflow). A policy trained only on renders will not transfer directly; options include real demonstrations (typically 50–100), sim and real co-training, or domain randomization of colors, lighting, and camera pose.
  - **Camera placement** matching the simulated poses, fields of view, and resolution, or retraining.
  - **Timing**: inference, camera capture, and servo bus latency must fit within the 50 ms control period.
  - **Safety**: joint-velocity limits, workspace bounds, and an emergency stop.
  - **Grasp physics**: real friction, mass, and soft jaw contact differ from simulation.
