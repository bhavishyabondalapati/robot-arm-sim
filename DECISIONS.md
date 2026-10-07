# Decisions log

Choices made while building this project, and why. Newest at the bottom.

## Setup
- **Robot model: SO-101 from MuJoCo Menagerie** (`robotstudio_so101`, commit
  `f054586`). Menagerie also has the older SO-100 (`trs_so_arm100`); SO-101 is
  the current LeRobot arm and its model includes a wrist camera, so it was
  preferred. Files are vendored unchanged into `assets/so101/` (Apache-2.0,
  license kept alongside) so the project works offline.
- **Our scene lives in a separate file** (`scene_pick_place.xml`) that
  `<include>`s the unmodified robot. Easy to update the robot later.
- **Conda env `robot-arm-sim`, Python 3.11**, pip packages pinned in
  `requirements.txt`.

## Scene
- **Cube: 3 cm, 30 g, red.** The SO-101 jaw opens ~9 cm at gripper=1.0 rad, so
  3 cm is easy to grasp yet small enough to need accuracy.
- **Top-down grasps only.** IK reachability maps showed a vertical gripper can
  reach floor-level points 0.14–0.24 m from the base within about ±40°, but the
  wrist_flex joint limit blocks vertical poses above ~6–8 cm at mid reach. So:
  cube spawns at radius 0.15–0.22 m, angle −40°…15°, and "lifted" height is
  ~7 cm (fingertip height).
- **Place target is fixed** (green disc at (0.127, 0.127), to the robot's left).
  Randomizing it too would make the cloning policy also need to locate the
  target from pixels, roughly doubling the visual learning problem. The cube
  position *and* yaw are random every episode.
- **Front camera** placed diagonally (front-right, above) after trying a few
  spots: the straight-on view had the arm hiding the cube.
- **Control at 20 Hz** (10 physics steps of 5 ms), absolute joint-position
  targets — the same interface as the real arm's servos in LeRobot.

## Inverse kinematics
- **Damped least squares (DLS), written from scratch** using MuJoCo's
  `mj_jacSite` Jacobian, instead of a library (e.g. `mink`, `ikpy`). It is
  ~100 lines, has no extra dependency, and is the thing worth understanding.
- **Target frame: the `gripperframe` site** at the fingertips. Its +x axis is
  the approach direction and +z the jaw-closing direction, so a top-down grasp
  is "site x = world −z".
- **The SO-101 has only 5 arm joints**, so it cannot hit an arbitrary 6-D pose.
  Top-down grasps still work because the arm is planar: pan picks the plane,
  lift/elbow/wrist_flex set position + pitch, wrist_roll sets the jaw yaw.
- **Orientation weight 0.25**: 1 rad of orientation error is weighted the same
  as 0.25 m of position error. Damping 0.02, at most 0.2 rad of joint change
  per iteration.
- **Joint limits**: clip after each step, and drop joints that are pinned at a
  limit and being pushed outward ("active set"), then re-solve.
- **Random restarts (default 20, only used if the first attempt fails).** On 500
  random reachable targets, 8 restarts solved 492 (position) / 484 (pose);
  20 restarts solved 499 / 500. All misses were targets *behind* the robot,
  which only a narrow range of folded arm poses can reach. The pick-and-place
  workspace is in front, where the first attempt almost always succeeds.

## Scripted pick-and-place
- **Waypoints in Cartesian space -> IK -> smooth joint-space motion.** 8
  waypoints (above cube, down, close, lift, carry, lower, open, back off), each
  solved with IK warm-started from the previous one, connected with a
  minimum-jerk curve. Joint-space interpolation (rather than a straight
  Cartesian line) is needed for the carry move: a straight line from the cube
  to the target cuts through the inner hole of the reachable workspace.
- **Jaw yaw**: a cube looks the same every 90°, so we pick the face direction
  closest to the arm's radial direction (least wrist roll).
- **Gripper closes to −0.17 rad** (past the cube), so the force-limited servo
  keeps squeezing.
- **Open-loop expert using the true cube pose.** It does not see pixels and does
  not react to slips. Result: **100/100** successes on seeds 0–99 (8 s per
  episode). It is the upper bound the learned policy is compared against.

## Demonstrations
- **What is recorded per step**: measured joint positions (6), front + wrist
  camera images (96×96 RGB, uint8), and the expert's commanded joint targets
  (6) as the action label. Stored as one compressed `.npz` per episode in
  `data/demos/` (git-ignored, ~1 MB per episode).
- **96×96 images**: big enough that the 3 cm cube is several pixels wide in the
  front view; small enough to keep 200 demos × 200 steps × 2 cameras (~2 GB)
  in GPU memory during training.
- **Both cameras**: the front camera gives the global layout (where is the
  cube?), the wrist camera gives a close-up for the final alignment and the
  cube's yaw. This mirrors the usual LeRobot SO-101 setup.
- **Noise injection (DART-style)**: the robot executes expert action +
  smoothly-varying noise, but the clean expert action is the label, so the data
  shows how to recover from small errors. 0.02 rad made 3 of 23 test demos
  fail (open-loop expert can't correct), so we use **0.01 rad**. Failed demos
  are discarded.
- **Seeds**: demos use seeds 100000+, validation rollouts 50000+, final
  evaluation 0–49, so the final evaluation uses cube poses never seen in training.
- **Episodes are 200 steps (10 s)**; the expert finishes at 8 s and then holds
  still, so the policy also learns to stop.

## Behavior-cloning policy
- **Inputs: both camera images + joint angles. Output: the next 10 joint-target
  actions** (an "action chunk", from ACT). Chunks average out the jerky,
  multi-modal moments (e.g. *exactly* when to close the gripper) and reduce
  compounding error. At run time, all chunk predictions covering the current
  step are averaged with weights exp(−0.1·i) ("temporal ensembling").
- **Small CNN + spatial softmax per camera (~0.6M parameters total)** rather
  than a pretrained ResNet-18 (11M). Spatial softmax outputs keypoint
  coordinates, which suits "where is the cube" tasks and learns fast from a few
  hundred demos. It trains in minutes on the M4 Pro.
- **Absolute joint targets, normalized** by dataset mean/std (stored inside the
  checkpoint so inference can't use the wrong statistics).
- **L1 loss** (as in ACT): less sensitive to occasional large errors than MSE
  and gives sharper action predictions.
- **Augmentation: random ±4 px image shifts** (DrQ-v2). Cheap, and the standard
  fix for image policies overfitting small datasets.
- **No time/phase input**: the policy must work out "where am I in the task"
  from the images and gripper state, like it would on a real robot.
- **Device: MPS (Apple GPU)**, CPU fallback. The whole dataset is kept on the
  device as uint8, so no DataLoader is needed. Smoke test: ~20 steps/s at batch
  128 on the M4 Pro.
- **Checkpoint selection by validation rollouts** on seeds 50000+, never the
  final evaluation seeds.
