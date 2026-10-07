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
- **Orientation weight 0.25** (1 rad of rotation error counts like 25 cm of
  position error... scaled down), damping 0.02, max 0.2 rad per iteration.
- **Joint limits**: clip after each step, and drop joints that are pinned at a
  limit and being pushed outward ("active set"), then re-solve.
- **Random restarts (default 20, only used if the first attempt fails).** On 500
  random reachable targets, 8 restarts solved 492 (position) / 484 (pose);
  20 restarts solved 499 / 500. All misses were targets *behind* the robot,
  which only a narrow range of folded arm poses can reach. The pick-and-place
  workspace is in front, where the first attempt almost always succeeds.
