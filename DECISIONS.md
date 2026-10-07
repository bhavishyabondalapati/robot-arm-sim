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
