# pitchmap

Per-player position heatmaps for football matches.

Processing is offline and stage-based: every stage reads artifacts produced by
earlier stages and writes its own, so any stage can be re-run in isolation.

The source is a wide broadcast camera that pans, tilts and zooms to keep the
outfield players in frame (goalkeepers may be cut off). Measured on the test
clip, a median of 20 players are on the pitch and in view per frame, so
positions are recovered for essentially the whole outfield throughout.

Because the camera moves, every frame gets its own homography. These are
estimated automatically from detected pitch landmarks, independently per frame,
so error cannot accumulate.

## Pipeline

```
video -> detect -> track -> annotate -> calibrate -> project -> heatmap
                                                  \-> overlay (sanity videos)
```

| Stage | Output | Purpose |
| --- | --- | --- |
| `detect` | `detections.parquet` | Per-frame player boxes, stored at a low confidence floor |
| `track` | `tracks.parquet`, `tracks_mot.txt` | Track ids across frames (MOT export for later tooling) |
| `annotate` | `calib/keyframes.json` | Optional manual landmark clicks, used to cross-check calibration |
| `calibrate` | `calib/homographies.npz` | Per-frame pixel-to-metre homography from detected pitch keypoints |
| `project` | `positions.parquet` | Foot points projected into pitch metres |
| `heatmap` | `heatmaps/*.png` | Per-tracklet occupancy maps on a pitch |
| `overlay` | `overlays/*.mp4` | Calibration and tracking sanity videos |

## Setup

```powershell
python -m venv hm_env
hm_env\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
hm_env\Scripts\python.exe -m pip install -r requirements.txt
hm_env\Scripts\python.exe -m pip install -e .
hm_env\Scripts\python.exe scripts\check_env.py
```

## Usage

```powershell
hm_env\Scripts\python.exe -m pitchmap init-run  --video data\raw\fixed_clip.mp4
hm_env\Scripts\python.exe -m pitchmap detect    --run data\runs\fixed_clip
hm_env\Scripts\python.exe -m pitchmap track     --run data\runs\fixed_clip
hm_env\Scripts\python.exe -m pitchmap annotate  --run data\runs\fixed_clip
hm_env\Scripts\python.exe -m pitchmap calibrate --run data\runs\fixed_clip
hm_env\Scripts\python.exe -m pitchmap project   --run data\runs\fixed_clip
hm_env\Scripts\python.exe -m pitchmap heatmap   --run data\runs\fixed_clip
hm_env\Scripts\python.exe -m pitchmap overlay   --run data\runs\fixed_clip --kind calib
```

Settings live in `configs/default.yaml`; most stages accept flags that override
the relevant entries for a single run.

## Layout

```
configs/    pipeline settings
data/       raw clips and per-clip run directories (not tracked)
docs/notes/ working notes and research links
models/     detector weights (not tracked)
scripts/    standalone helper scripts
src/pitchmap/
  cli/      one thin entrypoint per stage
  io/       video reading, run artifacts, MOT format
  detect/   detector implementations behind a shared protocol
  track/    tracker implementations behind a shared protocol
  calib/    pitch model, landmark clicking, homography fit and propagation
  project/  ground-plane projection
  heat/     heatmap accumulation and rendering
  viz/      sanity overlay videos
```

## Calibration accuracy

Measured on the test clip: 5499 of 5500 frames calibrated, median fit residual
0.52 m, and 0.35-0.80 m median drift for static pitch points tracked across
frames. That is comfortably below the heatmap smoothing kernel.

The `calibrate` stage caches raw keypoints, so fitting and smoothing can be
retuned with `--reuse-keypoints` without re-running the model.

## Watching it work

To see boxes drawn on the video while a stage is processing:

```powershell
hm_env\Scripts\python.exe -m pitchmap detect --run data
unsixed_clip --show
hm_env\Scripts\python.exe -m pitchmap track  --run data
unsixed_clip --show
```

Press `q` to stop early; whatever was processed is still saved.

To replay a finished run without re-running any model, which is much faster and
can be paused:

```powershell
hm_env\Scripts\python.exe -m pitchmap preview --run data
unsixed_clip --kind track
hm_env\Scripts\python.exe -m pitchmap preview --run data
unsixed_clip --kind detect
hm_env\Scripts\python.exe -m pitchmap preview --run data
unsixed_clip --kind calib
```

`space` pauses, `j` and `k` step frame by frame while paused, `q` quits.
`--start` jumps to a frame, `--speed` changes playback rate.

Note that `--limit` truncates a run's artifacts, so a stage refuses to overwrite
a longer existing run unless given `--force`. For smoke tests, make a separate
run directory with `init-run` instead.

## Checking the output

Three things to look at, in order of how much they tell you.

**1. Calibration overlay** - `data/runs/<clip>/overlays/calibration_check.mp4`.
The yellow pitch model is reprojected onto the footage. It should track the real
lines as the camera pans and zooms. A small constant offset is expected: it is
the keypoint model's localisation bias, measured at roughly half a metre.

**2. Tracking overlay** - `data/runs/<clip>/overlays/tracking_check.mp4`.
Boxes with track ids, plus a pitch minimap inset showing where each tracked
player was projected. Dots should stay inside the pitch and move smoothly. Track
ids changing on the same player is the known Part 1 limitation.

**3. Heatmaps** - `data/runs/<clip>/heatmaps/`. `player_<id>.png` per tracklet,
`all_tracklets_grid.png` for a contact sheet, `aggregate.png` for all positions.
Long tracklets should show role-shaped patterns rather than noise.

Numbers worth checking, printed by the stages themselves:

| Stage | Healthy on the test clip |
| --- | --- |
| `calibrate` | 5499/5500 frames, median residual 0.52 m |
| `project` | about 86 percent valid, median implied speed under 2 m/s |
| `heatmap` | 405 tracklets, 64 lasting over 20 s |

`project` reports why samples were rejected. `out_of_bounds` is mostly touchline
staff and substitutes; `teleport` marks jumps faster than a player can run,
which usually means the tracker swapped identities.

## Non-players

Detection runs at a low confidence floor to catch every player, so it also picks
up coaches, substitutes, officials and crowd. Filtering happens downstream:

- Crowd flickers disappear with the minimum tracklet length.
- Coaches and touchline staff are dropped by the on-pitch median test in the
  heatmap stage. Pass `--keep-off-pitch` to see them.
- Referees and assistants survive both tests, because they really are on the
  pitch and moving like players. Separating them needs appearance rather than
  geometry, which arrives with team-colour clustering and a detector that has a
  referee class.

Be careful reading heatmaps until then: on the test clip the longest and
cleanest tracklet is the referee, not a player. Officials are easier to track
than players because their kit is distinct and they are rarely occluded.

## Status

Part 1 produces heatmaps per *tracklet*, not per player: the current tracker
fragments identities (405 tracklets for about 22 players), so one player may
span several tracklets. Identity consolidation is the next milestone and is now
the main accuracy bottleneck.
