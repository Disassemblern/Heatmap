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
video -> detect -> calibrate -> track -> team -> refine -> project -> heatmap
                                                                  \-> overlay / preview
```

| Stage | Output | Purpose |
| --- | --- | --- |
| `detect` | `detections.parquet` | Per-frame player boxes, stored at a low confidence floor |
| `track` | `tracks.parquet`, `tracks_mot.txt`, `embeddings.npz` | Track ids across frames, with appearance embeddings |
| `team` | `team.parquet`, `team_features.npz` | Tracklets clustered into two teams by jersey colour |
| `refine` | `tracks.parquet` (rewritten), `tracks_raw.parquet` | Offline split of hidden id switches, reconnect of fragments |
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
hm_env\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Models

Three weight files belong in `models/`, which is not tracked:

| File | Used by | What it is | Licence |
| --- | --- | --- | --- |
| `yolov8m.pt` | `detect` | Ultralytics YOLOv8m, COCO `person` class | AGPL-3.0 |
| `yolo-football-pitch-detection.pt` | `calibrate` | YOLOv8x-pose fine-tuned for 32 pitch keypoints | AGPL-3.0 |
| `osnet_sports.pth.tar-60` | `track` (deep_eiou), `refine` | OSNet ReID trained on SportsMOT, 616 identities | none published |

`yolov8m.pt` downloads itself on first run. The other two are manual, and
`calibrate`, `track` and `refine` all fail without them.

**Pitch keypoints** - [martinjolif/yolo-football-pitch-detection](https://huggingface.co/martinjolif/yolo-football-pitch-detection)
on Hugging Face. The keypoint index to pitch-metre mapping this repo fits
against is in `src/pitchmap/calib/keypoints.py`, and it uses regulation
distances (16.5 m penalty box), not the 20.15 m in some published configs for
the same layout.

```powershell
curl.exe -L -o models\yolo-football-pitch-detection.pt https://huggingface.co/martinjolif/yolo-football-pitch-detection/resolve/main/yolo-football-pitch-detection.pt
```

**ReID** - the `sports_model.pth.tar-60` checkpoint from
[hsiangwei0903/Deep-EIoU](https://github.com/hsiangwei0903/Deep-EIoU), served
from the [Google Drive folder](https://drive.google.com/drive/folders/1wItcb0yeGaxOS08_G9yRWBTnpVf0vZ2w)
its README links. Download it by hand and save it as
`models/osnet_sports.pth.tar-60` - this repo uses the renamed copy, so the
rename is not optional. It publishes no licence; it is used here for personal,
non-commercial work only, like the vendored tracker it comes with (see
`src/pitchmap/vendor/README.md`).

## Usage

```powershell
hm_env\Scripts\python.exe -m pitchmap init-run  --video data\raw\fixed_clip.mp4
hm_env\Scripts\python.exe -m pitchmap detect    --run data\runs\fixed_clip
hm_env\Scripts\python.exe -m pitchmap calibrate --run data\runs\fixed_clip
hm_env\Scripts\python.exe -m pitchmap track     --run data\runs\fixed_clip
hm_env\Scripts\python.exe -m pitchmap team      --run data\runs\fixed_clip
hm_env\Scripts\python.exe -m pitchmap refine    --run data\runs\fixed_clip
hm_env\Scripts\python.exe -m pitchmap project   --run data\runs\fixed_clip --smooth
hm_env\Scripts\python.exe -m pitchmap heatmap   --run data\runs\fixed_clip
hm_env\Scripts\python.exe -m pitchmap overlay   --run data\runs\fixed_clip --kind track
```

Settings live in `configs/default.yaml`; most stages accept flags that override
the relevant entries for a single run.

## Layout

```
configs/    pipeline settings
data/       raw clips and per-clip run directories (not tracked)
docs/notes/ working notes and research links
models/     model weights (not tracked, see Models)
src/pitchmap/
  cli/      one thin entrypoint per stage
  io/       video reading, run artifacts, MOT format
  detect/   YOLO player detection
  track/    tracker implementations behind a shared protocol
  calib/    pitch model, landmark clicking, homography fit
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
hm_env\Scripts\python.exe -m pitchmap detect --run data\runs\fixed_clip --show
hm_env\Scripts\python.exe -m pitchmap track  --run data\runs\fixed_clip --show
```

Press `q` to stop early; whatever was processed is still saved.

To replay a finished run without re-running any model, which is much faster and
can be paused:

```powershell
hm_env\Scripts\python.exe -m pitchmap preview --run data\runs\fixed_clip --kind track
hm_env\Scripts\python.exe -m pitchmap preview --run data\runs\fixed_clip --kind detect
hm_env\Scripts\python.exe -m pitchmap preview --run data\runs\fixed_clip --kind calib
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
player was projected. Once the team stage has run, boxes and minimap dots are
drawn in each team's own average kit colour, so a mislabelled player shows up as
a box whose colour disagrees with the shirt inside it. Watch the id numbers
across occlusions: they should survive players crossing.

**3. Heatmaps** - `data/runs/<clip>/heatmaps/`. `player_<id>.png` per tracklet,
`all_tracklets_grid.png` for a contact sheet, `aggregate.png` for all positions.
Long tracklets should show role-shaped patterns rather than noise.

Numbers worth checking, printed by the stages themselves:

| Stage | Healthy on the test clip |
| --- | --- |
| `calibrate` | 5499/5500 frames, median residual 0.52 m |
| `track` | 188 tracklets, median 8.1 s, coverage 0.83 |
| `team` | fitted on 88 of 188, an even A/B split (50/48), 38 `other` |
| `refine` | 82 tracklets, 61 over 20 s, median 40 s, coverage 0.83 |
| `project` | about 88 percent valid, median implied speed under 2 m/s |
| `heatmap` | 50 rendered of 82, officials and touchline staff skipped |

Every stage records what it actually ran with in `data/runs/<clip>/meta.json`,
under its own entry. The `config` it stamps there lists only what differed from
`configs/default.yaml`, so an empty one means the stage ran on the defaults.

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

## Team assignment

Tracklets are clustered into two teams by the median colour of the player torso,
with pitch-green pixels masked out. Clustering runs in chroma (Lab without
lightness) when that separates the kits better than full Lab, which it usually
does: lightness mostly follows shadow and floodlighting rather than kit identity.
On the test clip chroma scores 0.63 against 0.52 for full Lab.

A tracklet whose distance to the nearer centroid exceeds `other_dist_ratio` of
the gap between the two centroids belongs to neither team and is labelled
`other`: referees, goalkeepers and staff. The threshold is relative so it does
not depend on the feature space or on how contrasting the two kits are. On the
test clip real players reach 0.48 and officials start at 0.73.

A wrong team colour is usually not a colour problem. It means the tracklet
contains more than one person, so no single label can be right. Tracklet purity
- how consistently the rows of a tracklet vote for one team - is therefore the
metric that matters, and it is what the split threshold was tuned against.

## Tracking

Identity is held by Deep-EIoU (sports-specific association plus OSNet appearance
embeddings trained on sports footage), then repaired offline: `refine` splits
tracklets whose embeddings cluster into more than one identity, and reconnects
fragments that belong to the same person, refusing any merge between tracklets
confidently assigned to different teams.

Measured on the test clip:

| Stage | Tracklets | Median length |
| --- | --- | --- |
| ByteTrack (previous) | 405 | 4.8 s |
| Deep-EIoU | 188 | 8.1 s |
| after `refine` | 82 | 40.2 s |

`refine` always works from `tracks_raw.parquet`, so re-running it with different
parameters is idempotent and never compounds. `track` deletes the refine and
team artifacts, since new tracks invalidate both.

## Measuring identity quality

Automatic checks on team labels reuse the same jersey colours that produced
them, so they cannot see a swap between two players of the same team. The
`review` and `eval` stages add a human-judged measurement that can.

```powershell
hm_env\Scripts\python.exe -m pitchmap review --run data\runs\fixed_clip
hm_env\Scripts\python.exe -m pitchmap eval   --run data\runs\fixed_clip
```

`review` renders each tracklet as a strip of crops spanning its life and writes
a CSV pre-filled with the pipeline's own guesses. Judging is then reading a few
images and correcting two columns: set `verdict` to `mixed` where a strip shows
more than one person, and fix `team` where the label is wrong. `eval` scores the
result.

Measured on the test clip, over the 61 tracklets long enough to judge: 92
percent are one person (94 percent by duration), and team labels agree with the
human on 95 percent of the 56 that carry one.

Judgements are keyed to track ids, and refine renumbers those on every run, so
`review` stores a fingerprint of the tracklets it described and `eval` refuses to
score against a different set. Re-run `review --overwrite` after changing
anything upstream.

## Status

Heatmaps are roughly one per player rather than per fragment, and officials are
excluded by default.

Measured against human judgement of the test clip: 92 percent of tracklets are a
single person, 95 percent carry the right team label. The three remaining label
errors are genuinely ambiguous - a dark coat against the navy kit - and are not
fixable by threshold tuning: accuracy is flat from 0.50 to 1.10.

Remaining gaps: some players still split into two tracklets across the clip, and
naming tracklets from the team sheet is still a manual step, by design.
