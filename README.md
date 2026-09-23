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
| `cut` | `data/raw/<clip>_p01.mp4`, ... | Optional: split a long recording into clips processed one at a time |
| `detect` | `detections.parquet` | Per-frame player boxes, stored at a low confidence floor |
| `track` | `tracks.parquet`, `tracks_mot.txt`, `embeddings.npz` | Track ids across frames, with appearance embeddings |
| `team` | `team.parquet`, `team_features.npz` | Tracklets clustered into two teams by jersey colour |
| `refine` | `tracks.parquet` (rewritten), `tracks_raw.parquet` | Offline split of hidden id switches, reconnect of fragments |
| `annotate` | `calib/keyframes.json` | Optional manual landmark clicks, used to cross-check calibration |
| `calibrate` | `calib/homographies.npz` | Per-frame pixel-to-metre homography from detected pitch keypoints |
| `project` | `positions.parquet` | Foot points projected into pitch metres |
| `heatmap` | `heatmaps/*.png` | Per-tracklet occupancy maps on a pitch |
| `overlay` | `overlays/*.mp4` | Calibration and tracking sanity videos |
| `combine` | a directory of `player_<name>.png` | One heatmap per named player, pooled across several runs |

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
| `yolo-football-player-detection.pt` | `detect` | YOLO11m, classes player / goalkeeper / referee / ball | AGPL-3.0 |
| `yolo-football-pitch-detection.pt` | `calibrate` | YOLOv8x-pose fine-tuned for 32 pitch keypoints | AGPL-3.0 |
| `osnet_sports.pth.tar-60` | `track` (deep_eiou), `refine` | OSNet ReID trained on SportsMOT, 616 identities | none published |

None of the three download themselves, and `detect`, `calibrate`, `track` and
`refine` all fail without them.

**Players** - [martinjolif/yolo-football-player-detection](https://huggingface.co/martinjolif/yolo-football-player-detection)
on Hugging Face. A football-specific detector rather than a general one matters
more than it sounds: a COCO `person` model separates players from crowd,
coaches and substitutes only by confidence, and that does not work on a wide
broadcast frame. This one separates them by class. Reported mAP50 is 0.9937 for
players and 0.9888 for referees.

```powershell
curl.exe -L -o models\yolo-football-player-detection.pt https://huggingface.co/martinjolif/yolo-football-player-detection/resolve/main/yolo-football-player-detection.pt
```

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

## Working through a full half

A 45-minute half takes about 4.5 hours on an RTX 3050 Laptop GPU, and one bad
setting means redoing all of it. Cutting it into 5-minute clips lets you check
each piece before moving on, and each piece is just an ordinary clip:

```powershell
hm_env\Scripts\python.exe -m pitchmap cut --video data\raw\half1.mp4 --every 5
```

That writes `half1_p01.mp4`, `half1_p02.mp4`, ... next to the recording and
prints the time range each one covers. For a single range instead, use
`--start 10:00 --end 15:00`. The video stream is copied, not re-encoded, so it
is instant and the detector sees identical pixels; cuts land on the nearest
keyframe, within a second or two of the requested time.

Then, per clip:

```powershell
$clip = "half1_p01"
& {
    function pm { hm_env\Scripts\python.exe -m pitchmap @args; if ($LASTEXITCODE) { throw "pitchmap $args failed" } }
    pm init-run  --video data\raw\$clip.mp4
    pm detect    --run data\runs\$clip
    pm calibrate --run data\runs\$clip
    pm track     --run data\runs\$clip
    pm team      --run data\runs\$clip
    pm refine    --run data\runs\$clip
    pm project   --run data\runs\$clip --smooth
    pm heatmap   --run data\runs\$clip
    pm review    --run data\runs\$clip
}
```

The block stops at the first stage that fails, so later stages never run on
missing files. A 5-minute clip takes about 30 minutes.

Check it with `preview --kind track`, then open `data\runs\<clip>\review.csv`
next to the contact sheets in `review\` and type a name or shirt number in the
`player` column for each tracklet you recognise. Blank rows are skipped; set
`verdict` to `mixed` for any strip that shows two different people.

Track ids are numbered from scratch in every run, so id 5 in one clip has
nothing to do with id 5 in the next. The names are what link the clips:

```powershell
hm_env\Scripts\python.exe -m pitchmap combine --out data\combined\half1 `
    --runs data\runs\half1_p01 data\runs\half1_p02 data\runs\half1_p03
```

`combine` pools the positions of every tracklet with the same name (case does
not matter) and draws one heatmap per player, so each clip counts in proportion
to how long the player was seen in it. It refuses a run whose `review.csv` was
written against different tracklets, which happens whenever `refine` re-runs.
Pool one half at a time: teams change ends at half time.

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
| `detect` | about 28 per frame: 23 player, 2.4 referee, 1.2 ball, 1.0 goalkeeper |
| `calibrate` | 5499/5500 frames, median residual 0.52 m |
| `track` | 255 raw tracklets, coverage 0.97 |
| `team` | an even A/B split (30/20 after refine), silhouette above 0.5 |
| `refine` | 62 tracklets, 40 over 20 s averaging 112 s, coverage 0.97 |
| `project` | about 89 percent valid, median implied speed under 2 m/s |
| `heatmap` | 35 rendered of 62, officials and touchline staff skipped |

Read `track`'s raw tracklet count with care: most of those 255 are single-frame
specks holding 0.3 percent of the rows between them, so the median length is a
meaningless statistic on that table. What matters is coverage, and how much data
ends up in tracklets long enough to use - after `refine`, 96 percent of valid
positions sit in the 40 tracklets lasting 20 s or more.

Every stage records what it actually ran with in `data/runs/<clip>/meta.json`,
under its own entry. The `config` it stamps there lists only what differed from
`configs/default.yaml`, so an empty one means the stage ran on the defaults.

`project` reports why samples were rejected. `out_of_bounds` is mostly touchline
staff and substitutes; `teleport` marks jumps faster than a player can run,
which usually means the tracker swapped identities.

## Non-players

The detector classifies rather than just locating, so most of this is settled
before tracking starts:

- **Referees** are their own class. `track` keeps only the classes in
  `track.classes` (player and goalkeeper by default) and skips the rest, so an
  official never becomes a tracklet. This used to be the worst trap in the
  pipeline: the longest, cleanest tracklet on the test clip was the referee, not
  a player, because officials wear distinct kit and are rarely occluded.
- **Crowd** is not a class the model has, so it is largely not detected at all.
  Whatever leaks through disappears with the minimum tracklet length.
- **Coaches and touchline staff** are dropped by the on-pitch median test in the
  heatmap stage. Pass `--keep-off-pitch` to see them.
- **Substitutes** are the remaining hard case. The model calls a warming-up
  substitute a player, correctly, and their training bib is neither team's kit,
  so they land in the `other` colour cluster rather than being excluded by class.

Detection still runs at a low confidence floor and stores every class, including
the referees and the ball it was told not to track. Changing your mind about
referees therefore costs a `track` re-run, not a `detect` re-run:

```powershell
hm_env\Scripts\python.exe -m pitchmap track --run data\runs\fixed_clip --classes player goalkeeper referee
```

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

| Setup | Tracklets >= 20 s | Mean length of those | Coverage |
| --- | --- | --- | --- |
| COCO `person` detections | 61 | 66 s | 0.83 |
| football detections | 51 | 88 s | 0.97 |
| football detections, 1 s split rule | 40 | 112 s | 0.97 |

The same 19 players' worth of data ends up in fewer, longer pieces: about 2.1
tracklets per player on the test clip.

The split rule matters more than its size suggests. `refine` finds identity
switches by clustering a tracklet's appearance embeddings, and it used to assign
each frame to a cluster on its own. One player's embedding wobbles with pose,
motion blur and shadow, so neighbouring frames alternated between two clusters
and the player became two ids flickering back and forth, even with nobody near
them. A real switch is sustained - the tracker moved to someone else and stayed
there - so a split now has to hold for `refine.min_split_s` (1 s): labels are
majority-voted over that window and shorter runs are absorbed. For players in
open space, frame-to-frame id switches fell from 3.67 to 0.28 percent, and
tracklets stayed as pure. Most switches left happen where players overlap, which
no appearance model can fully resolve. `refine --min-split 0` restores the old
behaviour for comparison.

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

With the earlier COCO `person` detector, human judgement of the test clip found
92 percent of tracklets to be one person and 95 percent of team labels right.
The detector and the split rule have both changed since, so those numbers are
unverified until the current contact sheets are judged again.

Judgements are keyed to track ids, and refine renumbers those on every run, so
`review` stores a fingerprint of the tracklets it described and `eval` refuses to
score against a different set. Re-run `review --overwrite` after changing
anything upstream.

## Status

Per clip, each player is covered by about two tracklets, and officials are
excluded by class before tracking. Naming tracklets in `review.csv` turns those
into exactly one heatmap per player, within a clip or pooled across a half with
`combine`.

Remaining gaps:

- Identity still breaks where players overlap: 3.4 percent of frame-to-frame
  pairs there, against 0.28 percent in open space. Two players in the same kit
  crossing is the case no appearance model can settle.
- Naming is manual, by design: a person recognises a shirt number in seconds,
  and nothing automatic can link two fragments of the same player reliably.
- Human-judged accuracy has not been re-measured since the detector changed.
