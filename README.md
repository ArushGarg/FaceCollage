# Face Collage

Native Android app (Kotlin) that processes a portrait video on-device, detects faces,
identifies the same person across separate appearances, picks a strong representative
shot per person, and builds a shareable collage.

## Demo video

Google Drive link (view-only, "Anyone with the link can view"):
**[INSERT YOUR DRIVE LINK HERE]**

## Build & run

1. Open the project root in Android Studio (Koala or newer) and let it sync — it will
   generate the Gradle wrapper on first sync. If you're building from the CLI instead
   and don't have a wrapper yet, run `gradle wrapper` once (requires Gradle 8.x
   installed locally), then use `./gradlew`.
2. `./gradlew assembleDebug`, or just Run from Android Studio on a device/emulator
   running API 26+.
3. In the app, tap **Pick a portrait video** and choose one of the three sample clips
   (or any similar portrait video — nothing about the pipeline is hardcoded to the
   sample set). Repeat for all three clips; each gets its own progress card and,
   once done, its own collage with Save/Share buttons.

No backend, no network calls — everything (frame extraction, detection, embedding,
clustering, collage rendering) runs on-device.

## Pipeline

video -> sampled frames (MediaMetadataRetriever, ~150ms steps, downscaled to 720px)
-> ML Kit face detection (bounding box, tracking ID, pose, eyes-open, smiling)
-> absolute-pixel size floor (110px) to drop faces too small to render legibly
-> TFLite FaceNet embedding per detected face
-> group frames into continuous "appearances" by ML Kit tracking ID (+ gap split)
-> cluster appearances into people by embedding similarity
-> pick best-scoring frame per person as the representative shot
-> brighten the shot if it's genuinely dark
-> render collage (one tile per person, generous crop, appearance-count badge)


All of it runs on `Dispatchers.Default`/`IO` behind a `Flow`; the UI only ever
collects progress events, so the main thread stays free the whole time (verified by
scrolling/interacting with the list while a video is mid-process).

**Memory:** a 30s clip sampled at 150ms is ~200 frames, and every appearance/cluster
needs its source frames retained until the representative shot is picked. At native
portrait resolution (e.g. 1080x1920, ~8MB per ARGB_8888 bitmap) that's well over a
gigabyte resident — a guaranteed OOM. `VideoFrameExtractor` downscales every decoded
frame to a 720px longest side before it's kept anywhere in the pipeline; that's still
far more resolution than the 160×160 embedder input or ML Kit detection needs.

### 1. Face detection — ML Kit

`FaceDetectorOptions` configured with `PERFORMANCE_MODE_ACCURATE`,
`LANDMARK_MODE_ALL`, `CLASSIFICATION_MODE_ALL`, and `enableTracking()`. This gives
us, per face per frame, for free: bounding box, head Euler angles (yaw/roll),
eyes-open probability, smiling probability, and a tracking ID — which is what makes
the "continuous visible segment" appearance definition tractable without writing a
custom tracker. `setMinFaceSize(0.15f)` filters out ML Kit's own smallest/lowest-
confidence detections up front.

On top of ML Kit's own relative size check, `VideoProcessor` additionally drops any
detected face whose bounding box is under 110px in either dimension (in our
downscaled 720px frame space). This is a separate, absolute floor: a face can be
"real" by ML Kit's relative threshold yet still be so small that, once cropped and
stretched ~5x to fill a collage tile, it reads as a low-detail, near-flat patch —
sharp and well-exposed by every metric we compute on the small source crop, but
visually poor once enlarged. 110px was tuned empirically after tracing exactly this
failure mode on real footage (see "Known limitations" for the honest caveat).

### 2. Face embeddings — on-device model

**Model:** FaceNet (Inception-ResNet v1), 512-dimensional output, int8-quantized
TFLite export, bundled at `app/src/main/assets/face_embedder.tflite` (~24MB).
Provenance and license documented in `NOTICE.md`. Input: 160x160 RGB, normalized to
`(pixel - 127.5) / 128.0`. Output embeddings are L2-normalized in `FaceEmbedder.kt`,
so cosine similarity reduces to a dot product.

This is a widely-used, freely-licensed (Apache 2.0) mobile FaceNet export rather
than something trained from scratch — the assignment scopes this as "use an
on-device embedding model," not train one.

Each detected face is cropped from the **full frame** with a 25% margin around the
ML Kit bounding box before being resized to 160x160 for embedding — a tight bbox
crop loses context and can clip the chin/forehead, which hurts embedding quality
more than it helps.

**Interpreter runs single-threaded** (`setNumThreads(1)`), deliberately, not for
speed. Multi-threaded TFLite reduction ops sum in a non-deterministic order,
producing tiny run-to-run differences in embedding values. Since the clustering
threshold sits on a narrow, empirically-found boundary, even a tiny embedding shift
on one borderline appearance can flip its cluster assignment — and because
clustering is greedy/incremental, that one flip cascades into a different result
for everything processed after it. Single-threaded inference trades a bit of speed
for the same video producing the same result on repeated runs.

### 3. Clustering — appearances, not raw frames

Rather than clustering every individual frame detection (noisy — a single blurry
frame gets its own bad embedding), `AppearanceTracker` first groups frames into
continuous segments by ML Kit tracking ID, splitting on a >1000ms gap (in case a
tracking ID is stale/reused after occlusion). Each appearance's representative
embedding is the mean of its 5 sharpest frames' embeddings, L2-renormalized.

`FaceClusterer` then does greedy incremental clustering over appearances in
chronological order: each appearance joins the most similar existing cluster if
cosine similarity clears the threshold, else starts a new cluster, updating a
running centroid. This is O(appearances × clusters) rather than full O(n²)
agglomerative clustering, and is easy to reason about against the worked example
in the brief.

**Similarity threshold: 0.55.** Tuned empirically by direct observation against
Sample 1's manually-confirmed ground truth (5 distinct people): 0.62 over-fragmented
into as many as 12 clusters; 0.45 over-merged down to 4; 0.57 over-fragmented again
to 7. 0.55 consistently lands at or very near 5 across repeated runs. This is the
one constant (`FaceClusterer(similarityThreshold = 0.55f)`) to retune first if a
specific clip under- or over-merges — raise it if distinct people are getting
merged into one, lower it if one person is splitting into multiple clusters.

### 4. Representative shot selection

`ShotScorer` combines five signals per candidate frame, weighted:

| Signal | Weight | How it's computed |
|---|---|---|
| Frontality | 0.30 | ML Kit head Euler yaw/roll — penalizes profile/tilted shots |
| Sharpness | 0.25 | Laplacian-variance on a 64×64 grayscale downsample of the face crop — this is what disqualifies motion-blurred whip-pan frames from ever winning |
| Exposure | 0.20 | Mean luminance of the same downsample, peaking at a comfortable midtone and decaying toward 0 for near-black or blown-out frames |
| Eyes open | 0.15 | ML Kit eyes-open probability, averaged L/R |
| Smiling | 0.10 | ML Kit smiling probability |

The exposure term exists because frontality/eyes/smile alone let a dark or backlit
frame win by default when it's the best option within a person's tracked frames —
none of those three signals detect "this frame is basically black."

The winning frame across *all* of a person's appearances (not just one) is cropped
generously (40% margin around the bbox, not a tight face-only crop) for the final
collage tile. Margin was originally 70%, but at that size the crop sometimes
extended past the actual face into either the frame edge or genuine black
letterboxing baked into the source footage, producing tiles that were half face,
half black. 40% is still well beyond a tight bbox crop while staying clear of edges
in most cases.

If the winning shot is still genuinely dark after scoring (i.e. every frame
available for that person is similarly dark — there's no better-lit option to pick
instead), `brightenIfDark` applies a flat RGB offset sized to the shortfall from a
target luminance. This corrects the *presentation* of the real chosen frame; it
does not select a different frame or fabricate content.

### 5. Appearance counting

An appearance = one continuous tracking-ID segment (see above). Two people sharing
a frame naturally count as one appearance each, since they get separate tracking
IDs. A whip-pan blur frame either fails detection entirely or scores near-zero on
sharpness and never becomes a representative shot, but if ML Kit does detect and
track through it, it still correctly counts as part of whichever appearance
segment it falls inside — it just never *wins* as the shown shot.

A cluster built from a single low-frame-count detection (`MIN_FACE_DETECTIONS = 2`
in `VideoProcessor`) is filtered out before rendering, on the reasoning that a real,
continuously-tracked person accumulates multiple detected frames across their
appearance(s); a cluster with only one is more likely a stray false-positive than a
real person.

### 6. Collage

`CollageBuilder` lays out a near-square grid (Instagram Story multi-photo style),
rounded tiles, dark background, and a small "N×" appearance-count badge per tile so
the count is visible on the shareable image itself, not just in the app's list UI.
Center-crop (not stretch) fits each generously-cropped shot into its square tile.

## Known limitations / what I'd improve with more time

- **Residual clustering variance.** Even with the embedder forced single-threaded,
  ML Kit's own bundled face detector runs its own internal TFLite interpreter with
  threading that isn't configurable via the public API. This can produce small
  run-to-run differences in detection confidence that occasionally flip a borderline
  clustering decision — observed as ±1 person on repeated runs of the same clip at
  the same threshold. This originates upstream of application code and can't be
  fully eliminated from within it.
- **Small-face floor is a real trade-off.** The 110px absolute size floor fixes
  visibly poor tiles from faces too small to enlarge cleanly, but it also means a
  person who is only ever captured small/distant in a clip may be filtered out
  entirely rather than shown poorly. There's no size threshold that's correct for
  every video; 110px was tuned against the specific sample clips.
- **Genuinely dark or single-brief-appearance people.** Brightening helps a dark
  shot become legible but can't recover detail that was never captured, and a person
  with only one brief appearance gets whatever pose/angle exists in that single
  moment — the scorer picks the best available frame, not an invented better one.
- Clustering is greedy/incremental rather than full agglomerative — good enough at
  30s-clip scale but would need revisiting for longer videos with many identities.
- No re-identification across separate video files (each of the three clips is
  clustered independently, as the brief's per-video collage requirement implies).
- Frame sampling at ~150ms is a fixed interval; adaptive sampling (denser during
  detected motion) would catch more/better candidate frames per appearance without
  processing every frame.
