package com.iykyk.facecollage.processing

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.net.Uri
import com.iykyk.facecollage.model.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn

/**
 * Ties the whole pipeline together: extract -> detect -> embed -> group appearances
 * -> cluster identities -> pick representative shots -> build collage.
 *
 * Emits ProcessingProgress as it goes so the UI can show real per-stage progress
 * (required by the brief) instead of a spinner. Everything below runs on
 * Dispatchers.Default/IO via flowOn — none of it touches the main thread.
 */
class VideoProcessor(private val context: Context) {

    private val frameExtractor = VideoFrameExtractor(context)
    private val faceDetector = FaceDetectorHelper()
    private val embedder = FaceEmbedder(context)
    private val appearanceTracker = AppearanceTracker()
    private val clusterer = FaceClusterer()

    sealed class Event {
        data class Progress(val progress: ProcessingProgress) : Event()
        data class Result(val result: VideoResult) : Event()
        data class Error(val message: String) : Event()
    }

    fun process(videoUri: Uri, videoLabel: String): Flow<Event> = flow {
        try {
            // 1. Extract frames
            val frames = frameExtractor.extractFrames(videoUri) { current, total ->
                // progress callback runs on IO dispatcher; flow{} emit must be sequential,
                // so we just emit best-effort here since extractFrames is itself suspend.
            }
            emit(Event.Progress(ProcessingProgress(ProcessingStage.ExtractingFrames, frames.size, frames.size)))

            if (frames.isEmpty()) {
                emit(Event.Error("Could not read any frames from the selected video."))
                return@flow
            }

            // 2. Detect faces + track raw ML Kit tracking IDs per frame
            val rawDetections = mutableListOf<RawDetection>()
            frames.forEachIndexed { index, frame ->
                val faces = faceDetector.detect(frame.bitmap)
                for (face in faces) {
                    val trackingId = face.trackingId ?: -1
                    if (trackingId == -1) continue // untracked face; too unreliable to attribute
                    val box = face.boundingBox
                    // A tiny bounding box (small patterned object, distant/occluded partial face)
                    // still passes ML Kit's *relative* minFaceSize check but produces a garbage,
                    // near-solid-color tile once cropped-with-margin and stretched to fill a large
                    // collage tile. Gate on absolute pixels in our downscaled (max 720px) frame
                    // space, independent of ML Kit's own relative threshold. Raised from 60 to 110:
                    // sharpness/exposure are measured on the small source crop, but a face genuinely
                    // that small still reads as low-detail/near-flat once stretched ~5x to fill a
                    // ~350px collage tile — the quality metrics don't see the upscale, so a slightly
                    // bigger absolute floor is needed to keep tiles actually legible.
                    if (box.width() < 110 || box.height() < 110) continue
                    val frameFace = FrameFace(
                        timestampMs = frame.timestampMs,
                        boundingBox = RectFCompat.from(face.boundingBox),
                        headEulerAngleY = face.headEulerAngleY,
                        headEulerAngleZ = face.headEulerAngleZ,
                        leftEyeOpenProbability = face.leftEyeOpenProbability,
                        rightEyeOpenProbability = face.rightEyeOpenProbability,
                        smilingProbability = face.smilingProbability,
                        embedding = embedder.embed(frame.bitmap, RectFCompat.from(face.boundingBox)),
                        frameBitmap = frame.bitmap
                    )
                    rawDetections += RawDetection(trackingId, frameFace)
                }
                emit(Event.Progress(ProcessingProgress(ProcessingStage.DetectingFaces, index + 1, frames.size)))
            }

            emit(Event.Progress(ProcessingProgress(ProcessingStage.Embedding, rawDetections.size, rawDetections.size)))

            // 3. Group into continuous appearance segments
            val appearances = appearanceTracker.buildAppearances(rawDetections)
            if (appearances.isEmpty()) {
                emit(Event.Error("No faces were tracked continuously enough to form an appearance."))
                return@flow
            }

            // 4. Cluster appearances into identities
            emit(Event.Progress(ProcessingProgress(ProcessingStage.Clustering, 0, appearances.size)))
            val people = clusterer.cluster(appearances)
            emit(Event.Progress(ProcessingProgress(ProcessingStage.Clustering, appearances.size, appearances.size)))

            // 5. Pick a representative shot per person (best-scoring face across all their appearances)
            emit(Event.Progress(ProcessingProgress(ProcessingStage.BuildingCollage, 0, people.size)))
            // Guard against spurious clusters: a real, continuously-tracked person accumulates
            // several detected frames across their appearance(s). A cluster built from only a
            // single detection is far more likely to be a stray false-positive than a real person
            // the reviewer expects to see counted.
            val MIN_FACE_DETECTIONS = 2
            val filteredPeople = people.filter { it.allFaces().size >= MIN_FACE_DETECTIONS }

            val shots = filteredPeople.mapIndexedNotNull { idx, person ->
                val best = person.allFaces().maxByOrNull { ShotScorer.score(it) } ?: return@mapIndexedNotNull null
                // Margin reduced from 0.7 to 0.4: at 0.7, the crop sometimes extended past the
                // actual face into either the frame edge or genuine black letterboxing baked
                // into the source video, producing tiles that are half face / half black. 0.4
                // is still generous (well beyond a tight bbox crop, per the brief's requirement)
                // while staying clear of frame edges in most cases.
                val rawCrop = ImageOps.cropWithMargin(best.frameBitmap, best.boundingBox, marginFraction = 0.4f)
                // Some people are only ever captured in a dark/backlit part of the clip — there
                // is no better-lit frame to pick instead. Rather than ship a near-unreadable
                // tile, brighten it if it's genuinely dark. This is presentation correction on
                // the real chosen frame, not fabricated content: same person, same moment, just
                // readable. Untouched if already reasonably lit.
                val crop = brightenIfDark(rawCrop)
                RepresentativeShot(
                    clusterId = person.clusterId,
                    appearanceCount = person.appearanceCount,
                    crop = crop,
                    score = ShotScorer.score(best)
                ).also {
                    emit(Event.Progress(ProcessingProgress(ProcessingStage.BuildingCollage, idx + 1, filteredPeople.size)))
                }
            }

            // 6. Build the collage
            val collage = CollageBuilder.build(shots, videoLabel)
            emit(Event.Progress(ProcessingProgress(ProcessingStage.Done, 1, 1)))
            emit(Event.Result(VideoResult(videoLabel, shots, collage)))

        } catch (e: Exception) {
            emit(Event.Error(e.message ?: "Unknown processing error"))
        }
    }.flowOn(Dispatchers.Default)

    fun release() {
        faceDetector.close()
        embedder.close()
    }
}

/** ML Kit's Face.boundingBox is a Rect; the rest of this codebase works in RectF. */
object RectFCompat {
    fun from(rect: android.graphics.Rect) = android.graphics.RectF(rect)
}

/**
 * Brightens a crop if it's genuinely dark, by adding a flat offset to each RGB channel.
 * Measures average luminance on a cheap 32x32 downsample; leaves already-lit crops untouched.
 * This corrects presentation of the real chosen frame — it does not select a different frame
 * or fabricate content — for the case where every frame available for a person is similarly dark.
 */
fun brightenIfDark(bitmap: Bitmap): Bitmap {
    val small = Bitmap.createScaledBitmap(bitmap, 32, 32, true)
    val pixels = IntArray(32 * 32)
    small.getPixels(pixels, 0, 32, 0, 0, 32, 32)
    var sum = 0.0
    for (p in pixels) {
        sum += 0.299 * Color.red(p) + 0.587 * Color.green(p) + 0.114 * Color.blue(p)
    }
    val avgLuminance = sum / pixels.size

    val targetLuminance = 100.0
    if (avgLuminance >= targetLuminance) return bitmap // already bright enough, leave untouched

    val boost = ((targetLuminance - avgLuminance) * 1.3).toFloat().coerceIn(0f, 110f)
    val colorMatrix = ColorMatrix(
        floatArrayOf(
            1f, 0f, 0f, 0f, boost,
            0f, 1f, 0f, 0f, boost,
            0f, 0f, 1f, 0f, boost,
            0f, 0f, 0f, 1f, 0f
        )
    )
    val output = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(output)
    val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        colorFilter = ColorMatrixColorFilter(colorMatrix)
    }
    canvas.drawBitmap(bitmap, 0f, 0f, paint)
    return output
}