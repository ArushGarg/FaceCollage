package com.iykyk.facecollage.model

import android.graphics.Bitmap
import android.graphics.RectF

/**
 * One face detected in one sampled video frame.
 *
 * [frameBitmap] is the *full* decoded frame (not a tight face crop) — kept only
 * long enough to compute [ShotScorer] metrics and, if this face wins as a
 * representative shot, to crop generously for the collage. Released otherwise.
 */
data class FrameFace(
    val timestampMs: Long,
    val boundingBox: RectF,
    val headEulerAngleY: Float,   // yaw; 0 = looking straight at camera
    val headEulerAngleZ: Float,   // roll
    val leftEyeOpenProbability: Float?,
    val rightEyeOpenProbability: Float?,
    val smilingProbability: Float?,
    val embedding: FloatArray,
    val frameBitmap: Bitmap
) {
    override fun equals(other: Any?) = this === other
    override fun hashCode() = System.identityHashCode(this)
}

/** A continuous visible run of the same person — what the brief calls "one appearance". */
data class Appearance(
    val startMs: Long,
    val endMs: Long,
    val faces: List<FrameFace>
)

/** All appearances in one video attributed to a single identity/cluster. */
data class PersonTrack(
    val clusterId: Int,
    val appearances: MutableList<Appearance> = mutableListOf()
) {
    val appearanceCount: Int get() = appearances.size
    fun allFaces(): List<FrameFace> = appearances.flatMap { it.faces }
}

/** Final chosen representative shot for a person, ready to place in the collage. */
data class RepresentativeShot(
    val clusterId: Int,
    val appearanceCount: Int,
    val crop: Bitmap,
    val score: Float
)

sealed class ProcessingStage(val label: String) {
    data object ExtractingFrames : ProcessingStage("Extracting frames")
    data object DetectingFaces : ProcessingStage("Detecting faces")
    data object Embedding : ProcessingStage("Computing face embeddings")
    data object Clustering : ProcessingStage("Grouping identities")
    data object BuildingCollage : ProcessingStage("Building collage")
    data object Done : ProcessingStage("Done")
}

data class ProcessingProgress(
    val stage: ProcessingStage,
    val current: Int = 0,
    val total: Int = 0
)

data class VideoResult(
    val videoLabel: String,
    val people: List<RepresentativeShot>,
    val collage: Bitmap
)
