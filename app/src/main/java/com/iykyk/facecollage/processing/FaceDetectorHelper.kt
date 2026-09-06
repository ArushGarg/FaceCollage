package com.iykyk.facecollage.processing

import android.graphics.Bitmap
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlinx.coroutines.tasks.await

/**
 * Thin coroutine wrapper around ML Kit's on-device face detector.
 *
 * PERFORMANCE_MODE_ACCURATE + classification + landmarks so we get
 * smiling / eyes-open probabilities and head Euler angles for free —
 * these directly feed ShotScorer, so we don't need a second attribute model.
 */
class FaceDetectorHelper {

    private val detector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .setMinFaceSize(0.15f) // ignore tiny/background faces
            .enableTracking()
            .build()
    )

    suspend fun detect(bitmap: Bitmap): List<Face> {
        val input = InputImage.fromBitmap(bitmap, 0)
        return try {
            detector.process(input).await()
        } catch (e: Exception) {
            emptyList()
        }
    }

    fun close() = detector.close()
}
