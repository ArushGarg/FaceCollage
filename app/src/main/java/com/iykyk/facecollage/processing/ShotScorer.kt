package com.iykyk.facecollage.processing

import android.graphics.Bitmap
import android.graphics.Color
import com.iykyk.facecollage.model.FrameFace
import kotlin.math.abs
import kotlin.math.min

/**
 * Scores a candidate frame of a person to pick the single best representative shot.
 * Combines five signals:
 *   - frontality (head pose)
 *   - sharpness (in-focus, not motion-blurred — this is what filters out whip-pans)
 *   - eyes open
 *   - smiling / pleasant expression
 *   - exposure (not a dark/backlit/near-black frame)
 *
 * The exposure term exists because the other four signals alone let a dark or
 * backlit frame win by default when it's the only/best option within a person's
 * tracked frames — none of frontality, eyes-open, or smile detect "this frame is
 * basically black", and a near-zero-variance dark frame can still score non-trivially
 * on those three. Without exposure, a legitimately visible frame elsewhere in the
 * same person's appearances can lose to a dark one that happens to face the camera.
 */
object ShotScorer {

    private const val W_FRONTALITY = 0.30f
    private const val W_SHARPNESS = 0.25f
    private const val W_EYES_OPEN = 0.15f
    private const val W_SMILE = 0.10f
    private const val W_EXPOSURE = 0.20f

    /** Debug-only breakdown of the individual component scores, for diagnosing low overall scores. */
    fun debugBreakdown(face: FrameFace): String {
        val lum = luminanceGrid(face)
        val frontality = frontalityScore(face.headEulerAngleY, face.headEulerAngleZ)
        val sharpness = sharpnessScore(lum)
        val exposure = exposureScore(lum)
        val eyesOpen = eyesOpenScore(face.leftEyeOpenProbability, face.rightEyeOpenProbability)
        val smile = face.smilingProbability ?: 0.5f
        return "frontality=$frontality sharpness=$sharpness exposure=$exposure eyesOpen=$eyesOpen smile=$smile"
    }

    fun score(face: FrameFace): Float {
        val lum = luminanceGrid(face)
        val frontality = frontalityScore(face.headEulerAngleY, face.headEulerAngleZ)
        val sharpness = sharpnessScore(lum)
        val exposure = exposureScore(lum)
        val eyesOpen = eyesOpenScore(face.leftEyeOpenProbability, face.rightEyeOpenProbability)
        val smile = face.smilingProbability ?: 0.5f // neutral prior if ML Kit couldn't tell

        return W_FRONTALITY * frontality +
                W_SHARPNESS * sharpness +
                W_EYES_OPEN * eyesOpen +
                W_SMILE * smile +
                W_EXPOSURE * exposure
    }

    /** 1.0 = dead-on frontal, decaying as yaw/roll grow. ML Kit angles are in degrees. */
    private fun frontalityScore(yaw: Float, roll: Float): Float {
        val yawPenalty = min(abs(yaw) / 45f, 1f)   // >45° yaw treated as fully profile
        val rollPenalty = min(abs(roll) / 30f, 1f)
        return 1f - (0.7f * yawPenalty + 0.3f * rollPenalty)
    }

    private fun eyesOpenScore(left: Float?, right: Float?): Float {
        if (left == null && right == null) return 0.5f
        val l = left ?: right!!
        val r = right ?: left!!
        return (l + r) / 2f
    }

    /** 64x64 grayscale luminance grid of the face crop, shared by sharpness and exposure. */
    private fun luminanceGrid(face: FrameFace): DoubleArray {
        val crop = ImageOps.cropWithMargin(face.frameBitmap, face.boundingBox, 0.15f)
        val small = Bitmap.createScaledBitmap(crop, 64, 64, true)
        val pixels = IntArray(64 * 64)
        small.getPixels(pixels, 0, 64, 0, 0, 64, 64)
        return DoubleArray(64 * 64) { i ->
            val p = pixels[i]
            0.299 * Color.red(p) + 0.587 * Color.green(p) + 0.114 * Color.blue(p)
        }
    }

    /**
     * Laplacian-variance sharpness. Whip-pan / motion-blurred frames collapse to
     * near-zero variance, which is exactly the signal used to disqualify "blurred
     * whip-pan passes count for nobody" frames from ever winning as the
     * representative shot.
     */
    private fun sharpnessScore(lum: DoubleArray): Float {
        var mean = 0.0
        var count = 0
        val lap = DoubleArray(64 * 64)
        for (y in 1 until 63) {
            for (x in 1 until 63) {
                val idx = y * 64 + x
                val value = -4 * lum[idx] + lum[idx - 1] + lum[idx + 1] + lum[idx - 64] + lum[idx + 64]
                lap[idx] = value
                mean += value
                count++
            }
        }
        mean /= count
        var variance = 0.0
        for (y in 1 until 63) {
            for (x in 1 until 63) {
                val idx = y * 64 + x
                val d = lap[idx] - mean
                variance += d * d
            }
        }
        variance /= count

        // Empirically, sharp ML-Kit-sized face crops land >150; heavy motion blur <20.
        // Normalize into 0..1 with a soft cap rather than a hard threshold.
        return min((variance / 300.0), 1.0).toFloat()
    }

    /**
     * 1.0 at a well-lit midtone mean luminance, decaying toward 0 for near-black
     * (backlit/underexposed) or near-white (blown-out) frames. This is the term
     * that keeps a dark, hard-to-see frame from winning just because it's the only
     * option scored on frontality/eyes/smile alone.
     */
    private fun exposureScore(lum: DoubleArray): Float {
        val mean = lum.average()
        return when {
            mean < 30.0 -> 0f                                   // effectively black
            mean < 90.0 -> ((mean - 30.0) / 60.0).toFloat()      // ramping up from dark
            mean <= 200.0 -> 1f                                 // comfortable midtone range
            mean < 240.0 -> (1f - ((mean - 200.0) / 40.0)).toFloat() // ramping down toward blown out
            else -> 0f                                          // blown out
        }.coerceIn(0f, 1f)
    }
}