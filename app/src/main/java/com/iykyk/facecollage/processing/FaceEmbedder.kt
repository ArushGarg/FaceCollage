package com.iykyk.facecollage.processing

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.sqrt

/**
 * On-device face embedding model: FaceNet (Inception-ResNet v1), 512-d output,
 * int8-quantized TFLite export. Bundled at assets/face_embedder.tflite.
 *
 * Model provenance (documented per assignment requirement) is repeated in README.md.
 *
 * Input: 160x160 RGB, normalized to [-1, 1] ((pixel - 127.5) / 128.0).
 * Output: 512-dim embedding, L2-normalized here so cosine similarity == dot product.
 *
 * Interpreter runs single-threaded (setNumThreads(1)) deliberately, not for speed:
 * multi-threaded TFLite reduction ops sum in a non-deterministic order, producing
 * tiny run-to-run differences in embedding values. Because the clustering threshold
 * in FaceClusterer sits on a narrow, empirically-found boundary, even a tiny
 * embedding shift on one borderline appearance can flip its cluster assignment —
 * and since clustering is greedy/incremental, that one flip cascades into a
 * different result for everything processed after it. Single-threaded inference
 * trades a bit of speed for the same video producing the same result every time,
 * which matters far more here than shaving milliseconds off a 30s clip.
 */
class FaceEmbedder(context: Context) {

    private val inputSize = 160
    private val embeddingDim = 512

    private val interpreter: Interpreter = run {
        val afd = context.assets.openFd("face_embedder.tflite")
        val inputStream = FileInputStream(afd.fileDescriptor)
        val buffer = inputStream.channel.map(
            FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength
        )
        Interpreter(buffer, Interpreter.Options().apply { setNumThreads(1) })
    }

    /**
     * @param frame the full decoded video frame
     * @param faceBox ML Kit bounding box in [frame]'s coordinate space
     */
    fun embed(frame: Bitmap, faceBox: RectF): FloatArray {
        val faceCrop = ImageOps.cropWithMargin(frame, faceBox, marginFraction = 0.25f)
        val resized = Bitmap.createScaledBitmap(faceCrop, inputSize, inputSize, true)

        val inputBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
            .order(ByteOrder.nativeOrder())
        val pixels = IntArray(inputSize * inputSize)
        resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        for (p in pixels) {
            val r = (p shr 16 and 0xFF)
            val g = (p shr 8 and 0xFF)
            val b = (p and 0xFF)
            inputBuffer.putFloat((r - 127.5f) / 128.0f)
            inputBuffer.putFloat((g - 127.5f) / 128.0f)
            inputBuffer.putFloat((b - 127.5f) / 128.0f)
        }

        val output = Array(1) { FloatArray(embeddingDim) }
        interpreter.run(inputBuffer, output)

        return l2Normalize(output[0])
    }

    private fun l2Normalize(v: FloatArray): FloatArray {
        var sumSq = 0f
        for (x in v) sumSq += x * x
        val norm = sqrt(sumSq).coerceAtLeast(1e-6f)
        return FloatArray(v.size) { v[it] / norm }
    }

    fun close() = interpreter.close()
}

object ImageOps {
    /** Crops generously around [box] (not a tight bbox) so downstream tiles aren't low-res mush. */
    fun cropWithMargin(bitmap: Bitmap, box: RectF, marginFraction: Float): Bitmap {
        val marginX = box.width() * marginFraction
        val marginY = box.height() * marginFraction
        val left = (box.left - marginX).coerceIn(0f, bitmap.width.toFloat())
        val top = (box.top - marginY).coerceIn(0f, bitmap.height.toFloat())
        val right = (box.right + marginX).coerceIn(0f, bitmap.width.toFloat())
        val bottom = (box.bottom + marginY).coerceIn(0f, bitmap.height.toFloat())
        val w = (right - left).toInt().coerceAtLeast(1)
        val h = (bottom - top).toInt().coerceAtLeast(1)
        return Bitmap.createBitmap(bitmap, left.toInt(), top.toInt(), w, h)
    }
}

fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
    var dot = 0f
    for (i in a.indices) dot += a[i] * b[i]
    return dot // both already L2-normalized, so dot product == cosine similarity
}