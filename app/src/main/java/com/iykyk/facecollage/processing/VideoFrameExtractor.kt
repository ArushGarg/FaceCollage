package com.iykyk.facecollage.processing

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

data class ExtractedFrame(val timestampMs: Long, val bitmap: Bitmap)

/**
 * Samples frames from a video at a fixed interval using MediaMetadataRetriever.
 * Runs entirely on Dispatchers.IO — the caller (VideoProcessor) is responsible
 * for keeping this off the main thread, which it is by construction.
 */
class VideoFrameExtractor(private val context: Context) {

    /**
     * @param sampleIntervalMs how often to grab a frame. 150ms (~6.6 fps) is dense
     * enough to catch whip-pans as blur (so they can be filtered by sharpness) while
     * keeping a 30s clip to ~200 frames, which is workable on-device.
     * @param maxDimensionPx every decoded frame is downscaled (aspect-preserving) so
     * its longest side is at most this. Retaining ~200 *full-resolution* portrait
     * frames (a 1080x1920 frame is ~8MB as ARGB_8888) for the whole pipeline would
     * OOM on a real device well before embedding even starts. 720px is still far
     * more resolution than either ML Kit detection or the 160x160 embedder input
     * needs, and keeps a 200-frame clip under ~150MB resident.
     */
    suspend fun extractFrames(
        videoUri: Uri,
        sampleIntervalMs: Long = 150,
        maxDimensionPx: Int = 720,
        onProgress: (current: Int, total: Int) -> Unit = { _, _ -> }
    ): List<ExtractedFrame> = withContext(Dispatchers.IO) {
        val retriever = MediaMetadataRetriever()
        val frames = mutableListOf<ExtractedFrame>()
        try {
            retriever.setDataSource(context, videoUri)
            val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull() ?: 0L

            if (durationMs <= 0) return@withContext emptyList()

            val totalSteps = (durationMs / sampleIntervalMs).toInt().coerceAtLeast(1)
            var t = 0L
            var step = 0
            while (t < durationMs) {
                val raw = retriever.getFrameAtTime(
                    t * 1000, // microseconds
                    MediaMetadataRetriever.OPTION_CLOSEST
                )
                if (raw != null) {
                    frames.add(ExtractedFrame(t, downscale(raw, maxDimensionPx)))
                }
                step++
                onProgress(step, totalSteps)
                t += sampleIntervalMs
            }
        } finally {
            retriever.release()
        }
        frames
    }

    private fun downscale(bitmap: Bitmap, maxDimensionPx: Int): Bitmap {
        val longest = maxOf(bitmap.width, bitmap.height)
        if (longest <= maxDimensionPx) return bitmap
        val scale = maxDimensionPx.toFloat() / longest
        val w = (bitmap.width * scale).toInt().coerceAtLeast(1)
        val h = (bitmap.height * scale).toInt().coerceAtLeast(1)
        val scaled = Bitmap.createScaledBitmap(bitmap, w, h, true)
        if (scaled !== bitmap) bitmap.recycle()
        return scaled
    }
}
