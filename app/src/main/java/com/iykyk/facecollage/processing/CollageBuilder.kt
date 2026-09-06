package com.iykyk.facecollage.processing

import android.graphics.*
import com.iykyk.facecollage.model.RepresentativeShot
import kotlin.math.ceil
import kotlin.math.sqrt

/**
 * Builds a single shareable collage bitmap from each video's representative shots.
 * Every person detected in the video appears exactly once, per the brief.
 *
 * Layout: a near-square grid (Instagram Story multi-photo grid style) with rounded
 * tiles, thin gutters, and a small "appearance count" badge per tile so the count
 * the brief asks to surface is visible directly on the shareable artifact, not
 * just in a separate UI list.
 */
object CollageBuilder {

    private const val TILE_CORNER_RADIUS = 28f
    private const val GUTTER = 10f
    private const val OUTPUT_WIDTH = 1080 // Instagram Story-safe width

    fun build(shots: List<RepresentativeShot>, videoLabel: String): Bitmap {
        val n = shots.size.coerceAtLeast(1)
        val columns = ceil(sqrt(n.toDouble())).toInt().coerceAtLeast(1)
        val rows = ceil(n / columns.toDouble()).toInt().coerceAtLeast(1)

        val tileSize = (OUTPUT_WIDTH - GUTTER * (columns + 1)) / columns
        val headerHeight = 110f
        val outputHeight = headerHeight + rows * tileSize + GUTTER * (rows + 1)

        val output = Bitmap.createBitmap(OUTPUT_WIDTH, outputHeight.toInt(), Bitmap.Config.ARGB_8888)
        val canvas = Canvas(output)
        canvas.drawColor(Color.parseColor("#111111"))

        // Header
        val titlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = 42f
            typeface = Typeface.create(Typeface.DEFAULT_BOLD, Typeface.BOLD)
        }
        val subPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#AAAAAA")
            textSize = 26f
        }
        canvas.drawText(videoLabel, GUTTER * 2, 50f, titlePaint)
        canvas.drawText("${shots.size} unique ${if (shots.size == 1) "person" else "people"} detected", GUTTER * 2, 88f, subPaint)

        shots.forEachIndexed { index, shot ->
            val col = index % columns
            val row = index / columns
            val left = GUTTER + col * (tileSize + GUTTER)
            val top = headerHeight + GUTTER + row * (tileSize + GUTTER)
            drawTile(canvas, shot, RectF(left, top, left + tileSize, top + tileSize))
        }

        return output
    }

    private fun drawTile(canvas: Canvas, shot: RepresentativeShot, rect: RectF) {
        val path = Path().apply {
            addRoundRect(rect, TILE_CORNER_RADIUS, TILE_CORNER_RADIUS, Path.Direction.CW)
        }
        canvas.save()
        canvas.clipPath(path)

        // Center-crop the (generously-cropped, non-tight-bbox) shot to fill the square tile.
        val src = centerCropRect(shot.crop.width, shot.crop.height, rect.width(), rect.height())
        canvas.drawBitmap(shot.crop, src, rect, null)
        canvas.restore()

        // Subtle border
        val borderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 2f
            color = Color.parseColor("#33FFFFFF")
        }
        canvas.drawPath(path, borderPaint)

        // Appearance-count badge, bottom-right of tile
        val badgeText = "${shot.appearanceCount}x"
        val badgePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = 30f
            typeface = Typeface.create(Typeface.DEFAULT_BOLD, Typeface.BOLD)
        }
        val badgeBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#CC000000")
        }
        val textWidth = badgePaint.measureText(badgeText)
        val badgeRect = RectF(
            rect.right - textWidth - 36f,
            rect.bottom - 56f,
            rect.right - 12f,
            rect.bottom - 12f
        )
        canvas.drawRoundRect(badgeRect, 16f, 16f, badgeBgPaint)
        canvas.drawText(badgeText, badgeRect.left + 14f, badgeRect.bottom - 16f, badgePaint)
    }

    /** Returns the source rect (in bitmap pixel space) that, scaled to dstW x dstH, center-crops without distortion. */
    private fun centerCropRect(srcW: Int, srcH: Int, dstW: Float, dstH: Float): Rect {
        val srcAspect = srcW.toFloat() / srcH
        val dstAspect = dstW / dstH
        return if (srcAspect > dstAspect) {
            // source wider than target -> crop left/right
            val cropW = (srcH * dstAspect).toInt()
            val left = (srcW - cropW) / 2
            Rect(left, 0, left + cropW, srcH)
        } else {
            // source taller than target -> crop top/bottom
            val cropH = (srcW / dstAspect).toInt()
            val top = (srcH - cropH) / 2
            Rect(0, top, srcW, top + cropH)
        }
    }
}
