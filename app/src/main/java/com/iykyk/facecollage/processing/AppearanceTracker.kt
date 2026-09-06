package com.iykyk.facecollage.processing

import com.iykyk.facecollage.model.Appearance
import com.iykyk.facecollage.model.FrameFace

/**
 * Raw per-frame face detection before we know which person it belongs to.
 * mlKitTrackingId comes from FaceDetectorOptions.enableTracking() — ML Kit keeps
 * this stable across consecutive frames while a face stays continuously visible
 * and in frame, which is exactly the "continuous visible segment" the brief
 * defines an appearance as.
 */
data class RawDetection(val mlKitTrackingId: Int, val face: FrameFace)

/**
 * Turns raw per-frame detections into [Appearance] segments.
 *
 * Two responsibilities:
 *  1. Group frames sharing the same ML Kit tracking ID.
 *  2. Split a group if there's a time gap bigger than [maxGapMs] — ML Kit's tracker
 *     is good but not perfect across brief occlusion/re-entry, and a stale tracking
 *     ID reused after a long gap should not silently merge two real appearances.
 *
 * Identity (which *person*) is deliberately NOT decided here — see FaceClusterer.
 * This class only answers "which frames are one continuous sighting", matching the
 * brief's definition of an appearance independent of who it turns out to be.
 */
class AppearanceTracker(private val maxGapMs: Long = 1000) {

    fun buildAppearances(detections: List<RawDetection>): List<Appearance> {
        val byTrackingId = detections.groupBy { it.mlKitTrackingId }
        val appearances = mutableListOf<Appearance>()

        for ((_, group) in byTrackingId) {
            val sorted = group.sortedBy { it.face.timestampMs }
            var segmentStart = 0
            for (i in 1 until sorted.size) {
                val gap = sorted[i].face.timestampMs - sorted[i - 1].face.timestampMs
                if (gap > maxGapMs) {
                    appearances += toAppearance(sorted.subList(segmentStart, i))
                    segmentStart = i
                }
            }
            appearances += toAppearance(sorted.subList(segmentStart, sorted.size))
        }
        return appearances.filter { it.faces.isNotEmpty() }
    }

    private fun toAppearance(group: List<RawDetection>): Appearance {
        val faces = group.map { it.face }
        return Appearance(
            startMs = faces.first().timestampMs,
            endMs = faces.last().timestampMs,
            faces = faces
        )
    }
}