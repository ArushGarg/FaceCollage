package com.iykyk.facecollage.processing

import com.iykyk.facecollage.model.Appearance
import com.iykyk.facecollage.model.PersonTrack
import kotlin.math.sqrt

/**
 * Clusters [Appearance] segments (not individual frames) into identities.
 *
 * Why cluster appearances rather than raw frames: an appearance already groups
 * frames ML Kit's tracker says are the same continuous sighting, so its embedding
 * centroid is far more stable than any single frame — a single blurry frame's bad
 * embedding gets averaged out instead of potentially founding its own spurious
 * cluster.
 *
 * Algorithm: greedy incremental clustering by cosine similarity against a running
 * centroid, processed in chronological order. Not full agglomerative clustering —
 * deliberately simple, O(n * clusters) instead of O(n^2), which matters more once
 * a video has many appearances, and is easy to reason about / debug against the
 * worked example in the brief.
 *
 * SIMILARITY_THRESHOLD = 0.53 (tuned empirically against Sample 1: 0.62 over-fragmented
 * into 12 clusters against a ground truth of 5; 0.45 over-merged down to 4. 0.53 sits
 * between those two observed failure points). Retune this first if a specific clip
 * still under/over-merges: raise it if distinct people are getting merged into one,
 * lower it if one person is splitting into multiple clusters.
 */
class FaceClusterer(private val similarityThreshold: Float = 0.55f) {

    private class Cluster(var centroid: FloatArray, var count: Int) {
        val appearances = mutableListOf<Appearance>()
    }

    fun cluster(appearances: List<Appearance>): List<PersonTrack> {
        val chronological = appearances.sortedBy { it.startMs }
        val clusters = mutableListOf<Cluster>()

        for (appearance in chronological) {
            val embedding = representativeEmbedding(appearance)
            val best = clusters.maxByOrNull { cosineSimilarity(it.centroid, embedding) }
            val bestSim = best?.let { cosineSimilarity(it.centroid, embedding) } ?: -1f

            if (best != null && bestSim >= similarityThreshold) {
                best.appearances += appearance
                best.centroid = runningMean(best.centroid, best.count, embedding)
                best.count += 1
            } else {
                clusters += Cluster(embedding, 1).apply { this.appearances += appearance }
            }
        }

        return clusters.mapIndexed { index, cluster ->
            PersonTrack(clusterId = index, appearances = cluster.appearances)
        }
    }

    /** Mean embedding of the up-to-5 sharpest frames in the appearance, then re-normalized. */
    private fun representativeEmbedding(appearance: Appearance): FloatArray {
        val topFaces = appearance.faces
            .sortedByDescending { ShotScorer.score(it) }
            .take(5)
        val dim = topFaces.first().embedding.size
        val sum = FloatArray(dim)
        for (f in topFaces) {
            for (i in 0 until dim) sum[i] += f.embedding[i]
        }
        for (i in 0 until dim) sum[i] = sum[i] / topFaces.size.toFloat()
        return l2Normalize(sum)
    }

    private fun runningMean(centroid: FloatArray, count: Int, next: FloatArray): FloatArray {
        val updated = FloatArray(centroid.size) { i ->
            (centroid[i] * count + next[i]) / (count + 1)
        }
        return l2Normalize(updated)
    }

    private fun l2Normalize(v: FloatArray): FloatArray {
        var sumSq = 0f
        for (x in v) sumSq += x * x
        val norm = sqrt(sumSq).coerceAtLeast(1e-6f)
        return FloatArray(v.size) { v[it] / norm }
    }
}