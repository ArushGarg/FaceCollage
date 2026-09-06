package com.iykyk.facecollage.ui

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.iykyk.facecollage.model.ProcessingProgress
import com.iykyk.facecollage.model.ProcessingStage
import com.iykyk.facecollage.model.VideoResult
import com.iykyk.facecollage.processing.VideoProcessor
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class VideoJob(
    val label: String,
    val uri: Uri,
    val progress: ProcessingProgress = ProcessingProgress(ProcessingStage.ExtractingFrames),
    val result: VideoResult? = null,
    val error: String? = null
)

class CollageViewModel(application: Application) : AndroidViewModel(application) {

    private val processor = VideoProcessor(application)

    private val _jobs = MutableStateFlow<List<VideoJob>>(emptyList())
    val jobs: StateFlow<List<VideoJob>> = _jobs.asStateFlow()

    fun processVideo(uri: Uri, label: String) {
        // Avoid double-processing the same picked video.
        if (_jobs.value.any { it.uri == uri }) return

        _jobs.value = _jobs.value + VideoJob(label = label, uri = uri)

        viewModelScope.launch {
            processor.process(uri, label).collect { event ->
                when (event) {
                    is VideoProcessor.Event.Progress -> updateJob(uri) { it.copy(progress = event.progress) }
                    is VideoProcessor.Event.Result -> updateJob(uri) { it.copy(result = event.result) }
                    is VideoProcessor.Event.Error -> updateJob(uri) { it.copy(error = event.message) }
                }
            }
        }
    }

    private fun updateJob(uri: Uri, transform: (VideoJob) -> VideoJob) {
        _jobs.value = _jobs.value.map { if (it.uri == uri) transform(it) else it }
    }

    override fun onCleared() {
        super.onCleared()
        processor.release()
    }
}
