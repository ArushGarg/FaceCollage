package com.iykyk.facecollage.ui

import android.graphics.Bitmap
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.iykyk.facecollage.model.ProcessingStage
import com.iykyk.facecollage.util.ShareUtils

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CollageScreen(viewModel: CollageViewModel = viewModel()) {
    val context = LocalContext.current
    val jobs by viewModel.jobs.collectAsState()
    var nextLabelIndex by remember { mutableStateOf(1) }

    val pickVideo = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri != null) {
            viewModel.processVideo(uri, "Sample ${nextLabelIndex}")
            nextLabelIndex++
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(title = { Text("Face Collage") })
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp)
        ) {
            Button(
                onClick = { pickVideo.launch("video/*") },
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Pick a portrait video")
            }

            Spacer(Modifier.height(16.dp))

            LazyColumn(verticalArrangement = Arrangement.spacedBy(20.dp)) {
                items(jobs) { job ->
                    JobCard(
                        job = job,
                        onSave = { bitmap ->
                            val uri = ShareUtils.saveToGallery(context, bitmap, job.label.replace(" ", "_"))
                            val message = if (uri != null) "Saved to Pictures/FaceCollage" else "Save failed"
                            android.widget.Toast.makeText(context, message, android.widget.Toast.LENGTH_SHORT).show()
                        },
                        onShare = { bitmap ->
                            val uri = ShareUtils.shareableUri(context, bitmap, job.label.replace(" ", "_"))
                            context.startActivity(ShareUtils.shareIntent(context, uri))
                        }
                    )
                }
            }
        }
    }
}

@Composable
private fun JobCard(
    job: VideoJob,
    onSave: (Bitmap) -> Unit,
    onShare: (Bitmap) -> Unit
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp)) {
            Text(job.label, style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(8.dp))

            when {
                job.error != null -> {
                    Text("Error: ${job.error}", color = MaterialTheme.colorScheme.error)
                }
                job.result != null -> {
                    val result = job.result
                    Text("${result.people.size} unique people, ${result.people.sumOf { it.appearanceCount }} total appearances")
                    Spacer(Modifier.height(8.dp))
                    Image(
                        bitmap = result.collage.asImageBitmap(),
                        contentDescription = "Collage for ${job.label}",
                        modifier = Modifier.fillMaxWidth()
                    )
                    Spacer(Modifier.height(8.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        Button(onClick = { onSave(result.collage) }) { Text("Save") }
                        OutlinedButton(onClick = { onShare(result.collage) }) { Text("Share") }
                    }
                }
                else -> {
                    val progress = job.progress
                    val fraction = if (progress.total > 0) progress.current.toFloat() / progress.total else 0f
                    Text(progress.stage.label)
                    Spacer(Modifier.height(4.dp))
                    LinearProgressIndicator(
                        progress = { fraction },
                        modifier = Modifier.fillMaxWidth()
                    )
                    if (progress.stage != ProcessingStage.Done && progress.total > 0) {
                        Spacer(Modifier.height(4.dp))
                        Text("${progress.current} / ${progress.total}", style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
        }
    }
}