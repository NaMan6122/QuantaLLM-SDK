package com.quantallm

import android.os.Environment
import android.util.Log
import java.io.File

data class DiscoveredModel(
    val name: String,
    val path: String,
    val sizeBytes: Long,
    val format: ModelFormat,
    val compatibleBackends: List<Backend>
)

enum class ModelFormat(val displayName: String) {
    GGUF("GGUF"),
    ONNX("ONNX")
}

object ModelScanner {

    private const val TAG = "ModelScanner"
    private const val MIN_MODEL_BYTES = 10_000_000L
    private const val MAX_ONNX_SCAN_DEPTH = 4

    fun scan(additionalPaths: List<String> = emptyList()): List<DiscoveredModel> {
        val models = mutableListOf<DiscoveredModel>()
        val searchPaths = getSearchPaths() + additionalPaths

        for (path in searchPaths) {
            val dir = File(path)
            if (!dir.exists() || !dir.isDirectory) continue

            models.addAll(scanGgufFiles(dir))
            models.addAll(scanOnnxDirs(dir, MAX_ONNX_SCAN_DEPTH))
        }

        return models.distinctBy { it.path }
    }

    fun scanGguf(additionalPaths: List<String> = emptyList()): List<DiscoveredModel> {
        val models = mutableListOf<DiscoveredModel>()
        for (path in getSearchPaths() + additionalPaths) {
            val dir = File(path)
            if (!dir.exists() || !dir.isDirectory) continue
            models.addAll(scanGgufFiles(dir))
        }
        return models.distinctBy { it.path }
    }

    fun scanOnnx(additionalPaths: List<String> = emptyList()): List<DiscoveredModel> {
        val models = mutableListOf<DiscoveredModel>()
        for (path in getSearchPaths() + additionalPaths) {
            val dir = File(path)
            if (!dir.exists() || !dir.isDirectory) continue
            models.addAll(scanOnnxDirs(dir, MAX_ONNX_SCAN_DEPTH))
        }
        return models.distinctBy { it.path }
    }

    fun validate(path: String): ValidationResult {
        val file = File(path)
        if (!file.exists()) return ValidationResult.Error("Path does not exist")
        if (!file.canRead()) return ValidationResult.Error("Cannot read path (check permissions)")

        return if (file.isDirectory) {
            validateOnnxDirectory(file)
        } else {
            validateGgufFile(file)
        }
    }

    private fun scanGgufFiles(dir: File): List<DiscoveredModel> {
        val files = dir.listFiles { f -> f.isFile && f.name.endsWith(".gguf", ignoreCase = true) }
            ?: return emptyList()

        return files.mapNotNull { file ->
            val size = file.length()
            if (size < MIN_MODEL_BYTES) {
                Log.w(TAG, "${file.name} too small (${size} bytes), skipping")
                return@mapNotNull null
            }
            DiscoveredModel(
                name = file.nameWithoutExtension,
                path = file.absolutePath,
                sizeBytes = size,
                format = ModelFormat.GGUF,
                compatibleBackends = listOf(Backend.LLAMA_CPP, Backend.LLAMA_CPP_HEXAGON)
            )
        }
    }

    private fun scanOnnxDirs(root: File, maxDepth: Int): List<DiscoveredModel> {
        if (maxDepth < 0 || !root.isDirectory) return emptyList()

        val entries = root.listFiles()?.filter { !it.name.startsWith(".") } ?: return emptyList()

        if (isOnnxModelDir(entries)) {
            val totalSize = root.walkTopDown()
                .onEnter { !it.name.startsWith(".") || it == root }
                .filter { it.isFile }
                .sumOf { it.length() }

            if (totalSize < MIN_MODEL_BYTES) return emptyList()

            return listOf(
                DiscoveredModel(
                    name = root.name,
                    path = root.absolutePath,
                    sizeBytes = totalSize,
                    format = ModelFormat.ONNX,
                    compatibleBackends = listOf(Backend.ONNX_CPU, Backend.ONNX_QNN)
                )
            )
        }

        if (maxDepth == 0) return emptyList()

        return entries
            .filter { it.isDirectory }
            .flatMap { scanOnnxDirs(it, maxDepth - 1) }
    }

    private fun isOnnxModelDir(entries: List<File>): Boolean {
        val hasOnnx = entries.any { it.isFile && it.name.endsWith(".onnx", ignoreCase = true) }
        val hasConfig = entries.any { it.isFile && it.name.equals("genai_config.json", ignoreCase = true) }
        return hasOnnx && hasConfig
    }

    private fun validateGgufFile(file: File): ValidationResult {
        if (!file.name.endsWith(".gguf", ignoreCase = true)) {
            return ValidationResult.Error("Not a .gguf file")
        }
        if (file.length() < MIN_MODEL_BYTES) {
            return ValidationResult.Error("File too small (${file.length()} bytes) — may be corrupted")
        }
        return ValidationResult.Valid
    }

    private fun validateOnnxDirectory(dir: File): ValidationResult {
        val entries = dir.listFiles()?.toList() ?: return ValidationResult.Error("Cannot list directory contents")

        if (!entries.any { it.isFile && it.name.endsWith(".onnx", ignoreCase = true) }) {
            return ValidationResult.Error("No .onnx file found in directory")
        }
        if (!entries.any { it.isFile && it.name.equals("genai_config.json", ignoreCase = true) }) {
            return ValidationResult.Error("genai_config.json not found")
        }
        if (!entries.any { it.isFile && it.name.startsWith("tokenizer", ignoreCase = true) }) {
            return ValidationResult.Error("Tokenizer files not found")
        }

        val totalSize = dir.walkTopDown()
            .onEnter { !it.name.startsWith(".") || it == dir }
            .filter { it.isFile }
            .sumOf { it.length() }

        if (totalSize < MIN_MODEL_BYTES) {
            return ValidationResult.Error("Model directory too small ($totalSize bytes) — may be incomplete")
        }

        return ValidationResult.Valid
    }

    private fun getSearchPaths(): List<String> {
        val ext = Environment.getExternalStorageDirectory().path
        return listOf("$ext/Download", "$ext/Downloads")
    }

    sealed class ValidationResult {
        data object Valid : ValidationResult()
        data class Error(val message: String) : ValidationResult()
    }
}
