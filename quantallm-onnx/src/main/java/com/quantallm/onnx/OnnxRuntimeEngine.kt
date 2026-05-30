package com.quantallm.onnx

import android.util.Log
import com.quantallm.Backend
import com.quantallm.GenerationParams
import com.quantallm.GenerationResult
import com.quantallm.InferenceEngine
import com.quantallm.ModelInfo
import org.json.JSONObject
import java.io.File

class OnnxRuntimeEngine(
    override val backend: Backend = Backend.ONNX_CPU
) : InferenceEngine {

    companion object {
        private const val TAG = "OnnxRuntimeEngine"
        private const val MAX_CONTEXT_LENGTH = 2048
    }

    override val displayName: String
        get() = backend.displayName

    private var ortModel: Any? = null
    private var ortTokenizer: Any? = null
    private var loaded = false
    private var currentPath: String? = null
    private var lastError: String = ""
    private var modelDirName: String = ""
    private var modelSizeBytes: Long = 0L
    private var modelContextLength: Int = MAX_CONTEXT_LENGTH

    private val executionProvider: String
        get() = if (backend == Backend.ONNX_QNN) "QNN" else "CPU"

    init {
        if (!backend.isOnnx) {
            throw IllegalArgumentException("OnnxRuntimeEngine only supports ONNX backends, got: $backend")
        }
    }

    override fun loadModel(modelPath: String, threads: Int): GenerationResult {
        return try {
            val modelDir = File(modelPath)
            if (!modelDir.exists() || !modelDir.isDirectory) {
                return GenerationResult.Error("ONNX model path must be a directory: $modelPath")
            }

            val hasOnnx = modelDir.listFiles()?.any {
                it.isFile && it.name.endsWith(".onnx", ignoreCase = true)
            } == true
            val hasConfig = File(modelDir, "genai_config.json").exists()
            val hasTokenizer = modelDir.listFiles()?.any {
                it.isFile && it.name.startsWith("tokenizer", ignoreCase = true)
            } == true

            if (!hasOnnx) return GenerationResult.Error("No .onnx model file found in $modelPath")
            if (!hasConfig) return GenerationResult.Error("genai_config.json not found in $modelPath")
            if (!hasTokenizer) return GenerationResult.Error("Tokenizer files not found in $modelPath")

            modelSizeBytes = modelDir.walkTopDown()
                .onEnter { !it.name.startsWith(".") || it == modelDir }
                .filter { it.isFile }
                .sumOf { it.length() }
            modelDirName = modelDir.name

            if (backend == Backend.ONNX_CPU) {
                patchGenaiConfigForAndroid(modelDir)
            }

            modelContextLength = readContextLength(modelDir)

            val config = ai.onnxruntime.genai.Config(modelPath)
            config.clearProviders()

            when (backend) {
                Backend.ONNX_CPU -> {
                    config.overlay("""{"model":{"decoder":{"session_options":{"enable_mem_pattern":false}}}}""")
                }
                Backend.ONNX_QNN -> {
                    config.appendProvider("QNN")
                    config.setProviderOption("QNN", "backend_path", "libQnnHtp.so")
                    config.setProviderOption("QNN", "htp_performance_mode", "sustained_high_performance")
                    config.setProviderOption("QNN", "htp_graph_finalization_optimization_mode", "3")
                    config.setProviderOption("QNN", "enable_htp_fp16_precision", "1")
                    config.setProviderOption("QNN", "qnn_context_cache_enable", "1")
                    config.setProviderOption("QNN", "qnn_context_cache_path", getContextCachePath(modelPath))
                }
                else -> {}
            }

            val model = ai.onnxruntime.genai.Model(config)
            config.close()
            val tokenizer = ai.onnxruntime.genai.Tokenizer(model)

            ortModel = model
            ortTokenizer = tokenizer
            loaded = true
            currentPath = modelPath
            lastError = ""

            GenerationResult.Success("ONNX model loaded: $modelDirName (${modelSizeBytes / (1024 * 1024)} MB)")
        } catch (e: Exception) {
            lastError = "Failed to load ONNX model: ${e.message}"
            Log.e(TAG, lastError, e)
            GenerationResult.Error(lastError, cause = e)
        }
    }

    override fun isModelLoaded(): Boolean = loaded

    override fun getModelInfo(): ModelInfo? {
        if (!loaded) return null
        return ModelInfo(
            name = modelDirName,
            format = "ONNX",
            sizeBytes = modelSizeBytes,
            contextLength = modelContextLength,
            backend = backend
        )
    }

    override fun getCurrentModelPath(): String? = currentPath

    override fun generate(prompt: String, params: GenerationParams): GenerationResult {
        if (!loaded) return GenerationResult.Error("No model loaded")
        if (prompt.isEmpty()) return GenerationResult.Error("Empty prompt")

        return try {
            val model = ortModel as ai.onnxruntime.genai.Model
            val tokenizer = ortTokenizer as ai.onnxruntime.genai.Tokenizer

            val startTime = System.currentTimeMillis()
            val inputSequences = tokenizer.encode(prompt)
            val genParams = ai.onnxruntime.genai.GeneratorParams(model)
            val inputTokenCount = inputSequences.getSequence(0).size
            val effectiveMaxLength = minOf(inputTokenCount + params.maxTokens, modelContextLength)
            genParams.setSearchOption("max_length", effectiveMaxLength.toDouble())
            genParams.setSearchOption("temperature", params.temperature.toDouble())

            val generator = ai.onnxruntime.genai.Generator(model, genParams)
            generator.appendTokenSequences(inputSequences)

            val tokenizerStream = tokenizer.createStream()
            val output = StringBuilder()
            var tokenCount = 0

            while (!generator.isDone()) {
                generator.generateNextToken()
                val token = tokenizerStream.decode(generator.getLastTokenInSequence(0))
                output.append(token)
                tokenCount++
            }

            generator.close()
            tokenizerStream.close()
            genParams.close()
            inputSequences.close()

            val duration = System.currentTimeMillis() - startTime
            GenerationResult.Success(output.toString(), tokensGenerated = tokenCount, durationMs = duration)
        } catch (e: Exception) {
            lastError = "Generation failed: ${e.message}"
            Log.e(TAG, lastError, e)
            GenerationResult.Error(lastError, cause = e)
        }
    }

    override fun generateStreaming(
        prompt: String,
        params: GenerationParams,
        callback: InferenceEngine.StreamingCallback
    ): GenerationResult {
        if (!loaded) return GenerationResult.Error("No model loaded")
        if (prompt.isEmpty()) return GenerationResult.Error("Empty prompt")

        return try {
            val model = ortModel as ai.onnxruntime.genai.Model
            val tokenizer = ortTokenizer as ai.onnxruntime.genai.Tokenizer

            val startTime = System.currentTimeMillis()
            val inputSequences = tokenizer.encode(prompt)
            val genParams = ai.onnxruntime.genai.GeneratorParams(model)
            val inputTokenCount = inputSequences.getSequence(0).size
            val effectiveMaxLength = minOf(inputTokenCount + params.maxTokens, modelContextLength)
            genParams.setSearchOption("max_length", effectiveMaxLength.toDouble())
            genParams.setSearchOption("temperature", params.temperature.toDouble())

            val generator = ai.onnxruntime.genai.Generator(model, genParams)
            generator.appendTokenSequences(inputSequences)

            val tokenizerStream = tokenizer.createStream()
            val output = StringBuilder()
            var tokenCount = 0

            while (!generator.isDone()) {
                generator.generateNextToken()
                val token = tokenizerStream.decode(generator.getLastTokenInSequence(0))
                output.append(token)
                tokenCount++
                callback.onToken(tokenCount, output.toString())
            }

            generator.close()
            tokenizerStream.close()
            genParams.close()
            inputSequences.close()

            val duration = System.currentTimeMillis() - startTime
            GenerationResult.Success(output.toString(), tokensGenerated = tokenCount, durationMs = duration)
        } catch (e: Exception) {
            lastError = "Streaming generation failed: ${e.message}"
            Log.e(TAG, lastError, e)
            GenerationResult.Error(lastError, cause = e)
        }
    }

    override fun abortGeneration() {
        // ONNX Runtime GenAI does not currently support mid-generation abort
    }

    override fun unloadModel() {
        if (!loaded) return
        try {
            (ortTokenizer as? AutoCloseable)?.close()
            (ortModel as? AutoCloseable)?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Unload failed", e)
        } finally {
            ortTokenizer = null
            ortModel = null
            loaded = false
            currentPath = null
            modelDirName = ""
            modelSizeBytes = 0L
            modelContextLength = MAX_CONTEXT_LENGTH
            lastError = ""
        }
    }

    override fun getLastError(): String = lastError

    private fun readContextLength(modelDir: File): Int {
        return try {
            val config = JSONObject(File(modelDir, "genai_config.json").readText())
            config.optJSONObject("model")?.optInt("context_length", MAX_CONTEXT_LENGTH)
                ?: MAX_CONTEXT_LENGTH
        } catch (_: Exception) {
            MAX_CONTEXT_LENGTH
        }
    }

    private fun getContextCachePath(modelPath: String): String {
        val cacheDir = File(modelPath, ".qnn_cache")
        cacheDir.mkdirs()
        return File(cacheDir, "context.bin").absolutePath
    }

    private fun patchGenaiConfigForAndroid(modelDir: File) {
        val configFile = File(modelDir, "genai_config.json")
        if (!configFile.exists()) return

        try {
            val original = configFile.readText()
            val config = JSONObject(original)
            var modified = false

            val modelObj = config.optJSONObject("model")
            if (modelObj != null) {
                val contextLength = modelObj.optInt("context_length", 0)
                if (contextLength > MAX_CONTEXT_LENGTH) {
                    modelObj.put("context_length", MAX_CONTEXT_LENGTH)
                    modified = true
                }

                val decoder = modelObj.optJSONObject("decoder")
                if (decoder != null) {
                    val sessionOpts = decoder.optJSONObject("session_options")
                    if (sessionOpts != null) {
                        val providers = sessionOpts.optJSONArray("provider_options")
                        if (providers != null) {
                            sessionOpts.remove("provider_options")
                            modified = true
                        }
                    }
                }
            }

            val searchObj = config.optJSONObject("search")
            if (searchObj != null) {
                val maxLen = searchObj.optInt("max_length", 0)
                if (maxLen > MAX_CONTEXT_LENGTH) {
                    searchObj.put("max_length", MAX_CONTEXT_LENGTH)
                    modified = true
                }
            }

            if (modified) {
                try {
                    configFile.writeText(config.toString(4))
                } catch (_: Exception) { }
            }
        } catch (_: Exception) { }
    }
}
