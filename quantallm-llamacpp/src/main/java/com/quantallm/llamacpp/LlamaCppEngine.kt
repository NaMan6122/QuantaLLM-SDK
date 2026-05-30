package com.quantallm.llamacpp

import android.util.Log
import com.naman.quantallm.LlamaJni
import com.quantallm.Backend
import com.quantallm.ChatSession
import com.quantallm.GenerationParams
import com.quantallm.GenerationResult
import com.quantallm.InferenceEngine
import com.quantallm.ModelInfo
import org.json.JSONObject

class LlamaCppEngine(
    override val backend: Backend = Backend.LLAMA_CPP
) : InferenceEngine {

    companion object {
        private const val TAG = "LlamaCppEngine"
    }

    override val displayName: String
        get() = backend.displayName

    private var nativeHandle: Long = 0L
    private var loaded = false
    private var currentPath: String? = null

    init {
        if (!backend.isLlamaCpp) {
            throw IllegalArgumentException("LlamaCppEngine only supports LLAMA_CPP backends, got: $backend")
        }
        nativeHandle = LlamaJni.nativeCreate()
        if (nativeHandle == 0L) {
            throw IllegalStateException("Failed to allocate native LlamaInstance")
        }
        if (backend == Backend.LLAMA_CPP_HEXAGON) {
            LlamaJni.nativeSetBackendMode(nativeHandle, 2)
        } else {
            LlamaJni.nativeSetBackendMode(nativeHandle, 1)
        }
    }

    override fun loadModel(modelPath: String, threads: Int): GenerationResult {
        return try {
            val cpuCount = Runtime.getRuntime().availableProcessors()
            val actualThreads = when {
                threads <= 0 -> maxOf(4, cpuCount - 2)
                threads > cpuCount -> cpuCount
                else -> threads
            }

            Log.d(TAG, "loadModel: path=$modelPath, threads=$actualThreads")
            val result = LlamaJni.nativeInit(nativeHandle, modelPath, actualThreads)

            if (result == 0) {
                loaded = true
                currentPath = modelPath
                GenerationResult.Success("Model loaded successfully")
            } else {
                val error = LlamaJni.nativeGetLastError(nativeHandle).ifBlank { "nativeInit returned $result" }
                GenerationResult.Error(error, result)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Exception loading model", e)
            GenerationResult.Error("Failed to load model: ${e.message}", cause = e)
        }
    }

    override fun isModelLoaded(): Boolean = loaded

    override fun getModelInfo(): ModelInfo? {
        if (!loaded) return null
        return try {
            val json = JSONObject(LlamaJni.nativeGetModelInfo(nativeHandle))
            ModelInfo(
                name = json.optString("model_name", "Unknown"),
                format = "GGUF",
                parameterCount = json.optLong("n_params", 0),
                sizeBytes = json.optLong("n_bytes", 0),
                contextLength = json.optInt("n_ctx_train", 0),
                vocabSize = json.optInt("n_vocab", 0),
                backend = backend
            )
        } catch (e: Exception) {
            Log.w(TAG, "getModelInfo failed", e)
            null
        }
    }

    override fun getCurrentModelPath(): String? = currentPath

    override fun generate(prompt: String, params: GenerationParams): GenerationResult {
        if (!loaded) return GenerationResult.Error("No model loaded")
        if (prompt.isEmpty()) return GenerationResult.Error("Empty prompt")

        return try {
            val startTime = System.currentTimeMillis()
            val result = LlamaJni.nativeGenerateAdvanced(
                nativeHandle,
                prompt,
                params.maxTokens,
                params.temperature,
                params.topK,
                params.topP,
                params.minP,
                params.repeatPenalty,
                params.penaltyLastN,
                params.frequencyPenalty,
                params.presencePenalty,
                params.seed
            )
            val duration = System.currentTimeMillis() - startTime

            if (result.startsWith("Error:")) {
                GenerationResult.Error(result.removePrefix("Error: ").trim())
            } else {
                GenerationResult.Success(result, durationMs = duration)
            }
        } catch (e: Exception) {
            Log.e(TAG, "generate failed", e)
            GenerationResult.Error("Generation failed: ${e.message}", cause = e)
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
            val startTime = System.currentTimeMillis()
            var tokenCount = 0
            val jniCallback = object : LlamaJni.StreamingCallback {
                override fun onProgress(generatedTokens: Int, partialText: String) {
                    tokenCount = generatedTokens
                    callback.onToken(generatedTokens, partialText)
                }
            }

            val result = LlamaJni.nativeGenerateStreamingAdvanced(
                nativeHandle,
                prompt,
                params.maxTokens,
                params.temperature,
                params.topK,
                params.topP,
                params.minP,
                params.repeatPenalty,
                params.penaltyLastN,
                params.frequencyPenalty,
                params.presencePenalty,
                params.seed,
                jniCallback
            )
            val duration = System.currentTimeMillis() - startTime

            if (result.startsWith("Error:")) {
                GenerationResult.Error(result.removePrefix("Error: ").trim())
            } else {
                GenerationResult.Success(result, tokensGenerated = tokenCount, durationMs = duration)
            }
        } catch (e: Exception) {
            Log.e(TAG, "generateStreaming failed", e)
            GenerationResult.Error("Streaming generation failed: ${e.message}", cause = e)
        }
    }

    override fun abortGeneration() {
        LlamaJni.nativeAbortGeneration(nativeHandle)
    }

    override fun unloadModel() {
        if (!loaded) return
        try {
            LlamaJni.nativeShutdown(nativeHandle)
        } catch (e: Exception) {
            Log.e(TAG, "unloadModel failed", e)
        } finally {
            loaded = false
            currentPath = null
        }
    }

    override fun close() {
        unloadModel()
        if (nativeHandle != 0L) {
            LlamaJni.nativeDestroy(nativeHandle)
            nativeHandle = 0L
        }
    }

    override fun getLastError(): String {
        return try {
            LlamaJni.nativeGetLastError(nativeHandle)
        } catch (_: Exception) {
            ""
        }
    }

    fun setContextSize(contextSize: Int): Boolean = LlamaJni.nativeSetContextSize(nativeHandle, contextSize)

    fun getContextSize(): Int = LlamaJni.nativeGetContextSize(nativeHandle)

    fun setCacheDir(cacheDir: String) = LlamaJni.nativeSetCacheDir(nativeHandle, cacheDir)

    fun setHexagonConfig(
        ndev: Int = 1,
        nhvx: Int = 0,
        verbose: Boolean = false,
        profile: Boolean = false
    ): Boolean = LlamaJni.nativeSetHexagonConfig(nativeHandle, ndev, nhvx, verbose, profile)

    fun createChatSession(systemPrompt: String? = null): LlamaCppChatSession {
        val uuid = java.util.UUID.randomUUID().toString()
        val nativeId = LlamaJni.nativeCreateChatSession(nativeHandle, systemPrompt, uuid)
        if (nativeId < 0) {
            throw IllegalStateException("Failed to create chat session: ${getLastError()}")
        }
        LlamaJni.nativeSetActiveChatSession(nativeHandle, nativeId)
        return LlamaCppChatSession(nativeId, uuid)
    }

    inner class LlamaCppChatSession internal constructor(
        private val nativeId: Int,
        override val id: String
    ) : ChatSession {

        override fun generateResponse(
            message: String,
            params: GenerationParams,
            callback: InferenceEngine.StreamingCallback?
        ): GenerationResult {
            if (!loaded) return GenerationResult.Error("No model loaded")

            LlamaJni.nativeSetActiveChatSession(nativeHandle, nativeId)

            return try {
                val startTime = System.currentTimeMillis()

                val result = if (callback != null) {
                    var tokenCount = 0
                    val jniCb = object : LlamaJni.StreamingCallback {
                        override fun onProgress(generatedTokens: Int, partialText: String) {
                            tokenCount = generatedTokens
                            callback.onToken(generatedTokens, partialText)
                        }
                    }
                    LlamaJni.nativeChatGenerateStreaming(
                        nativeHandle, message, params.maxTokens, params.temperature, jniCb
                    )
                } else {
                    LlamaJni.nativeChatGenerate(nativeHandle, message, params.maxTokens, params.temperature)
                }

                val duration = System.currentTimeMillis() - startTime
                val json = try { JSONObject(result) } catch (_: Exception) { null }

                if (json != null && json.has("error")) {
                    GenerationResult.Error(json.getString("error"))
                } else {
                    val responseText = json?.optString("response", result) ?: result
                    GenerationResult.Success(responseText, durationMs = duration)
                }
            } catch (e: Exception) {
                GenerationResult.Error("Chat generation failed: ${e.message}", cause = e)
            }
        }

        override fun close() {
            LlamaJni.nativeDeleteChatSession(nativeHandle, nativeId)
        }
    }
}
