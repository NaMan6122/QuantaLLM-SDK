package com.quantallm

interface InferenceEngine : AutoCloseable {

    val backend: Backend

    val displayName: String

    fun loadModel(modelPath: String, threads: Int = 0): GenerationResult

    fun isModelLoaded(): Boolean

    fun getModelInfo(): ModelInfo?

    fun getCurrentModelPath(): String?

    fun generate(
        prompt: String,
        params: GenerationParams = GenerationParams()
    ): GenerationResult

    fun generateStreaming(
        prompt: String,
        params: GenerationParams = GenerationParams(),
        callback: StreamingCallback
    ): GenerationResult

    fun abortGeneration()

    fun unloadModel()

    fun getLastError(): String

    override fun close() {
        unloadModel()
    }

    fun interface StreamingCallback {
        fun onToken(tokensGenerated: Int, partialText: String)
    }
}
