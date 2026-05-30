package com.quantallm

interface ChatSession : AutoCloseable {

    val id: String

    fun generateResponse(
        message: String,
        params: GenerationParams = GenerationParams(),
        callback: InferenceEngine.StreamingCallback? = null
    ): GenerationResult

    override fun close()
}
