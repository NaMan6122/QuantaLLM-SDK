package com.quantallm

sealed class GenerationResult {
    data class Success(
        val text: String,
        val tokensGenerated: Int = 0,
        val durationMs: Long = 0L
    ) : GenerationResult()

    data class Error(
        val message: String,
        val code: Int = -1,
        val cause: Throwable? = null
    ) : GenerationResult()
}
