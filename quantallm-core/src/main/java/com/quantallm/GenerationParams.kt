package com.quantallm

data class GenerationParams(
    val maxTokens: Int = 512,
    val temperature: Float = 0.7f,
    val topK: Int = 50,
    val topP: Float = 0.9f,
    val minP: Float = 0.0f,
    val repeatPenalty: Float = 1.1f,
    val penaltyLastN: Int = 64,
    val frequencyPenalty: Float = 0.0f,
    val presencePenalty: Float = 0.0f,
    val seed: Int = 0
)
