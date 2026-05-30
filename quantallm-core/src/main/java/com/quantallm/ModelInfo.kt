package com.quantallm

data class ModelInfo(
    val name: String,
    val format: String,
    val parameterCount: Long = 0L,
    val sizeBytes: Long = 0L,
    val contextLength: Int = 0,
    val vocabSize: Int = 0,
    val backend: Backend
)
