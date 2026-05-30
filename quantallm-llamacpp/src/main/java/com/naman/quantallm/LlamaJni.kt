package com.naman.quantallm

internal object LlamaJni {
    init {
        System.loadLibrary("llama_jni")
    }

    external fun nativeCreate(): Long
    external fun nativeDestroy(handle: Long)

    external fun nativeInit(handle: Long, modelPath: String, nThreads: Int): Int
    external fun nativeSetBackendMode(handle: Long, mode: Int): Boolean
    external fun nativeSetHexagonConfig(
        handle: Long,
        ndev: Int,
        nhvx: Int,
        verbose: Boolean,
        profile: Boolean
    ): Boolean
    external fun nativeGetLastError(handle: Long): String
    external fun nativeGetModelInfo(handle: Long): String

    external fun nativeGenerate(
        handle: Long,
        prompt: String,
        maxTokens: Int,
        temperature: Float
    ): String

    external fun nativeGenerateAdvanced(
        handle: Long,
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        minP: Float,
        repeatPenalty: Float,
        penaltyLastN: Int,
        frequencyPenalty: Float,
        presencePenalty: Float,
        seed: Int
    ): String

    external fun nativeGenerateStreaming(
        handle: Long,
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        callback: StreamingCallback
    ): String

    external fun nativeGenerateStreamingAdvanced(
        handle: Long,
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        minP: Float,
        repeatPenalty: Float,
        penaltyLastN: Int,
        frequencyPenalty: Float,
        presencePenalty: Float,
        seed: Int,
        callback: StreamingCallback
    ): String

    external fun nativeShutdown(handle: Long): Int
    external fun nativeAbortGeneration(handle: Long)
    external fun nativeSetContextSize(handle: Long, contextSize: Int): Boolean
    external fun nativeSetCacheDir(handle: Long, cacheDir: String)
    external fun nativeGetContextSize(handle: Long): Int

    external fun nativeCreateChatSession(
        handle: Long,
        systemPrompt: String?,
        sessionIdUuid: String
    ): Int

    external fun nativeSetActiveChatSession(handle: Long, sessionId: Int): Boolean
    external fun nativeDeleteChatSession(handle: Long, sessionId: Int)

    external fun nativeChatGenerate(
        handle: Long,
        message: String,
        maxTokens: Int,
        temperature: Float
    ): String

    external fun nativeChatGenerateStreaming(
        handle: Long,
        message: String,
        maxTokens: Int,
        temperature: Float,
        callback: StreamingCallback
    ): String

    external fun nativeGetChatSessionInfo(handle: Long, sessionId: Int): String
    external fun nativeListChatSessions(handle: Long): String

    interface StreamingCallback {
        fun onProgress(generatedTokens: Int, partialText: String)
    }
}
