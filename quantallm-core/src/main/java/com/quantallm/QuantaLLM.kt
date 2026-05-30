package com.quantallm

object QuantaLLM {

    fun createEngine(backend: Backend): InferenceEngine {
        val className = when {
            backend.isLlamaCpp -> "com.quantallm.llamacpp.LlamaCppEngine"
            backend.isOnnx -> "com.quantallm.onnx.OnnxRuntimeEngine"
            else -> throw IllegalArgumentException("Unsupported backend: $backend")
        }

        try {
            val clazz = Class.forName(className)
            val constructor = clazz.getConstructor(Backend::class.java)
            return constructor.newInstance(backend) as InferenceEngine
        } catch (e: ClassNotFoundException) {
            throw IllegalStateException(
                "Backend module for $backend not found on classpath. " +
                        "Add the corresponding dependency (e.g., quantallm-llamacpp or quantallm-onnx).",
                e
            )
        }
    }

    fun detectCapabilities(): DeviceCapability = DeviceCapability.detect()
}
