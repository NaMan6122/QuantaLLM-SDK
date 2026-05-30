package com.quantallm

import android.content.Context

object QuantaLLM {

    private var license: LicenseKey? = null
    private var initialized = false

    fun initialize(context: Context, licenseKey: String) {
        license = LicenseKey.validate(context.applicationContext, licenseKey)
        initialized = true
    }

    val isInitialized: Boolean get() = initialized

    val licenseTier: String get() = license?.tier ?: "none"

    fun createEngine(backend: Backend): InferenceEngine {
        val lic = license
            ?: throw LicenseException("QuantaLLM.initialize(context, licenseKey) must be called before createEngine()")
        if (lic.isExpired()) {
            throw LicenseException("License expired on ${lic.expiryDate}")
        }

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
