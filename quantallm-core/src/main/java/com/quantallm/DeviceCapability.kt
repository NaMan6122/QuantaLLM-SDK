package com.quantallm

import android.os.Build

data class DeviceCapability(
    val socName: String,
    val isQualcomm: Boolean,
    val htpVersion: Int,
    val supportedBackends: List<Backend>
) {
    companion object {
        fun detect(): DeviceCapability {
            val soc = Build.SOC_MODEL
            val mfr = Build.SOC_MANUFACTURER.lowercase()
            val isQualcomm = mfr == "qualcomm" || mfr == "qti"

            val htpVersion = if (isQualcomm) {
                when {
                    soc.startsWith("SM8750") -> 79
                    soc.startsWith("SM8650") -> 75
                    soc.startsWith("SM8550") || soc.startsWith("SM8475") -> 73
                    else -> 0
                }
            } else 0

            val backends = mutableListOf(Backend.LLAMA_CPP, Backend.ONNX_CPU)
            if (isQualcomm && htpVersion >= 73) {
                backends += Backend.LLAMA_CPP_HEXAGON
                backends += Backend.ONNX_QNN
            }

            return DeviceCapability(
                socName = soc,
                isQualcomm = isQualcomm,
                htpVersion = htpVersion,
                supportedBackends = backends
            )
        }
    }
}
