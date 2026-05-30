package com.quantallm

enum class Backend(val displayName: String) {
    LLAMA_CPP("llama.cpp (GGUF)"),
    LLAMA_CPP_HEXAGON("llama.cpp (Hexagon DSP)"),
    ONNX_CPU("ONNX Runtime (CPU)"),
    ONNX_QNN("ONNX Runtime (QNN/NPU)");

    val isLlamaCpp: Boolean get() = this == LLAMA_CPP || this == LLAMA_CPP_HEXAGON
    val isOnnx: Boolean get() = this == ONNX_CPU || this == ONNX_QNN
}
