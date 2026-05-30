# Keep engine class for reflection from QuantaLLM.createEngine()
-keep public class com.quantallm.llamacpp.LlamaCppEngine {
    public <init>(com.quantallm.Backend);
    public *;
}

# Keep JNI bridge (native method names must match C++ symbols)
-keep class com.naman.quantallm.LlamaJni { *; }
-keep class com.naman.quantallm.LlamaJni$StreamingCallback { *; }
