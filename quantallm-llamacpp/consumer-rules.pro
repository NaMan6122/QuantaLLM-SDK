# QuantaLLM LlamaCpp - Consumer ProGuard Rules

# Keep engine class (loaded via reflection by QuantaLLM.createEngine())
-keep public class com.quantallm.llamacpp.LlamaCppEngine {
    public <init>(com.quantallm.Backend);
    public *;
}
-keep public class com.quantallm.llamacpp.LlamaCppEngine$LlamaCppChatSession { *; }

# Keep JNI bridge (native method names must match C++ symbols)
-keep class com.naman.quantallm.LlamaJni { *; }
-keep interface com.naman.quantallm.LlamaJni$StreamingCallback { *; }
