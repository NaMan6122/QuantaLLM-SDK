# Keep public SDK API
-keep public class com.quantallm.QuantaLLM { public *; }
-keep public class com.quantallm.Backend { *; }
-keep public class com.quantallm.InferenceEngine { *; }
-keep public class com.quantallm.InferenceEngine$StreamingCallback { *; }
-keep public class com.quantallm.GenerationParams { *; }
-keep public class com.quantallm.GenerationResult { *; }
-keep public class com.quantallm.GenerationResult$Success { *; }
-keep public class com.quantallm.GenerationResult$Error { *; }
-keep public class com.quantallm.ModelInfo { *; }
-keep public class com.quantallm.ChatSession { *; }
-keep public class com.quantallm.DeviceCapability { *; }
-keep public class com.quantallm.LicenseException { *; }

# Obfuscate LicenseKey internals
-keep public class com.quantallm.LicenseKey {
    public boolean isExpired();
    public java.lang.String getExpiryDate();
    public java.lang.String getPackageName();
    public java.lang.String getTier();
}
