-keep class com.quantallm.llamacpp.** { *; }
-keepclassmembers class com.quantallm.llamacpp.internal.LlamaJni {
    native <methods>;
}
