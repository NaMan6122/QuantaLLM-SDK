# Keep engine class for reflection from QuantaLLM.createEngine()
-keep public class com.quantallm.onnx.OnnxRuntimeEngine {
    public <init>(com.quantallm.Backend);
    public *;
}
