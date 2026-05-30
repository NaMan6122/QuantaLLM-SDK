//==============================================================================
// LLaMA JNI Bridge - Android Native Interface
//==============================================================================
// Purpose: Bridge between Java/Kotlin and llama.cpp for on-device inference
// Last Modified: May 30, 2026
//==============================================================================

#include <jni.h>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <sstream>
#include <android/log.h>
#include <map>
#include <chrono>
#include <atomic>
#include <dlfcn.h>
#include <sys/stat.h>
#include <sched.h>
#include <unistd.h>
#include <dirent.h>
#include <cerrno>
#include "llama.h"   // from llama.cpp repo
#include "ggml-backend.h"

//==============================================================================
// SECTION: Logging & Configuration
//==============================================================================

#define LOG_TAG "LlamaJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

//==============================================================================
// SECTION: Backend Mode Enum
//==============================================================================

enum BackendMode : int {
    BACKEND_MODE_CPU = 1,
    BACKEND_MODE_HEXAGON = 2,
};

//==============================================================================
// SECTION: KV Cache Session Types
//==============================================================================

enum class SessionMode {
    DEFAULT,
    CHAT
};

struct Turn {
    int32_t turn_id;
    std::string user_message;
    std::string assistant_response;
    int32_t tokens_used;
    int64_t timestamp;
};

struct KVCacheSession {
    int32_t seq_id;
    int32_t n_past;
    std::vector<llama_token> tokens_history;
    int64_t last_access_time;
    bool is_active;

    SessionMode mode = SessionMode::DEFAULT;
    std::string session_id_uuid;
    std::string system_prompt;
    std::vector<Turn> turns;

    int32_t parent_seq_id = -1;
    std::vector<int32_t> inherited_seq_ids;
};

//==============================================================================
// SECTION: Per-Instance State
//==============================================================================

struct LlamaInstance {
    llama_model*   model = nullptr;
    llama_context* ctx   = nullptr;
    std::string    last_error;
    std::string    model_path;
    std::string    cache_dir;
    int            backend_mode = BACKEND_MODE_CPU;
    int            hexagon_ndev = 1;
    int            hexagon_nhvx = 0;
    bool           hexagon_verbose = false;
    bool           hexagon_profile = false;
    bool           hexagon_config_dirty = false;
    std::atomic<bool> abort_generation{false};
    std::map<int32_t, KVCacheSession> kv_sessions;
    int32_t        next_seq_id = 0;
    int32_t        current_session_id = -1;
    int32_t        n_ctx = 2048;
    int32_t        requested_n_ctx = 0;
};

//==============================================================================
// SECTION: Global State (process-wide only)
//==============================================================================

static bool g_backend_initialized = false;

// Static error string for the llama log callback (C callback, no instance pointer).
// Copied into inst->last_error when needed.
static std::string g_log_last_error;

//==============================================================================
// SECTION: Instance Helper
//==============================================================================

static LlamaInstance* get_instance(jlong handle) {
    return reinterpret_cast<LlamaInstance*>(handle);
}

//==============================================================================
// SECTION: Free Functions (no instance state)
//==============================================================================

static std::string detect_native_lib_dir() {
    Dl_info info = {};
    if (dladdr((void *) &llama_backend_init, &info) == 0 || info.dli_fname == nullptr) {
        return "";
    }

    std::string path(info.dli_fname);
    const size_t apk_sep = path.find("!/");
    if (apk_sep != std::string::npos) {
        return "";
    }
    const size_t sep = path.find_last_of('/');
    if (sep == std::string::npos) {
        return "";
    }
    return path.substr(0, sep);
}

// Read available RAM from /proc/meminfo (returns bytes, or 0 on failure)
static size_t get_available_ram_bytes() {
    FILE * f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    size_t avail_kb = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "MemAvailable:", 13) == 0) {
            sscanf(line + 13, " %zu", &avail_kb);
            break;
        }
    }
    fclose(f);
    return avail_kb * 1024;
}

// Set thread affinity to the fastest N cores (big.LITTLE aware)
static void pin_threads_to_perf_cores(int n_threads) {
    std::vector<std::pair<long, int>> core_freqs;
    for (int cpu = 0; cpu < 16; ++cpu) {
        char path[128];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu);
        FILE * f = fopen(path, "r");
        if (!f) continue;
        long freq = 0;
        if (fscanf(f, "%ld", &freq) == 1) {
            core_freqs.push_back({freq, cpu});
        }
        fclose(f);
    }
    if (core_freqs.empty()) {
        LOGD("Thread affinity: cannot read CPU frequencies, skipping");
        return;
    }
    std::sort(core_freqs.begin(), core_freqs.end(),
              [](const auto & a, const auto & b) { return a.first > b.first; });

    cpu_set_t mask;
    CPU_ZERO(&mask);
    int pinned = 0;
    for (const auto & [freq, cpu_id] : core_freqs) {
        if (pinned >= n_threads) break;
        CPU_SET(cpu_id, &mask);
        pinned++;
    }
    if (sched_setaffinity(0, sizeof(mask), &mask) == 0) {
        LOGI("Thread affinity: pinned to %d fastest cores (max freq %ld kHz)",
             pinned, core_freqs[0].first);
    } else {
        LOGW("Thread affinity: sched_setaffinity failed: %s", strerror(errno));
    }
}

static void configure_hexagon_runtime_env(int hexagon_ndev, int hexagon_nhvx,
                                           bool hexagon_verbose, bool hexagon_profile) {
    const std::string lib_dir = detect_native_lib_dir();
    if (!lib_dir.empty()) {
        setenv("ADSP_LIBRARY_PATH", lib_dir.c_str(), 1);
        if (getenv("LD_LIBRARY_PATH") == nullptr) {
            setenv("LD_LIBRARY_PATH", lib_dir.c_str(), 1);
        }
        LOGI("Configured ADSP_LIBRARY_PATH=%s", lib_dir.c_str());
    } else {
        LOGW("Unable to detect native lib dir for ADSP_LIBRARY_PATH");
    }

    if (getenv("GGML_HEXAGON_EXPERIMENTAL") == nullptr) {
        setenv("GGML_HEXAGON_EXPERIMENTAL", "1", 0);
    }

    const std::string ndev_str = std::to_string(hexagon_ndev);
    const std::string nhvx_str = std::to_string(hexagon_nhvx);

    setenv("GGML_HEXAGON_NDEV", ndev_str.c_str(), 1);
    setenv("GGML_HEXAGON_NHVX", nhvx_str.c_str(), 1);
    setenv("GGML_HEXAGON_VERBOSE", hexagon_verbose ? "1" : "0", 1);
    setenv("GGML_HEXAGON_PROFILE", hexagon_profile ? "1" : "0", 1);

    LOGI("Configured Hexagon env: EXPERIMENTAL=%s NDEV=%s NHVX=%s VERBOSE=%s PROFILE=%s",
         getenv("GGML_HEXAGON_EXPERIMENTAL"),
         ndev_str.c_str(),
         nhvx_str.c_str(),
         hexagon_verbose ? "1" : "0",
         hexagon_profile ? "1" : "0");
}

static bool contains_ignore_case(const char * text, const char * token) {
    if (text == nullptr || token == nullptr) {
        return false;
    }

    const std::string haystack(text);
    const std::string needle(token);
    if (needle.empty()) {
        return false;
    }

    for (size_t i = 0; i + needle.size() <= haystack.size(); ++i) {
        bool matches = true;
        for (size_t j = 0; j < needle.size(); ++j) {
            const char h = static_cast<char>(std::tolower(static_cast<unsigned char>(haystack[i + j])));
            const char n = static_cast<char>(std::tolower(static_cast<unsigned char>(needle[j])));
            if (h != n) {
                matches = false;
                break;
            }
        }
        if (matches) {
            return true;
        }
    }

    return false;
}

static bool is_hexagon_device(ggml_backend_dev_t device) {
    if (!device) {
        return false;
    }

    const char * dev_name = ggml_backend_dev_name(device);
    const char * dev_desc = ggml_backend_dev_description(device);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    const char * reg_name = reg ? ggml_backend_reg_name(reg) : nullptr;

    return contains_ignore_case(dev_name, "hexagon") ||
           contains_ignore_case(dev_desc, "hexagon") ||
           contains_ignore_case(reg_name, "hexagon") ||
           contains_ignore_case(dev_name, "htp") ||
           contains_ignore_case(dev_desc, "htp") ||
           contains_ignore_case(reg_name, "htp");
}

static ggml_backend_dev_t find_hexagon_device() {
    const size_t device_count = ggml_backend_dev_count();
    for (size_t i = 0; i < device_count; ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (is_hexagon_device(device)) {
            return device;
        }
    }
    return nullptr;
}

static size_t find_hexagon_devices(ggml_backend_dev_t * out_devices, size_t max_devices) {
    if (!out_devices || max_devices == 0) {
        return 0;
    }

    size_t found = 0;
    const size_t device_count = ggml_backend_dev_count();
    for (size_t i = 0; i < device_count && found < max_devices; ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (is_hexagon_device(device)) {
            out_devices[found++] = device;
        }
    }
    return found;
}

static void set_last_error(std::string & dest, const std::string & message) {
    dest = message;
}

static void llama_log_bridge(enum ggml_log_level level, const char * text, void * /* user_data */) {
    if (text == nullptr) {
        return;
    }

    switch (level) {
        case GGML_LOG_LEVEL_ERROR:
            LOGE("%s", text);
            if (!g_log_last_error.empty()) {
                g_log_last_error += " | ";
            }
            g_log_last_error += text;
            break;
        case GGML_LOG_LEVEL_WARN:
            LOGW("%s", text);
            break;
        case GGML_LOG_LEVEL_INFO:
            LOGI("%s", text);
            break;
        default:
            LOGD("%s", text);
            break;
    }
}

static bool preflight_model_file(std::string & last_error, const char * model_path) {
    if (model_path == nullptr || model_path[0] == '\0') {
        set_last_error(last_error, "Model path is empty");
        return false;
    }

    struct stat st {};
    if (stat(model_path, &st) != 0) {
        std::ostringstream oss;
        oss << "stat() failed for model path: errno=" << errno << " (" << strerror(errno) << ")";
        set_last_error(last_error, oss.str());
        return false;
    }

    FILE * fp = fopen(model_path, "rb");
    if (fp == nullptr) {
        std::ostringstream oss;
        oss << "fopen() failed for model path: errno=" << errno << " (" << strerror(errno) << ")";
        set_last_error(last_error, oss.str());
        return false;
    }

    unsigned char magic[4] = {0, 0, 0, 0};
    const size_t n_read = fread(magic, 1, sizeof(magic), fp);
    fclose(fp);
    if (n_read != sizeof(magic)) {
        set_last_error(last_error, "Failed to read GGUF magic from model file");
        return false;
    }

    if (!(magic[0] == 'G' && magic[1] == 'G' && magic[2] == 'U' && magic[3] == 'F')) {
        std::ostringstream oss;
        oss << "Invalid model magic bytes: 0x"
            << std::hex
            << (int) magic[0] << (int) magic[1] << (int) magic[2] << (int) magic[3]
            << " (expected GGUF)";
        set_last_error(last_error, oss.str());
        return false;
    }

    LOGI("Model preflight passed: size=%lld bytes", (long long) st.st_size);
    return true;
}

// FNV-1a 64-bit hash for cache key derivation
static uint64_t fnv1a_hash(const std::string & s) {
    uint64_t h = 14695981039346656037ULL;
    for (char c : s) {
        h ^= (uint8_t)c;
        h *= 1099511628211ULL;
    }
    return h;
}

static std::string get_system_prompt_cache_path(const std::string & cache_dir,
                                                 const std::string & model_path,
                                                 const std::string & system_prompt) {
    if (cache_dir.empty() || model_path.empty()) return "";
    uint64_t key = fnv1a_hash(model_path + "|" + system_prompt);
    char filename[64];
    snprintf(filename, sizeof(filename), "sp_kv_%016llx.bin", (unsigned long long)key);
    return cache_dir + "/" + filename;
}

/**
 * Properly escape a string for JSON embedding.
 */
static std::string json_escape_string(const std::string & input) {
    std::string escaped;
    escaped.reserve(input.size() + 64);
    for (char c : input) {
        switch (c) {
            case '"':  escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char hex[8];
                    snprintf(hex, sizeof(hex), "\\u%04x", (unsigned char)c);
                    escaped += hex;
                } else {
                    escaped += c;
                }
                break;
        }
    }
    return escaped;
}

/**
 * Build a JSON response string for chat generation results.
 */
static std::string build_chat_json_response(
        const std::string & response,
        int32_t turn_id,
        int32_t tokens_used,
        int32_t n_past) {
    std::string escaped = json_escape_string(response);
    std::ostringstream oss;
    oss << "{\"response\":\"" << escaped
        << "\",\"turn_id\":" << turn_id
        << ",\"tokens_used\":" << tokens_used
        << ",\"n_past\":" << n_past << "}";
    return oss.str();
}

//==============================================================================
// SECTION: KV Cache Utility Functions
//==============================================================================

static void kv_cache_clear_old_sessions(LlamaInstance* inst, int32_t keep_seq_id = -1) {
    if (!inst->ctx) return;
    llama_memory_t mem = llama_get_memory(inst->ctx);

    int64_t current_time = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    std::vector<int32_t> to_remove;
    for (auto& pair : inst->kv_sessions) {
        if (pair.first != keep_seq_id && !pair.second.is_active) {
            if (current_time - pair.second.last_access_time > 300) {
                to_remove.push_back(pair.first);
            }
        }
    }

    for (int32_t seq_id : to_remove) {
        llama_memory_seq_rm(mem, seq_id, -1, -1);
        inst->kv_sessions.erase(seq_id);
        LOGI("[KV] Removed old session seq_id=%d", seq_id);
    }
}

// Tokenize prompt into llama tokens
static bool tokenize_prompt(llama_model* model, const std::string & prompt, std::vector<llama_token> & out) {
    if (!model) return false;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int32_t n_tokens = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(), nullptr, 0, true, false);
    if (n_tokens == INT32_MIN) {
        LOGE("Tokenization overflow");
        return false;
    }
    if (n_tokens < 0) n_tokens = -n_tokens;
    out.resize(n_tokens);
    int32_t written = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(), out.data(), n_tokens, true, false);
    if (written < 0) {
        LOGE("Tokenization failed after allocation");
        return false;
    }
    out.resize(written);
    return true;
}

// Convert a single token to piece and append to string
static void append_token_piece(llama_model* model, llama_token token, std::string & out) {
    if (!model) return;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    char buf[16];
    int32_t len = llama_token_to_piece(vocab, token, buf, (int32_t)sizeof(buf), 0, true);
    if (len < 0 || len >= (int32_t)sizeof(buf)) {
        int32_t need = len < 0 ? -len : len + 1;
        std::vector<char> big(need);
        int32_t len2 = llama_token_to_piece(vocab, token, big.data(), need, 0, true);
        if (len2 > 0) out.append(big.data(), len2);
    } else if (len > 0) {
        out.append(buf, len);
    }
}

//==============================================================================
// SECTION: Instance Lifecycle JNI
//==============================================================================

extern "C" JNIEXPORT jlong JNICALL
Java_com_naman_quantallm_LlamaJni_nativeCreate(JNIEnv*, jobject) {
    return reinterpret_cast<jlong>(new LlamaInstance());
}

extern "C" JNIEXPORT void JNICALL
Java_com_naman_quantallm_LlamaJni_nativeDestroy(JNIEnv*, jobject, jlong handle) {
    auto* inst = get_instance(handle);
    if (inst->ctx) { llama_free(inst->ctx); }
    if (inst->model) { llama_model_free(inst->model); }
    delete inst;
}

//==============================================================================
// SECTION: Model Lifecycle
//==============================================================================

extern "C"
JNIEXPORT jint JNICALL
Java_com_naman_quantallm_LlamaJni_nativeInit(
        JNIEnv *env, jobject, jlong handle, jstring modelPath_, jint nThreads) {
    auto* inst = get_instance(handle);
    inst->last_error.clear();

    if (inst->ctx) {
        LOGI("nativeInit: already initialized, shutting down existing model");
        llama_free(inst->ctx);
        inst->ctx = nullptr;
        if (inst->model) {
            llama_model_free(inst->model);
            inst->model = nullptr;
        }
        inst->kv_sessions.clear();
        inst->next_seq_id = 0;
        inst->current_session_id = -1;
    }

    const char *modelPath = env->GetStringUTFChars(modelPath_, 0);
    inst->model_path = modelPath;
    LOGI("Loading model: %s", modelPath);
    const bool is_q8_0_model_path =
            contains_ignore_case(modelPath, "q8_0") ||
            contains_ignore_case(modelPath, "q8-0");
    configure_hexagon_runtime_env(inst->hexagon_ndev, inst->hexagon_nhvx,
                                   inst->hexagon_verbose, inst->hexagon_profile);

    if (!preflight_model_file(inst->last_error, modelPath)) {
        LOGE("%s", inst->last_error.c_str());
        env->ReleaseStringUTFChars(modelPath_, modelPath);
        return -1;
    }

    // Re-init backend if runtime knobs changed since last backend init.
    if (g_backend_initialized && inst->hexagon_config_dirty) {
        LOGI("Hexagon runtime config changed; re-initializing backend");
        llama_backend_free();
        g_backend_initialized = false;
        inst->hexagon_config_dirty = false;
    }

    // Initialize backend only once
    if (!g_backend_initialized) {
        llama_log_set(llama_log_bridge, nullptr);
        llama_backend_init();
        g_backend_initialized = true;
        LOGI("llama backend initialized");
    }
    LOGI("GPU offload support visible: %s", llama_supports_gpu_offload() ? "yes" : "no");

    // Configure model parameters
    llama_model_params mparams = llama_model_default_params();

    mparams.use_mmap = true;
    struct stat model_stat;
    if (stat(modelPath, &model_stat) == 0) {
        LOGI("mmap enabled for model (%lld MB)", (long long)(model_stat.st_size / (1024 * 1024)));
    }

    auto make_cpu_params = [&](const llama_model_params & source) {
        llama_model_params cpu_params = source;
        cpu_params.n_gpu_layers = 0;
        cpu_params.split_mode = LLAMA_SPLIT_MODE_NONE;
        cpu_params.main_gpu = -1;
        cpu_params.devices = nullptr;
        return cpu_params;
    };

    bool used_cpu_fallback = false;
    bool hexagon_requested = (inst->backend_mode == BACKEND_MODE_HEXAGON);
    constexpr size_t MAX_HEXAGON_DEVICES = 4;
    ggml_backend_dev_t hexagon_device = nullptr;
    ggml_backend_dev_t selected_devices[MAX_HEXAGON_DEVICES + 1] = { nullptr };

    if (hexagon_requested) {
        const size_t requested_ndev = static_cast<size_t>(std::clamp(inst->hexagon_ndev, 1, (int) MAX_HEXAGON_DEVICES));
        const size_t found_ndev = find_hexagon_devices(selected_devices, requested_ndev);

        if (found_ndev > 0) {
            hexagon_device = selected_devices[0];
            selected_devices[found_ndev] = nullptr;
            mparams.devices = selected_devices;
            if (found_ndev > 1) {
                mparams.split_mode = LLAMA_SPLIT_MODE_LAYER;
                mparams.main_gpu = 0;
            } else {
                mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
            }

            std::ostringstream oss;
            oss << "Hexagon backend requested, selected devices:";
            for (size_t i = 0; i < found_ndev; ++i) {
                const char * dev_name = ggml_backend_dev_name(selected_devices[i]);
                oss << " " << (dev_name ? dev_name : "unknown");
            }
            oss << " | split_mode=" << (found_ndev > 1 ? "LAYER" : "NONE");
            if (found_ndev < requested_ndev) {
                oss << " (requested " << requested_ndev << ", found " << found_ndev << ")";
            }
            LOGI("%s", oss.str().c_str());
        } else {
            LOGW("Hexagon backend requested but unavailable, using CPU mode");
            mparams = make_cpu_params(mparams);
            used_cpu_fallback = true;
        }
    } else {
        mparams = make_cpu_params(mparams);
        LOGI("CPU backend requested");
    }

    // Load model with selected backend first, then retry CPU-only if needed.
    inst->model = llama_model_load_from_file(modelPath, mparams);
    if (!inst->model && hexagon_requested && hexagon_device != nullptr) {
        LOGW("Hexagon load failed, retrying CPU-only");
        llama_model_params cpu_params = make_cpu_params(mparams);
        inst->model = llama_model_load_from_file(modelPath, cpu_params);
        used_cpu_fallback = (inst->model != nullptr);
    }

    if (!inst->model) {
        std::string reason = "Failed to load model from: ";
        reason += modelPath;
        reason += " (backend=";
        reason += (inst->backend_mode == BACKEND_MODE_HEXAGON ? "Hexagon" : "CPU");
        reason += ")";
        set_last_error(inst->last_error, reason);
        LOGE("%s", reason.c_str());
        env->ReleaseStringUTFChars(modelPath_, modelPath);
        return -1;
    }

    LOGI("Model loaded successfully");

    // Configure context parameters
    llama_context_params cparams = llama_context_default_params();

    cparams.n_threads = nThreads;
    const int n_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    cparams.n_threads_batch = (n_cpus > nThreads) ? n_cpus : nThreads;

    // Dynamic context window: auto-detect from model, RAM-aware cap
    {
        const int32_t n_ctx_train = llama_model_n_ctx_train(inst->model);
        if (inst->requested_n_ctx > 0) {
            inst->n_ctx = inst->requested_n_ctx;
        } else {
            const size_t avail_ram = get_available_ram_bytes();
            const int32_t n_head = llama_model_n_head(inst->model);
            const int32_t n_head_kv = llama_model_n_head_kv(inst->model);
            const bool is_gqa = (n_head_kv > 0 && n_head_kv < n_head);

            int32_t ctx_cap;
            if (avail_ram > (size_t)(8ULL * 1024 * 1024 * 1024)) {
                ctx_cap = is_gqa ? 8192 : 8192;
            } else if (avail_ram > (size_t)(6ULL * 1024 * 1024 * 1024)) {
                ctx_cap = is_gqa ? 8192 : 4096;
            } else {
                ctx_cap = 4096;
            }
            inst->n_ctx = std::min(n_ctx_train, ctx_cap);
            LOGI("GQA=%s (n_head=%d, n_head_kv=%d), RAM=%.1fGB",
                 is_gqa ? "yes" : "no", n_head, n_head_kv, avail_ram / (1024.0 * 1024 * 1024));
        }
        cparams.n_ctx = inst->n_ctx;
        LOGI("Context window set to %d (model trained with %d, requested %d)",
             inst->n_ctx, n_ctx_train, inst->requested_n_ctx);
    }

    cparams.n_batch = 512;
    cparams.n_ubatch = 256;

    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    cparams.type_k = GGML_TYPE_Q8_0;
    cparams.type_v = GGML_TYPE_F16;

    if (is_q8_0_model_path) {
        cparams.offload_kqv = false;
        LOGW("Applied Q8_0 safety profile: offload_kqv=OFF (prevents HTP KV buffer mapping failures)");
    } else if (inst->backend_mode == BACKEND_MODE_CPU || used_cpu_fallback) {
        cparams.offload_kqv = false;
        cparams.op_offload = false;
        LOGI("Using CPU-only context settings after offload fallback");
    }

    // Create context
    inst->ctx = llama_init_from_model(inst->model, cparams);
    if (!inst->ctx) {
        const std::string reason = "Failed to create llama context (possible OOM)";
        set_last_error(inst->last_error, reason);
        LOGE("%s", reason.c_str());
        llama_model_free(inst->model);
        inst->model = nullptr;
        env->ReleaseStringUTFChars(modelPath_, modelPath);
        return -2;
    }

    env->ReleaseStringUTFChars(modelPath_, modelPath);

    // Pin inference threads to performance cores for big.LITTLE SoCs
    if (inst->backend_mode == BACKEND_MODE_CPU || used_cpu_fallback) {
        pin_threads_to_perf_cores(nThreads);
    }

    return 0;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_naman_quantallm_LlamaJni_nativeGetLastError(
        JNIEnv *env, jobject, jlong handle) {
    auto* inst = get_instance(handle);
    if (!inst->last_error.empty()) {
        return env->NewStringUTF(inst->last_error.c_str());
    }
    return env->NewStringUTF(g_log_last_error.c_str());
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_naman_quantallm_LlamaJni_nativeSetBackendMode(
        JNIEnv *, jobject, jlong handle, jint mode) {
    auto* inst = get_instance(handle);
    if (mode != BACKEND_MODE_CPU && mode != BACKEND_MODE_HEXAGON) {
        set_last_error(inst->last_error, "Invalid backend mode requested");
        LOGE("Invalid backend mode: %d", mode);
        return JNI_FALSE;
    }

    inst->backend_mode = mode;
    LOGI("Backend mode set to %s%s",
         (inst->backend_mode == BACKEND_MODE_HEXAGON ? "Hexagon" : "CPU"),
         (inst->ctx ? " (will take effect on next model load)" : ""));
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_naman_quantallm_LlamaJni_nativeSetHexagonConfig(
        JNIEnv *, jobject, jlong handle, jint ndev, jint nhvx, jboolean verbose, jboolean profile) {
    auto* inst = get_instance(handle);
    if (ndev < 1 || ndev > 4) {
        set_last_error(inst->last_error, "Hexagon NDEV must be in range [1, 4]");
        LOGE("Invalid Hexagon NDEV=%d", ndev);
        return JNI_FALSE;
    }
    if (nhvx < 0 || nhvx > 8) {
        set_last_error(inst->last_error, "Hexagon NHVX must be in range [0, 8]");
        LOGE("Invalid Hexagon NHVX=%d", nhvx);
        return JNI_FALSE;
    }

    const bool new_verbose = (verbose == JNI_TRUE);
    const bool new_profile = (profile == JNI_TRUE);

    const bool changed = (inst->hexagon_ndev != ndev) ||
                         (inst->hexagon_nhvx != nhvx) ||
                         (inst->hexagon_verbose != new_verbose) ||
                         (inst->hexagon_profile != new_profile);

    inst->hexagon_ndev = ndev;
    inst->hexagon_nhvx = nhvx;
    inst->hexagon_verbose = new_verbose;
    inst->hexagon_profile = new_profile;

    if (changed) {
        inst->hexagon_config_dirty = true;
    }

    LOGI("Hexagon config set: NDEV=%d NHVX=%d VERBOSE=%s PROFILE=%s%s",
         inst->hexagon_ndev,
         inst->hexagon_nhvx,
         inst->hexagon_verbose ? "ON" : "OFF",
         inst->hexagon_profile ? "ON" : "OFF",
         inst->ctx ? " (will take effect on next model load)" : "");
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_naman_quantallm_LlamaJni_nativeSetContextSize(
        JNIEnv *, jobject, jlong handle, jint contextSize) {
    auto* inst = get_instance(handle);
    if (contextSize < 0 || contextSize > 131072) {
        LOGE("Invalid context size: %d (must be 0-131072)", contextSize);
        return JNI_FALSE;
    }
    inst->requested_n_ctx = contextSize;
    LOGI("Context size configured: %d (0 = auto-detect)", contextSize);
    return JNI_TRUE;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_naman_quantallm_LlamaJni_nativeSetCacheDir(
        JNIEnv *env, jobject, jlong handle, jstring cacheDir_) {
    auto* inst = get_instance(handle);
    const char *dir = env->GetStringUTFChars(cacheDir_, 0);
    inst->cache_dir = dir;
    env->ReleaseStringUTFChars(cacheDir_, dir);
    LOGI("Cache dir set to: %s", inst->cache_dir.c_str());
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_naman_quantallm_LlamaJni_nativeGetContextSize(
        JNIEnv *, jobject, jlong handle) {
    auto* inst = get_instance(handle);
    return inst->n_ctx;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_naman_quantallm_LlamaJni_nativeGetModelInfo(JNIEnv *env, jobject, jlong handle) {
    auto* inst = get_instance(handle);
    if (!inst->model) {
        return env->NewStringUTF("{\"error\":\"No model loaded\"}");
    }

    char info[512];
    const int32_t n_params = llama_model_n_params(inst->model);
    const int64_t n_bytes = llama_model_size(inst->model);
    const int32_t n_ctx_train = llama_model_n_ctx_train(inst->model);
    const llama_vocab * vocab = llama_model_get_vocab(inst->model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    snprintf(info, sizeof(info),
        "{\"params\":%d,\"size_mb\":%.2f,\"ctx_train\":%d,\"vocab\":%d,\"loaded\":true}",
        n_params,
        n_bytes / (1024.0 * 1024.0),
        n_ctx_train,
        n_vocab
    );

    return env->NewStringUTF(info);
}

//==============================================================================
// SECTION: KV Cache Session Management JNI APIs
//==============================================================================

extern "C"
JNIEXPORT jint JNICALL
Java_com_naman_quantallm_LlamaJni_nativeCreateKVSession(JNIEnv *env, jobject, jlong handle) {
    auto* inst = get_instance(handle);
    if (!inst->ctx) {
        LOGE("[KV] Cannot create session: context not initialized");
        return -1;
    }

    int32_t seq_id = inst->next_seq_id++;
    KVCacheSession session;
    session.seq_id = seq_id;
    session.n_past = 0;
    session.is_active = true;
    session.last_access_time = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    inst->kv_sessions[seq_id] = session;
    inst->current_session_id = seq_id;

    LOGI("[KV] Created new session seq_id=%d", seq_id);
    return seq_id;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_naman_quantallm_LlamaJni_nativeSetKVSession(JNIEnv *env, jobject, jlong handle, jint sessionId) {
    auto* inst = get_instance(handle);
    if (!inst->ctx) {
        LOGE("[KV] Cannot set session: context not initialized");
        return JNI_FALSE;
    }

    auto it = inst->kv_sessions.find(sessionId);
    if (it == inst->kv_sessions.end()) {
        LOGE("[KV] Session %d not found", sessionId);
        return JNI_FALSE;
    }

    inst->current_session_id = sessionId;
    it->second.is_active = true;
    it->second.last_access_time = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    LOGI("[KV] Set active session to seq_id=%d (n_past=%d)", sessionId, it->second.n_past);
    return JNI_TRUE;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_naman_quantallm_LlamaJni_nativeClearKVSession(JNIEnv *env, jobject, jlong handle, jint sessionId) {
    auto* inst = get_instance(handle);
    if (!inst->ctx) return;

    llama_memory_t mem = llama_get_memory(inst->ctx);

    if (sessionId == -1) {
        llama_memory_seq_rm(mem, -1, -1, -1);
        inst->kv_sessions.clear();
        inst->current_session_id = -1;
        LOGI("[KV] Cleared ALL sessions");
    } else {
        auto it = inst->kv_sessions.find(sessionId);
        if (it != inst->kv_sessions.end()) {
            llama_memory_seq_rm(mem, sessionId, -1, -1);
            inst->kv_sessions.erase(it);
            if (inst->current_session_id == sessionId) {
                inst->current_session_id = -1;
            }
            LOGI("[KV] Cleared session seq_id=%d", sessionId);
        }
    }
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_naman_quantallm_LlamaJni_nativeGetKVCacheStats(JNIEnv *env, jobject, jlong handle) {
    auto* inst = get_instance(handle);
    if (!inst->ctx) {
        return env->NewStringUTF("{\"error\":\"No context\"}");
    }

    int32_t total_tokens = 0;
    for (const auto& pair : inst->kv_sessions) {
        total_tokens += pair.second.n_past;
    }

    char stats[512];
    snprintf(stats, sizeof(stats),
        "{\"n_ctx\":%d,\"total_tokens\":%d,\"active_sessions\":%d,\"current_session\":%d}",
        inst->n_ctx, total_tokens, (int)inst->kv_sessions.size(), inst->current_session_id);

    return env->NewStringUTF(stats);
}

//==============================================================================
// SECTION: Legacy & Compatibility
//==============================================================================

extern "C"
JNIEXPORT jint JNICALL
Java_com_naman_quantallm_LlamaJni_nativeStartSession(
        JNIEnv *env, jobject, jlong handle, jbyteArray promptBytes_, jlong /*sessionId*/,
        jint /*maxTokens*/, jfloat /*temperature*/) {
    auto* inst = get_instance(handle);

    if (!inst->ctx) return -1;

    jsize len = env->GetArrayLength(promptBytes_);
    std::string prompt(len, ' ');
    env->GetByteArrayRegion(promptBytes_, 0, len, (jbyte*)prompt.data());

    LOGI("nativeStartSession called (placeholder) prompt_len=%d", (int)len);
    return 0;
}

//==============================================================================
// SECTION: Text Generation Functions
//==============================================================================

extern "C"
JNIEXPORT void JNICALL
Java_com_naman_quantallm_LlamaJni_nativeAbortGeneration(
        JNIEnv *, jobject, jlong handle) {
    auto* inst = get_instance(handle);
    inst->abort_generation.store(true, std::memory_order_release);
    LOGI("[ABORT] Generation abort flag set");
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_naman_quantallm_LlamaJni_nativeGenerate(
        JNIEnv * env, jobject, jlong handle, jstring prompt_, jint maxTokens, jfloat temperature) {
    auto* inst = get_instance(handle);

    if (!inst->ctx) {
        LOGE("nativeGenerate called without init");
        return env->NewStringUTF("");
    }

    inst->abort_generation.store(false, std::memory_order_release);

    const char * c_prompt = env->GetStringUTFChars(prompt_, 0);
    std::string prompt(c_prompt);
    env->ReleaseStringUTFChars(prompt_, c_prompt);

    std::vector<llama_token> prompt_tokens;
    if (!tokenize_prompt(inst->model, prompt, prompt_tokens)) {
        LOGE("Tokenization failed!");
        return env->NewStringUTF("");
    }

    llama_memory_t mem = llama_get_memory(inst->ctx);
    llama_memory_seq_rm(mem, 0, -1, -1);

    int32_t seq_id = 0;
    int32_t n_past = 0;
    LOGI("[KV] Default-mode generate: cleared seq_id=0, starting fresh");

    const int n_prompt = (int)prompt_tokens.size();

    const int n_batch = 512;
    for (int i = 0; i < n_prompt; i += n_batch) {
        if (inst->abort_generation.load(std::memory_order_acquire)) {
            LOGI("[ABORT] Generation aborted during prompt processing");
            return env->NewStringUTF("");
        }

        const int chunk_size = std::min(n_batch, n_prompt - i);
        llama_batch batch = llama_batch_init(chunk_size, 0, 1);

        for (int j = 0; j < chunk_size; ++j) {
            batch.token[j] = prompt_tokens[i + j];
            batch.pos[j] = n_past + i + j;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = seq_id;
            batch.logits[j] = (i + j == n_prompt - 1) ? 1 : 0;
        }
        batch.n_tokens = chunk_size;

        if (llama_decode(inst->ctx, batch) != 0) {
            LOGE("llama_decode (prompt batch) failed at offset %d", i);
            llama_batch_free(batch);
            return env->NewStringUTF("");
        }
        llama_batch_free(batch);
    }

    n_past += n_prompt;

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(50));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
    uint32_t seed = LLAMA_DEFAULT_SEED;
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));

    std::string output;
    int generated = 0;
    const llama_vocab * vocab = llama_model_get_vocab(inst->model);

    llama_batch gen_batch = llama_batch_init(1, 0, 1);
    gen_batch.n_seq_id[0] = 1;
    gen_batch.seq_id[0][0] = seq_id;
    gen_batch.logits[0] = 1;

    while (generated < maxTokens) {
        if (inst->abort_generation.load(std::memory_order_acquire)) {
            LOGI("[ABORT] Generation aborted by user at token %d", generated);
            break;
        }

        const llama_token id = llama_sampler_sample(smpl, inst->ctx, -1);
        if (id == LLAMA_TOKEN_NULL) {
            break;
        }
        if (llama_vocab_is_eog(vocab, id)) {
            break;
        }

        append_token_piece(inst->model, id, output);
        llama_sampler_accept(smpl, id);

        gen_batch.token[0] = id;
        gen_batch.pos[0] = n_past + generated;
        gen_batch.n_tokens = 1;

        if (llama_decode(inst->ctx, gen_batch) != 0) {
            LOGE("llama_decode (gen) failed at token %d", generated);
            break;
        }
        ++generated;
    }

    llama_batch_free(gen_batch);
    llama_sampler_free(smpl);

    return env->NewStringUTF(output.c_str());
}

//------------------------------------------------------------------------------
// Advanced Generation (with full sampler control)
//------------------------------------------------------------------------------

extern "C"
JNIEXPORT jstring JNICALL
Java_com_naman_quantallm_LlamaJni_nativeGenerateAdvanced(
        JNIEnv * env, jobject, jlong handle,
        jstring prompt_,
        jint maxTokens,
        jfloat temperature,
        jint topK,
        jfloat topP,
        jfloat minP,
        jfloat repeatPenalty,
        jint penaltyLastN,
        jfloat frequencyPenalty,
        jfloat presencePenalty,
        jint seed) {
    auto* inst = get_instance(handle);
    if (!inst->ctx) {
        return env->NewStringUTF("");
    }

    inst->abort_generation.store(false, std::memory_order_release);

    const char * c_prompt = env->GetStringUTFChars(prompt_, 0);
    std::string prompt(c_prompt);
    env->ReleaseStringUTFChars(prompt_, c_prompt);

    std::vector<llama_token> prompt_tokens;
    if (!tokenize_prompt(inst->model, prompt, prompt_tokens)) {
        return env->NewStringUTF("");
    }

    llama_memory_t mem = llama_get_memory(inst->ctx);
    llama_memory_seq_rm(mem, 0, -1, -1);

    int32_t seq_id = 0;
    int32_t n_past = 0;
    LOGI("[KV-ADV] Default-mode advanced generate: cleared seq_id=0, starting fresh");

    const int n_prompt = (int)prompt_tokens.size();
    const int n_batch = 512;
    for (int i = 0; i < n_prompt; i += n_batch) {
        if (inst->abort_generation.load(std::memory_order_acquire)) {
            return env->NewStringUTF("");
        }
        const int chunk_size = std::min(n_batch, n_prompt - i);
        llama_batch batch = llama_batch_init(chunk_size, 0, 1);
        for (int j = 0; j < chunk_size; ++j) {
            batch.token[j] = prompt_tokens[i + j];
            batch.pos[j] = n_past + i + j;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = seq_id;
            batch.logits[j] = (i + j == n_prompt - 1) ? 1 : 0;
        }
        batch.n_tokens = chunk_size;
        if (llama_decode(inst->ctx, batch) != 0) {
            llama_batch_free(batch);
            return env->NewStringUTF("");
        }
        llama_batch_free(batch);
    }
    n_past += n_prompt;

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    if (topK > 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(topK));
    }
    if (topP > 0.0f && topP < 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(topP, 1));
    }
    if (minP > 0.0f && minP < 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_min_p(minP, 1));
    }
    if (temperature != 1.0f && temperature > 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
    }
    if (penaltyLastN != 0 && (repeatPenalty != 1.0f || frequencyPenalty > 0.0f || presencePenalty > 0.0f)) {
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(penaltyLastN, repeatPenalty, frequencyPenalty, presencePenalty));
    }

    uint32_t useSeed = seed > 0 ? (uint32_t)seed : LLAMA_DEFAULT_SEED;
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(useSeed));

    std::string output;
    int generated = 0;
    const llama_vocab * vocab = llama_model_get_vocab(inst->model);

    llama_batch gen_batch = llama_batch_init(1, 0, 1);
    gen_batch.n_seq_id[0] = 1;
    gen_batch.seq_id[0][0] = seq_id;
    gen_batch.logits[0] = 1;

    while (generated < maxTokens) {
        if (inst->abort_generation.load(std::memory_order_acquire)) {
            LOGI("[ABORT] Advanced generation aborted by user at token %d", generated);
            break;
        }

        const llama_token id = llama_sampler_sample(smpl, inst->ctx, -1);
        if (id == LLAMA_TOKEN_NULL) {
            LOGE("[ADV] NULL token");
            break;
        }
        if (llama_vocab_is_eog(vocab, id)) {
            LOGI("[ADV] EOG reached");
            break;
        }
        append_token_piece(inst->model, id, output);
        llama_sampler_accept(smpl, id);

        gen_batch.token[0] = id;
        gen_batch.pos[0] = n_past + generated;
        gen_batch.n_tokens = 1;

        if (llama_decode(inst->ctx, gen_batch) != 0) {
            LOGE("[ADV] decode gen failed at token %d", generated);
            break;
        }
        ++generated;
    }

    llama_batch_free(gen_batch);
    llama_sampler_free(smpl);
    return env->NewStringUTF(output.c_str());
}

//------------------------------------------------------------------------------
// Streaming Generation (with progress callbacks)
//------------------------------------------------------------------------------

extern "C"
JNIEXPORT jstring JNICALL
Java_com_naman_quantallm_LlamaJni_nativeGenerateStreamingAdvanced(
        JNIEnv * env, jobject thiz, jlong handle,
        jstring prompt_,
        jint maxTokens,
        jfloat temperature,
        jint topK,
        jfloat topP,
        jfloat minP,
        jfloat repeatPenalty,
        jint penaltyLastN,
        jfloat frequencyPenalty,
        jfloat presencePenalty,
        jint seed,
        jobject callback) {
    auto* inst = get_instance(handle);

    if (!inst->ctx) {
        LOGE("nativeGenerateStreamingAdvanced called without init");
        return env->NewStringUTF("");
    }

    inst->abort_generation.store(false, std::memory_order_release);

    jclass callbackClass = env->GetObjectClass(callback);
    jmethodID onProgressMethod = env->GetMethodID(callbackClass, "onProgress", "(ILjava/lang/String;)V");
    if (!onProgressMethod) {
        LOGE("Failed to find onProgress callback method");
        return env->NewStringUTF("");
    }

    const char * c_prompt = env->GetStringUTFChars(prompt_, 0);
    std::string prompt(c_prompt);
    env->ReleaseStringUTFChars(prompt_, c_prompt);

    std::vector<llama_token> prompt_tokens;
    if (!tokenize_prompt(inst->model, prompt, prompt_tokens)) {
        LOGE("Tokenization failed!");
        return env->NewStringUTF("");
    }

    llama_memory_t mem = llama_get_memory(inst->ctx);
    llama_memory_seq_rm(mem, 0, -1, -1);

    int32_t seq_id = 0;
    int32_t n_past = 0;
    LOGI("[KV-STREAM-ADV] Default-mode streaming: cleared seq_id=0, starting fresh");

    const int n_prompt = (int)prompt_tokens.size();
    const int n_batch = 512;
    for (int i = 0; i < n_prompt; i += n_batch) {
        if (inst->abort_generation.load(std::memory_order_acquire)) return env->NewStringUTF("");
        const int chunk_size = std::min(n_batch, n_prompt - i);
        llama_batch batch = llama_batch_init(chunk_size, 0, 1);
        for (int j = 0; j < chunk_size; ++j) {
            batch.token[j] = prompt_tokens[i + j];
            batch.pos[j] = n_past + i + j;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = seq_id;
            batch.logits[j] = (i + j == n_prompt - 1) ? 1 : 0;
        }
        batch.n_tokens = chunk_size;
        if (llama_decode(inst->ctx, batch) != 0) {
            LOGE("llama_decode (prompt) failed");
            llama_batch_free(batch);
            return env->NewStringUTF("");
        }
        llama_batch_free(batch);
    }
    n_past += n_prompt;

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    if (topK > 0) llama_sampler_chain_add(smpl, llama_sampler_init_top_k(topK));
    if (topP > 0.0f && topP < 1.0f) llama_sampler_chain_add(smpl, llama_sampler_init_top_p(topP, 1));
    if (minP > 0.0f && minP < 1.0f) llama_sampler_chain_add(smpl, llama_sampler_init_min_p(minP, 1));
    if (temperature > 0.0f) llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
    if (penaltyLastN != 0 && (repeatPenalty != 1.0f || frequencyPenalty > 0.0f || presencePenalty > 0.0f)) {
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(penaltyLastN, repeatPenalty, frequencyPenalty, presencePenalty));
    }
    uint32_t useSeed = seed > 0 ? (uint32_t)seed : LLAMA_DEFAULT_SEED;
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(useSeed));

    std::string output;
    int generated = 0;
    const llama_vocab * vocab = llama_model_get_vocab(inst->model);

    llama_batch gen_batch = llama_batch_init(1, 0, 1);
    gen_batch.n_seq_id[0] = 1;
    gen_batch.seq_id[0][0] = seq_id;
    gen_batch.logits[0] = 1;

    while (generated < maxTokens) {
        if (inst->abort_generation.load(std::memory_order_acquire)) break;

        const llama_token id = llama_sampler_sample(smpl, inst->ctx, -1);
        if (id == LLAMA_TOKEN_NULL || llama_vocab_is_eog(vocab, id)) break;

        append_token_piece(inst->model, id, output);
        llama_sampler_accept(smpl, id);

        if ((generated + 1) % 5 == 0 || generated == 0) {
            jstring jPartialText = env->NewStringUTF(output.c_str());
            if (!jPartialText) break;
            env->CallVoidMethod(callback, onProgressMethod, generated + 1, jPartialText);
            env->DeleteLocalRef(jPartialText);
            if (env->ExceptionCheck()) { env->ExceptionClear(); break; }
        }

        gen_batch.token[0] = id;
        gen_batch.pos[0] = n_past + generated;
        gen_batch.n_tokens = 1;
        if (llama_decode(inst->ctx, gen_batch) != 0) break;
        ++generated;
    }

    jstring finalText = env->NewStringUTF(output.c_str());
    env->CallVoidMethod(callback, onProgressMethod, generated, finalText);
    env->DeleteLocalRef(finalText);

    llama_batch_free(gen_batch);
    llama_sampler_free(smpl);
    return env->NewStringUTF(output.c_str());
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_naman_quantallm_LlamaJni_nativeGenerateStreaming(
        JNIEnv * env, jobject thiz, jlong handle,
        jstring prompt_,
        jint maxTokens,
        jfloat temperature,
        jobject callback) {
    auto* inst = get_instance(handle);

    if (!inst->ctx) {
        LOGE("nativeGenerateStreaming called without init");
        return env->NewStringUTF("");
    }

    inst->abort_generation.store(false, std::memory_order_release);

    jclass callbackClass = env->GetObjectClass(callback);
    jmethodID onProgressMethod = env->GetMethodID(callbackClass, "onProgress", "(ILjava/lang/String;)V");

    if (!onProgressMethod) {
        LOGE("Failed to find onProgress callback method");
        return env->NewStringUTF("");
    }

    const char * c_prompt = env->GetStringUTFChars(prompt_, 0);
    std::string prompt(c_prompt);
    env->ReleaseStringUTFChars(prompt_, c_prompt);

    std::vector<llama_token> prompt_tokens;
    if (!tokenize_prompt(inst->model, prompt, prompt_tokens)) {
        LOGE("Tokenization failed!");
        return env->NewStringUTF("");
    }

    llama_memory_t mem = llama_get_memory(inst->ctx);
    llama_memory_seq_rm(mem, 0, -1, -1);

    int32_t seq_id = 0;
    int32_t n_past = 0;
    LOGI("[KV-STREAM] Default-mode streaming: cleared seq_id=0, starting fresh");

    const int n_prompt = (int)prompt_tokens.size();
    const int n_batch = 512;
    for (int i = 0; i < n_prompt; i += n_batch) {
        if (inst->abort_generation.load(std::memory_order_acquire)) {
            return env->NewStringUTF("");
        }
        const int chunk_size = std::min(n_batch, n_prompt - i);
        llama_batch batch = llama_batch_init(chunk_size, 0, 1);
        for (int j = 0; j < chunk_size; ++j) {
            batch.token[j] = prompt_tokens[i + j];
            batch.pos[j] = n_past + i + j;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = seq_id;
            batch.logits[j] = (i + j == n_prompt - 1) ? 1 : 0;
        }
        batch.n_tokens = chunk_size;
        if (llama_decode(inst->ctx, batch) != 0) {
            LOGE("llama_decode (prompt batch) failed");
            llama_batch_free(batch);
            return env->NewStringUTF("");
        }
        llama_batch_free(batch);
    }
    n_past += n_prompt;

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(50));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
    uint32_t seed = LLAMA_DEFAULT_SEED;
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));

    std::string output;
    int generated = 0;
    const llama_vocab * vocab = llama_model_get_vocab(inst->model);

    llama_batch gen_batch = llama_batch_init(1, 0, 1);
    gen_batch.n_seq_id[0] = 1;
    gen_batch.seq_id[0][0] = seq_id;
    gen_batch.logits[0] = 1;

    const int UPDATE_FREQUENCY = 5;

    while (generated < maxTokens) {
        if (inst->abort_generation.load(std::memory_order_acquire)) {
            LOGI("[ABORT] Streaming generation aborted by user at token %d", generated);
            break;
        }

        const llama_token id = llama_sampler_sample(smpl, inst->ctx, -1);
        if (id == LLAMA_TOKEN_NULL) {
            break;
        }
        if (llama_vocab_is_eog(vocab, id)) {
            break;
        }

        append_token_piece(inst->model, id, output);
        llama_sampler_accept(smpl, id);

        if ((generated + 1) % UPDATE_FREQUENCY == 0 || generated == 0) {
            jstring partialText = env->NewStringUTF(output.c_str());
            env->CallVoidMethod(callback, onProgressMethod, generated + 1, partialText);
            env->DeleteLocalRef(partialText);

            if (env->ExceptionCheck()) {
                env->ExceptionDescribe();
                env->ExceptionClear();
            }
        }

        gen_batch.token[0] = id;
        gen_batch.pos[0] = n_past + generated;
        gen_batch.n_tokens = 1;

        if (llama_decode(inst->ctx, gen_batch) != 0) {
            LOGE("llama_decode (gen) failed at token %d", generated);
            break;
        }
        ++generated;
    }

    jstring finalText = env->NewStringUTF(output.c_str());
    env->CallVoidMethod(callback, onProgressMethod, generated, finalText);
    env->DeleteLocalRef(finalText);

    llama_batch_free(gen_batch);
    llama_sampler_free(smpl);

    return env->NewStringUTF(output.c_str());
}

//==============================================================================
// SECTION: Cleanup & Shutdown
//==============================================================================

extern "C"
JNIEXPORT jint JNICALL
Java_com_naman_quantallm_LlamaJni_nativeShutdown(
        JNIEnv *env, jobject, jlong handle) {
    auto* inst = get_instance(handle);
    if (inst->ctx) {
        llama_free(inst->ctx);
        inst->ctx = nullptr;
        LOGI("Context freed");
    }

    if (inst->model) {
        llama_model_free(inst->model);
        inst->model = nullptr;
        LOGI("Model freed");
    }

    inst->kv_sessions.clear();
    inst->next_seq_id = 0;
    inst->current_session_id = -1;
    LOGI("KV cache sessions cleared");

    if (g_backend_initialized) {
        llama_backend_free();
        g_backend_initialized = false;
    }

    LOGI("Shutdown complete");
    return 0;
}

//==============================================================================
// SECTION: Chat Mode APIs
//==============================================================================

extern "C"
JNIEXPORT jint JNICALL
Java_com_naman_quantallm_LlamaJni_nativeCreateChatSession(
        JNIEnv *env,
        jobject,
        jlong handle,
        jstring systemPrompt,
        jstring sessionIdUuid
) {
    auto* inst = get_instance(handle);
    if (!inst->ctx) {
        LOGE("[CHAT] Cannot create session: context not initialized");
        return -1;
    }

    int32_t seq_id = inst->next_seq_id++;
    KVCacheSession session;
    session.seq_id = seq_id;
    session.mode = SessionMode::CHAT;
    session.n_past = 0;
    session.is_active = true;
    session.last_access_time = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    if (sessionIdUuid) {
        const char* uuid_chars = env->GetStringUTFChars(sessionIdUuid, 0);
        session.session_id_uuid = uuid_chars;
        env->ReleaseStringUTFChars(sessionIdUuid, uuid_chars);
    }

    if (systemPrompt) {
        const char* prompt_chars = env->GetStringUTFChars(systemPrompt, 0);
        if (prompt_chars && strlen(prompt_chars) > 0) {
            session.system_prompt = prompt_chars;

            std::string cache_path = get_system_prompt_cache_path(
                inst->cache_dir, inst->model_path, session.system_prompt);
            bool loaded_from_cache = false;

            if (!cache_path.empty()) {
                size_t n_token_count = 0;
                std::vector<llama_token> cached_tokens(4096);
                size_t ret = llama_state_seq_load_file(
                    inst->ctx, cache_path.c_str(), seq_id,
                    cached_tokens.data(), cached_tokens.size(), &n_token_count);
                if (ret > 0 && n_token_count > 0) {
                    session.n_past = (int32_t)n_token_count;
                    loaded_from_cache = true;
                    LOGI("[CHAT] Loaded system prompt KV cache from disk (%zu tokens)", n_token_count);
                }
            }

            if (!loaded_from_cache) {
                const llama_vocab* vocab = llama_model_get_vocab(inst->model);
                std::vector<llama_token> system_tokens;

                int32_t n_tokens = llama_tokenize(vocab, session.system_prompt.c_str(), (int32_t)session.system_prompt.size(), nullptr, 0, true, false);
                if (n_tokens < 0) n_tokens = -n_tokens;
                system_tokens.resize(n_tokens);
                int32_t written = llama_tokenize(vocab, session.system_prompt.c_str(), (int32_t)session.system_prompt.size(), system_tokens.data(), n_tokens, true, false);
                if (written > 0) {
                    system_tokens.resize(written);

                    const int n_sys = (int)system_tokens.size();
                    const int n_batch = 512;
                    for (int i = 0; i < n_sys; i += n_batch) {
                        const int chunk_size = std::min(n_batch, n_sys - i);
                        llama_batch batch = llama_batch_init(chunk_size, 0, 1);
                        for (int j = 0; j < chunk_size; ++j) {
                            batch.token[j] = system_tokens[i + j];
                            batch.pos[j] = i + j;
                            batch.n_seq_id[j] = 1;
                            batch.seq_id[j][0] = seq_id;
                            batch.logits[j] = (i + j == n_sys - 1) ? 1 : 0;
                        }
                        batch.n_tokens = chunk_size;
                        if (llama_decode(inst->ctx, batch) != 0) {
                            llama_batch_free(batch);
                            LOGE("[CHAT] Failed to decode system prompt at chunk %d", i);
                            break;
                        }
                        llama_batch_free(batch);
                    }

                    session.n_past = system_tokens.size();
                    LOGI("[CHAT] Processed system prompt: %d tokens", (int)system_tokens.size());

                    if (!cache_path.empty()) {
                        size_t saved = llama_state_seq_save_file(
                            inst->ctx, cache_path.c_str(), seq_id,
                            system_tokens.data(), system_tokens.size());
                        if (saved > 0) {
                            LOGI("[CHAT] Saved system prompt KV cache to disk (%zu bytes)", saved);
                        }
                    }
                }
            }
        }
        env->ReleaseStringUTFChars(systemPrompt, prompt_chars);
    }

    inst->kv_sessions[seq_id] = session;
    inst->current_session_id = seq_id;

    LOGI("[CHAT] Created session seq_id=%d, uuid=%s", seq_id, session.session_id_uuid.c_str());
    return seq_id;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_naman_quantallm_LlamaJni_nativeSetActiveChatSession(
        JNIEnv *env,
        jobject,
        jlong handle,
        jint sessionId
) {
    auto* inst = get_instance(handle);
    if (!inst->ctx) {
        LOGE("[CHAT] Cannot set session: context not initialized");
        return JNI_FALSE;
    }

    auto it = inst->kv_sessions.find(sessionId);
    if (it == inst->kv_sessions.end()) {
        LOGE("[CHAT] Session %d not found", sessionId);
        return JNI_FALSE;
    }

    inst->current_session_id = sessionId;
    it->second.is_active = true;
    it->second.last_access_time = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    LOGI("[CHAT] Set active session to seq_id=%d (n_past=%d)", sessionId, it->second.n_past);
    return JNI_TRUE;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_naman_quantallm_LlamaJni_nativeDeleteChatSession(
        JNIEnv *env,
        jobject,
        jlong handle,
        jint sessionId
) {
    auto* inst = get_instance(handle);
    if (!inst->ctx) return;

    auto it = inst->kv_sessions.find(sessionId);
    if (it != inst->kv_sessions.end()) {
        llama_memory_t mem = llama_get_memory(inst->ctx);
        llama_memory_seq_rm(mem, sessionId, -1, -1);
        inst->kv_sessions.erase(it);

        if (inst->current_session_id == sessionId) {
            inst->current_session_id = -1;
        }

        LOGI("[CHAT] Deleted session seq_id=%d", sessionId);
    }
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_naman_quantallm_LlamaJni_nativeChatGenerate(
        JNIEnv *env,
        jobject,
        jlong handle,
        jstring message,
        jint maxTokens,
        jfloat temperature
) {
    auto* inst = get_instance(handle);
    if (!inst->ctx || !inst->model) {
        return env->NewStringUTF("{\"error\":\"Model not initialized\"}");
    }

    inst->abort_generation.store(false, std::memory_order_release);

    if (inst->current_session_id < 0) {
        return env->NewStringUTF("{\"error\":\"No active chat session\"}");
    }

    auto it = inst->kv_sessions.find(inst->current_session_id);
    if (it == inst->kv_sessions.end()) {
        return env->NewStringUTF("{\"error\":\"Active session not found\"}");
    }

    KVCacheSession session = it->second;
    int32_t current_seq_id = inst->current_session_id;

    LOGI("[CHAT] Starting generation for session seq_id=%d, n_past=%d", current_seq_id, session.n_past);

    const char* msg_chars = env->GetStringUTFChars(message, 0);
    if (!msg_chars) {
        return env->NewStringUTF("{\"error\":\"Failed to get message string\"}");
    }
    std::string user_message(msg_chars);
    env->ReleaseStringUTFChars(message, msg_chars);

    std::vector<llama_chat_message> chat_messages;

    if (!session.system_prompt.empty() && session.n_past == 0) {
        llama_chat_message sys_msg;
        sys_msg.role = "system";
        sys_msg.content = session.system_prompt.c_str();
        chat_messages.push_back(sys_msg);
    }

    llama_chat_message user_msg;
    user_msg.role = "user";
    user_msg.content = user_message.c_str();
    chat_messages.push_back(user_msg);

    const size_t buf_size = user_message.size() * 4 + 1024;
    std::vector<char> formatted_buf(buf_size);

    int32_t formatted_len = llama_chat_apply_template(
        nullptr,
        chat_messages.data(),
        chat_messages.size(),
        true,
        formatted_buf.data(),
        buf_size
    );

    if (formatted_len < 0) {
        LOGW("[CHAT] Model template failed, trying chatml");
        formatted_len = llama_chat_apply_template(
            "chatml",
            chat_messages.data(),
            chat_messages.size(),
            true,
            formatted_buf.data(),
            buf_size
        );
    }

    if (formatted_len < 0) {
        LOGW("[CHAT] Chat template application failed, using simple format");
        std::string fallback;
        if (!session.system_prompt.empty() && session.n_past == 0) {
            fallback = "System: " + session.system_prompt + "\n\n";
        }
        fallback += "User: " + user_message + "\n\nAssistant:";
        formatted_len = fallback.size();
        formatted_buf.assign(fallback.begin(), fallback.end());
        formatted_buf.push_back('\0');
    } else if (formatted_len >= (int32_t)buf_size) {
        formatted_buf.resize(formatted_len + 1);
        formatted_len = llama_chat_apply_template(
            nullptr,
            chat_messages.data(),
            chat_messages.size(),
            true,
            formatted_buf.data(),
            formatted_buf.size()
        );
    }

    std::string formatted_prompt(formatted_buf.data(), formatted_len);
    LOGI("[CHAT] Applied chat template, formatted length: %d", formatted_len);
    LOGI("[CHAT] Formatted prompt: %s", formatted_prompt.c_str());

    const llama_vocab* vocab = llama_model_get_vocab(inst->model);
    std::vector<llama_token> prompt_tokens;

    int32_t n_tokens = llama_tokenize(vocab, formatted_prompt.c_str(), formatted_prompt.size(), nullptr, 0, true, false);
    if (n_tokens < 0) n_tokens = -n_tokens;
    if (n_tokens == 0) {
        return env->NewStringUTF("{\"error\":\"Empty prompt after tokenization\"}");
    }

    prompt_tokens.resize(n_tokens);
    int32_t written = llama_tokenize(vocab, formatted_prompt.c_str(), formatted_prompt.size(), prompt_tokens.data(), n_tokens, true, false);
    if (written < 0 || written == 0) {
        return env->NewStringUTF("{\"error\":\"Failed to tokenize formatted prompt\"}");
    }
    prompt_tokens.resize(written);
    LOGI("[CHAT] Tokenized to %d tokens", written);

    if (!prompt_tokens.empty()) {
        const int n_prompt = (int)prompt_tokens.size();
        const int n_batch = 512;
        for (int i = 0; i < n_prompt; i += n_batch) {
            if (inst->abort_generation.load(std::memory_order_acquire)) {
                return env->NewStringUTF("{\"error\":\"Aborted during prompt processing\"}");
            }
            const int chunk_size = std::min(n_batch, n_prompt - i);
            llama_batch batch = llama_batch_init(chunk_size, 0, 1);
            for (int j = 0; j < chunk_size; ++j) {
                batch.token[j] = prompt_tokens[i + j];
                batch.pos[j] = session.n_past + i + j;
                batch.n_seq_id[j] = 1;
                batch.seq_id[j][0] = current_seq_id;
                batch.logits[j] = (i + j == n_prompt - 1) ? 1 : 0;
            }
            batch.n_tokens = chunk_size;
            if (llama_decode(inst->ctx, batch) != 0) {
                llama_batch_free(batch);
                return env->NewStringUTF("{\"error\":\"Failed to decode prompt\"}");
            }
            llama_batch_free(batch);
        }
        session.n_past += n_prompt;
    }

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sparams);
    if (!smpl) {
        return env->NewStringUTF("{\"error\":\"Failed to init sampler\"}");
    }

    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(50));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
    uint32_t seed = LLAMA_DEFAULT_SEED;
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));

    std::string response;
    int32_t tokens_generated = 0;

    llama_batch gen_batch = llama_batch_init(1, 0, 1);
    gen_batch.n_seq_id[0] = 1;
    gen_batch.seq_id[0][0] = current_seq_id;
    gen_batch.logits[0] = 1;

    while (tokens_generated < maxTokens) {
        if (inst->abort_generation.load(std::memory_order_acquire)) {
            LOGI("[ABORT] Chat generation aborted by user at token %d", tokens_generated);
            break;
        }

        const llama_token id = llama_sampler_sample(smpl, inst->ctx, -1);

        if (id == LLAMA_TOKEN_NULL) {
            break;
        }

        if (llama_vocab_is_eog(vocab, id)) {
            LOGI("[CHAT] EOS token reached");
            break;
        }

        append_token_piece(inst->model, id, response);
        llama_sampler_accept(smpl, id);

        gen_batch.token[0] = id;
        gen_batch.pos[0] = session.n_past + tokens_generated;
        gen_batch.n_tokens = 1;

        if (llama_decode(inst->ctx, gen_batch) != 0) {
            break;
        }

        ++tokens_generated;
    }

    llama_batch_free(gen_batch);
    llama_sampler_free(smpl);

    inst->kv_sessions[current_seq_id].n_past = session.n_past + tokens_generated;

    std::string json_response = build_chat_json_response(
        response, 1, tokens_generated, inst->kv_sessions[current_seq_id].n_past);

    LOGI("[CHAT] Generated %d tokens successfully", tokens_generated);
    return env->NewStringUTF(json_response.c_str());
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_naman_quantallm_LlamaJni_nativeChatGenerateStreaming(
        JNIEnv *env,
        jobject,
        jlong handle,
        jstring message,
        jint maxTokens,
        jfloat temperature,
        jobject callback
) {
    auto* inst = get_instance(handle);
    jclass callbackClass = env->GetObjectClass(callback);
    jmethodID onProgressMethod = env->GetMethodID(callbackClass, "onProgress", "(ILjava/lang/String;)V");

    if (!onProgressMethod) {
        LOGE("[CHAT] Failed to find onProgress callback method");
        return env->NewStringUTF("{\"error\":\"Callback method not found\"}");
    }

    if (!inst->ctx || !inst->model) {
        return env->NewStringUTF("{\"error\":\"Model not initialized\"}");
    }

    inst->abort_generation.store(false, std::memory_order_release);

    if (inst->current_session_id < 0) {
        return env->NewStringUTF("{\"error\":\"No active chat session\"}");
    }

    auto it = inst->kv_sessions.find(inst->current_session_id);
    if (it == inst->kv_sessions.end()) {
        return env->NewStringUTF("{\"error\":\"Active session not found\"}");
    }

    KVCacheSession session = it->second;
    int32_t current_seq_id = inst->current_session_id;

    LOGI("[CHAT-STREAM] Starting streaming generation for session seq_id=%d", current_seq_id);

    const char* msg_chars = env->GetStringUTFChars(message, 0);
    if (!msg_chars) {
        return env->NewStringUTF("{\"error\":\"Failed to get message string\"}");
    }
    std::string user_message(msg_chars);
    env->ReleaseStringUTFChars(message, msg_chars);

    std::vector<llama_chat_message> chat_messages;

    if (!session.system_prompt.empty() && session.n_past == 0) {
        llama_chat_message sys_msg;
        sys_msg.role = "system";
        sys_msg.content = session.system_prompt.c_str();
        chat_messages.push_back(sys_msg);
    }

    llama_chat_message user_msg;
    user_msg.role = "user";
    user_msg.content = user_message.c_str();
    chat_messages.push_back(user_msg);

    const size_t buf_size = user_message.size() * 4 + 1024;
    std::vector<char> formatted_buf(buf_size);

    int32_t formatted_len = llama_chat_apply_template(
        nullptr,
        chat_messages.data(),
        chat_messages.size(),
        true,
        formatted_buf.data(),
        buf_size
    );

    if (formatted_len < 0) {
        LOGW("[CHAT-STREAM] Model template failed, trying chatml");
        formatted_len = llama_chat_apply_template(
            "chatml",
            chat_messages.data(),
            chat_messages.size(),
            true,
            formatted_buf.data(),
            buf_size
        );
    }

    if (formatted_len < 0) {
        LOGW("[CHAT-STREAM] Chat template failed, using simple format");
        std::string fallback;
        if (!session.system_prompt.empty() && session.n_past == 0) {
            fallback = "System: " + session.system_prompt + "\n\n";
        }
        fallback += "User: " + user_message + "\n\nAssistant:";
        formatted_len = fallback.size();
        formatted_buf.assign(fallback.begin(), fallback.end());
        formatted_buf.push_back('\0');
    } else if (formatted_len >= (int32_t)buf_size) {
        formatted_buf.resize(formatted_len + 1);
        formatted_len = llama_chat_apply_template(
            nullptr,
            chat_messages.data(),
            chat_messages.size(),
            true,
            formatted_buf.data(),
            formatted_buf.size()
        );
    }

    std::string formatted_prompt(formatted_buf.data(), formatted_len);
    LOGI("[CHAT-STREAM] Applied chat template, length: %d", formatted_len);
    LOGI("[CHAT-STREAM] Formatted prompt: %s", formatted_prompt.c_str());

    const llama_vocab* vocab = llama_model_get_vocab(inst->model);
    std::vector<llama_token> prompt_tokens;

    int32_t n_tokens = llama_tokenize(vocab, formatted_prompt.c_str(), formatted_prompt.size(), nullptr, 0, true, false);
    if (n_tokens < 0) n_tokens = -n_tokens;
    if (n_tokens == 0) {
        return env->NewStringUTF("{\"error\":\"Empty prompt after tokenization\"}");
    }

    prompt_tokens.resize(n_tokens);
    int32_t written = llama_tokenize(vocab, formatted_prompt.c_str(), formatted_prompt.size(), prompt_tokens.data(), n_tokens, true, false);
    if (written < 0 || written == 0) {
        return env->NewStringUTF("{\"error\":\"Failed to tokenize formatted prompt\"}");
    }
    prompt_tokens.resize(written);

    int32_t n_ctx = llama_n_ctx(inst->ctx);
    if (session.n_past + (int32_t)prompt_tokens.size() + maxTokens > n_ctx) {
        LOGE("[CHAT-STREAM] Context overflow: n_past=%d + prompt=%d + max_gen=%d > ctx=%d",
             session.n_past, (int)prompt_tokens.size(), maxTokens, n_ctx);
        int32_t available = n_ctx - session.n_past - (int32_t)prompt_tokens.size() - 4;
        if (available <= 0) {
            return env->NewStringUTF("{\"error\":\"Context window full. Start a new session.\"}");
        }
        maxTokens = available;
        LOGW("[CHAT-STREAM] Clamped maxTokens to %d", maxTokens);
    }

    if (!prompt_tokens.empty()) {
        const int n_prompt = (int)prompt_tokens.size();
        const int n_batch = 512;
        for (int i = 0; i < n_prompt; i += n_batch) {
            if (inst->abort_generation.load(std::memory_order_acquire)) {
                return env->NewStringUTF("{\"error\":\"Aborted during prompt processing\"}");
            }
            const int chunk_size = std::min(n_batch, n_prompt - i);
            llama_batch batch = llama_batch_init(chunk_size, 0, 1);
            for (int j = 0; j < chunk_size; ++j) {
                batch.token[j] = prompt_tokens[i + j];
                batch.pos[j] = session.n_past + i + j;
                batch.n_seq_id[j] = 1;
                batch.seq_id[j][0] = current_seq_id;
                batch.logits[j] = (i + j == n_prompt - 1) ? 1 : 0;
            }
            batch.n_tokens = chunk_size;
            if (llama_decode(inst->ctx, batch) != 0) {
                llama_batch_free(batch);
                return env->NewStringUTF("{\"error\":\"Failed to decode message\"}");
            }
            llama_batch_free(batch);
        }
        session.n_past += n_prompt;
    }

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sparams);
    if (!smpl) {
        return env->NewStringUTF("{\"error\":\"Failed to init sampler\"}");
    }

    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(50));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
    uint32_t seed = LLAMA_DEFAULT_SEED;
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));

    std::string response;
    int32_t tokens_generated = 0;

    llama_batch gen_batch = llama_batch_init(1, 0, 1);
    gen_batch.n_seq_id[0] = 1;
    gen_batch.seq_id[0][0] = current_seq_id;
    gen_batch.logits[0] = 1;

    while (tokens_generated < maxTokens) {
        if (inst->abort_generation.load(std::memory_order_acquire)) {
            LOGI("[ABORT] Chat streaming generation aborted by user at token %d", tokens_generated);
            break;
        }

        const llama_token id = llama_sampler_sample(smpl, inst->ctx, -1);

        if (id == LLAMA_TOKEN_NULL) {
            break;
        }

        if (llama_vocab_is_eog(vocab, id)) {
            break;
        }

        append_token_piece(inst->model, id, response);
        llama_sampler_accept(smpl, id);

        if ((tokens_generated + 1) % 5 == 0 || tokens_generated == 0) {
            jstring jPartialText = env->NewStringUTF(response.c_str());
            if (!jPartialText) {
                LOGE("[CHAT-STREAM] Failed to create Java string at token %d", tokens_generated);
                break;
            }
            env->CallVoidMethod(callback, onProgressMethod, tokens_generated + 1, jPartialText);
            env->DeleteLocalRef(jPartialText);

            if (env->ExceptionCheck()) {
                LOGE("[CHAT-STREAM] Java exception in callback at token %d", tokens_generated);
                env->ExceptionClear();
                break;
            }
        }

        gen_batch.token[0] = id;
        gen_batch.pos[0] = session.n_past + tokens_generated;
        gen_batch.n_tokens = 1;

        if (llama_decode(inst->ctx, gen_batch) != 0) {
            break;
        }

        ++tokens_generated;

        if (response.size() > 16384) {
            break;
        }
    }

    llama_batch_free(gen_batch);
    llama_sampler_free(smpl);

    inst->kv_sessions[current_seq_id].n_past = session.n_past + tokens_generated;

    std::string json_response = build_chat_json_response(
        response, 1, tokens_generated, inst->kv_sessions[current_seq_id].n_past);

    LOGI("[CHAT-STREAM] Generated %d tokens successfully", tokens_generated);
    return env->NewStringUTF(json_response.c_str());
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_naman_quantallm_LlamaJni_nativeGetChatSessionInfo(
        JNIEnv *env,
        jobject,
        jlong handle,
        jint sessionId
) {
    auto* inst = get_instance(handle);
    auto it = inst->kv_sessions.find(sessionId);
    if (it == inst->kv_sessions.end()) {
        return env->NewStringUTF("{\"error\":\"Session not found\"}");
    }

    const KVCacheSession& session = it->second;
    std::ostringstream oss;
    oss << "{\"seq_id\":" << session.seq_id
        << ",\"uuid\":\"" << json_escape_string(session.session_id_uuid) << "\""
        << ",\"mode\":\"" << (session.mode == SessionMode::CHAT ? "chat" : "default") << "\""
        << ",\"n_past\":" << session.n_past
        << ",\"turns\":" << (int)session.turns.size()
        << ",\"is_active\":" << (session.is_active ? "true" : "false")
        << "}";

    return env->NewStringUTF(oss.str().c_str());
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_naman_quantallm_LlamaJni_nativeListChatSessions(
        JNIEnv *env,
        jobject,
        jlong handle
) {
    auto* inst = get_instance(handle);
    std::string json = "[";
    bool first = true;

    for (const auto& pair : inst->kv_sessions) {
        if (pair.second.mode == SessionMode::CHAT) {
            if (!first) json += ",";
            first = false;

            std::ostringstream entry;
            entry << "{\"seq_id\":" << pair.second.seq_id
                  << ",\"uuid\":\"" << json_escape_string(pair.second.session_id_uuid) << "\""
                  << ",\"n_past\":" << pair.second.n_past
                  << ",\"turns\":" << (int)pair.second.turns.size()
                  << "}";
            json += entry.str();
        }
    }

    json += "]";
    return env->NewStringUTF(json.c_str());
}

//==============================================================================
// END OF FILE
//==============================================================================
