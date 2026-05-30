plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.quantallm.llamacpp"
    ndkVersion = "26.3.11579264"
    compileSdk = 36

    defaultConfig {
        minSdk = 31
        consumerProguardFiles("consumer-rules.pro")

        ndk {
            abiFilters += "arm64-v8a"
        }

        externalNativeBuild {
            cmake {
                cppFlags += listOf(
                    "-std=c++17",
                    "-O3",
                    "-DNDEBUG",
                    "-ffast-math",
                    "-fno-finite-math-only",
                    "-funroll-loops",
                    "-fomit-frame-pointer",
                    "-funsafe-math-optimizations",
                    "-fno-math-errno",
                    "-fno-trapping-math",
                    "-flto",
                    "-march=armv8.6-a+dotprod+i8mm+bf16+fp16",
                    "-mtune=cortex-a76",
                    "-DLLAMA_NATIVE=ON"
                )
                arguments += listOf(
                    "-DANDROID_STL=c++_static",
                    "-DANDROID_PLATFORM=android-31",
                    "-DANDROID_LD_FLAGS=-Wl,-z,max-page-size=16384",
                    "-DGGML_USE_ARM_DOTPROD=ON",
                    "-DGGML_USE_ARM_FP16=ON",
                    "-DGGML_USE_ARM_I8MM=ON",
                    "-DLLAMA_NATIVE=ON"
                )
            }
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/jni/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }
}

dependencies {
    api(project(":quantallm-core"))
}
