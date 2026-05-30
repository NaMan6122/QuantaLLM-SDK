plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
    id("maven-publish")
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

    buildTypes {
        release {
            isMinifyEnabled = false
            consumerProguardFiles("consumer-rules.pro")
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

    publishing {
        singleVariant("release") {
            withSourcesJar()
        }
    }
}

dependencies {
    api(project(":quantallm-core"))
}

afterEvaluate {
    publishing {
        publications {
            create<MavenPublication>("release") {
                from(components["release"])
                groupId = "com.quantallm"
                artifactId = "quantallm-llamacpp"
                version = project.findProperty("SDK_VERSION") as String? ?: "1.0.0"
            }
        }
        repositories {
            maven {
                name = "GitHubPackages"
                url = uri("https://maven.pkg.github.com/NaMan6122/quantallm-sdk")
                credentials {
                    username = project.findProperty("gpr.user") as String? ?: System.getenv("GITHUB_USER")
                    password = project.findProperty("gpr.token") as String? ?: System.getenv("GITHUB_TOKEN")
                }
            }
        }
    }
}
