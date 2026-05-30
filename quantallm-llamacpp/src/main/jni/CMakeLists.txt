cmake_minimum_required(VERSION 3.22)
project(llama_jni LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Root of vendored llama.cpp (headers only for JNI bridge compilation).
set(LLAMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp)
set(PREBUILT_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI})

set(REQUIRED_PREBUILT_LIBS
        libllama.so
        libggml.so
        libggml-base.so
        libggml-cpu.so
        libggml-hexagon.so
        libggml-htp-v68.so
        libggml-htp-v69.so
        libggml-htp-v73.so
        libggml-htp-v75.so
        libggml-htp-v79.so
        libggml-htp-v81.so
)

foreach(LIB_FILE ${REQUIRED_PREBUILT_LIBS})
    if (NOT EXISTS "${PREBUILT_LIB_DIR}/${LIB_FILE}")
        message(FATAL_ERROR
                "Missing prebuilt runtime ${LIB_FILE} at ${PREBUILT_LIB_DIR}. "
                "Build/install Snapdragon artifacts first and copy libs into app/src/main/jniLibs/${ANDROID_ABI}/")
    endif()
endforeach()

add_library(llama_jni SHARED llama_jni.cpp)

target_include_directories(llama_jni PRIVATE
        ${LLAMA_DIR}/include
        ${LLAMA_DIR}/ggml/include
)

add_library(llama_prebuilt SHARED IMPORTED)
set_target_properties(llama_prebuilt PROPERTIES
        IMPORTED_LOCATION "${PREBUILT_LIB_DIR}/libllama.so")

add_library(ggml_prebuilt SHARED IMPORTED)
set_target_properties(ggml_prebuilt PROPERTIES
        IMPORTED_LOCATION "${PREBUILT_LIB_DIR}/libggml.so")

add_library(ggml_base_prebuilt SHARED IMPORTED)
set_target_properties(ggml_base_prebuilt PROPERTIES
        IMPORTED_LOCATION "${PREBUILT_LIB_DIR}/libggml-base.so")

add_library(ggml_cpu_prebuilt SHARED IMPORTED)
set_target_properties(ggml_cpu_prebuilt PROPERTIES
        IMPORTED_LOCATION "${PREBUILT_LIB_DIR}/libggml-cpu.so")

add_library(ggml_hexagon_prebuilt SHARED IMPORTED)
set_target_properties(ggml_hexagon_prebuilt PROPERTIES
        IMPORTED_LOCATION "${PREBUILT_LIB_DIR}/libggml-hexagon.so")

find_library(ANDROID_LOG_LIB log)
find_library(ANDROID_DL_LIB dl)

target_link_libraries(llama_jni PRIVATE
        llama_prebuilt
        ggml_prebuilt
        ggml_base_prebuilt
        ggml_cpu_prebuilt
        ggml_hexagon_prebuilt
        ${ANDROID_LOG_LIB}
        ${ANDROID_DL_LIB}
)

target_compile_options(llama_jni PRIVATE
        $<$<CONFIG:Release>:-O3 -DNDEBUG>
        $<$<CONFIG:Debug>:-O0 -g>
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
        -Wno-unused-function
)

target_link_options(llama_jni PRIVATE
        -Wl,-z,max-page-size=16384
)
