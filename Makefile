# Adam — Embeddable AI Agent Library
# Makefile

ADAM_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
UNAME_S   := $(shell uname -s)
CC        ?= cc

# ============================================================================
# Platform / arch detection (overridable from CI: PLATFORM=android ARCH=arm64-v8a)
# ============================================================================

ifeq ($(OS),Windows_NT)
    PLATFORM ?= windows
    HOST     := windows
    CPUS     := $(shell powershell -Command "[Environment]::ProcessorCount" 2>/dev/null || echo 4)
else
    HOST := $(shell uname -s | tr '[:upper:]' '[:lower:]')
    ifeq ($(HOST),darwin)
        PLATFORM ?= macos
        CPUS     := $(shell sysctl -n hw.ncpu 2>/dev/null || echo 4)
    else
        PLATFORM ?= $(HOST)
        CPUS     := $(shell nproc 2>/dev/null || echo 4)
    endif
endif

# Non-empty when building for an Apple platform (macos / ios / ios-sim).
IS_APPLE := $(filter $(PLATFORM),macos ios ios-sim)

# CMake options pass-through from CI matrix
LLAMA     ?=
WHISPER   ?=
MINIAUDIO ?=

# ============================================================================
# Modules (git submodules in modules/) and shared build/ tree
# ============================================================================

MBEDTLS_DIR   := $(ADAM_ROOT)modules/mbedtls
CURL_DIR      := $(ADAM_ROOT)modules/curl
MINIAUDIO_DIR := $(ADAM_ROOT)modules/miniaudio
LLAMA_DIR     := $(ADAM_ROOT)modules/llama.cpp
WHISPER_DIR   := $(ADAM_ROOT)modules/whisper.cpp
SQLITE_MEMORY_DIR := $(ADAM_ROOT)modules/sqlite-memory
SQLITE_VECTOR_DIR := $(ADAM_ROOT)modules/sqlite-vector

# All dependency builds live under build/ at the repo root.
# Kept relative (not $(ADAM_ROOT)build) so target names like
# `build/llama.cpp.stamp` match what the CI workflow invokes directly.
BUILD_DIR     := build
DIST_DIR      := dist
LLAMA_BUILD   := $(BUILD_DIR)/llama.cpp
WHISPER_BUILD := $(BUILD_DIR)/whisper.cpp
MINIAUDIO_BUILD := $(BUILD_DIR)/miniaudio
MBEDTLS_BUILD := $(BUILD_DIR)/mbedtls
CURL_BUILD    := $(BUILD_DIR)/curl

# ============================================================================
# Compiler settings
# ============================================================================

SQLITE_DIR := $(ADAM_ROOT)modules/sqlite

# Section flags are Mach-O no-ops (Apple's linker uses subsection-via-symbols
# unconditionally), but -flto and -Wl,-dead_strip still apply there.
SIZE_CFLAGS := -ffunction-sections -fdata-sections -flto

CFLAGS := -std=gnu11 -D_POSIX_C_SOURCE=200809L -D_GNU_SOURCE -Wall -Wextra -Wpedantic -O2 -fPIC $(SIZE_CFLAGS)
CFLAGS += -Isrc -I$(MINIAUDIO_DIR) -I$(SQLITE_DIR)
CFLAGS += -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include -I$(LLAMA_DIR)/tools/mtmd
CFLAGS += -I$(WHISPER_DIR)/include
CFLAGS += -I$(SQLITE_MEMORY_DIR)/src -I$(SQLITE_VECTOR_DIR)/src -I$(SQLITE_VECTOR_DIR)/libs

# SQLite compile-time options
SQLITE_FLAGS := -DSQLITE_THREADSAFE=1 \
                -DSQLITE_ENABLE_FTS5=1 \
                -DSQLITE_DQS=0 \
                -DSQLITE_DEFAULT_MEMSTATUS=0 \
                -DSQLITE_DEFAULT_WAL_SYNCHRONOUS=1 \
                -DSQLITE_LIKE_DOESNT_MATCH_BLOBS \
                -DSQLITE_OMIT_DEPRECATED \
                -DSQLITE_OMIT_SHARED_CACHE \
                -DSQLITE_USE_ALLOCA \
                -DSQLITE_OMIT_AUTOINIT

# Android's bionic libc has pthread built in — no separate libpthread.so to link.
# -lz lives at the END of LIBS (single-pass GNU ld + strict musl loader);
# Windows pulls in -lws2_32 / -lcrypt32 / -lbcrypt there too. See the LIBS
# blocks below.
ifeq ($(PLATFORM),android)
LDFLAGS :=
else ifeq ($(PLATFORM),windows)
LDFLAGS :=
else
LDFLAGS := -lpthread
endif

ifneq ($(IS_APPLE),)
LDFLAGS += -Wl,-dead_strip -flto
else
LDFLAGS += -Wl,--gc-sections -flto
endif

# ============================================================================
# Platform-specific: Apple (NSURLSession) vs Other (libcurl + mbedtls)
# ============================================================================

# MinGW Ninja sometimes ignores -DCMAKE_STATIC_LIBRARY_PREFIX=lib for ggml
# sub-targets and emits `ggml.a` / `ggml-cpu.a` / `ggml-base.a` without
# the `lib` prefix. `ggml_lib` resolves to whichever variant exists so
# the LLAMA_LIBS list works on every platform.
ggml_lib = $(firstword $(wildcard $(LLAMA_BUILD)/ggml/src/lib$(1).a $(LLAMA_BUILD)/ggml/src/$(1).a))

LLAMA_LIBS := $(LLAMA_BUILD)/tools/mtmd/libmtmd.a \
              $(LLAMA_BUILD)/src/libllama.a \
              $(call ggml_lib,ggml) \
              $(call ggml_lib,ggml-cpu) \
              $(call ggml_lib,ggml-base)

# Optional GPU backends — wildcards resolve only when llama.cpp was built
# with -DGGML_VULKAN=ON / -DGGML_OPENCL=ON. Static archives go at the end
# of LLAMA_LIBS; matching `-lvulkan` / `-lOpenCL` runtime loaders are
# appended to LIBS by each platform block below.
GGML_VULKAN_LIB := $(firstword $(wildcard $(LLAMA_BUILD)/ggml/src/ggml-vulkan/libggml-vulkan.a $(LLAMA_BUILD)/ggml/src/ggml-vulkan/ggml-vulkan.a))
GGML_OPENCL_LIB := $(firstword $(wildcard $(LLAMA_BUILD)/ggml/src/ggml-opencl/libggml-opencl.a $(LLAMA_BUILD)/ggml/src/ggml-opencl/ggml-opencl.a))
ifneq ($(GGML_VULKAN_LIB),)
  LLAMA_LIBS += $(GGML_VULKAN_LIB)
endif
ifneq ($(GGML_OPENCL_LIB),)
  LLAMA_LIBS += $(GGML_OPENCL_LIB)
endif

# Whisper uses llama's ggml (symlinked: whisper.cpp/ggml → llama.cpp/ggml).
# Only libwhisper.a is needed — ggml symbols come from LLAMA_LIBS.
WHISPER_LIBS := $(WHISPER_BUILD)/src/libwhisper.a

# Apple platforms (macos, ios, ios-sim) all use NSURLSession.
# _DARWIN_C_SOURCE re-exposes BSD extensions (e.g. memmem) that the
# global _POSIX_C_SOURCE=200809L would otherwise hide.
ifneq ($(IS_APPLE),)
  CFLAGS  += -DADAM_NO_CURL -D_DARWIN_C_SOURCE=1
  LDFLAGS += -framework Foundation
  LDFLAGS += -framework SystemConfiguration -framework Security
  LDFLAGS += -framework CoreAudio -framework AudioToolbox -framework AVFoundation
  LDFLAGS += -framework Metal -framework MetalKit -framework Accelerate
  LDFLAGS += -lstdc++
  NET_SRC := src/adam_net_apple.m
  TTS_SYS_SRC := src/adam_tts_system.m
  LLAMA_LIBS += $(LLAMA_BUILD)/ggml/src/ggml-metal/libggml-metal.a
  # Pick up ggml-blas if llama.cpp was configured with -DGGML_BLAS=ON
  # (libggml.a then references blas backend symbols that need this lib).
  GGML_BLAS_LIB := $(wildcard $(LLAMA_BUILD)/ggml/src/ggml-blas/libggml-blas.a)
  ifneq ($(GGML_BLAS_LIB),)
    LLAMA_LIBS += $(GGML_BLAS_LIB)
  endif

  # PLATFORM_CFLAGS holds -arch + -isysroot bits that must apply to *every*
  # compilation unit in libadam.a (otherwise sqlite3.o, sqlite-vector,
  # sqlite-memory get built host-arch and the static archive has mixed
  # architectures). `-x objective-c` is added only to CFLAGS — for iOS,
  # miniaudio.h pulls in <AVFoundation/AVFoundation.h> which is Obj-C only.
  ifeq ($(PLATFORM),macos)
    ifndef ARCH
      PLATFORM_CFLAGS := -arch x86_64 -arch arm64
    else
      PLATFORM_CFLAGS := -arch $(ARCH)
    endif
    CFLAGS  += $(PLATFORM_CFLAGS)
    LDFLAGS += $(PLATFORM_CFLAGS)
  else ifeq ($(PLATFORM),ios)
    APPLE_SDK := -isysroot $(shell xcrun --sdk iphoneos --show-sdk-path) -miphoneos-version-min=14.0
    PLATFORM_CFLAGS := -arch arm64 $(APPLE_SDK)
    CFLAGS  += $(PLATFORM_CFLAGS) -x objective-c
    LDFLAGS += $(PLATFORM_CFLAGS)
  else ifeq ($(PLATFORM),ios-sim)
    APPLE_SDK := -isysroot $(shell xcrun --sdk iphonesimulator --show-sdk-path) -miphonesimulator-version-min=14.0
    PLATFORM_CFLAGS := -arch x86_64 -arch arm64 $(APPLE_SDK)
    CFLAGS  += $(PLATFORM_CFLAGS) -x objective-c
    LDFLAGS += $(PLATFORM_CFLAGS)
  endif

  LIBS := $(WHISPER_LIBS) $(LLAMA_LIBS)
  # Pick up CoreML stub when whisper was built with -DWHISPER_COREML=ON.
  WHISPER_COREML_LIB := $(wildcard $(WHISPER_BUILD)/src/libwhisper.coreml.a)
  ifneq ($(WHISPER_COREML_LIB),)
    LIBS    += $(WHISPER_COREML_LIB)
    LDFLAGS += -framework CoreML
  endif
else ifeq ($(PLATFORM),android)
  # Android: cross-compile via NDK, use libcurl + mbedtls.
  ifndef ARCH
    $(error Android ARCH must be set to ARCH=x86_64 or ARCH=arm64-v8a)
  endif
  ifndef ANDROID_NDK
    $(error ANDROID_NDK must point to the Android NDK install)
  endif
  ANDROID_NDK_BIN := $(ANDROID_NDK)/toolchains/llvm/prebuilt/$(HOST)-x86_64/bin
  ifneq (,$(filter $(ARCH),arm64 arm64-v8a))
    NDK_TRIPLE := aarch64-linux-android26
  else
    NDK_TRIPLE := $(ARCH)-linux-android26
  endif
  CC := $(ANDROID_NDK_BIN)/$(NDK_TRIPLE)-clang
  # Host ar/ranlib choke on ELF objects from the NDK; use llvm-ar from the NDK.
  AR := $(ANDROID_NDK_BIN)/llvm-ar
  CFLAGS  += -I$(CURL_DIR)/include -I$(MBEDTLS_DIR)/include
  LDFLAGS += -ldl -lm
  NET_SRC := src/adam_net_curl.c
  TTS_SYS_SRC := src/adam_tts_system.c
  LIBS    := $(WHISPER_LIBS) $(LLAMA_LIBS)
  LIBS    += $(CURL_BUILD)/lib/libcurl.a
  LIBS    += $(MBEDTLS_BUILD)/library/libmbedtls.a
  LIBS    += $(MBEDTLS_BUILD)/library/libmbedx509.a
  LIBS    += $(MBEDTLS_BUILD)/library/libmbedcrypto.a
  LIBS    += -lz
else ifeq ($(PLATFORM),linux)
  # Linux: use libcurl + mbedtls
  CFLAGS  += -I$(CURL_DIR)/include -I$(MBEDTLS_DIR)/include
  LDFLAGS += -ldl -lm -lstdc++
  NET_SRC := src/adam_net_curl.c
  TTS_SYS_SRC := src/adam_tts_system.c
  LIBS    := $(WHISPER_LIBS) $(LLAMA_LIBS)
  LIBS    += $(CURL_BUILD)/lib/libcurl.a
  LIBS    += $(MBEDTLS_BUILD)/library/libmbedtls.a
  LIBS    += $(MBEDTLS_BUILD)/library/libmbedx509.a
  LIBS    += $(MBEDTLS_BUILD)/library/libmbedcrypto.a
  ifneq ($(GGML_VULKAN_LIB),)
    LIBS += -lvulkan
  endif
  ifneq ($(GGML_OPENCL_LIB),)
    LIBS += -lOpenCL
  endif
  LIBS    += -lz
else
  # Windows/other: use libcurl + mbedtls.
  # -DCURL_STATICLIB: curl.h on Windows declares functions as dllimport
  # by default; without this macro, references to curl_* become __imp_*
  # which a static libcurl.a can't resolve.
  CFLAGS  += -I$(CURL_DIR)/include -I$(MBEDTLS_DIR)/include -DCURL_STATICLIB
  NET_SRC := src/adam_net_curl.c
  TTS_SYS_SRC := src/adam_tts_system.c
  LIBS    := $(WHISPER_LIBS) $(LLAMA_LIBS)
  LIBS    += $(CURL_BUILD)/lib/libcurl.a
  LIBS    += $(MBEDTLS_BUILD)/library/libmbedtls.a
  LIBS    += $(MBEDTLS_BUILD)/library/libmbedx509.a
  LIBS    += $(MBEDTLS_BUILD)/library/libmbedcrypto.a
  # Win32 system libs MUST come after static archives that reference them
  # (single-pass GNU ld). curl needs Winsock + crypt32; mbedtls needs
  # bcrypt for BCryptGenRandom; curl uses zlib for gzip.
  # The Vulkan loader DLL on Windows is `vulkan-1.dll`, so the import
  # library is `libvulkan-1.dll.a` and the flag is `-lvulkan-1`.
  LIBS    += -lws2_32 -lcrypt32 -lbcrypt
  ifneq ($(GGML_VULKAN_LIB),)
    LIBS += -lvulkan-1
  endif
  ifneq ($(GGML_OPENCL_LIB),)
    LIBS += -lOpenCL
  endif
  LIBS    += -lz
endif

# ============================================================================
# Sources
# ============================================================================

SRCS := src/arena.c src/adam.c src/adam_json.c src/adam_http.c src/adam_stream.c \
        src/adam_voice.c src/adam_audio.c src/adam_session.c src/adam_local.c \
        src/adam_stt_local.c src/adam_memory.c src/adam_evolve.c src/adam_research.c \
        src/adam_tools.c src/adam_cache.c
OBJS := $(SRCS:.c=.o)

# SQLite amalgamation (compiled separately with its own flags)
SQLITE_OBJ := $(SQLITE_DIR)/sqlite3.o

# Platform-specific objects (may be .c or .m)
NET_OBJ     := $(basename $(NET_SRC)).o
TTS_SYS_OBJ := $(basename $(TTS_SYS_SRC)).o

# sqlite-vector objects (compiled with -DSQLITE_CORE, no -Wpedantic)
SQLITE_VECTOR_SRCS := $(SQLITE_VECTOR_DIR)/src/sqlite-vector.c \
                      $(SQLITE_VECTOR_DIR)/src/distance-cpu.c \
                      $(SQLITE_VECTOR_DIR)/src/distance-neon.c \
                      $(SQLITE_VECTOR_DIR)/src/distance-avx2.c \
                      $(SQLITE_VECTOR_DIR)/src/distance-avx512.c \
                      $(SQLITE_VECTOR_DIR)/src/distance-sse2.c \
                      $(SQLITE_VECTOR_DIR)/src/distance-rvv.c
SQLITE_VECTOR_OBJS := $(patsubst $(SQLITE_VECTOR_DIR)/src/%.c,$(SQLITE_VECTOR_DIR)/src/%.o,$(SQLITE_VECTOR_SRCS))

# sqlite-memory objects (compiled with -DSQLITE_CORE, no -Wpedantic)
SQLITE_MEMORY_SRCS := $(SQLITE_MEMORY_DIR)/src/sqlite-memory.c \
                      $(SQLITE_MEMORY_DIR)/src/dbmem-utils.c \
                      $(SQLITE_MEMORY_DIR)/src/dbmem-parser.c \
                      $(SQLITE_MEMORY_DIR)/src/dbmem-search.c \
                      $(SQLITE_MEMORY_DIR)/src/md4c.c \
                      $(SQLITE_MEMORY_DIR)/src/dbmem-lembed.c
SQLITE_MEMORY_OBJS := $(patsubst $(SQLITE_MEMORY_DIR)/src/%.c,$(SQLITE_MEMORY_DIR)/src/%.o,$(SQLITE_MEMORY_SRCS))

# sqlite-memory common flags
DBMEM_CFLAGS := -std=gnu11 -O2 -DSQLITE_CORE -DDBMEM_OMIT_REMOTE_ENGINE \
                -D_POSIX_C_SOURCE=200809L -D_GNU_SOURCE -D_DARWIN_C_SOURCE=1 \
                -I$(SQLITE_MEMORY_DIR)/src -I$(SQLITE_VECTOR_DIR)/src \
                -I$(SQLITE_VECTOR_DIR)/libs -I$(SQLITE_DIR) \
                -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include

SQLITE_MEMORY_HTTP_OBJ :=
ifeq ($(UNAME_S),Linux)
  DBMEM_CFLAGS += -I$(CURL_DIR)/include -I$(MBEDTLS_DIR)/include
endif

# ============================================================================
# Targets
# ============================================================================

.PHONY: all clean test live voice talk chat memory vision deps \
        mbedtls curl llama whisper extension xcframework aar version

all: libadam.a adam

# --- Static library ---

AR ?= ar

libadam.a: $(OBJS) $(NET_OBJ) $(TTS_SYS_OBJ) $(SQLITE_OBJ) $(SQLITE_VECTOR_OBJS) $(SQLITE_MEMORY_OBJS) $(SQLITE_MEMORY_HTTP_OBJ)
	$(AR) rcs $@ $^

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# SQLite amalgamation — compiled with its own flags (no -Wpedantic, etc.)
$(SQLITE_OBJ): $(SQLITE_DIR)/sqlite3.c
	$(CC) -std=gnu11 -O2 -fPIC $(PLATFORM_CFLAGS) $(SQLITE_FLAGS) -c $< -o $@

# Objective-C compilation for Apple platform files. Gated on PLATFORM —
# otherwise the explicit .m → .o rule overrides the .c pattern rule on Linux,
# making gcc try to invoke cc1obj on adam_tts_system.m even though TTS_SYS_SRC
# is set to adam_tts_system.c there.
ifneq ($(IS_APPLE),)
src/adam_net_apple.o: src/adam_net_apple.m
	$(CC) $(CFLAGS) -c $< -o $@

src/adam_tts_system.o: src/adam_tts_system.m
	$(CC) $(CFLAGS) -c $< -o $@
endif

# sqlite-vector — compiled with -DSQLITE_CORE (no -Wpedantic)
$(SQLITE_VECTOR_DIR)/src/%.o: $(SQLITE_VECTOR_DIR)/src/%.c
	$(CC) -std=gnu11 -O3 -fPIC $(PLATFORM_CFLAGS) -DSQLITE_CORE -I$(SQLITE_VECTOR_DIR)/src -I$(SQLITE_VECTOR_DIR)/libs -I$(SQLITE_DIR) -c $< -o $@

# sqlite-memory — compiled with -DSQLITE_CORE (no -Wpedantic)
$(SQLITE_MEMORY_DIR)/src/%.o: $(SQLITE_MEMORY_DIR)/src/%.c
	$(CC) -fPIC $(PLATFORM_CFLAGS) $(DBMEM_CFLAGS) -c $< -o $@


# --- CLI ---

adam: libadam.a src/main.c
	$(CC) $(CFLAGS) -g src/main.c -L. -ladam $(LIBS) $(LDFLAGS) -o $@

# --- Tests ---

# `test` runs an extension-load smoke test against dist/adam.{dylib,so,dll}
# plus the test_adam mock-LLM unit-test binary.
#
# SANITIZE=0 disables -fsanitize=address,undefined for environments where
# libasan/libubsan aren't shipped (Alpine musl, MinGW). Local dev keeps
# sanitizers on by default.
#
# SKIP_UNITTEST=1 falls back to smoke-test-only — kept as an escape hatch
# for any platform where the test_adam binary can't be produced at all.
SQLITE3       ?= sqlite3
SKIP_UNITTEST ?= 0
SANITIZE      ?= 1

ifeq ($(SANITIZE),1)
TEST_SANITIZE := -fsanitize=address,undefined
else
TEST_SANITIZE :=
endif

# libadam.a contains miniaudio (via adam_audio.o) and libmtmd.a also embeds
# miniaudio (via mtmd-helper.cpp.o). Strict linkers (GNU ld, lld) reject
# the duplicate ma_atomic_global_lock symbol; Apple's ld merges silently.
# `--allow-multiple-definition` makes GNU ld / lld pick the first
# definition and continue — only needed for the test_adam link (the
# shared-extension link uses --gc-sections + archive semantics that avoid
# pulling mtmd-helper.o on most platforms).
ifneq ($(IS_APPLE),)
TEST_MULDEF :=
else
TEST_MULDEF := -Wl,--allow-multiple-definition
endif

# llama.cpp/whisper.cpp objects are C++. On Apple, `cc` auto-links libc++
# when it sees C++ symbols. On Android NDK and Linux, we need the C++
# driver explicitly so libstdc++/libc++ is brought in.
ifeq ($(PLATFORM),android)
TEST_LD          := $(ANDROID_NDK_BIN)/$(NDK_TRIPLE)-clang++
TEST_LD_EXTRA    := -static-libstdc++
else ifneq ($(IS_APPLE),)
TEST_LD          := $(CC)
TEST_LD_EXTRA    :=
else
TEST_LD          := c++
TEST_LD_EXTRA    :=
endif

TEST_DEPS := $(DIST_DIR)/$(EXT_FILE)
ifeq ($(SKIP_UNITTEST),0)
TEST_DEPS += test_adam
endif

test: $(TEST_DEPS)
	@echo "Running sqlite3 CLI smoke test (load + adam_version)..."
	$(SQLITE3) ":memory:" -cmd ".bail on" ".load ./$(DIST_DIR)/adam" "SELECT adam_version();"
ifeq ($(SKIP_UNITTEST),0)
	./test_adam
endif

test_adam: libadam.a test/test_adam.c
	$(CC) $(CFLAGS) -O0 -g $(TEST_SANITIZE) -c test/test_adam.c -o test_adam.o
	$(TEST_LD) $(PLATFORM_CFLAGS) test_adam.o -L. -ladam $(LIBS) $(LDFLAGS) $(TEST_SANITIZE) $(TEST_LD_EXTRA) $(TEST_MULDEF) -o $@

live: test_live
	./test_live

test_live: libadam.a test/test_live.c
	$(CC) $(CFLAGS) -g -fsanitize=address,undefined \
		test/test_live.c -L. -ladam $(LIBS) $(LDFLAGS) -o $@

voice: test_voice_interactive
	./test_voice_interactive

test_voice_interactive: libadam.a test/test_voice_interactive.c
	$(CC) $(CFLAGS) -g \
		test/test_voice_interactive.c \
		-L. -ladam $(LIBS) $(LDFLAGS) -o $@

memory: test_memory
	./test_memory

test_memory: libadam.a test/test_memory.c
	$(CC) $(CFLAGS) -g \
		test/test_memory.c \
		-L. -ladam $(LIBS) $(LDFLAGS) -o $@

chat: test_chat
ifdef GGUF
	./test_chat $(GGUF)
else
	./test_chat
endif

test_chat: libadam.a test/test_chat.c
	$(CC) $(CFLAGS) -g \
		test/test_chat.c \
		-L. -ladam $(LIBS) $(LDFLAGS) -o $@

talk: test_voice_talk
ifdef LOCAL
ifdef GGUF
	./test_voice_talk --local $(GGUF)
else
	./test_voice_talk --local
endif
else
	./test_voice_talk
endif

test_voice_talk: libadam.a test/test_voice_talk.c
	$(CC) $(CFLAGS) -g \
		test/test_voice_talk.c \
		-L. -ladam $(LIBS) $(LDFLAGS) -o $@

vision: test_vision
ifdef MMPROJ
	./test_vision $(GGUF) $(MMPROJ)
else
	./test_vision $(GGUF)
endif

test_vision: libadam.a test/test_vision.c
	$(CC) $(CFLAGS) -g \
		test/test_vision.c \
		-L. -ladam $(LIBS) $(LDFLAGS) -o $@

# --- Dependencies ---

# Cross-compile cmake options derived from PLATFORM/ARCH.
# These compose with whatever the CI matrix passes via $(LLAMA)/$(WHISPER)/$(MINIAUDIO).
ifeq ($(PLATFORM),macos)
  ifndef ARCH
    PLATFORM_OPTS := -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0
  else
    PLATFORM_OPTS := -DCMAKE_OSX_ARCHITECTURES="$(ARCH)" -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0
  endif
else ifeq ($(PLATFORM),ios)
  PLATFORM_OPTS := -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
else ifeq ($(PLATFORM),ios-sim)
  PLATFORM_OPTS := -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphonesimulator -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64"
else ifeq ($(PLATFORM),android)
  ifndef ARCH
    $(error Android ARCH must be set to ARCH=x86_64 or ARCH=arm64-v8a)
  endif
  ifndef ANDROID_NDK
    $(error ANDROID_NDK must point to the Android NDK install)
  endif
  ANDROID_NDK_BIN := $(ANDROID_NDK)/toolchains/llvm/prebuilt/$(HOST)-x86_64/bin
  PLATFORM_OPTS := -DCMAKE_TOOLCHAIN_FILE=$(ANDROID_NDK)/build/cmake/android.toolchain.cmake \
                   -DANDROID_ABI=$(ARCH) -DANDROID_PLATFORM=android-26 \
                   -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
                   -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF
else
  PLATFORM_OPTS :=
endif

deps: build/llama.cpp.stamp build/whisper.cpp.stamp build/miniaudio.stamp
ifeq ($(UNAME_S),Linux)
deps: build/mbedtls.stamp build/curl.stamp
endif

# Cmake dep builds get the same size flags as adam, plus -fvisibility=hidden
# so llama/ggml/whisper/curl/mbedtls internals don't pollute the final .so's
# dynamic symbol table (only adam_* and sqlite3_adam_init need to be exported).
# Pairs with -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON for cross-archive LTO.
DEP_CFLAGS   := $(SIZE_CFLAGS) -fPIC -fvisibility=hidden
DEP_CXXFLAGS := $(DEP_CFLAGS) -fvisibility-inlines-hidden

# Shared cmake options for every dep (llama / whisper / mbedtls / curl).
# -DCMAKE_POSITION_INDEPENDENT_CODE=ON is required so the resulting .a
# archives can be linked into the shared extension — GNU ld on Linux /
# Windows rejects non-PIC TLS relocations; macOS arm64 is implicitly PIC.
CMAKE_DEP_OPTS := \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
	-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
	-DCMAKE_C_FLAGS="$(DEP_CFLAGS)" \
	-DCMAKE_CXX_FLAGS="$(DEP_CXXFLAGS)"

# Stamp targets: cacheable in CI and idempotent for local dev.
$(BUILD_DIR)/llama.cpp.stamp:
	@mkdir -p $(BUILD_DIR)
	cmake -B $(LLAMA_BUILD) -S $(LLAMA_DIR) \
		-DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF \
		-DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
		-DCMAKE_STATIC_LIBRARY_PREFIX=lib \
		-DGGML_OPENMP=OFF \
		$(CMAKE_DEP_OPTS) $(PLATFORM_OPTS) $(LLAMA)
	cmake --build $(LLAMA_BUILD) --config Release -j$(CPUS) --target llama --target ggml --target mtmd
	touch $@

$(BUILD_DIR)/whisper.cpp.stamp: $(BUILD_DIR)/llama.cpp.stamp
	@mkdir -p $(BUILD_DIR)
	@# whisper.cpp/ggml must be symlinked to llama.cpp/ggml so whisper compiles
	@# against the same ggml that llama.cpp is built with.
	@test -L $(WHISPER_DIR)/ggml || (rm -rf $(WHISPER_DIR)/ggml && ln -s ../llama.cpp/ggml $(WHISPER_DIR)/ggml)
	cmake -B $(WHISPER_BUILD) -S $(WHISPER_DIR) \
		-DBUILD_SHARED_LIBS=OFF -DWHISPER_BUILD_TESTS=OFF \
		-DWHISPER_BUILD_EXAMPLES=OFF \
		-DCMAKE_STATIC_LIBRARY_PREFIX=lib \
		-DGGML_OPENMP=OFF \
		$(CMAKE_DEP_OPTS) $(PLATFORM_OPTS) $(LLAMA) $(WHISPER)
	cmake --build $(WHISPER_BUILD) --config Release -j$(CPUS) --target whisper
	touch $@

$(BUILD_DIR)/miniaudio.stamp:
	@mkdir -p $(MINIAUDIO_BUILD)
	@# miniaudio is header-only — the stamp exists so CI can cache the slot
	@# uniformly with llama/whisper.
	touch $@

$(BUILD_DIR)/mbedtls.stamp:
	@mkdir -p $(BUILD_DIR)
	cd $(MBEDTLS_DIR) && git submodule update --init
	@# MBEDTLS_FATAL_WARNINGS=OFF: mbedtls 3.6.5 has a `%d` printf format
	@# vs `time_t` mismatch on MinGW that hits -Werror=format. Same upstream
	@# code is fine on glibc/macOS where time_t is `long`, so disable globally
	@# rather than per-platform.
	cmake -B $(MBEDTLS_BUILD) -S $(MBEDTLS_DIR) \
		-DENABLE_TESTING=OFF -DENABLE_PROGRAMS=OFF \
		-DMBEDTLS_FATAL_WARNINGS=OFF \
		$(CMAKE_DEP_OPTS) $(PLATFORM_OPTS)
	cmake --build $(MBEDTLS_BUILD) --config Release -j$(CPUS)
	touch $@

$(BUILD_DIR)/curl.stamp: $(BUILD_DIR)/mbedtls.stamp
	cmake -B $(CURL_BUILD) -S $(CURL_DIR) \
		-DCURL_USE_MBEDTLS=ON -DCURL_USE_OPENSSL=OFF \
		-DCURL_DISABLE_NTLM=ON -DCURL_DISABLE_LDAP=ON -DCURL_DISABLE_LDAPS=ON \
		-DCURL_BROTLI=OFF -DCURL_ZSTD=OFF -DUSE_NGHTTP2=OFF \
		-DUSE_LIBIDN2=OFF -DCURL_USE_LIBPSL=OFF -DCURL_USE_LIBSSH2=OFF \
		-DBUILD_SHARED_LIBS=OFF -DBUILD_CURL_EXE=OFF -DBUILD_TESTING=OFF \
		-DMBEDTLS_INCLUDE_DIR=$(MBEDTLS_DIR)/include \
		-DMBEDTLS_LIBRARY=$(MBEDTLS_BUILD)/library/libmbedtls.a \
		-DMBEDX509_LIBRARY=$(MBEDTLS_BUILD)/library/libmbedx509.a \
		-DMBEDCRYPTO_LIBRARY=$(MBEDTLS_BUILD)/library/libmbedcrypto.a \
		$(CMAKE_DEP_OPTS) $(PLATFORM_OPTS)
	cmake --build $(CURL_BUILD) --config Release -j$(CPUS)
	touch $@

# Backwards-compatible aliases so existing developer commands still work.
llama:    $(BUILD_DIR)/llama.cpp.stamp
whisper:  $(BUILD_DIR)/whisper.cpp.stamp
mbedtls:  $(BUILD_DIR)/mbedtls.stamp
curl:     $(BUILD_DIR)/curl.stamp

# --- WASM (Emscripten) ---

WASM_SRCS := src/arena.c src/adam.c src/adam_json.c src/adam_http.c \
             src/adam_stream.c src/adam_evolve.c src/adam_research.c \
             src/adam_tools.c src/adam_cache.c
WASM_CFLAGS := -std=c11 -O2 -Isrc -D_POSIX_C_SOURCE=200809L \
               -DADAM_NO_LOCAL -DADAM_NO_CURL -DADAM_NO_PTHREADS \
               -DADAM_NO_SQLITE -DADAM_NO_VOICE \
               -DARENA_DEFAULT_ALIGN=8

wasm: adam.js

adam.js: $(WASM_SRCS)
	emcc $(WASM_CFLAGS) $(WASM_SRCS) \
		-sEXPORTED_FUNCTIONS='["_adam_init","_adam_cleanup","_adam_create_settings","_adam_settings_destroy","_adam_settings_set_provider","_adam_settings_set_base_url","_adam_settings_set_identity","_adam_settings_set_instructions","_adam_settings_set_stream","_adam_settings_add_tool","_adam_history_create","_adam_history_destroy","_adam_history_clear","_adam_history_count","_adam_history_append_user","_adam_run","_adam_run_simple","_adam_run_result_free","_adam_abort","_adam_abort_reset","_malloc","_free"]' \
		-sEXPORTED_RUNTIME_METHODS='["ccall","cwrap","UTF8ToString","stringToUTF8","lengthBytesUTF8","addFunction","removeFunction"]' \
		-sALLOW_MEMORY_GROWTH=1 -sASYNCIFY \
		-sALLOW_TABLE_GROWTH=1 \
		-sMODULARIZE=1 -sEXPORT_NAME=AdamModule \
		-o $@

# ============================================================================
# CI / packaging targets — built by .github/workflows/main.yml
# ============================================================================

# Print the version string from src/adam.h. Consumed by the release job.
version:
	@sed -n 's/^#define ADAM_VERSION_STRING[[:space:]]*"\([^"]*\)".*/\1/p' src/adam.h

# Loadable SQLite extension: dist/adam.{dylib,so,dll}.
# Wraps extensions/sqlite/Makefile after libadam.a + dependencies are built.
ifeq ($(PLATFORM),windows)
  EXT_FILE := adam.dll
else ifeq ($(PLATFORM),macos)
  EXT_FILE := adam.dylib
else ifeq ($(PLATFORM),ios)
  EXT_FILE := adam.dylib
else ifeq ($(PLATFORM),ios-sim)
  EXT_FILE := adam.dylib
else
  EXT_FILE := adam.so
endif

# Non-Apple platforms (linux, windows, android) need libcurl + mbedtls.
EXT_DEPS := $(BUILD_DIR)/llama.cpp.stamp $(BUILD_DIR)/whisper.cpp.stamp $(BUILD_DIR)/miniaudio.stamp
ifeq ($(IS_APPLE),)
  EXT_DEPS += $(BUILD_DIR)/mbedtls.stamp $(BUILD_DIR)/curl.stamp
endif

# Set STRIP_DIST=0 to keep symbols when debugging the shipped artifact.
STRIP_DIST ?= 1
ifeq ($(PLATFORM),android)
  STRIP       := $(ANDROID_NDK_BIN)/llvm-strip
  STRIP_FLAGS := --strip-all
else ifneq ($(IS_APPLE),)
  STRIP       := strip
  STRIP_FLAGS := -x
else
  STRIP       := strip
  STRIP_FLAGS := --strip-unneeded
endif

extension: $(DIST_DIR)/$(EXT_FILE)

$(DIST_DIR)/$(EXT_FILE): $(EXT_DEPS) libadam.a
	@mkdir -p $(DIST_DIR)
	$(MAKE) -C extensions/sqlite all PLATFORM=$(PLATFORM) ARCH=$(ARCH)
	cp extensions/sqlite/$(EXT_FILE) $(DIST_DIR)/$(EXT_FILE)
ifeq ($(STRIP_DIST),1)
	$(STRIP) $(STRIP_FLAGS) $(DIST_DIR)/$(EXT_FILE)
endif

# Apple XCFramework — builds adam.dylib three times (macos, ios, ios-sim) and
# bundles them into dist/adam.xcframework with framework metadata.
LIB_NAMES := ios.dylib ios-sim.dylib macos.dylib
FMWK_NAMES := ios-arm64 ios-arm64_x86_64-simulator macos-arm64_x86_64

.NOTPARALLEL: %.dylib
%.dylib:
	@# Clean enough to force a per-platform rebuild WITHOUT wiping dist/
	@# (each iteration produces a renamed dylib in dist/ that the xcframework
	@# bundling step needs).
	rm -rf $(BUILD_DIR) libadam.a src/*.o $(SQLITE_OBJ) $(SQLITE_VECTOR_OBJS) $(SQLITE_MEMORY_OBJS)
	$(MAKE) -C extensions/sqlite clean
	$(MAKE) extension PLATFORM=$* LLAMA="$(LLAMA)" WHISPER="$(WHISPER)" MINIAUDIO="$(MINIAUDIO)"
	mv $(DIST_DIR)/adam.dylib $(DIST_DIR)/$@

define ADAM_PLIST
<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\
<plist version=\"1.0\">\
<dict>\
<key>CFBundleDevelopmentRegion</key><string>en</string>\
<key>CFBundleExecutable</key><string>adam</string>\
<key>CFBundleIdentifier</key><string>ai.sqlite.adam</string>\
<key>CFBundleInfoDictionaryVersion</key><string>6.0</string>\
<key>CFBundlePackageType</key><string>FMWK</string>\
<key>CFBundleSignature</key><string>????</string>\
<key>CFBundleVersion</key><string>$(shell make -s version)</string>\
<key>CFBundleShortVersionString</key><string>$(shell make -s version)</string>\
<key>MinimumOSVersion</key><string>11.0</string>\
</dict></plist>
endef

define ADAM_MODULEMAP
framework module adam {\
  umbrella header \"adam.h\"\
  export *\
}
endef

$(DIST_DIR)/%.xcframework: $(LIB_NAMES)
	@$(foreach i,1 2,\
		lib=$(word $(i),$(LIB_NAMES)); \
		fmwk=$(word $(i),$(FMWK_NAMES)); \
		mkdir -p $(DIST_DIR)/$$fmwk/adam.framework/Headers; \
		mkdir -p $(DIST_DIR)/$$fmwk/adam.framework/Modules; \
		cp src/adam.h $(DIST_DIR)/$$fmwk/adam.framework/Headers; \
		printf "$(ADAM_PLIST)" > $(DIST_DIR)/$$fmwk/adam.framework/Info.plist; \
		printf "$(ADAM_MODULEMAP)" > $(DIST_DIR)/$$fmwk/adam.framework/Modules/module.modulemap; \
		mv $(DIST_DIR)/$$lib $(DIST_DIR)/$$fmwk/adam.framework/adam; \
		install_name_tool -id "@rpath/adam.framework/adam" $(DIST_DIR)/$$fmwk/adam.framework/adam; \
	)
	@lib=$(word 3,$(LIB_NAMES)); \
	fmwk=$(word 3,$(FMWK_NAMES)); \
	mkdir -p $(DIST_DIR)/$$fmwk/adam.framework/Versions/A/Headers; \
	mkdir -p $(DIST_DIR)/$$fmwk/adam.framework/Versions/A/Modules; \
	mkdir -p $(DIST_DIR)/$$fmwk/adam.framework/Versions/A/Resources; \
	cp src/adam.h $(DIST_DIR)/$$fmwk/adam.framework/Versions/A/Headers; \
	printf "$(ADAM_PLIST)" > $(DIST_DIR)/$$fmwk/adam.framework/Versions/A/Resources/Info.plist; \
	printf "$(ADAM_MODULEMAP)" > $(DIST_DIR)/$$fmwk/adam.framework/Versions/A/Modules/module.modulemap; \
	mv $(DIST_DIR)/$$lib $(DIST_DIR)/$$fmwk/adam.framework/Versions/A/adam; \
	install_name_tool -id "@rpath/adam.framework/adam" $(DIST_DIR)/$$fmwk/adam.framework/Versions/A/adam; \
	ln -sf A $(DIST_DIR)/$$fmwk/adam.framework/Versions/Current; \
	ln -sf Versions/Current/adam $(DIST_DIR)/$$fmwk/adam.framework/adam; \
	ln -sf Versions/Current/Headers $(DIST_DIR)/$$fmwk/adam.framework/Headers; \
	ln -sf Versions/Current/Modules $(DIST_DIR)/$$fmwk/adam.framework/Modules; \
	ln -sf Versions/Current/Resources $(DIST_DIR)/$$fmwk/adam.framework/Resources;
	xcodebuild -create-xcframework $(foreach fmwk,$(FMWK_NAMES),-framework $(DIST_DIR)/$(fmwk)/adam.framework) -output $@
	rm -rf $(foreach fmwk,$(FMWK_NAMES),$(DIST_DIR)/$(fmwk))

xcframework: $(DIST_DIR)/adam.xcframework

# Android AAR — builds adam.so for arm64-v8a + x86_64, drops them into
# packages/android/src/main/jniLibs/, then runs Gradle to assemble the AAR.
AAR_ARM64 := packages/android/src/main/jniLibs/arm64-v8a/
AAR_X86   := packages/android/src/main/jniLibs/x86_64/
aar:
	mkdir -p $(AAR_ARM64) $(AAR_X86)
	$(MAKE) clean
	$(MAKE) extension PLATFORM=android ARCH=arm64-v8a
	mv $(DIST_DIR)/adam.so $(AAR_ARM64)
	$(MAKE) clean
	$(MAKE) extension PLATFORM=android ARCH=x86_64
	mv $(DIST_DIR)/adam.so $(AAR_X86)
	cd packages/android && ./gradlew clean assembleRelease
	@mkdir -p $(DIST_DIR)
	cp packages/android/build/outputs/aar/android-release.aar $(DIST_DIR)/adam.aar

# --- Clean ---

clean:
	rm -f $(OBJS) $(SQLITE_OBJ) libadam.a adam test_adam test_live test_chat test_memory test_evolve test_voice_interactive test_voice_talk test_tools test_vision
	@# Wipe ALL platform-specific net/tts objects, not just the current PLATFORM's,
	@# so switching PLATFORM=android → PLATFORM=ios doesn't leave stale arch objects.
	rm -f src/adam_net_apple.o src/adam_net_curl.o src/adam_tts_system.o
	rm -f $(SQLITE_VECTOR_OBJS) $(SQLITE_MEMORY_OBJS) $(SQLITE_MEMORY_HTTP_OBJ)
	rm -rf *.dSYM
	rm -f adam.js adam.wasm
	rm -rf $(BUILD_DIR) $(DIST_DIR)
	$(MAKE) -C extensions/sqlite clean 2>/dev/null || true
