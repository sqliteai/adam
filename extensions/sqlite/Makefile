# SQLite Adam Extension
# Builds the loadable extension: adam.dylib (macOS/iOS) / adam.so (Linux/Android) / adam.dll (Windows)
#
# Driven by the top-level Makefile via `make extension`. Honors PLATFORM and ARCH
# the same way the parent does so cross-compile (Android, iOS) works end-to-end.

ADAM_ROOT := ../..

# ----------------------------------------------------------------------------
# Platform / arch detection (matches top-level Makefile)
# ----------------------------------------------------------------------------
ifeq ($(OS),Windows_NT)
    PLATFORM ?= windows
    HOST     := windows
else
    HOST := $(shell uname -s | tr '[:upper:]' '[:lower:]')
    ifeq ($(HOST),darwin)
        PLATFORM ?= macos
    else
        PLATFORM ?= $(HOST)
    endif
endif

# Compiler — Android overrides via NDK toolchain; everything else uses cc.
CC ?= cc

ifeq ($(PLATFORM),android)
  ifndef ARCH
    $(error Android ARCH must be set to ARCH=x86_64, ARCH=arm64-v8a, or ARCH=armeabi-v7a)
  endif
  ifndef ANDROID_NDK
    $(error ANDROID_NDK must point to the Android NDK install)
  endif
  ANDROID_NDK_BIN := $(ANDROID_NDK)/toolchains/llvm/prebuilt/$(HOST)-x86_64/bin
  ifneq (,$(filter $(ARCH),arm64 arm64-v8a))
    NDK_TRIPLE := aarch64-linux-android26
  else ifeq ($(ARCH),armeabi-v7a)
    NDK_TRIPLE := armv7a-linux-androideabi26
  else
    NDK_TRIPLE := $(ARCH)-linux-android26
  endif
  CC := $(ANDROID_NDK_BIN)/$(NDK_TRIPLE)-clang
endif

# ----------------------------------------------------------------------------
# Common flags
# ----------------------------------------------------------------------------
CFLAGS := -std=c11 -Wall -Wextra -O2 -fPIC
CFLAGS += -I$(ADAM_ROOT)/src
CFLAGS += -I$(ADAM_ROOT)/modules/sqlite
CFLAGS += -I$(ADAM_ROOT)/modules/miniaudio
CFLAGS += -I$(ADAM_ROOT)/modules/llama.cpp/include
CFLAGS += -I$(ADAM_ROOT)/modules/llama.cpp/ggml/include
CFLAGS += -I$(ADAM_ROOT)/modules/llama.cpp/tools/mtmd
CFLAGS += -I$(ADAM_ROOT)/modules/whisper.cpp/include
CFLAGS += -I$(ADAM_ROOT)/modules/sqlite-memory/src
CFLAGS += -I$(ADAM_ROOT)/modules/sqlite-vector/src
CFLAGS += -I$(ADAM_ROOT)/modules/sqlite-vector/libs
CFLAGS += -I../src

SHARED_SRCS := ../src/adam_ext_ctx.c ../src/adam_ext_config.c \
               ../src/adam_ext_session.c ../src/adam_ext_schema.c \
               ../src/adam_ext_chat.c
SQLITE_SRCS := sqlite_adam.c

ADAM_LIB := $(ADAM_ROOT)/libadam.a

# Dependency builds live under the top-level build/ tree.
LLAMA_BUILD   := $(ADAM_ROOT)/build/llama.cpp
WHISPER_BUILD := $(ADAM_ROOT)/build/whisper.cpp
MBEDTLS_BUILD := $(ADAM_ROOT)/build/mbedtls
CURL_BUILD    := $(ADAM_ROOT)/build/curl

LLAMA_LIBS := $(LLAMA_BUILD)/tools/mtmd/libmtmd.a \
              $(LLAMA_BUILD)/src/libllama.a \
              $(LLAMA_BUILD)/ggml/src/libggml.a \
              $(LLAMA_BUILD)/ggml/src/libggml-cpu.a \
              $(LLAMA_BUILD)/ggml/src/libggml-base.a
WHISPER_LIBS := $(WHISPER_BUILD)/src/libwhisper.a

# Pick up CoreML stub when whisper was built with -DWHISPER_COREML=ON.
# wildcard returns empty when the file isn't there, so non-CoreML builds
# are unaffected.
WHISPER_COREML_LIB := $(wildcard $(WHISPER_BUILD)/src/libwhisper.coreml.a)
ifneq ($(WHISPER_COREML_LIB),)
  WHISPER_LIBS += $(WHISPER_COREML_LIB)
  COREML_FRAMEWORK := -framework CoreML
endif

# Pick up ggml-blas if llama.cpp was configured with -DGGML_BLAS=ON.
GGML_BLAS_LIB := $(wildcard $(LLAMA_BUILD)/ggml/src/ggml-blas/libggml-blas.a)
ifneq ($(GGML_BLAS_LIB),)
  LLAMA_LIBS += $(GGML_BLAS_LIB)
endif

# Android's bionic libc has pthread built in — no separate libpthread.so to link.
ifeq ($(PLATFORM),android)
  LDFLAGS := -lz
else
  LDFLAGS := -lpthread -lz
endif

# ----------------------------------------------------------------------------
# Per-platform link / ext settings
# ----------------------------------------------------------------------------
ifeq ($(PLATFORM),macos)
  EXT := dylib
  CFLAGS  += -DADAM_NO_CURL
  ifndef ARCH
    CFLAGS  += -arch x86_64 -arch arm64
    LDFLAGS += -arch x86_64 -arch arm64
  else
    CFLAGS  += -arch $(ARCH)
    LDFLAGS += -arch $(ARCH)
  endif
  LDFLAGS += -dynamiclib -headerpad_max_install_names
  LDFLAGS += -framework Foundation -framework SystemConfiguration -framework Security
  LDFLAGS += -framework CoreAudio -framework AudioToolbox -framework AVFoundation
  LDFLAGS += -framework Metal -framework MetalKit -framework Accelerate $(COREML_FRAMEWORK)
  LDFLAGS += -lstdc++
  LLAMA_LIBS += $(LLAMA_BUILD)/ggml/src/ggml-metal/libggml-metal.a
  LIBS := $(WHISPER_LIBS) $(LLAMA_LIBS)
else ifeq ($(PLATFORM),ios)
  EXT := dylib
  CFLAGS  += -DADAM_NO_CURL -arch arm64 -x objective-c
  CFLAGS  += -isysroot $(shell xcrun --sdk iphoneos --show-sdk-path) -miphoneos-version-min=14.0
  LDFLAGS += -dynamiclib -headerpad_max_install_names
  LDFLAGS += -isysroot $(shell xcrun --sdk iphoneos --show-sdk-path) -miphoneos-version-min=14.0
  LDFLAGS += -framework Foundation -framework Security -framework AVFoundation
  LDFLAGS += -framework CoreAudio -framework AudioToolbox
  LDFLAGS += -framework Metal -framework Accelerate $(COREML_FRAMEWORK)
  LDFLAGS += -lstdc++
  LLAMA_LIBS += $(LLAMA_BUILD)/ggml/src/ggml-metal/libggml-metal.a
  LIBS := $(WHISPER_LIBS) $(LLAMA_LIBS)
else ifeq ($(PLATFORM),ios-sim)
  EXT := dylib
  CFLAGS  += -DADAM_NO_CURL -arch x86_64 -arch arm64 -x objective-c
  CFLAGS  += -isysroot $(shell xcrun --sdk iphonesimulator --show-sdk-path) -miphonesimulator-version-min=14.0
  LDFLAGS += -arch x86_64 -arch arm64 -dynamiclib -headerpad_max_install_names
  LDFLAGS += -isysroot $(shell xcrun --sdk iphonesimulator --show-sdk-path) -miphonesimulator-version-min=14.0
  LDFLAGS += -framework Foundation -framework Security -framework AVFoundation
  LDFLAGS += -framework CoreAudio -framework AudioToolbox
  LDFLAGS += -framework Metal -framework Accelerate $(COREML_FRAMEWORK)
  LDFLAGS += -lstdc++
  LLAMA_LIBS += $(LLAMA_BUILD)/ggml/src/ggml-metal/libggml-metal.a
  LIBS := $(WHISPER_LIBS) $(LLAMA_LIBS)
else ifeq ($(PLATFORM),windows)
  EXT := dll
  CFLAGS  += -I$(ADAM_ROOT)/modules/curl/include -I$(ADAM_ROOT)/modules/mbedtls/include
  LDFLAGS += -shared -lbcrypt -static-libgcc
  LDFLAGS += -Wl,--push-state,-Bstatic,-lstdc++,-lwinpthread,--pop-state
  LIBS := $(WHISPER_LIBS) $(LLAMA_LIBS)
  LIBS += $(CURL_BUILD)/lib/libcurl.a
  LIBS += $(MBEDTLS_BUILD)/library/libmbedtls.a
  LIBS += $(MBEDTLS_BUILD)/library/libmbedx509.a
  LIBS += $(MBEDTLS_BUILD)/library/libmbedcrypto.a
else ifeq ($(PLATFORM),android)
  EXT := so
  # -static-libstdc++ has no effect with NDK clang in this link config; omitted.
  LDFLAGS += -shared -ldl -lm -Wl,-z,max-page-size=16384
  CFLAGS  += -I$(ADAM_ROOT)/modules/curl/include -I$(ADAM_ROOT)/modules/mbedtls/include
  LIBS := $(WHISPER_LIBS) $(LLAMA_LIBS)
  LIBS += $(CURL_BUILD)/lib/libcurl.a
  LIBS += $(MBEDTLS_BUILD)/library/libmbedtls.a
  LIBS += $(MBEDTLS_BUILD)/library/libmbedx509.a
  LIBS += $(MBEDTLS_BUILD)/library/libmbedcrypto.a
else
  # linux (default)
  EXT := so
  LDFLAGS += -shared -ldl -lm -lstdc++
  CFLAGS  += -I$(ADAM_ROOT)/modules/curl/include -I$(ADAM_ROOT)/modules/mbedtls/include
  LIBS := $(WHISPER_LIBS) $(LLAMA_LIBS)
  LIBS += $(CURL_BUILD)/lib/libcurl.a
  LIBS += $(MBEDTLS_BUILD)/library/libmbedtls.a
  LIBS += $(MBEDTLS_BUILD)/library/libmbedx509.a
  LIBS += $(MBEDTLS_BUILD)/library/libmbedcrypto.a
endif

.PHONY: all clean test

all: adam.$(EXT)

adam.$(EXT): $(SQLITE_SRCS) $(SHARED_SRCS) $(ADAM_LIB)
	@# `-x none` resets the language after the source files so subsequent
	@# .a files aren't reinterpreted as source (e.g. when CFLAGS includes
	@# `-x objective-c` for iOS).
	$(CC) $(CFLAGS) $(LDFLAGS) $(SQLITE_SRCS) $(SHARED_SRCS) \
		-x none $(ADAM_LIB) $(LIBS) -o $@

test: adam.$(EXT)
	@echo "Testing adam SQLite extension..."
	@echo ".load ./adam" > /tmp/adam_ext_test.sql
	@echo "SELECT adam_config('provider', 'anthropic');" >> /tmp/adam_ext_test.sql
	@echo "SELECT adam_config('api_key', '$(ANTHROPIC_API_KEY)');" >> /tmp/adam_ext_test.sql
	@echo "SELECT adam('Say hello in 3 words');" >> /tmp/adam_ext_test.sql
	sqlite3 :memory: < /tmp/adam_ext_test.sql
	@rm -f /tmp/adam_ext_test.sql

clean:
	rm -f adam.dylib adam.so adam.dll adam.lib adam.def
