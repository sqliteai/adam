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
  LD := $(ANDROID_NDK_BIN)/$(NDK_TRIPLE)-clang++
else
  # Use the C++ driver for the final link on every platform. libllama /
  # libmtmd / libwhisper are C++ archives that reference typeinfo symbols
  # like _ZTVN10__cxxabiv117__class_type_infoE. With the C driver and GNU
  # ld's --as-needed default, the -lstdc++ in LDFLAGS gets dropped (it
  # appears before the objects), and dlopen fails at load. macOS' linker
  # is permissive enough that `cc` happened to work there, but we make it
  # consistent. `c++` resolves to clang++ on Apple, g++ on Linux/MinGW.
  LD := c++
endif

# ----------------------------------------------------------------------------
# Common flags
# ----------------------------------------------------------------------------
# Feature-test macros: keep BSD/GNU/POSIX extensions visible on every libc.
# glibc strict-C11 hides strdup / strcasecmp / clock_gettime; Apple needs
# _DARWIN_C_SOURCE; MinGW needs _GNU_SOURCE for strndup. Defining all three
# is safe — each platform only honors the ones it knows about.
# Section flags pair with -Wl,--gc-sections / -Wl,-dead_strip below.
CFLAGS := -std=gnu11 -Wall -Wextra -O2 -fPIC -ffunction-sections -fdata-sections -flto
CFLAGS += -D_POSIX_C_SOURCE=200809L -D_GNU_SOURCE -D_DARWIN_C_SOURCE=1
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

# ggml's CMakeLists sets VERSION/SOVERSION on its static targets, which
# on Windows MinGW + Ninja causes cmake to drop the `lib` prefix even
# with -DCMAKE_STATIC_LIBRARY_PREFIX=lib in the cache. Resolve each lib
# via $(wildcard …) so we pick up whichever name was produced.
ggml_lib = $(firstword $(wildcard $(LLAMA_BUILD)/ggml/src/lib$(1).a $(LLAMA_BUILD)/ggml/src/$(1).a))
LLAMA_LIBS := $(LLAMA_BUILD)/tools/mtmd/libmtmd.a \
              $(LLAMA_BUILD)/src/libllama.a \
              $(call ggml_lib,ggml) \
              $(call ggml_lib,ggml-cpu) \
              $(call ggml_lib,ggml-base)
WHISPER_LIBS := $(WHISPER_BUILD)/src/libwhisper.a

# Pick up CoreML stub when whisper was built with -DWHISPER_COREML=ON.
# wildcard returns empty when the file isn't there, so non-CoreML builds
# are unaffected.
WHISPER_COREML_LIB := $(wildcard $(WHISPER_BUILD)/src/libwhisper.coreml.a)
ifneq ($(WHISPER_COREML_LIB),)
  WHISPER_LIBS += $(WHISPER_COREML_LIB)
  COREML_FRAMEWORK := -framework CoreML
endif

# Pick up GPU backends if llama.cpp was configured with the matching flag.
# Check both naming conventions (lib prefix may be stripped on Windows MinGW).
GGML_BLAS_LIB   := $(firstword $(wildcard $(LLAMA_BUILD)/ggml/src/ggml-blas/libggml-blas.a $(LLAMA_BUILD)/ggml/src/ggml-blas/ggml-blas.a))
GGML_VULKAN_LIB := $(firstword $(wildcard $(LLAMA_BUILD)/ggml/src/ggml-vulkan/libggml-vulkan.a $(LLAMA_BUILD)/ggml/src/ggml-vulkan/ggml-vulkan.a))
GGML_OPENCL_LIB := $(firstword $(wildcard $(LLAMA_BUILD)/ggml/src/ggml-opencl/libggml-opencl.a $(LLAMA_BUILD)/ggml/src/ggml-opencl/ggml-opencl.a))
ifneq ($(GGML_BLAS_LIB),)
  LLAMA_LIBS += $(GGML_BLAS_LIB)
endif
ifneq ($(GGML_VULKAN_LIB),)
  LLAMA_LIBS += $(GGML_VULKAN_LIB)
endif
ifneq ($(GGML_OPENCL_LIB),)
  LLAMA_LIBS += $(GGML_OPENCL_LIB)
endif

# Default link libs. -lz is appended to LIBS (not LDFLAGS) for every
# platform that uses libcurl — libcurl references inflate*, and with
# single-pass GNU ld + --as-needed, -lz must sit AFTER libcurl.a or
# libz.so gets dropped from DT_NEEDED. glibc loaders happen to find
# inflateEnd through the global namespace anyway; musl is strict and
# fails dlopen with "Error relocating: inflateEnd: symbol not found".
ifeq ($(PLATFORM),android)
  LDFLAGS :=
else ifeq ($(PLATFORM),windows)
  LDFLAGS :=
else
  LDFLAGS := -lpthread
endif

# Dead-code elimination at link time + LTO across the wrapper objects and
# the input static archives (those built with -DCMAKE_INTERPROCEDURAL_-
# OPTIMIZATION=ON ship LLVM bitcode, so this can also collapse calls into
# llama / whisper / curl / mbedtls).
ifneq (,$(filter $(PLATFORM),macos ios ios-sim))
  LDFLAGS += -Wl,-dead_strip -flto
else
  LDFLAGS += -Wl,--gc-sections -flto
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
  # -DCURL_STATICLIB: curl.h on Windows declares functions as dllimport
  # by default; we link the static libcurl.a so undo that.
  CFLAGS  += -I$(ADAM_ROOT)/modules/curl/include -I$(ADAM_ROOT)/modules/mbedtls/include -DCURL_STATICLIB
  LDFLAGS += -shared -static-libgcc
  LDFLAGS += -Wl,--push-state,-Bstatic,-lstdc++,-lwinpthread,--pop-state
  LIBS := $(WHISPER_LIBS) $(LLAMA_LIBS)
  LIBS += $(CURL_BUILD)/lib/libcurl.a
  LIBS += $(MBEDTLS_BUILD)/library/libmbedtls.a
  LIBS += $(MBEDTLS_BUILD)/library/libmbedx509.a
  LIBS += $(MBEDTLS_BUILD)/library/libmbedcrypto.a
  # System libs MUST come after the static archives that reference them
  # (GNU ld is single-pass left-to-right). curl needs Winsock + crypt32;
  # mbedtls needs bcrypt for BCryptGenRandom; curl uses zlib for gzip
  # content-encoding.
  LIBS += -lws2_32 -lcrypt32 -lbcrypt -lz
else ifeq ($(PLATFORM),android)
  EXT := so
  # -static-libstdc++ now takes effect because the final link uses clang++
  # (see PLATFORM=android block above); it bundles libc++_static into
  # adam.so so it loads on devices without the NDK shared C++ runtime.
  LDFLAGS += -shared -static-libstdc++ -ldl -lm -Wl,-z,max-page-size=16384
  CFLAGS  += -I$(ADAM_ROOT)/modules/curl/include -I$(ADAM_ROOT)/modules/mbedtls/include
  LIBS := $(WHISPER_LIBS) $(LLAMA_LIBS)
  LIBS += $(CURL_BUILD)/lib/libcurl.a
  LIBS += $(MBEDTLS_BUILD)/library/libmbedtls.a
  LIBS += $(MBEDTLS_BUILD)/library/libmbedx509.a
  LIBS += $(MBEDTLS_BUILD)/library/libmbedcrypto.a
  LIBS += -lz
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
  LIBS += -lz
endif

# Vulkan / OpenCL runtime loaders. Must come AFTER the ggml-vulkan/-opencl
# static libs in LIBS (single-pass GNU ld). Only added when the matching
# backend was built (wildcard above resolved); Apple's Metal path skips
# these — neither GGML_VULKAN_LIB nor GGML_OPENCL_LIB is set on Apple.
# On Windows the Vulkan loader DLL is `vulkan-1.dll` (with the -1 suffix
# baked in) and the corresponding import library is libvulkan-1.dll.a, so
# the linker flag is `-lvulkan-1`, not `-lvulkan`.
ifneq ($(GGML_VULKAN_LIB),)
  ifeq ($(PLATFORM),windows)
    LIBS += -lvulkan-1
  else
    LIBS += -lvulkan
  endif
endif
ifneq ($(GGML_OPENCL_LIB),)
  LIBS += -lOpenCL
endif

.PHONY: all clean test

all: adam.$(EXT)

SQLITE_OBJS := $(SQLITE_SRCS:.c=.o)
SHARED_OBJS := $(SHARED_SRCS:.c=.o)

# Compile rules — use $(CC) (the C driver) with -std=c11.
$(SQLITE_OBJS): %.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(SHARED_OBJS): %.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Link rule — use $(LD), which is $(CC) everywhere except Android where it
# is clang++ so that -static-libstdc++ pulls in libc++_static.
adam.$(EXT): $(SQLITE_OBJS) $(SHARED_OBJS) $(ADAM_LIB)
	$(LD) $(LDFLAGS) $(SQLITE_OBJS) $(SHARED_OBJS) $(ADAM_LIB) $(LIBS) -o $@

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
	rm -f $(SQLITE_OBJS) $(SHARED_OBJS)
