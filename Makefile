# Adam — Embeddable AI Agent Library
# Makefile

ADAM_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
UNAME_S   := $(shell uname -s)
CC        ?= cc

# ============================================================================
# Modules (git submodules in modules/)
# ============================================================================

MBEDTLS_DIR   := $(ADAM_ROOT)modules/mbedtls
CURL_DIR      := $(ADAM_ROOT)modules/curl
MBEDTLS_BUILD := $(MBEDTLS_DIR)/build
CURL_BUILD    := $(CURL_DIR)/build
MINIAUDIO_DIR := $(ADAM_ROOT)modules/miniaudio
LLAMA_DIR     := $(ADAM_ROOT)modules/llama.cpp
LLAMA_BUILD   := $(LLAMA_DIR)/build
WHISPER_DIR   := $(ADAM_ROOT)modules/whisper.cpp
WHISPER_BUILD := $(WHISPER_DIR)/build

# ============================================================================
# Compiler settings
# ============================================================================

SQLITE_DIR := $(ADAM_ROOT)modules/sqlite

CFLAGS := -std=c11 -Wall -Wextra -Wpedantic -O2
CFLAGS += -Isrc -I$(MINIAUDIO_DIR) -I$(SQLITE_DIR)
CFLAGS += -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include
CFLAGS += -I$(WHISPER_DIR)/include

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

LDFLAGS := -lpthread -lz

# ============================================================================
# Platform-specific: Apple (NSURLSession) vs Other (libcurl + mbedtls)
# ============================================================================

LLAMA_LIBS := $(LLAMA_BUILD)/src/libllama.a \
              $(LLAMA_BUILD)/ggml/src/libggml.a \
              $(LLAMA_BUILD)/ggml/src/libggml-cpu.a \
              $(LLAMA_BUILD)/ggml/src/libggml-base.a

# Whisper uses only libwhisper.a — ggml symbols come from llama's ggml
WHISPER_LIBS := $(WHISPER_BUILD)/src/libwhisper.a

ifeq ($(UNAME_S),Darwin)
  # Apple: use NSURLSession — no curl/mbedtls needed
  CFLAGS  += -DADAM_NO_CURL
  LDFLAGS += -framework Foundation
  LDFLAGS += -framework SystemConfiguration -framework Security
  LDFLAGS += -framework CoreAudio -framework AudioToolbox
  LDFLAGS += -framework Metal -framework MetalKit -framework Accelerate
  LDFLAGS += -lstdc++
  NET_SRC := src/adam_net_apple.m
  TTS_SYS_SRC := src/adam_tts_system.m
  LLAMA_LIBS += $(LLAMA_BUILD)/ggml/src/ggml-metal/libggml-metal.a
  LLAMA_LIBS += $(LLAMA_BUILD)/ggml/src/ggml-blas/libggml-blas.a
  LDFLAGS += -framework AVFoundation
  LIBS    := $(WHISPER_LIBS) $(LLAMA_LIBS)
else ifeq ($(UNAME_S),Linux)
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
else
  # Windows/other: use libcurl + mbedtls
  CFLAGS  += -I$(CURL_DIR)/include -I$(MBEDTLS_DIR)/include
  NET_SRC := src/adam_net_curl.c
  TTS_SYS_SRC := src/adam_tts_system.c
  LIBS    := $(CURL_BUILD)/lib/libcurl.a
  LIBS    += $(MBEDTLS_BUILD)/library/libmbedtls.a
  LIBS    += $(MBEDTLS_BUILD)/library/libmbedx509.a
  LIBS    += $(MBEDTLS_BUILD)/library/libmbedcrypto.a
endif

# ============================================================================
# Sources
# ============================================================================

SRCS := src/arena.c src/adam.c src/adam_json.c src/adam_http.c src/adam_stream.c \
        src/adam_voice.c src/adam_audio.c src/adam_session.c src/adam_local.c \
        src/adam_stt_local.c
OBJS := $(SRCS:.c=.o)

# SQLite amalgamation (compiled separately with its own flags)
SQLITE_OBJ := $(SQLITE_DIR)/sqlite3.o

# Platform-specific objects (may be .c or .m)
NET_OBJ     := $(basename $(NET_SRC)).o
TTS_SYS_OBJ := $(basename $(TTS_SYS_SRC)).o

# ============================================================================
# Targets
# ============================================================================

.PHONY: all clean test live voice talk chat deps mbedtls curl llama whisper

all: libadam.a

# --- Static library ---

libadam.a: $(OBJS) $(NET_OBJ) $(TTS_SYS_OBJ) $(SQLITE_OBJ)
	ar rcs $@ $^

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# SQLite amalgamation — compiled with its own flags (no -Wpedantic, etc.)
$(SQLITE_OBJ): $(SQLITE_DIR)/sqlite3.c
	$(CC) -std=c11 -O2 $(SQLITE_FLAGS) -c $< -o $@

# Objective-C compilation for Apple platform files
src/adam_net_apple.o: src/adam_net_apple.m
	$(CC) $(CFLAGS) -c $< -o $@

src/adam_tts_system.o: src/adam_tts_system.m
	$(CC) $(CFLAGS) -c $< -o $@

# --- Tests ---

test: test_adam
	./test_adam

test_adam: libadam.a test/test_adam.c
	$(CC) $(CFLAGS) -g -fsanitize=address,undefined \
		test/test_adam.c -L. -ladam $(LIBS) $(LDFLAGS) -o $@

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

# --- Dependencies ---

deps: llama whisper
ifeq ($(UNAME_S),Linux)
deps: mbedtls curl
endif

llama:
	cmake -B $(LLAMA_BUILD) -S $(LLAMA_DIR) \
		-DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF \
		-DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
		-DCMAKE_BUILD_TYPE=Release
	cmake --build $(LLAMA_BUILD) -j$$(sysctl -n hw.ncpu 2>/dev/null || nproc) --target llama --target ggml

whisper:
	cmake -B $(WHISPER_BUILD) -S $(WHISPER_DIR) \
		-DBUILD_SHARED_LIBS=OFF -DWHISPER_BUILD_TESTS=OFF \
		-DWHISPER_BUILD_EXAMPLES=OFF \
		-DCMAKE_BUILD_TYPE=Release
	cmake --build $(WHISPER_BUILD) -j$$(sysctl -n hw.ncpu 2>/dev/null || nproc) --target whisper

mbedtls:
	cd $(MBEDTLS_DIR) && git submodule update --init
	cmake -B $(MBEDTLS_BUILD) -S $(MBEDTLS_DIR) \
		-DENABLE_TESTING=OFF -DENABLE_PROGRAMS=OFF \
		-DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON
	cmake --build $(MBEDTLS_BUILD) -j$$(sysctl -n hw.ncpu 2>/dev/null || nproc)

curl: mbedtls
	cmake -B $(CURL_BUILD) -S $(CURL_DIR) \
		-DCURL_USE_MBEDTLS=ON -DCURL_USE_OPENSSL=OFF \
		-DCURL_DISABLE_NTLM=ON -DCURL_DISABLE_LDAP=ON -DCURL_DISABLE_LDAPS=ON \
		-DCURL_BROTLI=OFF -DCURL_ZSTD=OFF -DUSE_NGHTTP2=OFF \
		-DUSE_LIBIDN2=OFF -DCURL_USE_LIBPSL=OFF -DCURL_USE_LIBSSH2=OFF \
		-DBUILD_SHARED_LIBS=OFF -DBUILD_CURL_EXE=OFF -DBUILD_TESTING=OFF \
		-DCMAKE_BUILD_TYPE=Release \
		-DMBEDTLS_INCLUDE_DIR=$(MBEDTLS_DIR)/include \
		-DMBEDTLS_LIBRARY=$(MBEDTLS_BUILD)/library/libmbedtls.a \
		-DMBEDX509_LIBRARY=$(MBEDTLS_BUILD)/library/libmbedx509.a \
		-DMBEDCRYPTO_LIBRARY=$(MBEDTLS_BUILD)/library/libmbedcrypto.a
	cmake --build $(CURL_BUILD) -j$$(sysctl -n hw.ncpu 2>/dev/null || nproc)

# --- Clean ---

clean:
	rm -f $(OBJS) $(NET_OBJ) $(TTS_SYS_OBJ) $(SQLITE_OBJ) libadam.a test_adam test_live test_chat test_voice_interactive test_voice_talk
	rm -rf test_adam.dSYM test_live.dSYM test_voice_interactive.dSYM test_voice_talk.dSYM
