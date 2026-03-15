# Adam — Embeddable AI Agent Library
# Makefile

ADAM_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# ============================================================================
# Modules (git submodules in modules/)
# ============================================================================

MBEDTLS_DIR   := $(ADAM_ROOT)modules/mbedtls
CURL_DIR      := $(ADAM_ROOT)modules/curl
MBEDTLS_BUILD := $(MBEDTLS_DIR)/build
CURL_BUILD    := $(CURL_DIR)/build

# ============================================================================
# Compiler settings
# ============================================================================

CC       ?= cc
CFLAGS   := -std=c11 -Wall -Wextra -Wpedantic -O2
CFLAGS   += -Isrc -I$(CURL_DIR)/include -I$(MBEDTLS_DIR)/include
CFLAGS   += -DADAM_NO_LOCAL -DADAM_NO_SQLITE

# Static libraries
LIBS     := $(CURL_BUILD)/lib/libcurl.a
LIBS     += $(MBEDTLS_BUILD)/library/libmbedtls.a
LIBS     += $(MBEDTLS_BUILD)/library/libmbedx509.a
LIBS     += $(MBEDTLS_BUILD)/library/libmbedcrypto.a

LDFLAGS  := -lpthread -lz

# macOS frameworks
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  LDFLAGS += -framework SystemConfiguration -framework Security -framework CoreFoundation
endif

# ============================================================================
# Sources
# ============================================================================

SRCS := src/arena.c src/adam.c src/adam_json.c src/adam_http.c src/adam_voice.c
OBJS := $(SRCS:.c=.o)

# ============================================================================
# Targets
# ============================================================================

.PHONY: all clean test live voice deps mbedtls curl

all: libadam.a

# --- Static library ---

libadam.a: $(OBJS)
	ar rcs $@ $^

src/%.o: src/%.c
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

test_voice_interactive: libadam.a test/test_voice_interactive.c src/adam_mic_macos.m
	$(CC) $(CFLAGS) -g \
		test/test_voice_interactive.c src/adam_mic_macos.m \
		-L. -ladam $(LIBS) $(LDFLAGS) \
		-framework AVFoundation -framework Foundation \
		-o $@

# --- Dependencies ---

deps: mbedtls curl

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
	rm -f $(OBJS) libadam.a test_adam test_live test_voice_interactive
	rm -rf test_adam.dSYM test_live.dSYM test_voice_interactive.dSYM
