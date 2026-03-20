# SQLite Adam Extension
# Builds adam.dylib (macOS) / adam.so (Linux) loadable extension

ADAM_ROOT := ../..
UNAME_S   := $(shell uname -s)
CC        ?= cc

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

LLAMA_BUILD   := $(ADAM_ROOT)/modules/llama.cpp/build
WHISPER_BUILD := $(ADAM_ROOT)/modules/whisper.cpp/build

LLAMA_LIBS := $(LLAMA_BUILD)/tools/mtmd/libmtmd.a \
              $(LLAMA_BUILD)/src/libllama.a \
              $(LLAMA_BUILD)/ggml/src/libggml.a \
              $(LLAMA_BUILD)/ggml/src/libggml-cpu.a \
              $(LLAMA_BUILD)/ggml/src/libggml-base.a
WHISPER_LIBS := $(WHISPER_BUILD)/src/libwhisper.a

LDFLAGS := -lpthread -lz

ifeq ($(UNAME_S),Darwin)
  CFLAGS  += -DADAM_NO_CURL
  EXT     := dylib
  LDFLAGS += -dynamiclib
  LDFLAGS += -framework Foundation -framework SystemConfiguration -framework Security
  LDFLAGS += -framework CoreAudio -framework AudioToolbox
  LDFLAGS += -framework Metal -framework MetalKit -framework Accelerate
  LDFLAGS += -lstdc++ -framework AVFoundation
  LLAMA_LIBS += $(LLAMA_BUILD)/ggml/src/ggml-metal/libggml-metal.a
  LLAMA_LIBS += $(LLAMA_BUILD)/ggml/src/ggml-blas/libggml-blas.a
  LIBS := $(WHISPER_LIBS) $(LLAMA_LIBS)
else
  EXT     := so
  LDFLAGS += -shared -ldl -lm -lstdc++
  CFLAGS  += -I$(ADAM_ROOT)/modules/curl/include -I$(ADAM_ROOT)/modules/mbedtls/include
  LIBS := $(WHISPER_LIBS) $(LLAMA_LIBS)
  LIBS += $(ADAM_ROOT)/modules/curl/build/lib/libcurl.a
  LIBS += $(ADAM_ROOT)/modules/mbedtls/build/library/libmbedtls.a
  LIBS += $(ADAM_ROOT)/modules/mbedtls/build/library/libmbedx509.a
  LIBS += $(ADAM_ROOT)/modules/mbedtls/build/library/libmbedcrypto.a
endif

.PHONY: all clean test

all: adam.$(EXT)

adam.$(EXT): $(SQLITE_SRCS) $(SHARED_SRCS) $(ADAM_LIB)
	$(CC) $(CFLAGS) $(LDFLAGS) $(SQLITE_SRCS) $(SHARED_SRCS) \
		$(ADAM_LIB) $(LIBS) -o $@

test: adam.$(EXT)
	@echo "Testing adam SQLite extension..."
	@echo ".load ./adam" > /tmp/adam_ext_test.sql
	@echo "SELECT adam_config('provider', 'anthropic');" >> /tmp/adam_ext_test.sql
	@echo "SELECT adam_config('api_key', '$(ANTHROPIC_API_KEY)');" >> /tmp/adam_ext_test.sql
	@echo "SELECT adam('Say hello in 3 words');" >> /tmp/adam_ext_test.sql
	sqlite3 :memory: < /tmp/adam_ext_test.sql
	@rm -f /tmp/adam_ext_test.sql

clean:
	rm -f adam.$(EXT) adam.dylib adam.so
