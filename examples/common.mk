# Common Makefile for Adam examples
# Include from each example: ADAM_ROOT := ../..
#                             include $(ADAM_ROOT)/examples/common.mk

UNAME_S := $(shell uname -s)
CC      ?= cc
BINARY  := $(notdir $(CURDIR))

CFLAGS := -std=c11 -Wall -Wextra -Wpedantic -O2 -g
CFLAGS += -I$(ADAM_ROOT)/src
CFLAGS += -I$(ADAM_ROOT)/modules/miniaudio
CFLAGS += -I$(ADAM_ROOT)/modules/sqlite
CFLAGS += -I$(ADAM_ROOT)/modules/llama.cpp/include
CFLAGS += -I$(ADAM_ROOT)/modules/llama.cpp/ggml/include
CFLAGS += -I$(ADAM_ROOT)/modules/llama.cpp/tools/mtmd
CFLAGS += -I$(ADAM_ROOT)/modules/whisper.cpp/include
CFLAGS += -I$(ADAM_ROOT)/modules/sqlite-memory/src
CFLAGS += -I$(ADAM_ROOT)/modules/sqlite-vector/src
CFLAGS += -I$(ADAM_ROOT)/modules/sqlite-vector/libs

LDFLAGS := -lpthread -lz

LLAMA_BUILD   := $(ADAM_ROOT)/modules/llama.cpp/build
WHISPER_BUILD := $(ADAM_ROOT)/modules/whisper.cpp/build

LLAMA_LIBS := $(LLAMA_BUILD)/tools/mtmd/libmtmd.a \
              $(LLAMA_BUILD)/src/libllama.a \
              $(LLAMA_BUILD)/ggml/src/libggml.a \
              $(LLAMA_BUILD)/ggml/src/libggml-cpu.a \
              $(LLAMA_BUILD)/ggml/src/libggml-base.a

WHISPER_LIBS := $(WHISPER_BUILD)/src/libwhisper.a

ifeq ($(UNAME_S),Darwin)
  CFLAGS  += -DADAM_NO_CURL
  LDFLAGS += -framework Foundation
  LDFLAGS += -framework SystemConfiguration -framework Security
  LDFLAGS += -framework CoreAudio -framework AudioToolbox
  LDFLAGS += -framework Metal -framework MetalKit -framework Accelerate
  LDFLAGS += -lstdc++
  LLAMA_LIBS += $(LLAMA_BUILD)/ggml/src/ggml-metal/libggml-metal.a
  LLAMA_LIBS += $(LLAMA_BUILD)/ggml/src/ggml-blas/libggml-blas.a
  LDFLAGS += -framework AVFoundation
  LIBS := $(WHISPER_LIBS) $(LLAMA_LIBS)
else
  # Linux / Windows / other: use libcurl + mbedtls
  CFLAGS  += -I$(ADAM_ROOT)/modules/curl/include -I$(ADAM_ROOT)/modules/mbedtls/include
  LDFLAGS += -lstdc++
  ifeq ($(UNAME_S),Linux)
    LDFLAGS += -ldl -lm
  endif
  LIBS := $(WHISPER_LIBS) $(LLAMA_LIBS)
  LIBS += $(ADAM_ROOT)/modules/curl/build/lib/libcurl.a
  LIBS += $(ADAM_ROOT)/modules/mbedtls/build/library/libmbedtls.a
  LIBS += $(ADAM_ROOT)/modules/mbedtls/build/library/libmbedx509.a
  LIBS += $(ADAM_ROOT)/modules/mbedtls/build/library/libmbedcrypto.a
endif

ADAM_LIB := $(ADAM_ROOT)/libadam.a

.PHONY: all clean run

all: $(BINARY)

$(BINARY): main.c $(ADAM_LIB)
	$(CC) $(CFLAGS) main.c -L$(ADAM_ROOT) -ladam $(LIBS) $(LDFLAGS) -o $@

run: $(BINARY)
	./$(BINARY)

clean:
	rm -f $(BINARY)
