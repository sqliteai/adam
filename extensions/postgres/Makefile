# Adam PostgreSQL Extension
# Uses PGXS build system

ADAM_ROOT := ../..

MODULE_big = adam
EXTENSION  = adam
DATA       = adam--0.1.sql

OBJS = pg_adam.o \
       ../src/adam_ext_ctx.o \
       ../src/adam_ext_config.o \
       ../src/adam_ext_session.o \
       ../src/adam_ext_schema.o \
       ../src/adam_ext_chat.o

# Adam library and dependencies
ADAM_LIB := $(ADAM_ROOT)/libadam.a
LLAMA_BUILD := $(ADAM_ROOT)/modules/llama.cpp/build
WHISPER_BUILD := $(ADAM_ROOT)/modules/whisper.cpp/build

SHLIB_LINK := -Wl,--no-as-needed $(ADAM_LIB) \
              $(WHISPER_BUILD)/src/libwhisper.a \
              -lgomp \
              $(LLAMA_BUILD)/tools/mtmd/libmtmd.a \
              $(LLAMA_BUILD)/src/libllama.a \
              $(LLAMA_BUILD)/ggml/src/libggml.a \
              $(LLAMA_BUILD)/ggml/src/libggml-cpu.a \
              $(LLAMA_BUILD)/ggml/src/libggml-base.a \
              -lstdc++ -lpthread -lz

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  PG_CFLAGS += -DADAM_NO_CURL
  SHLIB_LINK += $(LLAMA_BUILD)/ggml/src/ggml-metal/libggml-metal.a \
                $(LLAMA_BUILD)/ggml/src/ggml-blas/libggml-blas.a \
                -framework Foundation -framework SystemConfiguration \
                -framework Security -framework CoreAudio -framework AudioToolbox \
                -framework Metal -framework MetalKit -framework Accelerate \
                -framework AVFoundation
else
  SHLIB_LINK += $(ADAM_ROOT)/modules/curl/build/lib/libcurl.a \
                $(ADAM_ROOT)/modules/mbedtls/build/library/libmbedtls.a \
                $(ADAM_ROOT)/modules/mbedtls/build/library/libmbedx509.a \
                $(ADAM_ROOT)/modules/mbedtls/build/library/libmbedcrypto.a \
                -ldl -lm
endif

PG_CPPFLAGS += -I$(ADAM_ROOT)/src \
             -I$(ADAM_ROOT)/modules/sqlite \
             -I$(ADAM_ROOT)/modules/miniaudio \
             -I$(ADAM_ROOT)/modules/llama.cpp/include \
             -I$(ADAM_ROOT)/modules/llama.cpp/ggml/include \
             -I$(ADAM_ROOT)/modules/llama.cpp/tools/mtmd \
             -I$(ADAM_ROOT)/modules/whisper.cpp/include \
             -I$(ADAM_ROOT)/modules/sqlite-memory/src \
             -I$(ADAM_ROOT)/modules/sqlite-vector/src \
             -I$(ADAM_ROOT)/modules/sqlite-vector/libs \
             -I../src

# Force single architecture to match libadam.a
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),arm64)
  override CFLAGS := $(filter-out -arch x86_64,$(CFLAGS)) -arch arm64
  override LDFLAGS_SL := $(filter-out -arch x86_64,$(LDFLAGS_SL)) -arch arm64
endif

PG_CONFIG ?= pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
