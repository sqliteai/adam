FROM postgres:16-bookworm

RUN apt-get update && apt-get install -y \
    build-essential cmake git \
    postgresql-server-dev-16 \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . /adam
WORKDIR /adam

# Clean any macOS build artifacts that leaked through
RUN find . -name "*.o" -delete && find . -name "*.a" -delete \
    && rm -rf modules/llama.cpp/build modules/whisper.cpp/build \
       modules/mbedtls/build modules/curl/build

# Build llama.cpp (no GPU, no native optimizations — portable)
RUN cmake -B modules/llama.cpp/build -S modules/llama.cpp \
    -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
    -DGGML_NATIVE=OFF -DGGML_CPU_AARCH64=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_C_FLAGS="-Wno-error" -DCMAKE_CXX_FLAGS="-Wno-error" \
    && cmake --build modules/llama.cpp/build -j$(nproc) --target llama --target ggml --target mtmd

# Skip whisper.cpp — not needed for PG extension (no voice/STT)
# Create stub libwhisper.a so the Makefile doesn't complain
RUN mkdir -p modules/whisper.cpp/build/src \
    && ar rcs modules/whisper.cpp/build/src/libwhisper.a

# Build mbedtls
RUN cd modules/mbedtls && git submodule update --init \
    && cmake -B build -S . -DENABLE_TESTING=OFF -DENABLE_PROGRAMS=OFF \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    && cmake --build build -j$(nproc)

# Build curl
RUN cmake -B modules/curl/build -S modules/curl \
    -DCURL_USE_MBEDTLS=ON -DCURL_USE_OPENSSL=OFF \
    -DCURL_DISABLE_NTLM=ON -DCURL_DISABLE_LDAP=ON -DCURL_DISABLE_LDAPS=ON \
    -DCURL_BROTLI=OFF -DCURL_ZSTD=OFF -DUSE_NGHTTP2=OFF \
    -DUSE_LIBIDN2=OFF -DCURL_USE_LIBPSL=OFF -DCURL_USE_LIBSSH2=OFF \
    -DBUILD_SHARED_LIBS=OFF -DBUILD_CURL_EXE=OFF -DBUILD_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DMBEDTLS_INCLUDE_DIR=/adam/modules/mbedtls/include \
    -DMBEDTLS_LIBRARY=/adam/modules/mbedtls/build/library/libmbedtls.a \
    -DMBEDX509_LIBRARY=/adam/modules/mbedtls/build/library/libmbedx509.a \
    -DMBEDCRYPTO_LIBRARY=/adam/modules/mbedtls/build/library/libmbedcrypto.a \
    && cmake --build modules/curl/build -j$(nproc)

# Build libadam.a — remove macOS and voice files, inject -fPIC
RUN rm -f src/*.m src/adam_stt_local.c src/adam_voice.c src/adam_audio.c \
    && sed -i 's/^CFLAGS := /CFLAGS := -fPIC /' Makefile \
    && sed -i '/adam_tts_system\.m/d' Makefile \
    && sed -i '/adam_net_apple\.m/d' Makefile \
    && sed -i 's/^DBMEM_CFLAGS := /DBMEM_CFLAGS := -fPIC -D_GNU_SOURCE /' Makefile \
    && sed -i 's/-std=c11 -O3 -DSQLITE_CORE/-std=c11 -O3 -fPIC -DSQLITE_CORE/' Makefile \
    && sed -i 's/-std=c11 -O2 $(SQLITE_FLAGS)/-std=c11 -O2 -fPIC $(SQLITE_FLAGS)/' Makefile \
    && sed -i 's|src/adam_voice\.c ||' Makefile \
    && sed -i 's|src/adam_audio\.c ||' Makefile \
    && sed -i 's|src/adam_stt_local\.c ||' Makefile \
    && echo 'void adam_stt_local_cleanup(void *s){(void)s;} int adam_stt_local_transcribe(){return -1;}' > src/adam_stt_stub.c \
    && sed -i 's|src/adam_cache.c|src/adam_cache.c src/adam_stt_stub.c|' Makefile \
    && make libadam.a

# Build and install PG extension — capture full link output
RUN cd extensions/postgres && make 2>&1 | tee /tmp/pg_build.log || true
RUN grep -i "error\|reloc\|cannot\|fPIC\|recompile\|undefined" /tmp/pg_build.log || echo "No known error patterns"
RUN test -f extensions/postgres/adam.so && make -C extensions/postgres install || echo "adam.so NOT built — see errors above"

# Test script
COPY .env /tmp/.env
COPY extensions/postgres/test_adam.sh /tmp/test_adam.sh
RUN chmod +x /tmp/test_adam.sh

CMD ["/tmp/test_adam.sh"]
