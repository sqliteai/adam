//
//  adam_tools.c
//  Adam — Built-in tools: file I/O, shell, calculator, web search, SQL, HTTP
//
//  Platform support:
//    macOS/Linux/iOS/Android: POSIX (realpath, opendir, popen)
//    Windows: _fullpath, _popen, FindFirstFile
//    WASM: filesystem/shell tools disabled by default
//
//  Feature gates:
//    ADAM_NO_FILESYSTEM — disables file_read, file_write, list_directory
//    ADAM_NO_SHELL      — disables shell_exec
//
//  Created by Marco Bambini on 17/03/26.
//

#include "adam.h"
#include "adam_net.h"
#define JSMN_STATIC
#include "jsmn.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <math.h>

// WASM: disable filesystem and shell by default
#if defined(__EMSCRIPTEN__)
  #ifndef ADAM_NO_FILESYSTEM
    #define ADAM_NO_FILESYSTEM
  #endif
  #ifndef ADAM_NO_SHELL
    #define ADAM_NO_SHELL
  #endif
#endif

// ============================================================================
// MARK: - Platform compatibility layer
// ============================================================================

#ifdef _WIN32
  // Windows
  #include <windows.h>
  #include <direct.h>
  #include <io.h>

  #ifndef PATH_MAX
    #define PATH_MAX MAX_PATH
  #endif

  #define adam_getcwd   _getcwd
  #define adam_popen    _popen
  #define adam_pclose   _pclose
  #define adam_mkdir(p) _mkdir(p)

  static char *adam_realpath(const char *path, char *resolved) {
      return _fullpath(resolved, path, PATH_MAX);
  }

  // WEXITSTATUS equivalent — _pclose returns exit code directly on Windows
  #define ADAM_EXIT_STATUS(s) (s)

  // S_ISDIR / S_ISREG for Windows
  #include <sys/stat.h>
  #ifndef S_ISDIR
    #define S_ISDIR(m) (((m) & _S_IFMT) == _S_IFDIR)
  #endif
  #ifndef S_ISREG
    #define S_ISREG(m) (((m) & _S_IFMT) == _S_IFREG)
  #endif
  #define S_ISLNK(m) (0) // Windows has no symlink mode in stat

  // Path separator
  #define ADAM_PATH_SEP '\\'
  #define ADAM_PATH_SEP_STR "\\"

  // Directory listing (Windows FindFirstFile/FindNextFile)
  #ifndef ADAM_NO_FILESYSTEM
  typedef struct {
      HANDLE          h;
      WIN32_FIND_DATAA ffd;
      int             first;
  } adam_dir_t;

  static adam_dir_t *adam_opendir(const char *path) {
      adam_dir_t *d = calloc(1, sizeof(adam_dir_t));
      if (!d) return NULL;
      char pattern[PATH_MAX];
      snprintf(pattern, sizeof(pattern), "%s\\*", path);
      d->h = FindFirstFileA(pattern, &d->ffd);
      if (d->h == INVALID_HANDLE_VALUE) { free(d); return NULL; }
      d->first = 1;
      return d;
  }

  static const char *adam_readdir(adam_dir_t *d) {
      if (d->first) { d->first = 0; return d->ffd.cFileName; }
      if (!FindNextFileA(d->h, &d->ffd)) return NULL;
      return d->ffd.cFileName;
  }

  static void adam_closedir(adam_dir_t *d) {
      if (d) { FindClose(d->h); free(d); }
  }

  static int adam_dir_is_dir(adam_dir_t *d) {
      return (d->ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
  }
  #endif // ADAM_NO_FILESYSTEM

#else
  // POSIX (macOS, Linux, iOS, Android)
  #include <unistd.h>
  #include <limits.h>
  #ifndef ADAM_NO_FILESYSTEM
    #include <dirent.h>
    #include <sys/stat.h>
  #endif
  #ifndef ADAM_NO_SHELL
    #include <sys/wait.h>
  #endif

  #ifndef PATH_MAX
    #define PATH_MAX 4096
  #endif

  #define adam_getcwd   getcwd
  #define adam_popen    popen
  #define adam_pclose   pclose
  #define adam_realpath realpath
  #define adam_mkdir(p) mkdir(p, 0755)
  #define ADAM_EXIT_STATUS(s) WEXITSTATUS(s)
  #define ADAM_PATH_SEP '/'
  #define ADAM_PATH_SEP_STR "/"
#endif

// ============================================================================
// MARK: - SQLite (conditional)
// ============================================================================

#ifndef ADAM_NO_SQLITE
#include "sqlite3.h"
extern sqlite3 *adam_memory_db(adam_memory_t *mem);
#endif

// ============================================================================
// MARK: - jsmn helpers
// ============================================================================

static int jtok_eq(const char *json, const jsmntok_t *tok, const char *s) {
    size_t slen = strlen(s);
    return (tok->type == JSMN_STRING
         && (size_t)(tok->end - tok->start) == slen
         && memcmp(json + tok->start, s, slen) == 0);
}

static const char *jtok_arena_str(arena_t *arena, const char *json,
                                   const jsmntok_t *tok) {
    size_t len = (size_t)(tok->end - tok->start);
    char *s = arena_alloc(arena, len + 1);
    if (!s) return NULL;
    memcpy(s, json + tok->start, len);
    s[len] = '\0';
    return s;
}

static char *jtok_malloc_str(const char *json, const jsmntok_t *tok) {
    size_t len = (size_t)(tok->end - tok->start);
    char *s = malloc(len + 1);
    if (!s) return NULL;
    memcpy(s, json + tok->start, len);
    s[len] = '\0';
    return s;
}

// ============================================================================
// MARK: - Sandbox: path validation
// ============================================================================

#ifndef ADAM_NO_FILESYSTEM

static const char *sandbox_check(arena_t *arena, const adam_settings_t *s,
                                  const char *path) {
    if (!s || !path) return NULL;
    if (s->allowed_dir_count == 0) return NULL;

    char resolved[PATH_MAX];
    if (!adam_realpath(path, resolved)) {
        // File may not exist — resolve parent dir + basename
        char *path_copy = strdup(path);
        if (!path_copy) return NULL;

        char *last_sep = strrchr(path_copy, '/');
#ifdef _WIN32
        // Also check backslash on Windows
        char *last_bsep = strrchr(path_copy, '\\');
        if (last_bsep && (!last_sep || last_bsep > last_sep))
            last_sep = last_bsep;
#endif
        if (!last_sep) {
            // Relative to CWD
            free(path_copy);
            char cwd[PATH_MAX];
            if (!adam_getcwd(cwd, sizeof(cwd))) return NULL;
            for (size_t i = 0; i < s->allowed_dir_count; i++) {
                size_t dlen = strlen(s->allowed_dirs[i]);
                if (strncmp(cwd, s->allowed_dirs[i], dlen) == 0
                    && (cwd[dlen] == '/' || cwd[dlen] == '\\'
                        || cwd[dlen] == '\0')) {
                    snprintf(resolved, sizeof(resolved), "%s" ADAM_PATH_SEP_STR "%s",
                             cwd, path);
                    return arena_strdup(arena, resolved);
                }
            }
            return NULL;
        }

        *last_sep = '\0';
        const char *parent = path_copy;
        const char *basename = last_sep + 1;

        char parent_resolved[PATH_MAX];
        if (!adam_realpath(parent, parent_resolved)) {
            free(path_copy);
            return NULL;
        }

        snprintf(resolved, sizeof(resolved), "%s" ADAM_PATH_SEP_STR "%s",
                 parent_resolved, basename);
        free(path_copy);
    }

    // Check resolved path against allowed directories
    for (size_t i = 0; i < s->allowed_dir_count; i++) {
        size_t dlen = strlen(s->allowed_dirs[i]);
        if (strncmp(resolved, s->allowed_dirs[i], dlen) == 0
            && (resolved[dlen] == '/' || resolved[dlen] == '\\'
                || resolved[dlen] == '\0')) {
            return arena_strdup(arena, resolved);
        }
    }

    return NULL;
}

#endif // ADAM_NO_FILESYSTEM

// ============================================================================
// MARK: - Tool: file_read
// ============================================================================

#ifndef ADAM_NO_FILESYSTEM

adam_tool_result_t adam_tool_file_read(arena_t *arena, void *ctx,
                                       const char *args_json, size_t args_len) {
    adam_tool_result_t res = { .for_llm = NULL, .for_user = NULL, .success = 0 };
    adam_settings_t *s = (adam_settings_t *)ctx;
    if (!s || !args_json) {
        res.for_llm = arena_strdup(arena, "Error: settings not configured");
        return res;
    }

    jsmntok_t tokens[16];
    jsmn_parser parser;
    jsmn_init(&parser);
    int ntok = jsmn_parse(&parser, args_json, args_len, tokens, 16);
    if (ntok < 1) {
        res.for_llm = arena_strdup(arena, "Error: invalid JSON");
        return res;
    }

    const char *path = NULL;
    int max_bytes = 65536;

    for (int i = 1; i < ntok - 1; i++) {
        if (jtok_eq(args_json, &tokens[i], "path") &&
            tokens[i + 1].type == JSMN_STRING) {
            path = jtok_arena_str(arena, args_json, &tokens[i + 1]);
            i++;
        } else if (jtok_eq(args_json, &tokens[i], "max_bytes") &&
                   tokens[i + 1].type == JSMN_PRIMITIVE) {
            max_bytes = atoi(args_json + tokens[i + 1].start);
            i++;
        }
    }

    if (!path) {
        res.for_llm = arena_strdup(arena, "Error: 'path' is required");
        return res;
    }

    const char *resolved = sandbox_check(arena, s, path);
    if (!resolved) {
        res.for_llm = arena_strdup(arena,
            "Error: access denied (path not in allowed directories)");
        return res;
    }

    FILE *f = fopen(resolved, "rb");
    if (!f) {
        char buf[256];
        snprintf(buf, sizeof(buf), "Error: cannot open file: %s", strerror(errno));
        res.for_llm = arena_strdup(arena, buf);
        return res;
    }

    if (max_bytes <= 0) max_bytes = 65536;
    if (max_bytes > 1048576) max_bytes = 1048576;

    char *buf = arena_alloc(arena, (size_t)max_bytes + 1);
    if (!buf) { fclose(f); res.for_llm = arena_strdup(arena, "Error: OOM"); return res; }

    size_t nread = fread(buf, 1, (size_t)max_bytes, f);
    fclose(f);
    buf[nread] = '\0';

    res.for_llm = buf;
    res.success = 1;
    return res;
}

// ============================================================================
// MARK: - Tool: file_write
// ============================================================================

adam_tool_result_t adam_tool_file_write(arena_t *arena, void *ctx,
                                        const char *args_json, size_t args_len) {
    adam_tool_result_t res = { .for_llm = NULL, .for_user = NULL, .success = 0 };
    adam_settings_t *s = (adam_settings_t *)ctx;
    if (!s || !args_json) {
        res.for_llm = arena_strdup(arena, "Error: settings not configured");
        return res;
    }

    jsmntok_t tokens[16];
    jsmn_parser parser;
    jsmn_init(&parser);
    int ntok = jsmn_parse(&parser, args_json, args_len, tokens, 16);
    if (ntok < 1) {
        res.for_llm = arena_strdup(arena, "Error: invalid JSON");
        return res;
    }

    const char *path = NULL;
    const char *content = NULL;
    int append = 0;

    for (int i = 1; i < ntok - 1; i++) {
        if (jtok_eq(args_json, &tokens[i], "path") &&
            tokens[i + 1].type == JSMN_STRING) {
            path = jtok_arena_str(arena, args_json, &tokens[i + 1]);
            i++;
        } else if (jtok_eq(args_json, &tokens[i], "content") &&
                   tokens[i + 1].type == JSMN_STRING) {
            content = jtok_arena_str(arena, args_json, &tokens[i + 1]);
            i++;
        } else if (jtok_eq(args_json, &tokens[i], "append") &&
                   tokens[i + 1].type == JSMN_PRIMITIVE) {
            append = (args_json[tokens[i + 1].start] == 't');
            i++;
        }
    }

    if (!path || !content) {
        res.for_llm = arena_strdup(arena,
            "Error: 'path' and 'content' are required");
        return res;
    }

    const char *resolved = sandbox_check(arena, s, path);
    if (!resolved) {
        res.for_llm = arena_strdup(arena,
            "Error: access denied (path not in allowed directories)");
        return res;
    }

    FILE *f = fopen(resolved, append ? "a" : "w");
    if (!f) {
        char buf[256];
        snprintf(buf, sizeof(buf), "Error: cannot open file for writing: %s",
                 strerror(errno));
        res.for_llm = arena_strdup(arena, buf);
        return res;
    }

    size_t len = strlen(content);
    size_t written = fwrite(content, 1, len, f);
    fclose(f);

    if (written != len) {
        res.for_llm = arena_strdup(arena, "Error: incomplete write");
        return res;
    }

    char msg[128];
    snprintf(msg, sizeof(msg), "Wrote %zu bytes to %s", written, path);
    res.for_llm = arena_strdup(arena, msg);
    res.success = 1;
    return res;
}

// ============================================================================
// MARK: - Tool: list_directory
// ============================================================================

adam_tool_result_t adam_tool_list_directory(arena_t *arena, void *ctx,
                                            const char *args_json,
                                            size_t args_len) {
    adam_tool_result_t res = { .for_llm = NULL, .for_user = NULL, .success = 0 };
    adam_settings_t *s = (adam_settings_t *)ctx;
    if (!s || !args_json) {
        res.for_llm = arena_strdup(arena, "Error: settings not configured");
        return res;
    }

    jsmntok_t tokens[8];
    jsmn_parser parser;
    jsmn_init(&parser);
    int ntok = jsmn_parse(&parser, args_json, args_len, tokens, 8);
    if (ntok < 1) {
        res.for_llm = arena_strdup(arena, "Error: invalid JSON");
        return res;
    }

    const char *path = NULL;
    for (int i = 1; i < ntok - 1; i++) {
        if (jtok_eq(args_json, &tokens[i], "path") &&
            tokens[i + 1].type == JSMN_STRING) {
            path = jtok_arena_str(arena, args_json, &tokens[i + 1]);
            i++;
        }
    }

    if (!path) {
        res.for_llm = arena_strdup(arena, "Error: 'path' is required");
        return res;
    }

    const char *resolved = sandbox_check(arena, s, path);
    if (!resolved) {
        res.for_llm = arena_strdup(arena,
            "Error: access denied (path not in allowed directories)");
        return res;
    }

    size_t buf_size = 8192;
    char *buf = arena_alloc(arena, buf_size);
    if (!buf) { res.for_llm = arena_strdup(arena, "Error: OOM"); return res; }

    size_t pos = 0;
    int count = 0;

#ifdef _WIN32
    adam_dir_t *d = adam_opendir(resolved);
    if (!d) {
        res.for_llm = arena_strdup(arena, "Error: cannot open directory");
        return res;
    }

    const char *name;
    while ((name = adam_readdir(d)) != NULL && pos < buf_size - 256) {
        if (name[0] == '.' && (name[1] == '\0' ||
            (name[1] == '.' && name[2] == '\0')))
            continue;

        const char *type = adam_dir_is_dir(d) ? "dir" : "file";
        pos += (size_t)snprintf(buf + pos, buf_size - pos, "%s\t%s\n", name, type);
        count++;
    }
    adam_closedir(d);
#else
    DIR *d = opendir(resolved);
    if (!d) {
        char errbuf[256];
        snprintf(errbuf, sizeof(errbuf), "Error: cannot open directory: %s",
                 strerror(errno));
        res.for_llm = arena_strdup(arena, errbuf);
        return res;
    }

    struct dirent *ent;
    while ((ent = readdir(d)) != NULL && pos < buf_size - 256) {
        if (ent->d_name[0] == '.' && (ent->d_name[1] == '\0' ||
            (ent->d_name[1] == '.' && ent->d_name[2] == '\0')))
            continue;

        char full[PATH_MAX];
        snprintf(full, sizeof(full), "%s/%s", resolved, ent->d_name);
        struct stat st;
        const char *type = "?";
        long size = 0;
        if (stat(full, &st) == 0) {
            if (S_ISDIR(st.st_mode)) type = "dir";
            else if (S_ISREG(st.st_mode)) { type = "file"; size = (long)st.st_size; }
            else if (S_ISLNK(st.st_mode)) type = "link";
        }

        if (size > 0) {
            pos += (size_t)snprintf(buf + pos, buf_size - pos,
                "%s\t%s\t%ld\n", ent->d_name, type, size);
        } else {
            pos += (size_t)snprintf(buf + pos, buf_size - pos,
                "%s\t%s\n", ent->d_name, type);
        }
        count++;
    }
    closedir(d);
#endif

    if (count == 0) {
        res.for_llm = arena_strdup(arena, "(empty directory)");
    } else {
        buf[pos] = '\0';
        res.for_llm = buf;
    }
    res.success = 1;
    return res;
}

#else // ADAM_NO_FILESYSTEM — stubs

adam_tool_result_t adam_tool_file_read(arena_t *arena, void *ctx,
                                       const char *args_json, size_t args_len) {
    UNUSED_PARAM(ctx); UNUSED_PARAM(args_json); UNUSED_PARAM(args_len);
    return (adam_tool_result_t){
        .for_llm = arena_strdup(arena, "Error: filesystem access not available on this platform"),
        .success = 0 };
}

adam_tool_result_t adam_tool_file_write(arena_t *arena, void *ctx,
                                        const char *args_json, size_t args_len) {
    UNUSED_PARAM(ctx); UNUSED_PARAM(args_json); UNUSED_PARAM(args_len);
    return (adam_tool_result_t){
        .for_llm = arena_strdup(arena, "Error: filesystem access not available on this platform"),
        .success = 0 };
}

adam_tool_result_t adam_tool_list_directory(arena_t *arena, void *ctx,
                                            const char *args_json, size_t args_len) {
    UNUSED_PARAM(ctx); UNUSED_PARAM(args_json); UNUSED_PARAM(args_len);
    return (adam_tool_result_t){
        .for_llm = arena_strdup(arena, "Error: filesystem access not available on this platform"),
        .success = 0 };
}

#endif // ADAM_NO_FILESYSTEM

// ============================================================================
// MARK: - Tool: shell_exec
// ============================================================================

#ifndef ADAM_NO_SHELL

adam_tool_result_t adam_tool_shell_exec(arena_t *arena, void *ctx,
                                        const char *args_json, size_t args_len) {
    adam_tool_result_t res = { .for_llm = NULL, .for_user = NULL, .success = 0 };
    adam_settings_t *s = (adam_settings_t *)ctx;
    if (!s || !args_json) {
        res.for_llm = arena_strdup(arena, "Error: settings not configured");
        return res;
    }

    if (s->allowed_dir_count == 0) {
        res.for_llm = arena_strdup(arena,
            "Error: shell access denied (no allowed directories configured)");
        return res;
    }

    jsmntok_t tokens[16];
    jsmn_parser parser;
    jsmn_init(&parser);
    int ntok = jsmn_parse(&parser, args_json, args_len, tokens, 16);
    if (ntok < 1) {
        res.for_llm = arena_strdup(arena, "Error: invalid JSON");
        return res;
    }

    const char *command = NULL;
    int timeout = 30;

    for (int i = 1; i < ntok - 1; i++) {
        if (jtok_eq(args_json, &tokens[i], "command") &&
            tokens[i + 1].type == JSMN_STRING) {
            command = jtok_arena_str(arena, args_json, &tokens[i + 1]);
            i++;
        } else if (jtok_eq(args_json, &tokens[i], "timeout") &&
                   tokens[i + 1].type == JSMN_PRIMITIVE) {
            timeout = atoi(args_json + tokens[i + 1].start);
            i++;
        }
    }

    if (!command) {
        res.for_llm = arena_strdup(arena, "Error: 'command' is required");
        return res;
    }

    if (timeout <= 0) timeout = 30;
    if (timeout > 300) timeout = 300;

    size_t cmd_size = strlen(command) + strlen(s->allowed_dirs[0]) + 128;
    char *full_cmd = arena_alloc(arena, cmd_size);
    if (!full_cmd) {
        res.for_llm = arena_strdup(arena, "Error: allocation failed");
        return res;
    }

#ifdef _WIN32
    snprintf(full_cmd, cmd_size,
        "cd /d \"%s\" && cmd /c \"%s\" 2>&1",
        s->allowed_dirs[0], command);
#elif defined(__APPLE__) || defined(__ANDROID__) || defined(__ios__)
    // macOS/iOS/Android: no GNU timeout command
    snprintf(full_cmd, cmd_size,
        "cd \"%s\" && /bin/sh -c '%s' 2>&1",
        s->allowed_dirs[0], command);
#else
    // Linux: has GNU timeout
    snprintf(full_cmd, cmd_size,
        "cd \"%s\" && timeout %d /bin/sh -c '%s' 2>&1",
        s->allowed_dirs[0], timeout, command);
#endif

    FILE *p = adam_popen(full_cmd, "r");
    if (!p) {
        res.for_llm = arena_strdup(arena, "Error: failed to execute command");
        return res;
    }

    size_t buf_size = 65536;
    char *buf = arena_alloc(arena, buf_size);
    if (!buf) { adam_pclose(p); res.for_llm = arena_strdup(arena, "Error: OOM"); return res; }

    size_t pos = 0;
    size_t nread;
    while (pos < buf_size - 1 &&
           (nread = fread(buf + pos, 1, buf_size - 1 - pos, p)) > 0) {
        pos += nread;
    }
    buf[pos] = '\0';

    int status = adam_pclose(p);
    int exit_code = ADAM_EXIT_STATUS(status);

    if (pos + 32 < buf_size) {
        pos += (size_t)snprintf(buf + pos, buf_size - pos,
            "\n[exit code: %d]", exit_code);
    }

    res.for_llm = buf;
    res.success = (exit_code == 0) ? 1 : 0;
    return res;
}

#else // ADAM_NO_SHELL — stub

adam_tool_result_t adam_tool_shell_exec(arena_t *arena, void *ctx,
                                        const char *args_json, size_t args_len) {
    UNUSED_PARAM(ctx); UNUSED_PARAM(args_json); UNUSED_PARAM(args_len);
    return (adam_tool_result_t){
        .for_llm = arena_strdup(arena, "Error: shell execution not available on this platform"),
        .success = 0 };
}

#endif // ADAM_NO_SHELL

// ============================================================================
// MARK: - Tool: calculator (pure C, fully portable)
// ============================================================================

typedef struct { const char *s; size_t pos; } calc_ctx_t;

static double calc_expr(calc_ctx_t *c);

static void calc_skip_ws(calc_ctx_t *c) {
    while (c->s[c->pos] == ' ') c->pos++;
}

static double calc_number(calc_ctx_t *c) {
    calc_skip_ws(c);
    double val = 0;
    int neg = 0;
    if (c->s[c->pos] == '-') { neg = 1; c->pos++; }

    if (c->s[c->pos] == '(') {
        c->pos++;
        val = calc_expr(c);
        calc_skip_ws(c);
        if (c->s[c->pos] == ')') c->pos++;
    } else {
        char *end = NULL;
        val = strtod(c->s + c->pos, &end);
        if (end == c->s + c->pos) return 0.0 / 0.0;
        c->pos = (size_t)(end - c->s);
    }

    return neg ? -val : val;
}

static double calc_power(calc_ctx_t *c) {
    double base = calc_number(c);
    calc_skip_ws(c);
    if (c->s[c->pos] == '^') {
        c->pos++;
        double exp = calc_power(c);
        return pow(base, exp);
    }
    return base;
}

static double calc_term(calc_ctx_t *c) {
    double val = calc_power(c);
    calc_skip_ws(c);
    while (c->s[c->pos] == '*' || c->s[c->pos] == '/' || c->s[c->pos] == '%') {
        char op = c->s[c->pos++];
        double right = calc_power(c);
        if (op == '*') val *= right;
        else if (op == '/') val = (right != 0) ? val / right : 0.0 / 0.0;
        else val = (right != 0) ? fmod(val, right) : 0.0 / 0.0;
        calc_skip_ws(c);
    }
    return val;
}

static double calc_expr(calc_ctx_t *c) {
    double val = calc_term(c);
    calc_skip_ws(c);
    while (c->s[c->pos] == '+' || c->s[c->pos] == '-') {
        char op = c->s[c->pos++];
        double right = calc_term(c);
        if (op == '+') val += right;
        else val -= right;
        calc_skip_ws(c);
    }
    return val;
}

adam_tool_result_t adam_tool_calculator(arena_t *arena, void *ctx,
                                        const char *args_json, size_t args_len) {
    adam_tool_result_t res = { .for_llm = NULL, .for_user = NULL, .success = 0 };
    UNUSED_PARAM(ctx);

    if (!args_json) {
        res.for_llm = arena_strdup(arena, "Error: no arguments");
        return res;
    }

    jsmntok_t tokens[8];
    jsmn_parser parser;
    jsmn_init(&parser);
    int ntok = jsmn_parse(&parser, args_json, args_len, tokens, 8);
    if (ntok < 1) {
        res.for_llm = arena_strdup(arena, "Error: invalid JSON");
        return res;
    }

    const char *expression = NULL;
    for (int i = 1; i < ntok - 1; i++) {
        if (jtok_eq(args_json, &tokens[i], "expression") &&
            tokens[i + 1].type == JSMN_STRING) {
            expression = jtok_arena_str(arena, args_json, &tokens[i + 1]);
            i++;
        }
    }

    if (!expression || strlen(expression) == 0) {
        res.for_llm = arena_strdup(arena, "Error: 'expression' is required");
        return res;
    }

    calc_ctx_t cc = { .s = expression, .pos = 0 };
    double result = calc_expr(&cc);

    char buf[64];
    if (isnan(result)) {
        snprintf(buf, sizeof(buf), "Error: invalid expression");
        res.for_llm = arena_strdup(arena, buf);
    } else if (isinf(result)) {
        res.for_llm = arena_strdup(arena, "Infinity");
        res.success = 1;
    } else if (result == (double)(long long)result && fabs(result) < 1e15) {
        snprintf(buf, sizeof(buf), "%lld", (long long)result);
        res.for_llm = arena_strdup(arena, buf);
        res.success = 1;
    } else {
        snprintf(buf, sizeof(buf), "%.10g", result);
        res.for_llm = arena_strdup(arena, buf);
        res.success = 1;
    }

    return res;
}

// ============================================================================
// MARK: - Tool: web_search (Brave Search API, needs HTTP)
// ============================================================================

adam_tool_result_t adam_tool_web_search(arena_t *arena, void *ctx,
                                        const char *args_json, size_t args_len) {
    adam_tool_result_t res = { .for_llm = NULL, .for_user = NULL, .success = 0 };
    adam_settings_t *s = (adam_settings_t *)ctx;

    if (!args_json) {
        res.for_llm = arena_strdup(arena, "Error: no arguments");
        return res;
    }

    jsmntok_t tokens[16];
    jsmn_parser parser;
    jsmn_init(&parser);
    int ntok = jsmn_parse(&parser, args_json, args_len, tokens, 16);
    if (ntok < 1) {
        res.for_llm = arena_strdup(arena, "Error: invalid JSON");
        return res;
    }

    const char *query = NULL;
    int count = 5;

    for (int i = 1; i < ntok - 1; i++) {
        if (jtok_eq(args_json, &tokens[i], "query") &&
            tokens[i + 1].type == JSMN_STRING) {
            query = jtok_arena_str(arena, args_json, &tokens[i + 1]);
            i++;
        } else if (jtok_eq(args_json, &tokens[i], "count") &&
                   tokens[i + 1].type == JSMN_PRIMITIVE) {
            count = atoi(args_json + tokens[i + 1].start);
            i++;
        }
    }

    if (!query) {
        res.for_llm = arena_strdup(arena, "Error: 'query' is required");
        return res;
    }

    const char *api_key = getenv("BRAVE_API_KEY");
    if (!api_key && s) api_key = s->api_key;
    if (!api_key) {
        res.for_llm = arena_strdup(arena,
            "Error: BRAVE_API_KEY not set (set env var or configure API key)");
        return res;
    }

    // URL-encode
    size_t qlen = strlen(query);
    char *encoded = arena_alloc(arena, qlen * 3 + 1);
    if (!encoded) { res.for_llm = arena_strdup(arena, "Error: OOM"); return res; }
    size_t epos = 0;
    for (size_t i = 0; i < qlen; i++) {
        char c = query[i];
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.') {
            encoded[epos++] = c;
        } else if (c == ' ') {
            encoded[epos++] = '+';
        } else {
            epos += (size_t)snprintf(encoded + epos, 4, "%%%02X",
                                      (unsigned char)c);
        }
    }
    encoded[epos] = '\0';

    UNUSED_PARAM(count);
    UNUSED_PARAM(api_key);

    // TODO: implement with adam_net GET support
    res.for_llm = arena_strdup(arena,
        "Error: web_search requires HTTP GET support (not yet implemented)");
    return res;
}

// ============================================================================
// MARK: - Tool: http_post (needs HTTP)
// ============================================================================

adam_tool_result_t adam_tool_http_post(arena_t *arena, void *ctx,
                                       const char *args_json, size_t args_len) {
    adam_tool_result_t res = { .for_llm = NULL, .for_user = NULL, .success = 0 };

    if (!args_json) {
        res.for_llm = arena_strdup(arena, "Error: no arguments");
        return res;
    }

    jsmntok_t tokens[32];
    jsmn_parser parser;
    jsmn_init(&parser);
    int ntok = jsmn_parse(&parser, args_json, args_len, tokens, 32);
    if (ntok < 1) {
        res.for_llm = arena_strdup(arena, "Error: invalid JSON");
        return res;
    }

    const char *url = NULL;
    const char *body = NULL;

    for (int i = 1; i < ntok - 1; i++) {
        if (jtok_eq(args_json, &tokens[i], "url") &&
            tokens[i + 1].type == JSMN_STRING) {
            url = jtok_arena_str(arena, args_json, &tokens[i + 1]);
            i++;
        } else if (jtok_eq(args_json, &tokens[i], "body") &&
                   tokens[i + 1].type == JSMN_STRING) {
            body = jtok_arena_str(arena, args_json, &tokens[i + 1]);
            i++;
        }
    }

    if (!url) {
        res.for_llm = arena_strdup(arena, "Error: 'url' is required");
        return res;
    }

    adam_settings_t *s = (adam_settings_t *)ctx;
    if (!s) {
        res.for_llm = arena_strdup(arena, "Error: settings not configured");
        return res;
    }

#if !defined(ADAM_NO_CURL) || defined(__APPLE__)
    adam_net_response_t resp = adam_net_post_json(
        s, arena, url, NULL, body ? body : "{}", NULL, 0);

    if (resp.error != ADAM_OK || !resp.data) {
        char buf[128];
        snprintf(buf, sizeof(buf), "Error: HTTP request failed (status %ld)",
                 resp.http_code);
        res.for_llm = arena_strdup(arena, buf);
        return res;
    }

    size_t result_size = resp.data_len + 32;
    char *result_buf = arena_alloc(arena, result_size);
    if (result_buf) {
        snprintf(result_buf, result_size, "[HTTP %ld]\n%.*s",
                 resp.http_code, (int)resp.data_len, (char *)resp.data);
        res.for_llm = result_buf;
    } else {
        res.for_llm = arena_strdup(arena, (char *)resp.data);
    }

    res.success = (resp.http_code >= 200 && resp.http_code < 300) ? 1 : 0;
#else
    UNUSED_PARAM(body);
    res.for_llm = arena_strdup(arena,
        "Error: HTTP not available (build with curl or on Apple platform)");
#endif

    return res;
}

// ============================================================================
// MARK: - Tool: sql_query
// ============================================================================

#ifndef ADAM_NO_SQLITE

adam_tool_result_t adam_tool_sql_query(arena_t *arena, void *ctx,
                                       const char *args_json, size_t args_len) {
    adam_tool_result_t res = { .for_llm = NULL, .for_user = NULL, .success = 0 };
    adam_memory_t *mem = (adam_memory_t *)ctx;
    if (!mem || !args_json) {
        res.for_llm = arena_strdup(arena, "Error: database not configured");
        return res;
    }

    sqlite3 *db = adam_memory_db(mem);
    if (!db) {
        res.for_llm = arena_strdup(arena, "Error: no database connection");
        return res;
    }

    jsmntok_t tokens[32];
    jsmn_parser parser;
    jsmn_init(&parser);
    int ntok = jsmn_parse(&parser, args_json, args_len, tokens, 32);
    if (ntok < 1) {
        res.for_llm = arena_strdup(arena, "Error: invalid JSON");
        return res;
    }

    char *sql = NULL;
    for (int i = 1; i < ntok - 1; i++) {
        if (jtok_eq(args_json, &tokens[i], "sql") &&
            tokens[i + 1].type == JSMN_STRING) {
            sql = jtok_malloc_str(args_json, &tokens[i + 1]);
            i++;
        }
    }

    if (!sql) {
        res.for_llm = arena_strdup(arena, "Error: 'sql' is required");
        return res;
    }

    // Block destructive statements
    char upper[16] = {0};
    for (int i = 0; i < 15 && sql[i]; i++)
        upper[i] = (sql[i] >= 'a' && sql[i] <= 'z')
                    ? (char)(sql[i] - 32) : sql[i];
    if (strncmp(upper, "DROP", 4) == 0 || strncmp(upper, "TRUNCATE", 8) == 0) {
        free(sql);
        res.for_llm = arena_strdup(arena, "Error: destructive SQL not allowed");
        return res;
    }

    sqlite3_stmt *vm = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &vm, NULL);
    free(sql);

    if (rc != SQLITE_OK) {
        char buf[512];
        snprintf(buf, sizeof(buf), "Error: %s", sqlite3_errmsg(db));
        res.for_llm = arena_strdup(arena, buf);
        return res;
    }

    size_t buf_size = 8192;
    char *buf = arena_alloc(arena, buf_size);
    if (!buf) {
        sqlite3_finalize(vm);
        res.for_llm = arena_strdup(arena, "Error: allocation failed");
        return res;
    }

    size_t pos = 0;
    int ncol = sqlite3_column_count(vm);
    int rows = 0;

    for (int c = 0; c < ncol && pos < buf_size - 64; c++) {
        const char *name = sqlite3_column_name(vm, c);
        pos += (size_t)snprintf(buf + pos, buf_size - pos,
            "%s%s", c > 0 ? "\t" : "", name ? name : "?");
    }
    if (ncol > 0) buf[pos++] = '\n';

    while (sqlite3_step(vm) == SQLITE_ROW && rows < 100) {
        for (int c = 0; c < ncol && pos < buf_size - 128; c++) {
            const char *val = (const char *)sqlite3_column_text(vm, c);
            pos += (size_t)snprintf(buf + pos, buf_size - pos,
                "%s%s", c > 0 ? "\t" : "", val ? val : "NULL");
        }
        if (pos < buf_size - 1) buf[pos++] = '\n';
        rows++;
    }
    sqlite3_finalize(vm);

    if (rows == 0 && ncol == 0) {
        int changes = sqlite3_changes(db);
        snprintf(buf, buf_size, "OK (%d rows affected)", changes);
        pos = strlen(buf);
    }

    buf[pos] = '\0';
    res.for_llm = buf;
    res.success = 1;
    return res;
}

#endif // ADAM_NO_SQLITE
