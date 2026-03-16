//
//  adam_net_apple.m
//  Adam — HTTP layer via NSURLSession (macOS, iOS)
//
//  Replaces libcurl on Apple platforms. Uses the system TLS stack
//  (Security.framework), HTTP/2, and connection pooling natively.
//  No external dependencies required.
//
//  Created by Marco Bambini on 16/03/26.
//

#ifdef __APPLE__

#include "adam_net.h"
#include <string.h>
#include <stdlib.h>

#import <Foundation/Foundation.h>

// ============================================================================
// MARK: - Shared session (connection pooling)
// ============================================================================

static NSURLSession *g_session = nil;

static NSURLSession *get_session(void) {
    if (!g_session) {
        NSURLSessionConfiguration *config =
            [NSURLSessionConfiguration defaultSessionConfiguration];
        config.timeoutIntervalForRequest = 120.0;
        config.timeoutIntervalForResource = 120.0;
        config.HTTPMaximumConnectionsPerHost = 4;
        g_session = [NSURLSession sessionWithConfiguration:config];
    }
    return g_session;
}

// ============================================================================
// MARK: - Cleanup
// ============================================================================

void adam_net_cleanup(adam_settings_t *s) {
    UNUSED_PARAM(s);
    // persistent curl handles are not used on Apple
    // session is global and reused
}

// ============================================================================
// MARK: - Helper: synchronous data task
// ============================================================================

typedef struct {
    uint8_t *data;
    size_t   len;
    long     http_code;
    int      error;
} sync_result_t;

static sync_result_t sync_request(NSURLRequest *request, arena_t *arena) {
    __block sync_result_t result = {0};
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);

    NSURLSessionDataTask *task =
        [get_session() dataTaskWithRequest:request
         completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        if (error) {
            result.error = 1;
        } else {
            NSHTTPURLResponse *http = (NSHTTPURLResponse *)response;
            result.http_code = (long)http.statusCode;
            if (data && data.length > 0) {
                if (arena) {
                    result.data = arena_alloc(arena, data.length + 1);
                    if (result.data) {
                        memcpy(result.data, data.bytes, data.length);
                        result.data[data.length] = '\0';
                        result.len = data.length;
                    }
                } else {
                    result.data = malloc(data.length + 1);
                    if (result.data) {
                        memcpy(result.data, data.bytes, data.length);
                        result.data[data.length] = '\0';
                        result.len = data.length;
                    }
                }
            }
        }
        dispatch_semaphore_signal(sem);
    }];

    [task resume];
    dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
    return result;
}

// ============================================================================
// MARK: - Helper: build NSMutableURLRequest with headers
// ============================================================================

static NSMutableURLRequest *make_request(const char *url,
                                          const char *auth_header,
                                          const char **extra_headers) {
    NSURL *nsurl = [NSURL URLWithString:[NSString stringWithUTF8String:url]];
    NSMutableURLRequest *req = [NSMutableURLRequest requestWithURL:nsurl];
    req.HTTPMethod = @"POST";

    if (auth_header) {
        // Parse "Key: Value" format
        const char *colon = strchr(auth_header, ':');
        if (colon) {
            NSString *key = [[NSString alloc]
                initWithBytes:auth_header
                       length:(NSUInteger)(colon - auth_header)
                     encoding:NSUTF8StringEncoding];
            const char *val = colon + 1;
            while (*val == ' ') val++;
            NSString *value = [NSString stringWithUTF8String:val];
            [req setValue:value forHTTPHeaderField:key];
        }
    }

    if (extra_headers) {
        for (int i = 0; extra_headers[i]; i++) {
            const char *colon = strchr(extra_headers[i], ':');
            if (colon) {
                NSString *key = [[NSString alloc]
                    initWithBytes:extra_headers[i]
                           length:(NSUInteger)(colon - extra_headers[i])
                         encoding:NSUTF8StringEncoding];
                const char *val = colon + 1;
                while (*val == ' ') val++;
                [req setValue:[NSString stringWithUTF8String:val]
                    forHTTPHeaderField:key];
            }
        }
    }

    return req;
}

// ============================================================================
// MARK: - POST JSON
// ============================================================================

adam_net_response_t adam_net_post_json(
    adam_settings_t *s, arena_t *arena,
    const char *url, const char *auth_header,
    const char *body, const char **extra_headers, int handle_id
) {
    UNUSED_PARAM(s); UNUSED_PARAM(handle_id);

    adam_net_response_t resp = {0};

    @autoreleasepool {
        NSMutableURLRequest *req = make_request(url, auth_header, extra_headers);
        [req setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
        req.HTTPBody = [NSData dataWithBytes:body length:strlen(body)];

        sync_result_t r = sync_request(req, arena);
        resp.data = r.data;
        resp.data_len = r.len;
        resp.http_code = r.http_code;
        if (r.error) resp.error = ADAM_ERR_CURL; // reuse error code
    }

    return resp;
}

// ============================================================================
// MARK: - POST Multipart
// ============================================================================

adam_net_response_t adam_net_post_multipart(
    adam_settings_t *s, arena_t *arena,
    const char *url, const char *auth_header,
    const adam_net_field_t *fields, size_t field_count, int handle_id
) {
    UNUSED_PARAM(s); UNUSED_PARAM(handle_id);

    adam_net_response_t resp = {0};

    @autoreleasepool {
        NSString *boundary = [NSString stringWithFormat:@"adam-boundary-%u", arc4random()];
        NSMutableData *body = [NSMutableData data];

        for (size_t i = 0; i < field_count; i++) {
            [body appendData:[[NSString stringWithFormat:@"--%@\r\n", boundary]
                dataUsingEncoding:NSUTF8StringEncoding]];

            if (fields[i].value) {
                // Text field
                [body appendData:[[NSString stringWithFormat:
                    @"Content-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n",
                    fields[i].name, fields[i].value]
                    dataUsingEncoding:NSUTF8StringEncoding]];
            } else {
                // Binary field
                NSString *disposition = [NSString stringWithFormat:
                    @"Content-Disposition: form-data; name=\"%s\"; filename=\"%s\"\r\n",
                    fields[i].name,
                    fields[i].filename ? fields[i].filename : "data"];
                [body appendData:[disposition dataUsingEncoding:NSUTF8StringEncoding]];

                if (fields[i].content_type) {
                    [body appendData:[[NSString stringWithFormat:
                        @"Content-Type: %s\r\n", fields[i].content_type]
                        dataUsingEncoding:NSUTF8StringEncoding]];
                }
                [body appendData:[@"\r\n" dataUsingEncoding:NSUTF8StringEncoding]];
                [body appendBytes:fields[i].data length:fields[i].data_len];
                [body appendData:[@"\r\n" dataUsingEncoding:NSUTF8StringEncoding]];
            }
        }
        [body appendData:[[NSString stringWithFormat:@"--%@--\r\n", boundary]
            dataUsingEncoding:NSUTF8StringEncoding]];

        NSMutableURLRequest *req = make_request(url, auth_header, NULL);
        NSString *ct = [NSString stringWithFormat:
            @"multipart/form-data; boundary=%@", boundary];
        [req setValue:ct forHTTPHeaderField:@"Content-Type"];
        req.HTTPBody = body;

        sync_result_t r = sync_request(req, arena);
        resp.data = r.data;
        resp.data_len = r.len;
        resp.http_code = r.http_code;
        if (r.error) resp.error = ADAM_ERR_CURL;
    }

    return resp;
}

// ============================================================================
// MARK: - POST Streaming (delegate-based for chunk-by-chunk delivery)
// ============================================================================

@interface AdamStreamDelegate : NSObject <NSURLSessionDataDelegate>
@property (nonatomic, assign) adam_net_stream_fn callback;
@property (nonatomic, assign) void *callbackCtx;
@property (nonatomic, assign) long httpCode;
@property (nonatomic, assign) int error;
@property (nonatomic, strong) dispatch_semaphore_t semaphore;
@end

@implementation AdamStreamDelegate

- (void)URLSession:(NSURLSession *)session
          dataTask:(NSURLSessionDataTask *)dataTask
didReceiveResponse:(NSURLResponse *)response
 completionHandler:(void (^)(NSURLSessionResponseDisposition))completionHandler {
    UNUSED_PARAM(session); UNUSED_PARAM(dataTask);
    NSHTTPURLResponse *http = (NSHTTPURLResponse *)response;
    self.httpCode = (long)http.statusCode;
    completionHandler(NSURLSessionResponseAllow);
}

- (void)URLSession:(NSURLSession *)session
          dataTask:(NSURLSessionDataTask *)dataTask
    didReceiveData:(NSData *)data {
    UNUSED_PARAM(session); UNUSED_PARAM(dataTask);
    if (self.callback && data.length > 0) {
        int ret = self.callback(self.callbackCtx,
                                (const uint8_t *)data.bytes, data.length);
        if (ret != 0) {
            [dataTask cancel];
        }
    }
}

- (void)URLSession:(NSURLSession *)session
              task:(NSURLSessionTask *)task
didCompleteWithError:(NSError *)error {
    UNUSED_PARAM(session); UNUSED_PARAM(task);
    if (error) self.error = 1;
    dispatch_semaphore_signal(self.semaphore);
}

@end

adam_net_response_t adam_net_post_streaming(
    adam_settings_t *s, const char *url, const char *auth_header,
    const char *body, const char **extra_headers,
    adam_net_stream_fn on_chunk, void *stream_ctx,
    int handle_id
) {
    UNUSED_PARAM(s); UNUSED_PARAM(handle_id);

    adam_net_response_t resp = {0};

    @autoreleasepool {
        AdamStreamDelegate *delegate = [[AdamStreamDelegate alloc] init];
        delegate.callback = on_chunk;
        delegate.callbackCtx = stream_ctx;
        delegate.semaphore = dispatch_semaphore_create(0);

        NSURLSessionConfiguration *config =
            [NSURLSessionConfiguration defaultSessionConfiguration];
        config.timeoutIntervalForRequest = 30.0;
        NSURLSession *session =
            [NSURLSession sessionWithConfiguration:config
                                          delegate:delegate
                                     delegateQueue:nil];

        NSMutableURLRequest *req = make_request(url, auth_header, extra_headers);
        [req setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
        req.HTTPBody = [NSData dataWithBytes:body length:strlen(body)];

        NSURLSessionDataTask *task = [session dataTaskWithRequest:req];
        [task resume];

        dispatch_semaphore_wait(delegate.semaphore, DISPATCH_TIME_FOREVER);
        [session invalidateAndCancel];

        resp.http_code = delegate.httpCode;
        if (delegate.error) resp.error = ADAM_ERR_CURL;
    }

    return resp;
}

#endif // __APPLE__
