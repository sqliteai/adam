//
//  adam_mic_macos.m
//  Adam — Microphone recording on macOS via AVFoundation
//
//  Minimal Objective-C helper. Records from the default input device
//  to a WAV file. Press Enter to stop recording.
//
//  Created by Marco Bambini on 15/03/26.
//

#ifdef __APPLE__

#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>

// Record from the default microphone to a WAV file.
// Blocks until recording_seconds elapse or the stop flag is set.
// Returns 0 on success.
int adam_mic_record_wav(const char *output_path, int max_seconds,
                        volatile int *stop_flag) {
    @autoreleasepool {
        NSString *path = [NSString stringWithUTF8String:output_path];
        NSURL *url = [NSURL fileURLWithPath:path];

        NSDictionary *settings = @{
            AVFormatIDKey: @(kAudioFormatLinearPCM),
            AVSampleRateKey: @16000.0,
            AVNumberOfChannelsKey: @1,
            AVLinearPCMBitDepthKey: @16,
            AVLinearPCMIsFloatKey: @NO,
            AVLinearPCMIsBigEndianKey: @NO,
        };

        NSError *error = nil;
        AVAudioRecorder *recorder =
            [[AVAudioRecorder alloc] initWithURL:url
                                        settings:settings
                                           error:&error];
        if (error || !recorder) {
            NSLog(@"adam_mic: failed to create recorder: %@", error);
            return -1;
        }

        [recorder record];

        // Wait for stop signal or timeout
        NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:max_seconds];
        while (!stop_flag || !(*stop_flag)) {
            [[NSRunLoop currentRunLoop] runUntilDate:
                [NSDate dateWithTimeIntervalSinceNow:0.1]];
            if ([[NSDate date] compare:deadline] != NSOrderedAscending)
                break;
        }

        [recorder stop];
        return 0;
    }
}

#endif // __APPLE__
