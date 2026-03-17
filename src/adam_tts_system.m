//
//  adam_tts_system.m
//  Adam — System TTS via AVSpeechSynthesizer (macOS / iOS)
//
//  Created by Marco Bambini on 16/03/26.
//

#ifdef __APPLE__

#include "adam.h"
#import <AVFoundation/AVFoundation.h>

// ============================================================================
// MARK: - Speech delegate (detects completion)
// ============================================================================

@interface AdamSpeechDelegate : NSObject <AVSpeechSynthesizerDelegate>
@property (nonatomic, assign) volatile int done;
@end

@implementation AdamSpeechDelegate

- (void)speechSynthesizer:(AVSpeechSynthesizer *)synthesizer
 didFinishSpeechUtterance:(AVSpeechUtterance *)utterance {
    UNUSED_PARAM(synthesizer); UNUSED_PARAM(utterance);
    self.done = 1;
}

- (void)speechSynthesizer:(AVSpeechSynthesizer *)synthesizer
didCancelSpeechUtterance:(AVSpeechUtterance *)utterance {
    UNUSED_PARAM(synthesizer); UNUSED_PARAM(utterance);
    self.done = 1;
}

@end

// ============================================================================
// MARK: - Public API
// ============================================================================

adam_status_t adam_tts_system_speak(const char *text, const char *language) {
    if (!text || !text[0]) return ADAM_OK;

    @autoreleasepool {
        NSString *nsText = [NSString stringWithUTF8String:text];
        if (!nsText || nsText.length == 0) return ADAM_OK;

        AVSpeechUtterance *utterance =
            [AVSpeechUtterance speechUtteranceWithString:nsText];
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate;
        utterance.pitchMultiplier = 1.0f;
        utterance.volume = 1.0f;

        if (language) {
            utterance.voice = [AVSpeechSynthesisVoice
                voiceWithLanguage:[NSString stringWithUTF8String:language]];
        }

        AdamSpeechDelegate *delegate = [[AdamSpeechDelegate alloc] init];

        AVSpeechSynthesizer *synth = [[AVSpeechSynthesizer alloc] init];
        synth.delegate = delegate;
        [synth speakUtterance:utterance];

        // Poll for completion — allows SIGINT to interrupt
        while (!delegate.done) {
            [[NSRunLoop currentRunLoop] runUntilDate:
                [NSDate dateWithTimeIntervalSinceNow:0.1]];
        }
    }

    return ADAM_OK;
}

#endif // __APPLE__
