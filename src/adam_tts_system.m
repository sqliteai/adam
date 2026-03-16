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
@property (nonatomic, strong) dispatch_semaphore_t semaphore;
@end

@implementation AdamSpeechDelegate

- (void)speechSynthesizer:(AVSpeechSynthesizer *)synthesizer
 didFinishSpeechUtterance:(AVSpeechUtterance *)utterance {
    UNUSED_PARAM(synthesizer); UNUSED_PARAM(utterance);
    dispatch_semaphore_signal(self.semaphore);
}

- (void)speechSynthesizer:(AVSpeechSynthesizer *)synthesizer
didCancelSpeechUtterance:(AVSpeechUtterance *)utterance {
    UNUSED_PARAM(synthesizer); UNUSED_PARAM(utterance);
    dispatch_semaphore_signal(self.semaphore);
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

        // Set language if specified, otherwise let the system auto-detect
        if (language) {
            utterance.voice = [AVSpeechSynthesisVoice
                voiceWithLanguage:[NSString stringWithUTF8String:language]];
        }

        dispatch_semaphore_t sem = dispatch_semaphore_create(0);

        AdamSpeechDelegate *delegate = [[AdamSpeechDelegate alloc] init];
        delegate.semaphore = sem;

        AVSpeechSynthesizer *synth = [[AVSpeechSynthesizer alloc] init];
        synth.delegate = delegate;
        [synth speakUtterance:utterance];

        // Wait for speech to complete (timeout: 60s)
        dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW,
                                                    60 * NSEC_PER_SEC));
    }

    return ADAM_OK;
}

#endif // __APPLE__
