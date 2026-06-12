#[derive(Debug, Clone, Eq, PartialEq)]
pub enum SpeechUnlockType {
    OpenWakeWordAlexa,
    OpenWakeWordHeyMycroft,
    OpenWakeWordHeyJarvis,
    OpenWakeWordAhojHugo,
}
#[derive(Debug, Clone)]
pub struct UnlockConfig {
    pub unlock_type: SpeechUnlockType,
    pub detection_threshold: f32,
    pub quite_threshold: i16,
    pub endpoint_duration_secs: f32,
    pub save_wavs: bool,
    pub unlock_time_secs: u8,
}

impl Default for UnlockConfig {
    fn default() -> Self {
        UnlockConfig {
            unlock_type: SpeechUnlockType::OpenWakeWordAlexa, // use OpenWakeVoice
            detection_threshold: 0.3,                        // detect hugo with more than 50% prob
            quite_threshold: 1,                            // min level of RMS to run unlock detection
            endpoint_duration_secs: 0.2,                     // expects 1 secs quite time after unlock sentence
            save_wavs: false,                                 // true if storing and uploading wavs to cloud
            #[cfg(not(debug_assertions))]
            unlock_time_secs: 30, // number of seconds for AI to be unlocked in listening state with GCP
            #[cfg(debug_assertions)]
            unlock_time_secs: 15, // number of seconds for AI to be unlocked in listening state with GCP (shorter for dev)
        }
    }
}
