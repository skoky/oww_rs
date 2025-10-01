use serde_derive::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq)]
pub enum SpeechUnlockType {
    OpenWakeWordAlexa,
}
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct UnlockConfig {
    pub unlock_type: SpeechUnlockType,
    pub(crate) yelling_threshold: i16,
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
            yelling_threshold: 5000,                         // yelling 5x more than normal speech
            detection_threshold: 0.5,                        // detect hugo with more than 50% prob
            quite_threshold: 10,                            // min level of RMS to run unlock detection
            endpoint_duration_secs: 1.0,                     // expects 1 secs quite time after unlock sentence
            save_wavs: false,                                 // true if storing and uploading wavs to cloud
            #[cfg(not(debug_assertions))]
            unlock_time_secs: 30, // number of seconds for AI to be unlocked in listening state with GCP
            #[cfg(debug_assertions)]
            unlock_time_secs: 15, // number of seconds for AI to be unlocked in listening state with GCP (shorter for dev)
        }
    }
}

// pub fn get_unlock_time_secs(config_file: PathBuf) -> u8 {
//     load_config(config_file).unlock_time_secs
// }

// pub fn load_config(config_file_name: PathBuf) -> UnlockConfig {
//     confy::load_path(&config_file_name).unwrap_or_else(|e| {
//         warn!("Can't load {:?}; error {}. Creating defaults", config_file_name, e);
//         let new_config = UnlockConfig::default();
//         if let Err(e) = confy::store_path(config_file_name.clone(), &new_config) {
//             panic!(
//                 "Can't store {:?}; error '{}', current dir: {:?}. Install Hugo in read-write location",
//                 &config_file_name,
//                 e,
//                 env::current_dir()
//             );
//         }
//         new_config
//     })
// }

// pub fn save_config(config_file_name: PathBuf, unlock_config: &UnlockConfig) -> Result<(), String> {
//     let x = confy::store_path(config_file_name, unlock_config);
//     match x {
//         Ok(_) => Ok(()),
//         Err(e) => Err(format!("{}", e)),
//     }
// }
