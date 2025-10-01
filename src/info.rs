use crate::config::SpeechUnlockType::{OpenWakeWordAlexa};
use crate::config::{UnlockConfig};
use log::{debug};

pub const OWW_CZ_NAME_AHOJ_HUGO: &str = "Český - Ahoj Hugo";
pub const OWW_CZ_NAME_ALEXA: &str = "Český - Alexa";

#[derive(Debug)]
pub struct LanguageModel {
    pub name: String,
    pub selected: bool,
}

impl LanguageModel {
    pub fn new(name: &str, selected: bool) -> Self {
        LanguageModel { name: name.to_string(), selected }
    }
}

pub fn get_trigger_phases(unlock_config: &UnlockConfig) -> Vec<String> {
    match unlock_config.unlock_type {
        OpenWakeWordAlexa => vec!["Alexa".to_string()],
    }
}

pub fn set_unlock_model(language_model: &LanguageModel) -> Option<UnlockConfig> {
    let unlock_config = UnlockConfig::default(); // load_config(unlock_config_file.clone());

    let model_type = match language_model.name.as_str() {
        OWW_CZ_NAME_ALEXA => OpenWakeWordAlexa,
        _ => {
            panic!("Unexpected language model {:?}", language_model);
        }
    };
    let mut new_unlock_config = unlock_config.clone();
    new_unlock_config.unlock_type = model_type;

    debug!("New unlock model config {:?}", new_unlock_config);
    // if let Err(e) = save_config(unlock_config_file, &new_unlock_config) {
    //     warn!("Failed to save unlock config {}", e);
    // }
    Some(new_unlock_config)
}
