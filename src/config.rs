
use std::{fs, path::PathBuf};
use dirs::home_dir;
use serde::{Deserialize, Serialize};

const CONFIG_DIR: &str = ".synch_tool";
const CONFIG_FILE: &str = "config.json";

const DEFAULT_REMOTE_WORK_DIR: &str = "/data/work/irdali.durrani";

#[derive(Serialize, Deserialize, Debug)]
pub struct Config{
    pub local_repo: String,
    pub submodules: Vec<String>,
    pub servers: Vec<String>,
    pub default_server: Option<String>,
    pub remote_work_dir: String,
}

impl Config{

    fn get_config_path() -> PathBuf{
        std::env::current_dir()
            .expect("Failed to get current directory")
            .join(CONFIG_DIR)
            .join(CONFIG_FILE)
    }

    pub fn init() -> Self {
        let config_path = Self::get_config_path();
        if config_path.exists() {
            eprintln!("Already initialized. Config already exists at {:?}", config_path);
            std::process::exit(1);
        }
        fs::create_dir_all(config_path.parent().unwrap()).expect("Failed to create config directory");
        let config = Config {
            local_repo: std::env::current_dir()
                .expect("Failed to get current directory")
                .to_string_lossy()
                .into_owned(),
            submodules: Vec::new(),
            servers: Vec::new(),
            default_server: None,
            remote_work_dir: DEFAULT_REMOTE_WORK_DIR.to_string(),
        };
        let json = serde_json::to_string_pretty(&config).expect("Failed to serialize config");
        fs::write(&config_path, json).expect("Failed to write config file");
        println!("Initialized config at {:?}", config_path);
        config
    }

    pub fn parse_ssh_config() -> Vec<String> {
        let ssh_config_path = dirs::home_dir()
            .expect("Failed to get home directory")
            .join(".ssh")
            .join("config");

        if !ssh_config_path.exists() {
            eprintln!("No SSH config found at {:?}", ssh_config_path);
            return Vec::new();
        }

        let contents = fs::read_to_string(&ssh_config_path).expect("Failed to read SSH config");
        contents
            .lines()
            .filter_map(|line| {
                let trimmed = line.trim();
                if trimmed.to_lowercase().starts_with("host ") {
                    let name = trimmed[5..].trim().to_string();
                    if name != "*" { Some(name) } else { None }
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn save(&self) {
        let config_path = Self::get_config_path();
        let json = serde_json::to_string_pretty(self).expect("Failed to serialize config");
        fs::write(&config_path, json).expect("Failed to write config file");
    }

    pub fn load() -> Self {
        let config_path = Self::get_config_path();
        if !config_path.exists() {
            eprintln!("Not initialized. Run `synch-tool init` first.");
            std::process::exit(1);
        }
        let contents = fs::read_to_string(&config_path).expect("Failed to read config file");
        serde_json::from_str(&contents).expect("Failed to parse config file")
    }
}