
use std::{fs, io::{self, BufRead, Write}, path::PathBuf};
use serde::{Deserialize, Serialize};

const CONFIG_DIR: &str = ".synch_tool";
const CONFIG_FILE: &str = "config.json";

#[derive(Serialize, Deserialize, Debug)]
pub struct Config{
    pub local_repo: String,
    pub submodules: Vec<String>,
    pub servers: Vec<String>,
    pub default_server: Option<String>,
    pub remote_work_dir: String,
    pub excludes: Vec<String>,
    pub max_file_size_mb: u64,
}

impl Config{

    // For init: always use current directory
    fn init_config_path() -> PathBuf {
        std::env::current_dir()
            .expect("Failed to get current directory")
            .join(CONFIG_DIR)
            .join(CONFIG_FILE)
    }

    // For load/save: walk up from current directory to find config
    fn find_config_path() -> Option<PathBuf> {
        let mut dir = std::env::current_dir().expect("Failed to get current directory");
        loop {
            let candidate = dir.join(CONFIG_DIR).join(CONFIG_FILE);
            if candidate.exists() {
                return Some(candidate);
            }
            if !dir.pop() {
                return None;
            }
        }
    }

    pub fn init() -> Self {
        let config_path = Self::init_config_path();
        if config_path.exists() {
            eprintln!("Already initialized. Config already exists at {:?}", config_path);
            std::process::exit(1);
        }
        if let Some(parent_config) = Self::find_config_path() {
            eprintln!("Warning: a parent directory is already initialized at {:?}", parent_config);
            eprintln!("Run `synch-tool init --force` or initialize from the root directory.");
            std::process::exit(1);
        }
        fs::create_dir_all(config_path.parent().unwrap()).expect("Failed to create config directory");
        let config = Config {
            local_repo: std::env::current_dir()
                .expect("Failed to get current directory")
                .to_string_lossy()
                .into_owned(),
            submodules: Vec::new(),
            servers: Self::parse_ssh_config(),
            default_server: None,
            remote_work_dir: Self::prompt_remote_work_dir(),
            max_file_size_mb: 5,
            excludes: vec![
                ".venv/".to_string(),
                "venv/".to_string(),
                "env/".to_string(),
                ".env/".to_string(),
                "__pycache__/".to_string(),
                "*.pyc".to_string(),
                ".conda/".to_string(),
            ],
        };
        let json = serde_json::to_string_pretty(&config).expect("Failed to serialize config");
        fs::write(&config_path, json).expect("Failed to write config file");
        println!("Initialized config at {:?}", config_path);
        config
    }

    fn prompt_remote_work_dir() -> String {
        print!("Remote working directory (leave blank to skip): ");
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin().lock().read_line(&mut input).unwrap();
        input.trim().to_string()
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
        let config_path = Self::find_config_path()
            .expect("Config not found. Run `synch-tool init` first.");
        let json = serde_json::to_string_pretty(self).expect("Failed to serialize config");
        fs::write(&config_path, json).expect("Failed to write config file");
    }

    pub fn load() -> Self {
        let config_path = match Self::find_config_path() {
            Some(p) => p,
            None => {
                eprintln!("Not initialized. Run `synch-tool init` first.");
                std::process::exit(1);
            }
        };
        let contents = fs::read_to_string(&config_path).expect("Failed to read config file");
        serde_json::from_str(&contents).expect("Failed to parse config file")
    }
}