use std::path::PathBuf;
use std::process::Command;

pub struct Git;

impl Git {
    pub fn is_repo(path: &PathBuf) -> bool {
        Command::new("git")
            .args(["-C", path.to_str().unwrap(), "rev-parse", "--git-dir"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    pub fn init_if_needed(path: &PathBuf) {
        if Self::is_repo(path) {
            println!("Already a git repo.");
        } else {
            let status = Command::new("git")
                .args(["-C", path.to_str().unwrap(), "init"])
                .status()
                .expect("Failed to run git init");
            if status.success() {
                println!("Git repo initialized.");
            } else {
                eprintln!("git init failed.");
                std::process::exit(1);
            }
        }
    }

    pub fn get_submodules(path: &PathBuf) -> Vec<String> {
        let output = Command::new("git")
            .args(["-C", path.to_str().unwrap(), "submodule", "status"])
            .output()
            .expect("Failed to run git submodule status");
        if !output.status.success() {
            return Vec::new();
        }
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter_map(|line| {
                // format: " <hash> <path> (<branch>)"
                let parts: Vec<&str> = line.trim().splitn(3, ' ').collect();
                if parts.len() >= 2 { Some(parts[1].to_string()) } else { None }
            })
            .collect()
    }

    pub fn submodule_update(path: &PathBuf) {
        let status = Command::new("git")
            .args(["-C", path.to_str().unwrap(), "submodule", "update", "--init", "--recursive"])
            .status()
            .expect("Failed to run git submodule update");
        if !status.success() {
            eprintln!("git submodule update failed.");
            std::process::exit(1);
        }
    }

    pub fn submodule_add(path: &PathBuf, url: &str, subpath: &str) {
        let status = Command::new("git")
            .args(["-C", path.to_str().unwrap(), "submodule", "add", url, subpath])
            .status()
            .expect("Failed to run git submodule add");
        if !status.success() {
            eprintln!("git submodule add failed.");
            std::process::exit(1);
        }
    }

    pub fn submodule_remove(path: &PathBuf, subpath: &str) {
        let s1 = Command::new("git")
            .args(["-C", path.to_str().unwrap(), "submodule", "deinit", "-f", subpath])
            .status()
            .expect("Failed to run git submodule deinit");
        let s2 = Command::new("git")
            .args(["-C", path.to_str().unwrap(), "rm", "-f", subpath])
            .status()
            .expect("Failed to run git rm");
        if !s1.success() || !s2.success() {
            eprintln!("submodule remove failed.");
            std::process::exit(1);
        }
    }

    pub fn submodule_pull(path: &PathBuf) {
        let status = Command::new("git")
            .args(["-C", path.to_str().unwrap(), "submodule", "foreach", "git pull"])
            .status()
            .expect("Failed to run git submodule foreach git pull");
        if !status.success() {
            eprintln!("submodule pull failed.");
            std::process::exit(1);
        }
    }
}
