use std::path::PathBuf;
use std::process::Command;

pub struct Sync;

impl Sync {
    // Check if remote path has uncommitted changes
    fn check_remote_clean(server: &str, remote_path: &str) -> bool {
        let output = Command::new("ssh")
            .args([server, &format!("git -C {} status --porcelain -u", remote_path)])
            .output()
            .expect("Failed to run ssh");
        if !output.status.success() {
            // Remote path may not exist yet — treat as clean so we can push fresh
            return true;
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        if !stdout.trim().is_empty() {
            eprintln!("Remote has uncommitted or untracked files:");
            eprintln!("{}", stdout.trim());
            return false;
        }
        true
    }

    // Check if local path has uncommitted changes
    fn check_local_clean(local_path: &PathBuf) -> bool {
        let output = Command::new("git")
            .args(["-C", local_path.to_str().unwrap(), "status", "--porcelain", "-u"])
            .output()
            .expect("Failed to run git status");
        let stdout = String::from_utf8_lossy(&output.stdout);
        if !stdout.trim().is_empty() {
            eprintln!("Local has uncommitted or untracked files:");
            eprintln!("{}", stdout.trim());
            return false;
        }
        true
    }

    // Get HEAD commit hash, returns None if repo has no commits or path doesn't exist
    fn get_remote_head(server: &str, remote_path: &str) -> Option<String> {
        let output = Command::new("ssh")
            .args([server, &format!("git -C {} rev-parse HEAD 2>/dev/null", remote_path)])
            .output()
            .expect("Failed to run ssh");
        if output.status.success() {
            let hash = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if hash.is_empty() { None } else { Some(hash) }
        } else {
            None
        }
    }

    fn get_local_head(local_path: &PathBuf) -> Option<String> {
        let output = Command::new("git")
            .args(["-C", local_path.to_str().unwrap(), "rev-parse", "HEAD"])
            .output()
            .expect("Failed to run git rev-parse");
        if output.status.success() {
            let hash = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if hash.is_empty() { None } else { Some(hash) }
        } else {
            None
        }
    }

    // Returns true if ancestor is an ancestor of descendent
    fn is_ancestor(local_path: &PathBuf, ancestor: &str, descendent: &str) -> bool {
        Command::new("git")
            .args(["-C", local_path.to_str().unwrap(), "merge-base", "--is-ancestor", ancestor, descendent])
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    fn get_commit_message(local_path: &PathBuf, hash: &str) -> String {
        let output = Command::new("git")
            .args(["-C", local_path.to_str().unwrap(), "log", "--oneline", "-1", hash])
            .output()
            .expect("Failed to run git log");
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }

    // Check commit relationship before push. Returns true if safe to proceed.
    fn check_commits_for_push(local_path: &PathBuf, server: &str, remote_path: &str, force: bool) -> bool {
        let local_head = match Self::get_local_head(local_path) {
            Some(h) => h,
            None => {
                eprintln!("Local repo has no commits.");
                return false;
            }
        };
        let remote_head = match Self::get_remote_head(server, remote_path) {
            Some(h) => h,
            None => return true, // remote has no commits yet, safe to push
        };

        if local_head == remote_head {
            return true; // in sync
        }

        if Self::is_ancestor(local_path, &remote_head, &local_head) {
            // remote is behind local — safe to push
            return true;
        }

        if Self::is_ancestor(local_path, &local_head, &remote_head) {
            // remote is ahead of local
            eprintln!("Remote is ahead of local. Run `synch-tool pull` first.");
            eprintln!("  Remote HEAD: {}", Self::get_commit_message(local_path, &remote_head));
            return false;
        }

        // diverged
        eprintln!("WARNING: branches have diverged.");
        eprintln!("  Local HEAD:  {}", Self::get_commit_message(local_path, &local_head));
        eprintln!("  Remote HEAD: {}", Self::get_commit_message(local_path, &remote_head));
        if force {
            eprintln!("Proceeding with --force.");
            true
        } else {
            eprintln!("Use --force to overwrite anyway.");
            false
        }
    }

    // Check commit relationship before pull. Returns true if safe to proceed.
    fn check_commits_for_pull(local_path: &PathBuf, server: &str, remote_path: &str, force: bool) -> bool {
        let local_head = match Self::get_local_head(local_path) {
            Some(h) => h,
            None => return true, // local has no commits yet, safe to pull
        };
        let remote_head = match Self::get_remote_head(server, remote_path) {
            Some(h) => h,
            None => return true, // remote has no git repo or no commits yet, safe to pull
        };

        if local_head == remote_head {
            return true;
        }

        if Self::is_ancestor(local_path, &local_head, &remote_head) {
            // local is behind remote — safe to pull
            return true;
        }

        if Self::is_ancestor(local_path, &remote_head, &local_head) {
            // local is ahead of remote
            eprintln!("Local is ahead of remote. You would overwrite newer local commits.");
            eprintln!("  Local HEAD: {}", Self::get_commit_message(local_path, &local_head));
            return false;
        }

        // diverged
        eprintln!("WARNING: branches have diverged.");
        eprintln!("  Local HEAD:  {}", Self::get_commit_message(local_path, &local_head));
        eprintln!("  Remote HEAD: {}", Self::get_commit_message(local_path, &remote_head));
        if force {
            eprintln!("Proceeding with --force.");
            true
        } else {
            eprintln!("Use --force to overwrite anyway.");
            false
        }
    }

    fn run_rsync(local_path: &PathBuf, server: &str, remote_path: &str, push: bool, excludes: &[String], max_file_size_mb: u64) {
        let local = format!("{}/", local_path.to_str().unwrap());
        let remote = format!("{}:{}/", server, remote_path);
        let (src, dst) = if push { (&local, &remote) } else { (&remote, &local) };
        let mut args = vec![
            "-av".to_string(),
            "--filter=:- .gitignore".to_string(),
            "--delete".to_string(),
            format!("--max-size={}m", max_file_size_mb),
        ];
        for ex in excludes {
            args.push(format!("--exclude={}", ex));
        }
        args.push(src.to_string());
        args.push(dst.to_string());
        let status = Command::new("rsync")
            .args(&args)
            .status()
            .expect("Failed to run rsync");
        if !status.success() {
            eprintln!("rsync failed.");
            std::process::exit(1);
        }
    }

    pub fn push(local_path: &PathBuf, server: &str, remote_path: &str, force: bool, excludes: &[String], max_file_size_mb: u64) {
        if !force && !Self::check_local_clean(local_path) {
            std::process::exit(1);
        }
        if !force && !Self::check_remote_clean(server, remote_path) {
            std::process::exit(1);
        }
        if !Self::check_commits_for_push(local_path, server, remote_path, force) {
            std::process::exit(1);
        }
        Self::run_rsync(local_path, server, remote_path, true, excludes, max_file_size_mb);
    }

    pub fn pull(local_path: &PathBuf, server: &str, remote_path: &str, force: bool, excludes: &[String], max_file_size_mb: u64) {
        if !force && !Self::check_local_clean(local_path) {
            std::process::exit(1);
        }
        if !Self::check_commits_for_pull(local_path, server, remote_path, force) {
            std::process::exit(1);
        }
        Self::run_rsync(local_path, server, remote_path, false, excludes, max_file_size_mb);
    }

    pub fn exec(server: &str, remote_path: &str, command: &str) {
        let full_cmd = format!("cd {} && {}", remote_path, command);
        // Inherit stdio for live streaming output
        let status = Command::new("ssh")
            .args([server, &full_cmd])
            .status()
            .expect("Failed to run ssh");
        if !status.success() {
            std::process::exit(status.code().unwrap_or(1));
        }
    }

    pub fn open_shell(server: &str) {
        let status = Command::new("ssh")
            .arg(server)
            .status()
            .expect("Failed to run ssh");
        if !status.success() {
            std::process::exit(status.code().unwrap_or(1));
        }
    }
}
