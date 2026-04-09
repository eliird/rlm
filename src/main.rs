mod config;
mod git;
mod sync;
mod claude;
use config::Config;
use git::Git;
use sync::Sync;
use claude::Claude;
use clap::{Parser, Subcommand};
use std::io::BufRead;

fn require_remote_dir(cfg: &config::Config) {
    if cfg.remote_work_dir.is_empty() {
        eprintln!("Remote working directory is not set. Run `st remote-dir set <path>` first.");
        std::process::exit(1);
    }
}

#[derive(Parser)]
struct Cli{
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands{
    Init,
    Status,
    Servers {
        #[command(subcommand)]
        command: ServerCommands,
    },
    RemoteDir {
        #[command(subcommand)]
        command: RemoteDirCommands,
    },
    Submodules {
        #[command(subcommand)]
        command: SubmoduleCommands,
    },
    Push {
        /// Submodule name to push (pushes all if omitted)
        submodule: Option<String>,
        #[arg(long)]
        force: bool,
    },
    Pull {
        /// Submodule name to pull (pulls all if omitted)
        submodule: Option<String>,
        #[arg(long)]
        force: bool,
    },
    Exec {
        /// Command to run on the remote server in remote_work_dir
        command: Vec<String>,
    },
    Run {
        /// Command to run in the persistent remote tmux session
        command: Vec<String>,
    },
    Ssh,
    /// Install a Claude Code skill for st
    InitClaude {
        #[arg(long)]
        global: bool,
    },
}

#[derive(Subcommand)]
enum SubmoduleCommands {
    List,
    Update,
    Add { url: String, path: String },
    Remove { path: String },
    Pull,
}

#[derive(Subcommand)]
enum RemoteDirCommands{
    Set { path: String },
    Clear,
}

#[derive(Subcommand)]
enum ServerCommands{
    Add { name: String },
    Update,
    Default { name: String },
    Remove { name: String },
}


fn main(){
    let cli = Cli::parse();

    match cli.command{
        Commands::Init => {
            Config::init();
            Git::init_if_needed(&std::env::current_dir().unwrap());
        }
        Commands::Status => {
            let cfg = Config::load();
            println!("{:?}", cfg);
        }
        Commands::RemoteDir { command } => {
            let mut cfg = Config::load();
            match command {
                RemoteDirCommands::Set { path } => {
                    cfg.remote_work_dir = path.clone();
                    cfg.save();
                    println!("Remote work dir set to '{}'.", path);
                }
                RemoteDirCommands::Clear => {
                    cfg.remote_work_dir = String::new();
                    cfg.save();
                    println!("Remote work dir cleared.");
                }
            }
        }
        Commands::Servers { command } => {
            match command {
                ServerCommands::Add { name } => {
                    let mut cfg = Config::load();
                    if cfg.servers.contains(&name) {
                        eprintln!("Server '{}' already in list.", name);
                        std::process::exit(1);
                    }
                    cfg.servers.push(name.clone());
                    cfg.save();
                    println!("Added server '{}'.", name);
                }
                ServerCommands::Default { name } => {
                    let mut cfg = Config::load();
                    if !cfg.servers.contains(&name) {
                        eprintln!("Server '{}' not found in list. Run `servers update` first.", name);
                        std::process::exit(1);
                    }
                    cfg.default_server = Some(name.clone());
                    cfg.save();
                    println!("Default server set to '{}'.", name);
                }
                ServerCommands::Remove { name } => {
                    let mut cfg = Config::load();
                    if !cfg.servers.contains(&name) {
                        eprintln!("Server '{}' not found in list.", name);
                        std::process::exit(1);
                    }
                    cfg.servers.retain(|s| s != &name);
                    if cfg.default_server.as_deref() == Some(&name) {
                        cfg.default_server = None;
                        println!("Default server cleared.");
                    }
                    cfg.save();
                    println!("Removed server '{}'.", name);
                }
                ServerCommands::Update => {
                    let mut cfg = Config::load();
                    let servers = Config::parse_ssh_config();
                    println!("Found {} servers: {:?}", servers.len(), servers);
                    cfg.servers = servers;
                    cfg.save();
                    println!("Servers saved to config.");
                }
            }
        }
        Commands::Push { submodule, force } => {
            let cfg = Config::load();
            require_remote_dir(&cfg);
            let server = match &cfg.default_server {
                Some(s) => s.clone(),
                None => {
                    eprintln!("No default server set. Run `servers default <name>` first.");
                    std::process::exit(1);
                }
            };
            let repo_path = std::path::PathBuf::from(&cfg.local_repo);
            let submodules = match submodule {
                Some(name) => vec![name],
                None => {
                    let local_subs = Git::get_submodules(&repo_path);
                    // Check for remote dirs that no longer exist locally
                    let remote_dirs = Sync::list_remote_dirs(&server, &cfg.remote_work_dir);
                    for remote_dir in &remote_dirs {
                        if !local_subs.contains(remote_dir) {
                            eprint!("'{}' exists on server but not locally. Remove from server? [y/N] ", remote_dir);
                            let mut input = String::new();
                            std::io::stdin().read_line(&mut input).unwrap();
                            if input.trim().eq_ignore_ascii_case("y") {
                                let remote_path = format!("{}/{}", cfg.remote_work_dir, remote_dir);
                                Sync::remove_remote_dir(&server, &remote_path);
                                println!("Removed '{}' from server.", remote_dir);
                            }
                        }
                    }
                    local_subs
                }
            };
            if submodules.is_empty() {
                eprintln!("No submodules found.");
                std::process::exit(1);
            }
            for sub in &submodules {
                let local_path = repo_path.join(sub);
                let remote_path = format!("{}/{}", cfg.remote_work_dir, sub);
                println!("Pushing '{}'...", sub);
                Sync::push(&local_path, &server, &remote_path, force, &cfg.excludes, cfg.max_file_size_mb);
            }
        }
        Commands::Pull { submodule, force } => {
            let cfg = Config::load();
            require_remote_dir(&cfg);
            let server = match &cfg.default_server {
                Some(s) => s.clone(),
                None => {
                    eprintln!("No default server set. Run `servers default <name>` first.");
                    std::process::exit(1);
                }
            };
            let repo_path = std::path::PathBuf::from(&cfg.local_repo);
            let submodules = match submodule {
                Some(name) => vec![name],
                None => Git::get_submodules(&repo_path),
            };
            if submodules.is_empty() {
                eprintln!("No submodules found.");
                std::process::exit(1);
            }
            for sub in &submodules {
                let local_path = repo_path.join(sub);
                let remote_path = format!("{}/{}", cfg.remote_work_dir, sub);
                println!("Pulling '{}'...", sub);
                Sync::pull(&local_path, &server, &remote_path, force, &cfg.excludes, cfg.max_file_size_mb);
            }
        }
        Commands::Run { command } => {
            let cfg = Config::load();
            let server = match &cfg.default_server {
                Some(s) => s.clone(),
                None => {
                    eprintln!("No default server set. Run `servers default <name>` first.");
                    std::process::exit(1);
                }
            };
            Sync::run(&server, &command.join(" "));
        }
        Commands::Exec { command } => {
            let cfg = Config::load();
            require_remote_dir(&cfg);
            let server = match &cfg.default_server {
                Some(s) => s.clone(),
                None => {
                    eprintln!("No default server set. Run `servers default <name>` first.");
                    std::process::exit(1);
                }
            };
            Sync::exec(&server, &cfg.remote_work_dir, &command.join(" "));
        }
        Commands::InitClaude { global } => {
            Claude::init_skill(global);
        }
        Commands::Ssh => {
            let cfg = Config::load();
            let server = match &cfg.default_server {
                Some(s) => s.clone(),
                None => {
                    eprintln!("No default server set. Run `servers default <name>` first.");
                    std::process::exit(1);
                }
            };
            Sync::open_shell(&server);
        }
        Commands::Submodules { command } => {
            let cfg = Config::load();
            let repo_path = std::path::PathBuf::from(&cfg.local_repo);
            match command {
                SubmoduleCommands::List => {
                    let subs = Git::get_submodules(&repo_path);
                    if subs.is_empty() {
                        println!("No submodules found.");
                    } else {
                        for s in subs {
                            println!("{}", s);
                        }
                    }
                }
                SubmoduleCommands::Update => {
                    Git::submodule_update(&repo_path);
                    println!("Submodules updated.");
                }
                SubmoduleCommands::Add { url, path } => {
                    Git::submodule_add(&repo_path, &url, &path);
                    println!("Submodule '{}' added.", path);
                }
                SubmoduleCommands::Remove { path } => {
                    Git::submodule_remove(&repo_path, &path);
                    println!("Submodule '{}' removed.", path);
                }
                SubmoduleCommands::Pull => {
                    Git::submodule_pull(&repo_path);
                    println!("Submodules pulled.");
                }
            }
        }
    }
}