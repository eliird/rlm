mod config;
use config::Config;
use clap::{Parser, ValueEnum, Subcommand};

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
            let cfg = Config::load();
            match command {
                ServerCommands::Add { name } => {
                    println!("{}", name);
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
    }
}