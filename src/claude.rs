use std::fs;
use std::path::PathBuf;

pub struct Claude;

impl Claude {
    pub fn init_skill(global: bool) {
        let commands_dir = if global {
            dirs::home_dir()
                .expect("Failed to get home directory")
                .join(".claude")
                .join("commands")
        } else {
            std::env::current_dir()
                .expect("Failed to get current directory")
                .join(".claude")
                .join("commands")
        };

        fs::create_dir_all(&commands_dir).expect("Failed to create commands directory");

        let skill_path = commands_dir.join("st.md");
        let scope = if global { "user-scoped (global)" } else { "project-scoped (local)" };

        if skill_path.exists() {
            eprintln!("Skill already exists at {:?}", skill_path);
            std::process::exit(1);
        }

        let content = Self::skill_content();
        fs::write(&skill_path, content).expect("Failed to write skill file");
        println!("Claude skill installed ({}) at {:?}", scope, skill_path);
        println!("Use /st in Claude Code to invoke it.");
    }

    fn skill_content() -> String {
        r#"# st — synch-tool skill

You have access to `st`, a CLI tool for syncing local git repos and submodules to remote servers over SSH/rsync.

## What st does

- Manages a config at `.synch_tool/config.json` in the project root (walk-up search from current dir)
- Tracks servers (from `~/.ssh/config`), a default server, a remote working directory, and submodules
- Syncs submodules to/from the default server via rsync, respecting `.gitignore`
- Checks for uncommitted changes and commit divergence before syncing — aborts or warns if unsafe

## When the user asks you to use st

Use the Bash tool to run `st` commands directly. Always run `st status` first if you need to know the current config state.

## Available commands

| Command | What it does |
|---|---|
| `st init` | Init config + git init if needed |
| `st status` | Print current config |
| `st servers update` | Refresh server list from `~/.ssh/config` |
| `st servers add <name>` | Add a server |
| `st servers default <name>` | Set the default server |
| `st servers remove <name>` | Remove a server |
| `st remote-dir set <path>` | Set the remote working directory |
| `st remote-dir clear` | Clear the remote working directory |
| `st submodules list` | List git submodules |
| `st submodules update` | Run `git submodule update --init --recursive` |
| `st submodules add <url> <path>` | Add a submodule |
| `st submodules remove <path>` | Remove a submodule |
| `st submodules pull` | Pull latest in all submodules |
| `st push [submodule] [--force]` | Rsync submodule(s) to default server |
| `st pull [submodule] [--force]` | Rsync submodule(s) from default server |
| `st exec <command>` | Run a one-shot command on default server in remote_work_dir (streamed) |
| `st run <command>` | Run command in persistent `st` tmux session on default server, streams output back live |
| `st ssh` | Attach to the `st` tmux session on default server (creates it if absent) |
| `st init-claude [--global]` | Install this Claude skill |

## Safeguards

- `push` aborts if remote has uncommitted changes or remote is ahead of local
- `pull` aborts if local has uncommitted changes or local is ahead of remote
- Both warn and require `--force` if commits have diverged
- All sync commands require a default server to be set

## How to handle user requests

- "push my changes" → run `st push`
- "pull from server" → run `st pull`
- "run X on the server" → use `st run X` for env-persistent execution (conda/venv retained), or `st exec X` for a quick one-shot command
- "what's my config" → run `st status`
- "add a submodule" → run `st submodules add <url> <path>`
- If a command fails due to dirty state or diverged commits, explain what st reported and suggest next steps
"#.to_string()
    }
}
