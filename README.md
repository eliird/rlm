# st — synch-tool

A CLI tool for syncing local git repos with submodules to remote GPU servers over SSH/rsync.

## Motivation

Built for a workflow where Claude Code runs locally but compute lives on a remote server behind a VPN. The idea:

- Edit everything locally, push/pull individual submodules to the server as needed
- Keep a persistent tmux session on the server to retain conda/venv environments between commands
- Guard against accidentally overwriting uncommitted work in either direction

If your flow is purely push-to-server → pull-from-git, `git remote` would be simpler. This tool is for when you want direct server↔local sync alongside your normal git workflow.

## Installation

```sh
cargo install --path .
```

## Usage

```sh
st <command> [args]
```

Run `st status` to see the current config at any time.

## Commands

| Command | Description |
|---|---|
| `init` | Initialize config in current directory, `git init` if needed |
| `status` | Print current config |
| `servers update` | Populate server list from `~/.ssh/config` |
| `servers add <name>` | Add a server by name |
| `servers default <name>` | Set the default server |
| `servers remove <name>` | Remove a server |
| `remote-dir set <path>` | Set the remote working directory |
| `remote-dir clear` | Clear the remote working directory |
| `submodules list` | List git submodules |
| `submodules update` | Run `git submodule update --init --recursive` |
| `submodules add <url> <path>` | Add a submodule |
| `submodules remove <path>` | Remove a submodule |
| `submodules pull` | Pull latest in all submodules |
| `push [submodule] [--force]` | Rsync submodule(s) to server — aborts if remote has uncommitted changes or is ahead |
| `pull [submodule] [--force]` | Rsync submodule(s) from server — aborts if local has uncommitted changes or is ahead |
| `exec <command>` | Run a one-shot command on the server in the remote working directory |
| `run <command>` | Run a command in the persistent `st` tmux session, streaming output live |
| `ssh` | Attach to the `st` tmux session on the server (creates it if absent) |
| `init-claude [--global]` | Install a Claude Code skill for `st` |
