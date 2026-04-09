# st — synch-tool

A CLI tool for syncing local git repos with submodules to remote GPU servers over SSH/rsync.

## Motivation

Built for a workflow where Claude Code runs locally but compute lives on a remote server behind a VPN. The idea:

- Edit everything locally, push/pull individual submodules to the server as needed
- Keep a persistent tmux session on the server to retain conda/venv environments between commands
- Guard against accidentally overwriting uncommitted work in either direction
- Skills so claude can execute code remotely on the tmux session.

If your flow is purely push-to-server → pull-from-git, `git remote` would be simpler. This tool is for when you want direct server↔local sync alongside your normal git workflow. Also convenient to not have to setup claude related skills and plugins on both local and remote machines. Another thing is it keeps the results of what commands claude executed on server in the tmux kernel which is useful for reviewing things later.

## Installation

```sh
cargo install --path .
```

## Usage

```sh
st <command> [args]
```

Run `st status` to see the current config at any time.

## Example Workflow

Say you want a single folder that holds all your projects, kept in sync with a GPU server.

```sh
# Create a workspace and initialize st
mkdir ~/work && cd ~/work
st init
# → prompts for remote working directory (e.g. /home/you/work on the server)
# → auto-populates servers from ~/.ssh/config

# Set your default server
st servers default SBZ-H100-01

# Add your repos as submodules
st submodules add git@github.com:you/project-a.git project-a
st submodules add git@github.com:you/project-b.git project-b

# Push everything to the server
st push

# Make changes locally, push a specific repo
cd project-a
# ... edit files ...
st push project-a

# Activate your environment once in the persistent tmux session
st run conda activate myenv

# Run code remotely and stream output back
st run python project-a/train.py

# Drop into the tmux session to check on things interactively
st ssh

# Run a one-shot command (no tmux, just ssh)
st exec nvidia-smi

# Pull changes back from the server
st pull project-a
```

With Claude Code, install the skill so Claude can run `st` commands for you:

```sh
st init-claude --global
# Then in Claude Code: /st check GPUs on the server
```

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
