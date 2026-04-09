# synch-tool

A CLI tool for syncing local repos to remote servers.

## Commands

| Command | Description |
|---|---|
| `init` | Initialize config in current directory and run `git init` if not already a git repo |
| `status` | Print current config |
| `servers update` | Populate server list from `~/.ssh/config` |
| `servers add <name>` | Add a server by name |
| `servers default <name>` | Set the default server |
| `servers remove <name>` | Remove a server from the list |
| `remote-dir set <path>` | Set the remote working directory |
| `remote-dir clear` | Clear the remote working directory |
| `submodules list` | List all git submodules in the repo |
| `submodules update` | Run `git submodule update --init --recursive` |
| `submodules add <url> <path>` | Add a new submodule |
| `submodules remove <path>` | Remove a submodule |
| `submodules pull` | Pull latest changes in all submodules |
| `push [submodule] [--force]` | Rsync submodule(s) to default server, with dirty and commit checks |
| `pull [submodule] [--force]` | Rsync submodule(s) from default server, with dirty and commit checks |
| `exec <command>` | Run a command on the default server in the remote working directory |
| `ssh` | Open a shell on the default server |
| `init-claude` | Install a Claude Code skill for st in the current project |
| `init-claude --global` | Install a Claude Code skill for st globally for all projects |
