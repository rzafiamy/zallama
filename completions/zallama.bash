# bash completion for zallama
#
# The zallama CLI itself is the single source of truth: this script just asks it
# for candidates via the hidden `__complete` subcommand. New commands/models are
# picked up automatically with no edits here.
#
# Install: source this file from ~/.bashrc, or drop it in the bash-completion
# dir (e.g. /etc/bash_completion.d/ or /usr/share/bash-completion/completions/).

_zallama_complete() {
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    local candidates
    candidates="$(zallama __complete "$cur" "$prev" "$COMP_CWORD" 2>/dev/null)"

    # No candidates from the CLI -> fall back to default filename completion
    # (useful for `zallama add <name> <path-to.gguf>`).
    if [[ -z "$candidates" ]]; then
        COMPREPLY=( $(compgen -f -- "$cur") )
        return 0
    fi

    COMPREPLY=( $(compgen -W "$candidates" -- "$cur") )
    return 0
}

complete -F _zallama_complete zallama
