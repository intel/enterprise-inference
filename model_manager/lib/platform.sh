#!/usr/bin/env bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Platform detection — which Kubernetes platform is the cluster?
#
# The platforms and how to recognise each are data, not code: see
# platforms.yaml (MM_PLATFORMS_FILE). Adding a platform is a new entry there.
#
# Resolved once per run (platform_resolve, called from main) into PLATFORM:
# the name of the first platform whose probe matches, else "kubernetes".
#
# Precedence:
#   1. MM_PLATFORM env — a platform name, kubernetes, or auto (empty = auto).
#      The Ansible install sets it from the inference_platform role, so the
#      CLI does not probe the cluster a second time.
#   2. Auto-detect from the cluster.
#
# A probe error (unreachable, Unauthorized, Forbidden) before a match aborts:
# guessing kubernetes on, say, OpenShift would take the paths that break it.

MM_PLATFORMS_FILE="${MM_PLATFORMS_FILE:-$SCRIPT_DIR/platforms.yaml}"
PLATFORM=""

_platform_field() { yq -r ".platforms[$1].$2 // \"\"" "$MM_PLATFORMS_FILE"; }

_platform_probe_failed() {
    err "Could not detect the cluster platform (probe for '$1' failed): $2"
    abort "Fix cluster access (kubectl get nodes), or set MM_PLATFORM to a platform name."
}

platform_resolve() {
    [[ -n "$PLATFORM" ]] && return 0
    [[ -f "$MM_PLATFORMS_FILE" ]] || abort "Platform registry not found: $MM_PLATFORMS_FILE"

    local names
    names=$(yq -r '.platforms[].name' "$MM_PLATFORMS_FILE")

    local requested="${MM_PLATFORM:-auto}"
    requested="${requested,,}"
    if [[ "$requested" != "auto" ]]; then
        if [[ "$requested" == "kubernetes" ]] || grep -qxF -- "$requested" <<< "$names"; then
            PLATFORM="$requested"
            info "Platform: $(platform_label) (MM_PLATFORM)"
            return 0
        fi
        abort "Invalid MM_PLATFORM='${MM_PLATFORM}'. Valid: auto, kubernetes, $(paste -sd, <<< "$names")."
    fi

    local count i name api version_re probe_err server_version="" matched=""
    count=$(yq -r '.platforms | length' "$MM_PLATFORMS_FILE")
    for (( i = 0; i < count; i++ )); do
        name=$(_platform_field "$i" name)
        api=$(_platform_field "$i" probe.api)
        version_re=$(_platform_field "$i" probe.version)

        if [[ -n "$api" ]]; then
            # Keep stderr only: it carries the NotFound marker; the body is unused.
            if probe_err=$(kubectl get --raw "$api" --request-timeout=15s 2>&1 >/dev/null); then
                matched="$name"
            elif [[ "$probe_err" != *"(NotFound)"* ]]; then
                _platform_probe_failed "$name" "$probe_err"
            fi
        elif [[ -n "$version_re" ]]; then
            if [[ -z "$server_version" ]]; then
                # stdout only: kubectl warns (e.g. version skew) on stderr, which
                # would corrupt the JSON. Re-run for the message on failure.
                local version_json
                if ! version_json=$(kubectl version -o json --request-timeout=15s 2>/dev/null); then
                    probe_err=$(kubectl version --request-timeout=15s 2>&1 >/dev/null || true)
                    _platform_probe_failed "$name" "$probe_err"
                fi
                server_version=$(yq -p json -r '.serverVersion.gitVersion // ""' <<< "$version_json")
            fi
            [[ "$server_version" =~ $version_re ]] && matched="$name"
        else
            abort "Platform '${name}' in $MM_PLATFORMS_FILE has no probe (api or version)."
        fi
        [[ -n "$matched" ]] && break
    done

    PLATFORM="${matched:-kubernetes}"
    info "Platform: $(platform_label) (detected)"
}

# platform_is <name> — true when the resolved platform is <name>.
platform_is() { [[ "$PLATFORM" == "$1" ]]; }

# Display name from the registry; plain Kubernetes is not listed there.
platform_label() {
    [[ "$PLATFORM" == "kubernetes" ]] && { echo "Kubernetes"; return; }
    local label
    label=$(yq -r ".platforms[] | select(.name == \"$PLATFORM\") | .label // \"\"" "$MM_PLATFORMS_FILE")
    echo "${label:-$PLATFORM}"
}
