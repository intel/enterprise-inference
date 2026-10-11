#!/usr/bin/env bash
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Configuration loading and model resolution

# Configuration loading and model/runtime resolution.
# Converts models.yaml → JSON once, then uses jq for all lookups (fast).
# resolve_model sets MODEL_* (name/id/kind/namespace/cpu/memory/replicas/
# category/image/node/routing + args/env/servers JSON, plus chat_template and its
# MODEL_CHAT_TEMPLATE_IS_FILE classification), then _select_server_version
# resolves MODEL_RUNTIME + MODEL_SERVER_VERSION.

_MM_CONFIG_JSON=""
declare -A _RUNTIME_CACHE=()

load_config() {
    local config_file="${1:-$MM_CONFIG}"
    [[ -f "$config_file" ]] || abort "Config not found: $config_file"

    # Convert once — all subsequent lookups use jq on cached JSON
    _MM_CONFIG_JSON=$(yq e -o=json '.' "$config_file")

    DEFAULT_NAMESPACE=$(echo "$_MM_CONFIG_JSON" | jq -r '.defaults.namespace // "llm-inference"')
    DEFAULT_RUNTIME=$(echo "$_MM_CONFIG_JSON" | jq -r '.defaults.runtime // "vllm"')
    DEFAULT_REPLICAS=$(echo "$_MM_CONFIG_JSON" | jq -r '.defaults.replicas // 1')
    DEFAULT_REQUIRE_AMX=$(echo "$_MM_CONFIG_JSON" | jq -r '.defaults.require_amx // false')
    STORAGE_PVC_NAME=$(echo "$_MM_CONFIG_JSON" | jq -r '.storage.pvc_name // "model-store"')

    local cfg_http_proxy cfg_https_proxy cfg_no_proxy
    cfg_http_proxy=$(echo "$_MM_CONFIG_JSON" | jq -r '.network.http_proxy // ""')
    cfg_https_proxy=$(echo "$_MM_CONFIG_JSON" | jq -r '.network.https_proxy // ""')
    cfg_no_proxy=$(echo "$_MM_CONFIG_JSON" | jq -r '.network.no_proxy // ""')

    MM_HTTP_PROXY="${cfg_http_proxy:-${HTTP_PROXY:-${http_proxy:-}}}"
    MM_HTTPS_PROXY="${cfg_https_proxy:-${HTTPS_PROXY:-${https_proxy:-}}}"
    MM_NO_PROXY="${cfg_no_proxy:-${NO_PROXY:-${no_proxy:-}}}"
    MM_CONNECTIVITY_URL=$(echo "$_MM_CONFIG_JSON" | jq -r '.network.connectivity_check // "https://huggingface.co"')
}

check_network() {
    [[ -n "$MM_CONNECTIVITY_URL" ]] || return 0
    local proxy_args=()
    [[ -n "$MM_HTTPS_PROXY" ]] && proxy_args=(--proxy "$MM_HTTPS_PROXY")
    if ! curl -sf --max-time 10 "${proxy_args[@]}" -o /dev/null "$MM_CONNECTIVITY_URL" 2>/dev/null; then
        abort "Network check failed: cannot reach $MM_CONNECTIVITY_URL (proxy: ${MM_HTTPS_PROXY:-none}). Check network/proxy settings in models.yaml or environment."
    fi
}

# Assert a JSON blob is the expected container type (object|array), aborting
# with a user-friendly message that names the offending field. Guards the
# env/args merge against malformed models.yaml (e.g. env written as a list).
_validate_json_type() {
    local what="$1" json="$2" want="$3"
    local got
    got=$(jq -rn --argjson v "$json" '$v | type' 2>/dev/null) || got="invalid"
    if [[ "$got" != "$want" ]]; then
        local hint="a mapping of KEY: VALUE"
        [[ "$want" == "array" ]] && hint="a list of strings"
        abort "Invalid $what: expected $want ($hint) in $MM_CONFIG, got $got."
    fi
}

resolve_model() {
    local name="$1"

    # Scalar fields. Server + version selection is handled separately by
    # _select_server_version (below), so runtime/server_version/servers are NOT
    # read here.
    local fields
    fields=$(echo "$_MM_CONFIG_JSON" | jq -r --arg n "$name" --arg dns "$DEFAULT_NAMESPACE" --arg drep "$DEFAULT_REPLICAS" '
        .models[] | select(.name == $n) |
        [
            .name // "",
            (.model_id // ""),
            (.kind // ""),
            (.namespace // $dns),
            (.cpu // "" | tostring),
            (.memory // "" | tostring),
            (.replicas // ($drep | tonumber) | tostring),
            (.category // ""),
            (.image // ""),
            (.node // ""),
            (.routing // "")
        ] | join("")') || true  # \x01 (SOH) field delimiter, safe in model names

    [[ -z "$fields" ]] && abort "Model '$name' not found in $MM_CONFIG"

    IFS=$'\x01' read -r MODEL_NAME MODEL_ID MODEL_KIND MODEL_NAMESPACE \
        MODEL_CPU MODEL_MEMORY MODEL_REPLICAS MODEL_CATEGORY MODEL_IMAGE MODEL_NODE MODEL_ROUTING \
        <<< "$fields"

    # Speculative-decoding draft model, fetched next to the weights (see the download job).
    MODEL_DRAFT_ID=$(echo "$_MM_CONFIG_JSON" | jq -r --arg n "$name" \
        '.models[] | select(.name == $n) | .draft_model_id // ""')

    # chat_template is resolved separately, NOT through the joined `read` above:
    # a `|` block scalar is legitimately multi-line and `read` stops at the first
    # newline, which silently truncated the template to its opening line. vLLM
    # then got a bare `{%- for ... -%}` and died with a Jinja
    # "Unexpected end of template" error.
    MODEL_CHAT_TEMPLATE=$(echo "$_MM_CONFIG_JSON" | jq -r --arg n "$name" \
        '.models[] | select(.name == $n) | .chat_template // ""')

    # `chat_template_file:` references a reusable template file (under
    # MM_CHAT_TEMPLATES, or an absolute path on the deploy host). Its CONTENT is
    # read and treated exactly like inline Jinja (→ ConfigMap → /etc/chat-template),
    # so large templates live in files and can be shared across models instead of
    # being duplicated inline in models.yaml. Mutually exclusive with chat_template.
    local _ct_file
    _ct_file=$(echo "$_MM_CONFIG_JSON" | jq -r --arg n "$name" \
        '.models[] | select(.name == $n) | .chat_template_file // ""')
    if [[ -n "$_ct_file" ]]; then
        [[ -n "$MODEL_CHAT_TEMPLATE" ]] && \
            abort "Model '$name': set either chat_template OR chat_template_file, not both."
        local _ct_path="$_ct_file"
        [[ "$_ct_path" != /* ]] && _ct_path="${MM_CHAT_TEMPLATES:-$SCRIPT_DIR/chat-templates}/$_ct_file"
        [[ -f "$_ct_path" ]] || \
            abort "Model '$name' chat_template_file not found: $_ct_path"
        MODEL_CHAT_TEMPLATE=$(cat "$_ct_path")
    fi

    # Reject a non-scalar chat_template. Some HF repos (e.g. Hermes-3) publish
    # tokenizer_config.json's chat_template as a LIST of {name, template} entries;
    # copy-pasting that shape here would otherwise be stringified to JSON and
    # handed to vLLM as a bogus template. vLLM's --chat-template takes one
    # template, so make the user pick.
    local _ct_type
    _ct_type=$(echo "$_MM_CONFIG_JSON" | jq -r --arg n "$name" \
        '.models[] | select(.name == $n) | .chat_template | type')
    if [[ "$_ct_type" == "array" || "$_ct_type" == "object" ]]; then
        abort "Invalid model '$name' chat_template: expected a string (inline Jinja or a file path) in $MM_CONFIG, got $_ct_type. Some HuggingFace repos publish a list of named templates — copy the single template you want (e.g. the one named \"default\") as a block scalar."
    fi

    # Classify the value: an absolute path is used as-is (the file must already
    # exist in the pod, e.g. on the model PVC under /mnt/models); anything else is
    # treated as inline Jinja and shipped via ConfigMap, because inline Jinja in
    # the spec breaks the KServe controller (see apply_chat_template_cm).
    #
    # vLLM applies the same path-vs-literal split but by content, not by shape:
    # _load_chat_template tries open() first and only falls back to treating the
    # string as a literal when it contains one of JINJA_CHARS = "{}\n". We key on
    # a leading "/" instead — an explicit path is unambiguous, and a template
    # never starts with one.
    MODEL_CHAT_TEMPLATE_IS_FILE=false
    [[ "$MODEL_CHAT_TEMPLATE" == /* ]] && MODEL_CHAT_TEMPLATE_IS_FILE=true

    # Structured fields captured as JSON. args/env are per-model overrides
    # consumed by build_*_block; the rest feed _select_server_version.
    MODEL_ARGS_JSON=$(echo "$_MM_CONFIG_JSON" | jq -c --arg n "$name" \
        '.models[] | select(.name == $n) | .args // []')
    MODEL_ENV_JSON=$(echo "$_MM_CONFIG_JSON" | jq -c --arg n "$name" \
        '.models[] | select(.name == $n) | .env // {}')

    # Validate shape early with a friendly message, so a mistake in models.yaml
    # doesn't surface later as a cryptic jq "object and array cannot be
    # multiplied" error during env/args merge.
    _validate_json_type "model '$name' env"  "$MODEL_ENV_JSON"  object
    _validate_json_type "model '$name' args" "$MODEL_ARGS_JSON" array

    # Server binding — the model → server(s) mapping. Preferred shape is a
    # `servers:` map; `runtime:`/`server_version:` are single-server shorthand.
    MODEL_SERVERS_JSON=$(echo "$_MM_CONFIG_JSON" | jq -c --arg n "$name" \
        '.models[] | select(.name == $n) | .servers // {}')
    MODEL_DEFAULT_SERVER=$(echo "$_MM_CONFIG_JSON" | jq -r --arg n "$name" \
        '.models[] | select(.name == $n) | .default_server // ""')
    MODEL_RUNTIME_RAW=$(echo "$_MM_CONFIG_JSON" | jq -r --arg n "$name" \
        '.models[] | select(.name == $n) | .runtime // ""')
    MODEL_SERVER_VERSION_RAW=$(echo "$_MM_CONFIG_JSON" | jq -r --arg n "$name" \
        '.models[] | select(.name == $n) | .server_version // ""')

    _select_server_version
}

# Resolve the (server, version) a model deploys to, from either the `servers:`
# map or the `runtime:`/`server_version:` shorthand, honoring CLI --server /
# --server-version overrides. Collapses the result into MODEL_RUNTIME +
# MODEL_SERVER_VERSION so all downstream code is unchanged.
#
# Selection (highest wins):
#   server :  --server  >  default_server  >  sole/first server key  >  DEFAULT_RUNTIME
#   version:  --server-version  >  server's `default`  >  first listed version  >  ""(→ runtime default)
#
# Validation — an EXPLICIT `servers:` map is authoritative:
#   - the chosen server must be declared, and
#   - when that server lists non-empty `versions:`, the chosen version must be in
#     it — this is the per-model "known-good" allow-list (reviewer point 2).
#   The shorthand imposes no allow-list (preserves prior free-override behavior).
_select_server_version() {
    # Reference directly — a `${v:-{}}` brace-default is mangled by bash param
    # expansion. MODEL_SERVERS_JSON is always initialized (global + resolve_model).
    local servers="$MODEL_SERVERS_JSON"
    [[ -z "$servers" ]] && servers="{}"
    local explicit=true
    if [[ "$servers" == "{}" ]]; then
        explicit=false
        # Desugar shorthand into a servers map. versions:[] (empty) means no
        # per-model allow-list, so CLI version overrides stay unrestricted.
        local rt="${MODEL_RUNTIME_RAW:-$DEFAULT_RUNTIME}"
        servers=$(jq -cn --arg rt "$rt" --arg d "$MODEL_SERVER_VERSION_RAW" \
            '{($rt): ({versions: []} + (if $d == "" then {} else {default: $d} end))}')
        MODEL_DEFAULT_SERVER="$rt"
    fi

    # Pick the server.
    local default_server="$MODEL_DEFAULT_SERVER"
    [[ -z "$default_server" ]] && default_server=$(jq -rn --argjson s "$servers" '$s | keys[0] // ""')
    local server="${OPT_SERVER:-$default_server}"
    [[ -z "$server" ]] && server="$DEFAULT_RUNTIME"

    if [[ "$explicit" == true ]]; then
        local declared_server
        declared_server=$(jq -rn --argjson s "$servers" --arg k "$server" '$s | has($k)')
        [[ "$declared_server" == "true" ]] || abort \
            "Model '$MODEL_NAME' does not declare server '$server'. Declared: $(jq -rn --argjson s "$servers" '$s | keys | join(", ")')"
    fi

    # Pick the version from the chosen server's entry.
    # NB: with `jq -n`, a bare `.versions` reads from null — must use `$e.versions`.
    local entry declared server_default
    entry=$(jq -cn --argjson s "$servers" --arg k "$server" '$s[$k] // {}')
    declared=$(jq -cn --argjson e "$entry" '$e.versions // []')
    server_default=$(jq -rn --argjson e "$entry" '$e.default // ($e.versions[0] // "")')
    local version="${OPT_SERVER_VERSION:-$server_default}"

    # Per-model allow-list check (only when a non-empty versions list is declared).
    local ndecl
    ndecl=$(jq -rn --argjson d "$declared" '$d | length')
    if [[ "$ndecl" -gt 0 && -n "$version" ]]; then
        local in_list
        in_list=$(jq -rn --argjson d "$declared" --arg v "$version" '($d | index($v)) != null')
        [[ "$in_list" == "true" ]] || abort \
            "Model '$MODEL_NAME' is not declared to support '$server' version '$version'. Declared versions: $(jq -rn --argjson d "$declared" '$d | join(", ")')"
    fi

    MODEL_RUNTIME="$server"
    MODEL_SERVER_VERSION="$version"
}

# Return a runtime's config as JSON, deep-merged from two complementary sources:
#   1. runtimes/<name>/values.yaml       — engine plumbing (device, kind, routing,
#                                           shared env, per-category args/defaults)
#   2. models.yaml  →  .runtimes.<name>   — version control surface (default_version,
#                                           versions, and any plumbing overrides)
# The two are recursively merged with the models.yaml entry winning on any key
# clash (`$base * $overlay`), so version selection stays in models.yaml while the
# verbose plumbing lives in values.yaml. Either source may be absent — a runtime
# defined in only one file still resolves. Result is memoized in _RUNTIME_CACHE.
# All runtime consumers (load_runtime, apply_category_defaults, build_*_block) go
# through this single resolver, so the source layout can change without touching
# them.
runtime_json() {
    local runtime="$1"
    if [[ -n "${_RUNTIME_CACHE[$runtime]:-}" ]]; then
        printf '%s' "${_RUNTIME_CACHE[$runtime]}"
        return 0
    fi

    # Base: engine plumbing from the per-directory values.yaml (if present).
    local base="{}"
    local runtime_file="$MM_RUNTIMES/$runtime/values.yaml"
    [[ -f "$runtime_file" ]] && base=$(yq e -o=json '.' "$runtime_file")

    # Overlay: the `runtimes:` entry in the already-parsed models.yaml (if present).
    local overlay="{}"
    if [[ -n "${_MM_CONFIG_JSON:-}" ]]; then
        overlay=$(jq -c --arg n "$runtime" '.runtimes[$n] // {}' <<< "$_MM_CONFIG_JSON" 2>/dev/null)
    fi

    # A runtime must exist in at least one source.
    if [[ "$base" == "{}" && "$overlay" == "{}" ]]; then
        abort "Runtime '$runtime' not found under 'runtimes:' in $MM_CONFIG or at $runtime_file"
    fi

    # Deep merge; overlay (models.yaml) wins on key clash.
    local rj
    rj=$(jq -cn --argjson base "$base" --argjson overlay "$overlay" '$base * $overlay')

    _RUNTIME_CACHE[$runtime]="$rj"
    printf '%s' "$rj"
}

# Resolve runtime defaults (image, args, env) for a runtime name.
load_runtime() {
    local runtime="${1:-$MODEL_RUNTIME}"
    local rj
    rj=$(runtime_json "$runtime")

    RUNTIME_DEVICE=$(echo "$rj" | jq -r '.device // "cpu"')
    RUNTIME_KIND=$(echo "$rj" | jq -r '.kind // "LLMInferenceService"')
    RUNTIME_FORMAT=$(echo "$rj" | jq -r '.format // ""')
    RUNTIME_NAME=$(echo "$rj" | jq -r '.runtime_name // ""')
    RUNTIME_ROUTING=$(echo "$rj" | jq -r '.routing // ""')
    RUNTIME_DOWNLOADER_IMAGE=$(echo "$rj" | jq -r '.model_downloader_image // ""')
    # Optional pointer to a raw ClusterServingRuntime manifest (relative to
    # MM_RUNTIMES). Empty → deploy falls back to a per-dir serving-runtime.yaml.
    RUNTIME_SERVING_MANIFEST=$(echo "$rj" | jq -r '.serving_runtime_manifest // ""')

    # Validate the runtime's shared env + selected category args are well-shaped,
    # so a malformed runtimes.<name> entry fails with a clear message rather than
    # a cryptic jq merge error later.
    _validate_json_type "runtime '$runtime' env" "$(echo "$rj" | jq -c '.env // {}')" object
    if [[ -n "${MODEL_CATEGORY:-}" ]]; then
        local _cat_args _cat_env
        _cat_args=$(echo "$rj" | jq -c --arg c "$MODEL_CATEGORY" '.categories[$c].args // []')
        _validate_json_type "runtime '$runtime' category '$MODEL_CATEGORY' args" "$_cat_args" array
        _cat_env=$(echo "$rj" | jq -c --arg c "$MODEL_CATEGORY" '.categories[$c].env // {}')
        _validate_json_type "runtime '$runtime' category '$MODEL_CATEGORY' env" "$_cat_env" object
    fi

    # Server-version resolution.
    #   Versioned runtime: a `versions:` map keyed by version string, each entry
    #     an {image, env?, args?} object, plus a top-level `default_version:`.
    #   Flat runtime: a top-level `image:` (no versions map) — e.g. openvino.
    # The requested version is whatever _select_server_version resolved into
    # MODEL_SERVER_VERSION (which already honored --server-version, the model's
    # server binding, and any per-model allow-list); here we just fall back to
    # the runtime's default_version and confirm the version exists in the runtime.
    local has_versions requested
    has_versions=$(echo "$rj" | jq -r 'has("versions")')
    if [[ "$has_versions" == "true" ]]; then
        requested="${MODEL_SERVER_VERSION:-}"
        [[ -z "$requested" ]] && requested=$(echo "$rj" | jq -r '.default_version // ""')
        [[ -z "$requested" ]] && requested=$(echo "$rj" | jq -r '.versions | keys[0] // ""')

        local vj
        vj=$(echo "$rj" | jq -c --arg v "$requested" '.versions[$v] // empty')
        [[ -z "$vj" ]] && abort "Runtime '$runtime' has no version '$requested'. Available: $(echo "$rj" | jq -r '.versions | keys | join(", ")')"

        RUNTIME_VERSION="$requested"
        MODEL_SERVER_VERSION="$requested"
        RUNTIME_IMAGE=$(echo "$vj" | jq -r '.image // empty')
        [[ -z "$RUNTIME_IMAGE" ]] && abort "Runtime '$runtime' version '$requested' defines no image"
        # Per-version env/args deltas (optional) — layered above shared env and
        # below per-model overrides in the build_* helpers.
        RUNTIME_VERSION_ENV_JSON=$(echo "$vj" | jq -c '.env // {}')
        RUNTIME_VERSION_ARGS_JSON=$(echo "$vj" | jq -c '.args // []')
        _validate_json_type "runtime '$runtime' version '$requested' env"  "$RUNTIME_VERSION_ENV_JSON"  object
        _validate_json_type "runtime '$runtime' version '$requested' args" "$RUNTIME_VERSION_ARGS_JSON" array
    else
        # Flat runtime (no versions: map), e.g. openvino. It has a single image
        # and no version axis. A version that came from the model's servers:
        # declaration is treated as a label and ignored — this keeps the servers:
        # shape uniform across versioned and flat engines. But an EXPLICIT CLI
        # --server-version is a meaningless request here, so we still reject it.
        if [[ -n "${OPT_SERVER_VERSION:-}" ]]; then
            abort "Runtime '$runtime' does not define a 'versions:' map, so --server-version '${OPT_SERVER_VERSION}' cannot be selected."
        fi
        RUNTIME_VERSION=""
        MODEL_SERVER_VERSION=""   # drop the label; downstream treats flat runtime as versionless
        RUNTIME_IMAGE=$(echo "$rj" | jq -r '.image')
        RUNTIME_VERSION_ENV_JSON="{}"
        RUNTIME_VERSION_ARGS_JSON="[]"
    fi
}

# Apply category defaults from runtime values
apply_category_defaults() {
    local category="${MODEL_CATEGORY:-llm}"
    local runtime="${MODEL_RUNTIME}"
    local rj
    rj=$(runtime_json "$runtime")

    if [[ -z "$MODEL_CPU" ]]; then
        MODEL_CPU=$(echo "$rj" | jq -r --arg c "$category" '.categories[$c].defaults.cpu // 8')
    fi
    if [[ -z "$MODEL_MEMORY" ]]; then
        MODEL_MEMORY=$(echo "$rj" | jq -r --arg c "$category" '.categories[$c].defaults.memory // "16Gi"')
    fi
}

# Auto-detect category from HuggingFace (requires curl + jq)
detect_category() {
    local model_id="$1"
    [[ -z "$model_id" ]] && echo "llm" && return

    local proxy_args=()
    [[ -n "${MM_HTTPS_PROXY:-}" ]] && proxy_args=(--proxy "$MM_HTTPS_PROXY")
    local response
    response=$(curl -sf --max-time 10 "${proxy_args[@]}" "https://huggingface.co/api/models/$model_id" 2>/dev/null)
    [[ -z "$response" ]] && echo "llm" && return

    local tag
    tag=$(echo "$response" | jq -r '.pipeline_tag // ""' 2>/dev/null)

    case "$tag" in
        text-generation|text2text-generation|conversational) echo "llm" ;;
        feature-extraction|sentence-similarity|fill-mask) echo "embed" ;;
        text-classification) echo "rerank" ;;
        image-text-to-text|visual-question-answering|image-to-text) echo "vlm" ;;
        *) echo "llm" ;;
    esac
}

# List all model names from config
list_model_names() {
    echo "$_MM_CONFIG_JSON" | jq -r '.models[].name'
}
