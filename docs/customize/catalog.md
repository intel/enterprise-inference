# Model Catalog (`models.yaml`)

## What is it?

`models.yaml` is the one file you edit to control what runs and how. It is the single
source of truth behind every `model-manager` command — you rarely touch anything else.

The installer seeds `env/<env>/models.yaml` from this repo's
`model_manager/models.yaml`. **Edit the environment copy** to add or remove models; the
repo copy is the template for new environments.

It has three parts:

```yaml
defaults:            # 1. Fallbacks applied to every model (namespace, engine, replicas)
storage:             #    Where downloaded weights live
network:             #    Proxy / connectivity settings for pulling from Hugging Face

models:              # 2. The models you can deploy — one entry per model
- name: qwen3-0-6b
  model_id: Qwen/Qwen3-0.6B     # the Hugging Face repo to pull
  category: llm                 # workload type → sensible CPU/memory/flags
  servers:                      # which engine + versions this model may use
    vllm: { versions: ["0.24.0", "0.19.1"], default: "0.24.0" }

runtimes:            # 3. The engines themselves — images, versions, shared settings
  vllm: { ... }
  openvino: { ... }
```

What that gives you:

| Capability | What it means for you |
|------------|-----------------------|
| **One catalog** | List a model once; deploy it by name with `./model-manager deploy <name>` |
| **Sensible defaults** | Only `name` and `model_id` are required — category, CPU, memory, and engine flags are filled in |
| **Deliberate versioning** | Declare which engine versions a model may use; an untested version is refused, not silently deployed |
| **Central, reusable settings** | Engine images and shared options live in one `runtimes:` block, reused across every model |
| **Override anywhere** | Per-model `args`/`env`, or one-off `--cpu`/`--arg`/`--env` at deploy time — the CLI always wins |
| **Auto-deploy** | Flag a model `autodeploy: true` to bring it up during install |

## Global sections

### `defaults`

Applied to every model that doesn't override them.

| Field | Default | Meaning |
|---|---|---|
| `namespace` | `llm-inference` | Target namespace for deployments |
| `runtime` | `vllm` | Engine used when a model omits one |
| `replicas` | `1` | Replica count |
| `require_amx` | `true` | Only schedule model pods on nodes with Intel® AMX. Relies on the NFD labels applied by the `nfd` role |

### `storage`

| Field | Default | Meaning |
|---|---|---|
| `pvc_name` | `model-store` | The shared PVC holding downloaded weights. Must be **ReadWriteMany** on a multi-node cluster |

### `network`

Used by the download Job when pulling from Hugging Face. Empty values inherit from the
installer's environment.

| Field | Meaning |
|---|---|
| `http_proxy` / `https_proxy` | Proxy for weight downloads, e.g. `http://proxy:911` |
| `no_proxy` | Comma-separated bypass list |
| `connectivity_check` | URL probed for reachability before a download starts (default `https://huggingface.co`) |

## Per-model fields

Only `name` and `model_id` are required.

```yaml
models:
- name: qwen3-0-6b
  model_id: Qwen/Qwen3-0.6B
  category: llm
  cpu: 8
  memory: 16Gi
```

| Field | Meaning |
|---|---|
| `name` | Deploy name, and the `model` id used in API requests |
| `model_id` | The Hugging Face repo to pull weights from |
| `category` | Workload class — `llm`, `embed`, `rerank`, or `vlm`. Drives default args, CPU, and memory |
| `cpu` / `memory` | Resource requests (default: from the category). `cpu: auto` sizes CPU to a NUMA node — see [below](#cpu-auto) |
| `replicas` | Replica count (default: 1) |
| `tp` | Tensor parallelism — split the model across N CPU sockets |
| `namespace` | Target namespace (default: `llm-inference`) |
| `node` | Pin to a specific node |
| `routing` | `epp` or `direct` |
| `image` | Override the runtime image entirely |
| `args` | Extra engine args, appended after the category and version args |
| `env` | Extra env vars, merged over the runtime env |
| `autodeploy` | `true` → bring the model up during install. See [below](#auto-deploy-on-install) |
| `chat_template` | Inline Jinja or an absolute path → vLLM `--chat-template`. See [below](#chat-templates) |
| `chat_template_file` | A Jinja file whose content is used as `chat_template`, so models share one template. See [below](#chat-templates) |
| `draft_model_id` | Hugging Face repo of a speculative-decoding draft model (vLLM). See [below](#speculative-decoding) |

Plus the server binding, described next.

## Choose which server and version a model may use

Two equivalent forms, depending on how much control you want:

```yaml
# Preferred: declare the versions this model is known to work with (an allow-list)
- name: qwen3-0-6b
  model_id: Qwen/Qwen3-0.6B
  category: llm
  servers:
    vllm: { versions: ["0.24.0", "0.19.1"], default: "0.24.0" }
  default_server: vllm

# Shorthand: one server, optionally pinned to a version — no allow-list
- name: llama3-8b-awq
  model_id: casperhansen/llama-3-8b-instruct-awq
  category: llm
  runtime: vllm
  server_version: "0.24.0"
```

With the `servers:` form, a deploy that asks for a version *not* in the list is
**rejected**, so nobody accidentally ships an untested combination. The shorthand form
places no such restriction.

Version selection precedence, highest first:

```
--server-version   >   the model's server_version:   >   the runtime's default_version
```

Select at deploy time with `--server <engine> --server-version <v>`, and run variants
side by side with `--as <alt-name>`. See [Deploy a Model](../deploy/deploy_models.md#pick-an-engine-and-version).

The engines themselves — images, available versions, and shared settings — are defined
once under `runtimes:`. See [Runtimes](runtimes.md).

## CPU auto

`cpu: auto` sizes a model's CPU to the node it lands on instead of a fixed number:

```yaml
- name: qwen2-5-coder-14b
  model_id: Qwen/Qwen2.5-Coder-14B-Instruct
  cpu: auto
```

| CPU policy | `cpu: auto` becomes |
|---|---|
| `nri-balloons` | The free physical cores of one NUMA node per TP rank, minus the reserved CPUs, on the smallest schedulable node — so the pod fits wherever it is placed |
| `kubelet-static`, `best-effort` | The category's default `cpu` |

- A model that is already deployed keeps its current CPU, so re-running a deploy or the
  install leaves it unchanged.
- `--cpu N` overrides it for one deploy, and `--cpu auto` asks for the sizing from the
  command line. `--dry-run` shows the number it resolves to.
- Under `nri-balloons` it reads the `nri.intel.com/max-tp` node labels the solutions
  repo's NRI balloons role sets. When fewer CPUs than the category default are free, the
  deploy stops with `cpu: auto — only N free CPUs per NUMA node …`: undeploy a model,
  add a node, or pass `--cpu`.

See [CPU Pinning & NUMA](../reference/cpu_pinning.md).

## Chat templates

`chat_template` accepts either inline Jinja or an absolute path.

**Inline Jinja** (use a `|` block scalar for multi-line) is stored in a
`<model>-chat-template` ConfigMap and mounted read-only at `/etc/chat-template`. It is
deliberately *not* inlined into the `LLMInferenceService`, because the KServe
controller runs the spec through Go `text/template` and rejects `{{ ... }}`. The
ConfigMap is created on deploy and removed on undeploy.

**An absolute path** (leading `/`) is passed through untouched and must already exist in
the pod — for example a file you placed on the model PVC, visible at
`/mnt/models/<file>`. vLLM exits if it is missing, so prefer inline Jinja unless the
file is genuinely external.

> [!NOTE]
> It must be a **single** string. If a Hugging Face repo publishes a *list* of named
> templates, copy just the one you want (usually `default`). To preserve a literal CR,
> use a double-quoted scalar with `\r` escapes — a `|` block scalar cannot carry CR,
> because YAML normalizes line breaks.

**A shared file** — `chat_template_file` names a Jinja file whose content is used exactly
like inline Jinja (the same ConfigMap and `--chat-template`), so several models share one
template instead of each carrying a copy:

```yaml
- name: qwen2-5-coder-14b
  model_id: Qwen/Qwen2.5-Coder-14B-Instruct
  chat_template_file: tool_chat_template_qwen3coder.jinja
```

- A relative name is looked up in `model_manager/chat-templates/`, or in the directory
  `MM_CHAT_TEMPLATES` points at — never next to the catalog, so a seeded
  `env/<env>/models.yaml` works unchanged.
- An absolute path is read on the host running `model-manager`, not in the pod.
- Setting both `chat_template` and `chat_template_file`, or naming a file that does not
  exist, stops the deploy before anything changes on the cluster.

Bundled templates:

| File | Used for |
|---|---|
| `tool_chat_template_qwen3coder.jinja` | Qwen2.5-Coder, Qwen3-Coder and Qwen3-30B-A3B tool calling |
| `tool_chat_template_gemma4.jinja` | Gemma 4 tool calling (`gemma-4-26b`) |
| `tool_chat_template_hermes.jinja` | Hermes-style tool calling |

## Speculative decoding

`draft_model_id` names a small draft model for vLLM speculative decoding. The download
Job fetches it into the model's own directory on the model store, so the pod — which
mounts only that directory and serves offline — finds it at `/mnt/models/draft`.
Point vLLM at that path in `args`:

```yaml
- name: llama-3-1-8b-specdec
  model_id: meta-llama/Llama-3.1-8B-Instruct
  draft_model_id: meta-llama/Llama-3.2-1B-Instruct
  args: [ "--speculative-config.method", "draft_model",
          "--speculative-config.model", "/mnt/models/draft",
          "--speculative-config.num_speculative_tokens", "5" ]
```

- The draft has its own download marker: adding one to a model whose weights are
  already on disk fetches only the draft.
- It downloads with the same `HF_TOKEN`. A gated draft — such as
  `meta-llama/Llama-3.2-1B-Instruct` — without a token stops the deploy before the Job
  starts.
- Write JSON-valued vLLM flags in dot notation (`--flag.key value`): KServe evaluates the
  arg string, which strips the quotes from inline JSON.

## Auto-deploy on install

By default **no models are deployed during install** — the inference layer only stands
up serving infrastructure. Bringing a model up automatically takes **two** settings:

1. Flag the model in the catalog:

   ```yaml
   - name: qwen3-0-6b
     model_id: Qwen/Qwen3-0.6B
     autodeploy: true
   ```

2. Enable the auto-deploy phase in this repo's `config.yaml`, which ships **disabled**:

   ```yaml
   llm_services_deploy_models: true
   ```

> [!IMPORTANT]
> Both are required. `config.yaml` sets `llm_services_deploy_models: false`, which gates
> the entire phase — with it off, `autodeploy: true` has no effect. This is deliberate:
> a full install never serves the whole catalog, which may include large models.

Once enabled, only models flagged `autodeploy: true` are deployed.

> [!NOTE]
> Two kinds of model deploy without either setting: models a solution layer requests by
> name in `llm_services_autodeploy_models` (the agent toolkit sets its default model
> there), and the guardrail's guard model (`guardrail_model`) whenever the guardrail is
> on. The guard model deploys first, so models with `cpu: auto` size themselves around it.
> They come from the same catalog. The environment copy is seeded once, at `init`, so an
> environment created before a catalog update lacks the new entries: a requested model
> missing there is taken from `model_manager/models.yaml` (the rest of
> `env/<env>/models.yaml` still applies) and the install prints a notice. To customise
> such a model, copy its entry into `env/<env>/models.yaml`. See
> [Configuration](configuration.md#llm-services-and-model-deployment).

**Auto-deploy is fault-tolerant.** If a flagged model fails — a gated model with no
`HF_TOKEN`, a transient download error — the install **does not abort**. The serving
infrastructure is already up, the failure is reported in a summary, and the model can be
deployed manually afterward. Set `model_deploy_strict: true` to make any auto-deploy
failure fatal instead.

To skip the deploy phase for a single run without editing config, run
`SKIP_MODEL_DEPLOYMENT=true ./es_auto_installer.sh install inference` (or append `-- -e skip_models=true`).
With the content-safety guardrail on, its guard model is still deployed: the guardrail blocks every request without it.

Extra flags can be appended to every auto-deploy invocation via `model_deploy_flags`,
e.g. `"--wait-timeout 1800"` for large models.
