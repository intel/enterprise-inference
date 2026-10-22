# CPU Pinning & NUMA

## Why it matters

On CPU clusters, inference throughput depends heavily on *which* cores a model runs on.
A model whose threads are spread across NUMA nodes pays a memory-latency penalty on
every token, and a model sharing cores with a noisy neighbour loses throughput
unpredictably.

This layer pins model pods to physical cores that respect the node's NUMA topology, so
each model gets exclusive, locality-aware CPU.

## The three policies

One switch in [`config.yaml`](../customize/configuration.md#cpu-policy) controls all
CPU-pinning infrastructure. Every dependent setting — NRI install, kubelet flags, model
annotations — is derived from it.

```yaml
kubernetes_cpu_policy: "nri-balloons"   # nri-balloons | kubelet-static | best-effort
```

| Policy | Mechanism | Use when |
|---|---|---|
| `nri-balloons` (default) | NRI balloons resource policy — NUMA-aware pinning into named CPU pools | The default. Best throughput and the most control |
| `kubelet-static` | kubelet static CPU manager — Guaranteed QoS grants exclusive whole cores | You cannot run NRI, or you already standardized on the static CPU manager |
| `best-effort` | No pinning; the standard scheduler places threads | Development, oversubscribed nodes, or debugging a pinning problem |

> [!WARNING]
> This must match the value the cluster was **installed** with. It decides both how
> kubelet is configured (solutions repo) and how model manifests are rendered (this
> repo). Changing it on an existing cluster requires re-running
> `./es_auto_installer.sh install kubernetes` so kubelet is reconfigured.

model-manager records the resolved policy on each model pod:

```yaml
annotations:
  cpu-policy.model-manager.io/type: nri-balloons
```

If NRI is expected but cannot be resolved, the pod is annotated
`type: best-effort` with `reason: nri-resolution-failed` rather than being pinned
incorrectly — the model still serves, without exclusive cores.

## NRI balloons

The NRI balloons policy groups CPUs into *balloons* and assigns pods to them, keeping
each pod's cores within one NUMA node where possible.

Tuning lives in the solutions repo, because it configures the nodes:

```yaml
# nri_reserved_cpu_list: ""            # Linux cpuset for system CPUs (auto-sized if empty)
# nri_deployment_profile: "standard"   # minimal(8) | standard(12) | observability(16) | full(20)
```

The profile sizes the CPU reservation for platform workloads — the number in parentheses
is roughly how many cores are held back. Pick a larger profile when the cluster also runs
observability or RAG workloads, so model pods don't contend with them.

### Pre-flight capacity check

Before changing anything, `model-manager deploy` verifies the request actually fits:

- `cpu` × `replicas` against the available balloon pool, and
- for `--tp N`, that the shards can spread across NUMA nodes

If it would oversubscribe, the deploy is **blocked** with actionable guidance rather than
creating a pod that will never schedule.

| Override | Effect |
|---|---|
| `MM_NRI_PREFLIGHT=warn` | Report the shortfall, then proceed anyway |
| `MM_NRI_SKIP_PREFLIGHT=1` | Skip the check entirely |
| `MM_NRI_ENABLED=false` | Skip NRI annotations altogether |

Re-deploying an unchanged model is a **no-op**: model-manager diffs the rendered spec
server-side and skips both pre-flight and apply when nothing changed, so a running
model's own cores are never counted against it. Set `MM_FORCE_REDEPLOY=1` to force a full
re-apply.

### Inspecting what got pinned

A helper script reports the actual CPU assignment for running vLLM pods — pod, balloon,
node, NUMA node, CPUs, and hyperthread siblings:

```bash
pip install kubernetes
python3 ext/enterprise.ai-inference/model_manager/scripts/nri-cpu-report.py \
  --namespace llm-inference
```

## kubelet static CPU manager

Under `kubelet-static`, model pods are rendered as **Guaranteed QoS** — integer `cpu`
with requests equal to limits — so kubelet's static CPU manager grants them exclusive
whole cores.

With `full-pcpus-only` enabled (the solutions default), a pod's `cpu` request must be a
multiple of the SMT width, or kubelet rejects the pod with `SMTAlignmentError`.
model-manager rounds `cpu` up to satisfy this automatically.

On an SMT-disabled cluster, tell it the width is 1:

```yaml
mm_kubelet_threads_per_core: 1     # exported as MM_KUBELET_THREADS_PER_CORE
```

## Node placement

Pinning decides *which cores*; node affinity decides *which node*. The `nfd` role labels
every node with its hardware capabilities:

```
feature.node.kubernetes.io/cpu-security.amx.enabled=true
feature.node.kubernetes.io/cpu-cpuid.AVX512F=true
feature.node.kubernetes.io/memory-numa.node_count=2
```

model-manager turns these into `nodeAffinity` rules. The catalog's
`require_amx: true` default keeps model pods off nodes without Intel® AMX — see
[Model Catalog](../customize/catalog.md#defaults). Pin a model to one specific node with
`--cpu N --node <name>`.

## Overriding per deploy

```bash
# Force a policy for one model (must match how the cluster was installed)
./model-manager deploy qwen3-0-6b --cpu-policy best-effort --wait

# Give a model 16 cores across 2 sockets
./model-manager deploy llama3-8b-awq --cpu 16 --tp 2 --wait

# See what would be rendered, including annotations, without applying
./model-manager deploy qwen3-0-6b --dry-run
```

Related: [Configuration](../customize/configuration.md#cpu-policy) ·
[Deploy a Model](../deploy/deploy_models.md#deploy-options) ·
[Troubleshooting](../troubleshooting.md)
