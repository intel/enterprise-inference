# Intel® Arc™ (XPU) GPU Deployment Guide

<!-- Copyright (C) 2025-2026 Intel Corporation -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

## Overview

This guide describes how to serve models on nodes with Intel® discrete GPUs (Arc
Battlemage B-series cards such as B60/B70) using the vLLM XPU backend. This is a second
compute platform alongside the default CPU (Xeon) path, selected via
`kubernetes_accelerator: xpu` in AI Solutions and served by the `vllm-xpu` runtime.

## Supported Models

Deploy with `model-manager deploy <name>`. These map 1:1 to the `*-xpu` entries in
[`model_manager/models.yaml`](../../model_manager/models.yaml), which hold the per-model
vLLM flags, tool-call parser, and tensor-parallel size. `Tensor Parallel` indicates how
many GPUs a single model replica is sharded across.

| Model ID | Catalog name | Tensor Parallel |
|----------|--------------|:---:|
| Qwen/Qwen3-Coder-30B-A3B-Instruct | `qwen3-coder-30b-xpu` | 4 |
| Qwen/Qwen2.5-Coder-14B-Instruct | `qwen2-5-coder-14b-xpu` | 2 |
| meta-llama/Llama-3.1-8B-Instruct | `llama-3-1-8b-xpu` | 1 |
| Qwen/Qwen3-Coder-Next | `qwen3-coder-next-xpu` | 4 |
| google/gemma-4-31B-it | `gemma-4-31b-xpu` | 4 |
| Qwen/Qwen3.6-27B | `qwen3-6-27b-xpu` | 4 |

Update `models.yaml` if you need to change memory utilization, context length, or add a
new model. Settings shared by every XPU model are in the `vllm-xpu` runtime
([`model_manager/runtimes/vllm-xpu/values.yaml`](../../model_manager/runtimes/vllm-xpu/values.yaml)
and the `vllm-xpu` entry under `runtimes:` in `models.yaml`).

## Prerequisites

Complete the AI Solutions prerequisites first, then the following XPU-specific items on
**every worker node that has an Intel discrete GPU**:

### 1. Hardware

- Intel® Arc™ / Flex / Max discrete GPU (this guide has been exercised on Battlemage B-series
  cards).
- Confirm the render node is visible on the host:
  ```bash
  ls /dev/dri/
  # Expect renderD1xx entries, e.g. renderD128
  ```

### 2. GPU driver and Level Zero runtime

Install the Intel GPU driver/compute runtime for your kernel (Xe or i915) so that
`libze_intel_gpu.so.1` and `libze_loader.so.1` exist under
`/usr/lib/x86_64-linux-gnu/`. The vLLM XPU container image does **not** ship these — the
`vllm-xpu` runtime mounts them from the host via `hostPath` volumes (`host_lib_dir` in
`runtimes/vllm-xpu/values.yaml`), so they must already be present on the node before you
deploy.

### 3. Intel GPU Plugin for Kubernetes (automated)

When `kubernetes_accelerator: xpu` is set, AI Solutions installs the `intel_gpu_plugin`
component before any model is deployed. It applies the "base" overlay manifest from
[intel/intel-device-plugins-for-kubernetes](https://github.com/intel/intel-device-plugins-for-kubernetes)
— no NFD or node labeling required — and waits for the resulting DaemonSet to roll out.

The branch/tag it deploys from defaults to `main` and is controlled by
`intel_gpu_plugin_ref`; pin it to a specific release tag for reproducible deployments.

After the run, confirm the resource is advertised on your node:

```bash
kubectl describe node <node-name> | grep -A2 'Allocatable:' | grep gpu.intel.com
```

You should see `gpu.intel.com/xe` (newer Xe kernel driver, e.g. Ubuntu 24.04+/25.10 on
Arc B-series) or `gpu.intel.com/i915` (legacy i915 driver) listed with a count. If it's
missing, re-run the component on its own to see the failure:

```bash
./es_auto_installer.sh install intel_gpu_plugin
```

### 4. Driver resource name override (`xe` vs `i915`)

The `vllm-xpu` runtime defaults to `gpu.intel.com/xe`. If your node plugin
exposes `gpu.intel.com/i915` instead, change it before deploying:

```yaml
# model_manager/runtimes/vllm-xpu/values.yaml
xpu:
  #resource: gpu.intel.com/i915
  resource: gpu.intel.com/xe
```

### 5. Supplemental group IDs

`xpu.supplemental_groups: [990, 44]` in `runtimes/vllm-xpu/values.yaml` grants the
vLLM container access to `/dev/dri`. These GIDs are host-specific — verify and
adjust them if pods fail with a GPU permission error:

```bash
getent group render video   # e.g. render:x:990 video:x:44
stat -c '%g %G' /dev/dri/renderD128
```

## Configuration

Set the accelerator in AI Solutions' `env/<env>/global_config.yaml`:

```yaml
kubernetes_accelerator: "xpu"
```

With `xpu`, NRI CPU balloons are not deployed (they apply to CPU serving only).

## Deploying

```bash
# Install the platform (including the Intel GPU plugin) and the inference layer
./es_auto_installer.sh install inference

# Deploy an XPU model
cd model_manager
./model-manager deploy gemma-4-31b-xpu --wait
```

model-manager renders the model with the XPU template: it requests
`gpu.intel.com/xe` equal to the tensor-parallel size, mounts the host GPU libraries, and
sources the oneAPI environment (`setvars.sh`) through `BASH_ENV` before vLLM starts.

## Verifying the Deployment

```bash
# Confirm the model is up
kubectl get llminferenceservice -n llm-inference | grep xpu

# Watch pod status
kubectl get pods -n llm-inference | grep xpu

# Test inference through the gateway (litellm mode shown; adjust the model name per the table above)
curl -k https://litellm.<base_domain_name>/v1/completions \
  -X POST \
  -d '{"model": "gemma-4-31b-xpu", "prompt": "What is Deep Learning?", "max_tokens": 50, "temperature": 0}' \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $TOKEN"
```

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| Pod stuck `Pending`, event shows `Insufficient gpu.intel.com/xe` | Intel GPU Plugin not installed, or wrong `xpu.resource` | Install the plugin (see Prerequisites §3) and confirm the resource name matches your driver (§4) |
| `libze_loader.so.1: cannot open shared object file` | Host is missing the Level Zero runtime | Install Intel GPU compute runtime on the node so the libs exist under `/usr/lib/x86_64-linux-gnu/` |
| `CreateContainerConfigError` / permission denied on `/dev/dri` | `supplemental_groups` GIDs don't match the host's `render`/`video` groups | Re-check with `getent group render video` and update `runtimes/vllm-xpu/values.yaml` |

## References

- [Intel GPU Plugin for Kubernetes](https://github.com/intel/intel-device-plugins-for-kubernetes)
- [vLLM XPU Backend Documentation](https://docs.vllm.ai/en/latest/getting_started/xpu-installation.html)
- [Intel discrete GPU driver installation guides](https://dgpu-docs.intel.com/driver/client/overview.html)
