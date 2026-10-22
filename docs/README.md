# Intel® AI for Enterprise Inference Documentation

The technical documentation for the inference layer. Start with **Deploy** to get
a model serving, use **Customize** to add models and tune the engines, then look
things up in **Reference**.

> New here? The three-step quick start is in the [project README](../README.md#quick-start).
> This layer is a component of [Intel® AI for Enterprise Solutions](https://github.com/intel/enterprise-ai-solutions) — install that platform first.

## Deploy

Get a model running and call it.

| Guide | What it covers |
|-------|----------------|
| [Deploy a Model](deploy/deploy_models.md) | Every `model-manager` command and flag — deploy, update, undeploy, gated models, ad-hoc Hugging Face deploys |
| [Accessing Models](deploy/accessing_models.md) | Endpoints and credentials for both auth modes, and how to list what's serving |

## Customize

Add models, choose engines and versions, and change any setting.

| Guide | What it covers |
|-------|----------------|
| [Model Catalog](customize/catalog.md) | The `models.yaml` catalog — every field, and auto-deploy at install time |
| [Runtimes](customize/runtimes.md) | vLLM and OpenVINO — versions, images, and per-workload defaults |
| [Configuration](customize/configuration.md) | Every option in `config.yaml`, plus the `MM_*` environment variables |

## Reference

How the layer is built and how a request travels through it.

| Guide | What it covers |
|-------|----------------|
| [Architecture](reference/architecture.md) | Components, Ansible roles, and how this repo plugs into the platform |
| [Inference Request Flow](reference/request_flow.md) | End-to-end request routing for both auth modes, with a verified network trace |
| [CPU Pinning & NUMA](reference/cpu_pinning.md) | The three CPU policies, the pre-flight capacity check, and NUMA placement |
| [Database Architecture](reference/database.md) | Shared PostgreSQL schema and provisioning for LiteLLM and Langfuse |

## Support

| Guide | What it covers |
|-------|----------------|
| [Troubleshooting](troubleshooting.md) | Common symptoms and their fixes |
