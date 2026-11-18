# Copilot code review instructions

This repo deploys an LLM inference stack on Kubernetes: Ansible roles (`roles/`), Helm charts
(`charts/`), and the `model-manager` CLI (Bash + some Python, `model_manager/`).

Evaluate every PR in two categories: **Security** and **Code quality**. Score each 0-5.
Comment only on lines the PR adds or changes. Do not comment on unchanged code.

## 1. Security

Check new code and new or changed dependencies (container images, Helm chart versions,
Ansible collections, Python packages, binaries downloaded at runtime).

Code and commands, flag:
- Shell injection: unquoted variables, `eval`, `bash -c "$VAR"`, user input put into commands,
  `kubectl`/`helm` args, or YAML templates without validation.
- `curl | bash`, downloads without checksum or signature checks, `curl -k`, `--insecure`,
  `validate_certs: false`, turned-off TLS verification.
- Hardcoded secrets, tokens, passwords, or API keys; secrets printed to logs or shell traces
  (`set -x`); Ansible tasks that handle secrets without `no_log: true`.
- Unsafe file handling: `chmod 777`, predictable `/tmp` paths, `rm -rf` with a variable that
  may be empty.
- Kubernetes/Helm: `privileged: true`, `hostNetwork`, `hostPath`, missing `securityContext`,
  running as root, `allowPrivilegeEscalation`, extra capabilities, broad RBAC
  (`*` verbs/resources, cluster-admin), services exposed without auth.
- Python: `subprocess(..., shell=True)`, `yaml.load` without SafeLoader, `pickle`,
  `eval`/`exec`.

Dependencies, flag:
- Container images on `latest` or with no pinned tag/digest.
- New or upgraded packages, images, or charts with known CVEs, or end-of-life versions.
- Dependencies from unofficial or unknown registries or repos.
- Version downgrades with no reason given.

## 2. Code quality

Main focus: **the code must not be cluttered with comments.**
- Flag comments that only repeat what the code says (`# install package` above an install
  task, `# loop over items`).
- Flag commented-out code, leftover debug output, and TODOs with no owner or issue link.
- Flag comment blocks longer than the code they describe.
- Comments that explain *why* (a workaround, a non-obvious limit, a link to an upstream issue)
  are good. Do not flag them.
- Explain once in the PR description, not over and over in comments.

Also flag, briefly:
- Duplicated logic that should use an existing helper (`model_manager/lib/*.sh`) or a shared
  Ansible task or Helm template.
- Bash scripts without `set -euo pipefail` (or a reason to skip it), and problems shellcheck
  would report.
- Ansible tasks that are not idempotent (`shell`/`command` without `changed_when`/`creates`)
  or that use `shell` when a module exists.
- Hardcoded values that should be in `values.yaml`, role defaults, or `config.yaml`.

## Scoring

| Score | Meaning |
|-------|---------|
| 5 | No issues found |
| 4 | Small nits only |
| 3 | Some problems that should be fixed, none blocking |
| 2 | At least one real issue that should block merge |
| 1 | Several serious issues |
| 0 | Critical issue (e.g. leaked secret, remote code execution, very broad permissions) |

## Output format

Start the review summary with this block:

```
### Review scores
| Category     | Score | Summary          |
|--------------|-------|------------------|
| Security     | X/5   | <one line>       |
| Code quality | X/5   | <one line>       |
```

Then list findings grouped by category, most severe first. For each finding give the file and
line, what is wrong, and a concrete fix. If a category has no findings, say so in one line.
Do not pad the review with praise or general advice.
