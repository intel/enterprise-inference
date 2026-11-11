# Review rules

Claude applies these when reviewing pull requests.

## Verdict

- **Blocking** findings give `Verdict: needs changes`: a bug, a security issue, a committed secret, missing tests for changed behaviour, or a break of the repo conventions below. Each one states how it fails: the input or setup, and the wrong result.
- **Suggestions** (rules 1–4 and 6: size, readability, structure, comments, naming) never change the verdict. List at most 5, the most valuable first.
- `Verdict: LGTM` when there is no blocking finding, with or without suggestions.
- **Earlier decisions.** The author's replies to earlier reviews are in the author notes. Do not raise a point again that a reply answered (fixed, or declined with a reason) unless the current code makes that answer wrong; then say what changed.

## Principles

1. **Minimal code.** No code for hypothetical needs, no unused parameters, no scaffolding "for later". If a change can be smaller and still correct, say where.
2. **Easy to read over clever.** Straightforward control flow and plain names beat abstraction. Flag any interface with one implementation, factory for one product, or config for a value that never changes.
3. **SOLID and DRY.** One responsibility per module, class, chart template, or script. Repeated blocks become a function, helper template, or shared module. Dependencies injected, not hard-coded.
4. **No or minimal comments.** Code explains itself. A comment is justified only for a non-obvious constraint (external bug, protocol quirk, security reason). Flag comments that restate the code.
5. **Correctness first.** Bugs, committed secrets, unhandled errors on external calls, Helm values without defaults, and missing tests for changed behavior always outrank style and are blocking.
6. **Integrity.** Changes fit the existing structure of the component (Ansible role layout and the `components.yaml` registry, Helm chart conventions, `model_manager` module patterns) and follow the patterns already present.

## Repo conventions

A violation in changed code is a blocking finding.

- **Native Ansible modules.** Use `ansible.builtin.*`, `kubernetes.core.*` (`k8s`, `k8s_info`, `k8s_exec`, `helm`) and other collections. No `shell`. `command` only where no module fits, with `argv`, explicit `changed_when` and `failed_when`, and a comment saying why. Behaviour of `kubernetes.core` is that of 6.3.0, the version the solutions repo pins.
- **Idempotent and check-mode safe.** A second run reports no changes. Anything carrying a secret has `no_log: true`; no secrets in values, logs or docs.
- **Component lifecycle.** A component is registered in `components.yaml` and its role has `install`, `teardown` and `validate` tasks. Teardown removes what install created (releases, CRDs, finalizers, namespaces) and tolerates a partly installed cluster.
- **Both modes.** Code and docs work for `routing_mode` `subdomain` and `path`, and for `auth_type` `jwt` and `litellm_vkey`.
- **Security baseline.** Namespaces keep Pod Security `restricted` and Istio ambient labels where the role sets them; workloads run as non-root with dropped capabilities; mesh exceptions are as narrow as the port and source allow.
- **model_manager is Python 3.9+.** It must run on Python 3.9 with PyYAML 5.4.1 as well as current versions (both CI rows): no `match` or other 3.10+ syntax, and `X | Y` types only in annotations, which `from __future__ import annotations` keeps unevaluated. Standard library and PyYAML only; a new dependency needs a reason. Type hints on functions; shared data as dataclasses in `core/config.py`.
- **model_manager safety.** External tools run through `subprocess` with an argument list (`kubectl` through `core/k8s.py`), never `shell=True`, `os.system` or `eval`; each `# nosec` names the Bandit check and why it holds. YAML through `yaml.safe_load`. Output and failures go through `core/ui.py` (`info`, `ok`, `warn`, `abort`) and never include tokens or keys.
- **Tests.** Changed `model_manager` behaviour has tests in `model_manager/tests` (unit tests, and parity tests against the `fakebin` stand-ins for `kubectl` and `curl`, no live cluster), which CI runs.

Report blocking findings first, then suggestions, each with `file:line`. No praise, no diff summary.
