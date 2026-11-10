# Review rules

Claude applies these when reviewing pull requests. A PR gets `Verdict: LGTM` only if all hold.

1. **Minimal code.** No code for hypothetical needs, no unused parameters, no scaffolding "for later". If a change can be smaller and still correct, say where.
2. **Easy to read over clever.** Straightforward control flow and plain names beat abstraction. Flag any interface with one implementation, factory for one product, or config for a value that never changes.
3. **SOLID and DRY.** One responsibility per module, class, chart template, or script. Repeated blocks become a function, helper template, or shared module. Dependencies injected, not hard-coded.
4. **No or minimal comments.** Code explains itself. A comment is justified only for a non-obvious constraint (external bug, protocol quirk, security reason). Flag comments that restate the code.
5. **Correctness first.** Bugs, committed secrets, unhandled errors on external calls, Helm values without defaults, and missing tests for changed behavior always outrank style and are always `needs changes`.
6. **Integrity.** Changes fit the existing structure of the component (Ansible role layout and the `components.yaml` registry, Helm chart conventions, `model_manager` module patterns) and follow the patterns already present.

## Repo conventions

A violation in changed code is a finding.

- **Native Ansible modules.** Use `ansible.builtin.*`, `kubernetes.core.*` (`k8s`, `k8s_info`, `k8s_exec`, `helm`) and other collections. No `shell`. `command` only where no module fits, with `argv`, explicit `changed_when` and `failed_when`, and a comment saying why. Behaviour of `kubernetes.core` is that of 6.3.0, the version the solutions repo pins.
- **Idempotent and check-mode safe.** A second run reports no changes. Anything carrying a secret has `no_log: true`; no secrets in values, logs or docs.
- **Component lifecycle.** A component is registered in `components.yaml` and its role has `install`, `teardown` and `validate` tasks. Teardown removes what install created (releases, CRDs, finalizers, namespaces) and tolerates a partly installed cluster.
- **Both modes.** Code and docs work for `routing_mode` `subdomain` and `path`, and for `auth_type` `jwt` and `litellm_vkey`.
- **Security baseline.** Namespaces keep Pod Security `restricted` and Istio ambient labels where the role sets them; workloads run as non-root with dropped capabilities; mesh exceptions are as narrow as the port and source allow.
- **Tests.** Changed `model_manager` behaviour has tests in `model_manager/tests`, which CI runs.

Report all findings, each with `file:line`, ordered by severity. No praise, no diff summary.
Group them for easier reading if more than 10.
