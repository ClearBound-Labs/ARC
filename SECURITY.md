# Security Policy

## Reporting a Vulnerability

If you discover a security issue in ARC, please report it privately via
GitHub Security Advisories:

https://github.com/a-kaushik2209/ARC/security/advisories/new

Do not file public issues for sensitive findings. For non-sensitive
hardening (linter rules, dependency upgrades, lint of CI workflows, etc.)
feel free to open a normal PR.

We aim to acknowledge reports within seven days.

## Trust Boundary: Checkpoint Files

ARC's checkpoint loading uses `torch.load()`, which deserializes data with
Python's `pickle` protocol. Pickle executes arbitrary code during
deserialization through the `__reduce__` mechanism, so a checkpoint file
is effectively executable Python.

**Treat any checkpoint file you did not produce yourself as untrusted
code.** This includes, at minimum:

- Checkpoints downloaded from the internet (including model zoos and
  Hugging Face Hub repos you do not control)
- Checkpoints shared by other users on a multi-tenant system
- Checkpoints sitting in a directory writable by other accounts on the
  same machine (shared GPU servers, CI runners, training clusters)

### What ARC does

Since 5.2.0 every ARC loader (`arc._serialization.safe_torch_load`) uses
`weights_only=True`, which restricts unpickling to tensors and plain Python
values and rejects payloads that try to call arbitrary code via
`__reduce__`. A file that needs more than that fails to load with
`pickle.UnpicklingError`. **There is no automatic fallback.**

Unsafe loading exists only as an explicit opt-in, for files you produced
yourself: `allow_unsafe=True` on `MetaModelTrainer.load_checkpoint` and
`ArcV2.load`, `AdaptiveCheckpointConfig(allow_unsafe_load=True)`. It warns
every time it is used.

Checkpoints ARC writes itself (`Controller.save_checkpoint`, the adaptive
checkpointer's disk tier) contain only tensors and plain values, so they
always load safely.

### Integrity checks

`AdaptiveCheckpointConfig(sign_checkpoints=True, signing_key=...)` stores a
SHA-256 content hash and an HMAC signature with each checkpoint and verifies
both on restore. The Controller's recovery ledger can be HMAC-keyed with
`ControllerConfig(ledger_key=...)` or the `ARC_LEDGER_KEY` environment
variable; without a key, its hash chain only reveals accidental edits.
Neither replaces verifying checkpoints you download from elsewhere.

## Hardening Recommendations

1. **Use per-user checkpoint directories.** ARC's defaults are now under
   `~/.cache/arc/` rather than `/tmp/`. The previous `/tmp/` defaults
   were world-writable on Linux and let any local user plant a malicious
   pickle that ARC's autonomous rollback would later load. If you
   override the default, point it at a directory that other users on
   the system cannot write to.

2. **Audit checkpoint sources.** For checkpoints from external sources
   (other researchers, model hubs, internet downloads), inspect the
   producer and transport before loading. A signed hash from a trusted
   channel is the simplest verification.

3. **Never opt in to unsafe loading for files you didn't write.** If an
   ARC load fails with `UnpicklingError`, the file contains objects
   beyond tensors and plain values; for anything from an untrusted
   source, that is the refusal working as intended.

4. **Run on a separate user account for untrusted training data.** If
   you must work with checkpoints from outside your team, sandbox the
   training process in a dedicated low-privilege account so a successful
   exploit has minimal blast radius.