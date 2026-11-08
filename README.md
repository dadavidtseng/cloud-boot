# cloud-boot

Reproducible Ubuntu cloud-init testing and diagnostics on QEMU/KVM.

cloud-boot boots an official Ubuntu cloud image in a local KVM virtual machine, feeds it
NoCloud seed data from a declarative scenario, checks the resulting guest state over SSH, and
keeps the console, cloud-init logs, and a JSON report as evidence.

> **Status: Milestone 0 (pre-release).** This is a working vertical slice, not a release. It
> verifies the image with the published SHA-256 checksum only; it does **not** yet authenticate
> the checksum file's GPG signature, and it is not a security boundary for untrusted images or
> user-data. Do not rely on it for anything security-sensitive.

## Requirements

- An x86_64 Linux host with KVM. Verified on Ubuntu 24.04 running under WSL2 with nested
  virtualization (QEMU 8.2.2, OpenSSH 9.6p1, Python 3.12.3).
- Host packages:

  ```bash
  sudo apt install qemu-system-x86 qemu-utils openssh-client
  ```

- Read and write access to `/dev/kvm`. Join the `kvm` group, then start a new login session
  (under WSL, run `wsl --shutdown` from Windows and reopen the distribution):

  ```bash
  sudo usermod -aG kvm "$USER"
  ```

- Python 3.12 and [uv](https://docs.astral.sh/uv/).
- Network access to `https://cloud-images.ubuntu.com/` for the first run, which downloads the
  Ubuntu 24.04 (Noble) amd64 image, about 600 MB. Later runs reuse the cached image.

## Install

```bash
git clone https://github.com/dadavidtseng/cloud-boot.git
cd cloud-boot
uv sync
```

`uv sync` creates `.venv` with cloud-boot installed in editable mode and the `cloud-boot`
command on its path.

## Quick start

```bash
uv run cloud-boot run scenarios/basic-boot.yaml
```

Progress goes to stderr. A passing run with a cached image takes about 20-30 seconds:

```text
INFO preflight: checking host commands and KVM access
INFO workspace: writing evidence to /home/you/.local/state/cloud-boot/runs/<run-id>
INFO image: resolving noble amd64
INFO prepare: creating the run key, seed media, and overlay
INFO boot: starting QEMU (300 s limit)
INFO ssh: waiting for SSH on 127.0.0.1:<port>
INFO cloud-init: waiting for cloud-init to finish
INFO assertions: evaluating 3 assertions
INFO collect: copying cloud-init logs
INFO shutdown: asking the guest to power off
INFO shutdown: stopping QEMU
INFO report: passed (exit code 0)
```

Stdout carries exactly one line, the run directory, so scripts can capture it:

```text
/home/you/.local/state/cloud-boot/runs/<run-id>
```

A failed assertion is listed on stderr, and the command exits with code 7:

```text
FAIL marker-created (failed): /var/tmp/boot-marker does not contain the expected text
```

## Where results go

Each run gets its own directory, named `<UTC timestamp>-<random id>`, under
`$XDG_STATE_HOME/cloud-boot/runs/` (by default `~/.local/state/cloud-boot/runs/`). Directories
are mode 0700 and files 0600.

```text
<run-id>/
|-- report.json                  versioned JSON report: identities, assertions, outcome, exit code
`-- boots/001-boot/
    |-- console.log              the guest serial console
    |-- cloud-init.log           copied from the guest
    |-- cloud-init-output.log    copied from the guest
    |-- qemu-command.json        the exact QEMU argument array
    `-- qemu.stderr.log          QEMU's own diagnostics
```

Seed media, the qcow2 overlay, and the run's SSH key live in `transient/` during the run and are
deleted afterwards, even when the run fails. Failed runs keep all the evidence above.

Downloaded images are cached by SHA-256 under `$XDG_CACHE_HOME/cloud-boot`
(by default `~/.cache/cloud-boot`).

## Exit codes

The exit code always matches `exit_code` in `report.json` when a run directory was created.

| Code | Meaning |
|---:|---|
| 0 | Every required assertion passed |
| 2 | Invalid command line or scenario |
| 3 | A required host command or `/dev/kvm` access is missing |
| 4 | Image, key, seed, or overlay preparation failed |
| 5 | QEMU failed to start |
| 6 | The guest never became reachable over SSH |
| 7 | A required assertion failed or was blocked |
| 8 | The run exceeded the scenario's time limit |
| 9 | The run's evidence or report could not be written |
| 10 | Unexpected internal error |
| 130 | Interrupted with Ctrl-C (cleanup and the report still run) |

## The basic scenario

[`scenarios/basic-boot.yaml`](scenarios/basic-boot.yaml) needs no guest network access: its
user-data writes `/var/tmp/boot-marker` and installs nothing. It asserts that:

1. cloud-init finished with the extended status `done` (a `degraded done` boot fails);
2. `/var/tmp/boot-marker` exists and contains `boot-ok`, proving the seed's user-data ran;
3. the `openssh-server` package is installed.

`openssh-server` is checked on purpose. Every assertion runs over SSH, so by the time this check
runs, the image's SSH service has already proven usable. If a future image removes or replaces
the package, the failure means cloud-boot's transport assumption has changed, not that an
arbitrary package disappeared.

## Milestone 0 limitations

- **Checksum only:** the image digest is checked against `SHA256SUMS`, but the GPG signature on
  `SHA256SUMS` is not verified yet. Reports record `signature_authenticated: false`.
- **No secrets:** `${secret:...}` references are rejected because output redaction does not
  exist yet.
- **One VM, one boot:** each run boots a single guest once. Multi-boot, reboot, and fault
  scenarios are not supported yet.
- **Fixed environment:** NoCloud seed media only, QEMU user-mode networking (`nat`) only, KVM
  required, and the guest must be reachable over SSH as the `ubuntu` user.
- **No options or configuration files:** only `cloud-boot run <scenario>`, `--help`, and
  `--version` exist. Directories follow the XDG variables above.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

The end-to-end test boots a real VM, so it is skipped unless you opt in. It reports the reason
when it skips, and never reports a skipped boot as passed:

```bash
CLOUD_BOOT_E2E=1 uv run pytest tests/e2e -v
```

## License

Apache-2.0. See [LICENSE](LICENSE).
