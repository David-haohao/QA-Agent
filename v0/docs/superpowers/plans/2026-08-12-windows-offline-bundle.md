# Windows Offline Bundle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a self-contained Windows x64 offline installation bundle that can create the project virtual environment and install every Python dependency without network access.

**Architecture:** Bundle the official CPython 3.12 x64 installer, a fully pinned Windows wheelhouse, integrity metadata, and PowerShell installation and verification scripts under `v0/download`. The existing `v0/local_models` directory remains the model payload and is referenced in the guide instead of duplicated.

**Tech Stack:** CPython 3.12, pip wheelhouse, PowerShell 5.1+, SHA-256, Python unittest.

---

### Task 1: Record the reproducible dependency resolution

**Files:**
- Create: `download/requirements-offline.txt`
- Create: `download/resolution.json`

- [ ] **Step 1: Resolve the Windows CPython 3.12 dependency graph from `requirements.txt`**

Run `python -m pip install --dry-run --ignore-installed --report download/resolution.json -r requirements.txt` using a CPython interpreter that has pip.

- [ ] **Step 2: Derive pinned requirements from the report**

Write one normalized `distribution==version` line per resolved package to `download/requirements-offline.txt`.

- [ ] **Step 3: Download only Windows x64 binary distributions**

Run `python -m pip download --only-binary=:all: --dest download/wheels -r download/requirements-offline.txt`.

- [ ] **Step 4: Verify the wheelhouse covers the lock file**

Run a Python unittest that checks every locked package has a matching wheel (or an allowed `py3-none-any` wheel) in `download/wheels`.

### Task 2: Add and test offline installer behavior

**Files:**
- Create: `tests/test_offline_bundle.py`
- Create: `download/install_offline.ps1`
- Create: `download/verify_offline.ps1`

- [ ] **Step 1: Write a failing test for required installer safety checks**

Assert that `install_offline.ps1` requires the bundled installer and wheelhouse, uses `--no-index`, creates `.venv` only when absent, and runs `pip check`.

- [ ] **Step 2: Run the test and confirm it fails because the scripts are absent**

Run `python -m unittest tests.test_offline_bundle -v`.

- [ ] **Step 3: Add the minimal PowerShell scripts**

Implement per-user CPython installation, SHA-256 verification, venv creation, offline pip installation, and post-install verification without contacting a package index.

- [ ] **Step 4: Re-run the installer-content test**

Run `python -m unittest tests.test_offline_bundle -v` and confirm pass.

### Task 3: Package runtime, manifests, and validate offline installation

**Files:**
- Create: `download/python/python-3.12.x-amd64.exe`
- Create: `download/SHA256SUMS.txt`
- Create: `download/bundle-manifest.json`
- Create: `download/README.md`

- [ ] **Step 1: Download the official CPython 3.12 x64 installer**

Use Python.org's release artifact and save it under `download/python`.

- [ ] **Step 2: Generate integrity metadata**

Compute SHA-256 values for every regular bundle file and record package/runtime metadata in `bundle-manifest.json`.

- [ ] **Step 3: Verify in an isolated venv with no index**

Create a temporary verification venv using the bundled-compatible CPython 3.12, install with `--no-index --find-links download/wheels -r download/requirements-offline.txt`, and run `pip check` plus imports needed by the application.

- [ ] **Step 4: Document the transfer and execution sequence**

Explain that `download` and `local_models` must both be copied to the isolated Windows host; document the commands and the need to set LLM connection configuration separately.

### Task 4: Final validation and handoff

**Files:**
- Verify: `requirements.txt`
- Verify: `download/*`
- Verify: `tests/test_offline_bundle.py`

- [ ] **Step 1: Run the full project test suite**

Run `.venv\\Scripts\\python.exe -m unittest discover -s tests -v`.

- [ ] **Step 2: Check the download manifest and SHA-256 verification**

Run `download\\verify_offline.ps1 -BundleRoot download -SkipImportCheck`.

- [ ] **Step 3: Inspect Git scope before commit**

Ensure only offline scripts, documentation, tests, and intended source changes are staged; do not stage the user-owned root files or large binary artifact directory.
