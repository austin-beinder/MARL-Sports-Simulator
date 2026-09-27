This document was written in its entirety by Claude.

# Updating Requirements

Dependencies are managed with [pip-tools](https://github.com/jazzband/pip-tools) inside a project virtual environment (`.venv`).

| File | Edited by | Purpose |
| --- | --- | --- |
| `requirements.in` | You | Top-level packages the project imports (e.g. `numpy~=2.5.3`) |
| `requirements.txt` | `pip-compile` | Exact pinned versions of everything, generated from `requirements.in` |
| `requirements-dev.in` | You | Tooling used to manage requirements (`pip-tools`, `pipreqs`) |
| `requirements-dev.txt` | `pip-compile` | Pinned versions of the tooling |

Never edit the `.txt` files by hand. Commit all four files.

All commands below are run from the repo root in PowerShell.

## First-time setup

1. Create the virtual environment (Python 3.13):

```shell
py -3.13 -m venv .venv
```

2. Activate it (your prompt should then start with `(.venv)`):

```shell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks the script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again. In Git Bash, use `source .venv/Scripts/activate` instead.

3. Install the tooling, then sync the environment:

```shell
python -m pip install pip-tools
```

```shell
pip-sync requirements.txt requirements-dev.txt
```

4. In VS Code, run **Python: Select Interpreter** and choose `.\.venv\Scripts\python.exe`.

## Updating requirements

Activate the venv first (`.venv\Scripts\Activate.ps1`).

1. Update `requirements.in`. Either edit it by hand, or regenerate it from the project's imports:

```shell
pipreqs . --savepath requirements.in --force --mode compat --ignore .venv,__pycache__,.git
```

`pipreqs` overwrites the file, so review the result with `git diff requirements.in` before continuing. It can guess the wrong pip package name when it differs from the import name (e.g. `import cv2` is `opencv-python`).

2. Regenerate the pinned lock files:

```shell
pip-compile --strip-extras requirements.in
```

```shell
pip-compile --strip-extras requirements-dev.in
```

Add `--upgrade` to either command to bump pins to the newest versions allowed by the `.in` file.

3. Sync the venv to exactly match the lock files:

```shell
pip-sync requirements.txt requirements-dev.txt
```

Always pass **both** files. `pip-sync` uninstalls anything not listed, so syncing only `requirements.txt` would remove `pipreqs` and pip-tools' own dependencies.

4. Run the tests to confirm nothing broke:

```shell
python -m unittest discover -s pybullet/tests -p "*_test.py" -t pybullet
```

5. Commit `requirements.in`, `requirements.txt`, `requirements-dev.in`, and `requirements-dev.txt`.

## Pulling someone else's requirement changes

With the venv activated:

```shell
pip-sync requirements.txt requirements-dev.txt
```

## Gotchas

- **Do not pass `--no-index` to `pip-compile`.** In pip-tools 7.x it means "don't use PyPI" and resolution will fail. pip-tools 7.6 still prints `--no-index` in the generated file header even when it was not passed; ignore it and use the commands above.
- **Python version matters.** The lock files are generated with Python 3.13. Everyone should create their venv with the same version, or `pip-compile` may resolve different pins.
- **Check you're in the venv** with `python -c "import sys; print(sys.executable)"`; it should end in `\.venv\Scripts\python.exe`.
