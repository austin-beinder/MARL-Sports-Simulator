# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-11-26
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-11-26: Generated this file: a restartable run queue for scripts/train.py
#     (a worker pool over a plain-text queue file that is re-read as slots free
#     up, --init-from dependencies, done markers, a status file) (backlog task 16).
# ==============================================================================
"""Run a queue of ``scripts/train.py`` jobs, a few at a time, until the queue is empty.

The queue is a plain-text file, one job per line (blank lines and ``#`` comments are skipped)::

    runs/task16/keepaway_kick_2v1_pybullet_seed0 | --config configs/keepaway_1v1.toml --seed 0 --set shared.n_blue=2
    @workers 8

* ``<out_dir> | <train.py arguments>``: the runner adds ``--out-dir <out_dir> --overwrite`` itself.
* ``@workers N`` sets how many jobs run at once (the last such line wins; ``--workers`` is the default).

Jobs start in file order. The file is re-read every poll, so lines can be appended (or a pending
job's arguments edited, or ``@workers`` changed) while the queue runs.

* **Done** means train.py exited 0 and left ``checkpoints/final.pt`` and ``eval.csv``; the runner then
  writes ``<out_dir>/QUEUE_DONE``. Done jobs are skipped, so after a crash or reboot just start the
  queue again: finished runs stay, unfinished ones restart from scratch (train.py can't resume).
* **Dependencies:** a job whose arguments hold ``--init-from X`` waits until X exists and the queued
  job whose out_dir contains X (if any) is done. If that job failed, the job is ``blocked``.
* **Failures** don't stop the queue. They are listed in the status file with their log; a later
  start of the queue retries them.
* Logs: ``<queue file's folder>/logs/<out_dir name>.log``. Status: ``<queue file>.status.txt``,
  rewritten whenever something changes.

Run it from the repo root with the venv active::

    python scripts/queue_runs.py runs/task16/queue.txt --workers 10
    python scripts/queue_runs.py runs/task16/queue.txt --dry-run     # parse it and show each job's state

A queue started from an editor's terminal dies with the editor; for a long queue start it in its own
window (PowerShell: ``Start-Process``).
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TextIO

REPO = Path(__file__).resolve().parent.parent
DONE_MARKER = "QUEUE_DONE"


@dataclass
class Job:
    """One queue line: where the run goes and the train.py arguments that make it."""

    out_dir: str
    args: list[str]
    state: str = "pending"  # pending | waiting | running | done | failed | blocked
    note: str = ""
    proc: subprocess.Popen | None = field(default=None, repr=False)
    log: TextIO | None = field(default=None, repr=False)
    started: float = 0.0
    wall: float = 0.0

    @property
    def name(self) -> str:
        return Path(self.out_dir).name

    @property
    def init_from(self) -> str | None:
        """The checkpoint this job warm-starts from (``--init-from X`` or ``--init-from=X``)."""
        for k, arg in enumerate(self.args):
            if arg == "--init-from" and k + 1 < len(self.args):
                return self.args[k + 1]

            if arg.startswith("--init-from="):
                return arg.split("=", 1)[1]

        return None


def parse_queue(path: Path) -> tuple[list[tuple[str, list[str]]], int | None]:
    """The queue file's jobs as (out_dir, args), in order, plus its ``@workers`` value (or None)."""
    jobs: list[tuple[str, list[str]]] = []
    workers = None
    seen: dict[str, int] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("@workers"):
            workers = int(line.split()[1])
            if workers < 1:
                raise ValueError(f"{path}:{lineno}: @workers must be >= 1")
            continue

        out_dir, sep, rest = line.partition("|")
        out_dir = out_dir.strip().replace("\\", "/")
        if not sep or not out_dir:
            raise ValueError(f"{path}:{lineno}: expected '<out_dir> | <train.py arguments>'")

        args = shlex.split(rest)
        if any(a in ("--out-dir", "--overwrite") or a.startswith("--out-dir=") for a in args):
            raise ValueError(f"{path}:{lineno}: leave out --out-dir/--overwrite; the runner adds them")

        name = Path(out_dir).name
        if name in seen:
            raise ValueError(f"{path}:{lineno}: run name {name!r} already used on line {seen[name]} "
                             "(log files are named after it)")
        seen[name] = lineno
        jobs.append((out_dir, args))

    return jobs, workers


def is_done(out_dir: str) -> bool:
    return (REPO / out_dir / DONE_MARKER).exists()


def outputs_ok(out_dir: str) -> str:
    """'' if the run left what a finished run leaves, else what's missing."""
    run = REPO / out_dir
    missing = [p for p in ("checkpoints/final.pt", "eval.csv") if not (run / p).exists()]
    return f"missing {', '.join(missing)}" if missing else ""


class Queue:
    """The worker pool: keeps up to ``workers`` jobs running, in queue order, until none are left."""

    def __init__(self, path: Path, workers: int, poll: float):
        self.path = path
        self.default_workers = workers
        self.workers = workers
        self.poll = poll
        self.jobs: dict[str, Job] = {}  # by out_dir, in queue order
        self.log_dir = path.parent / "logs"
        self.status_path = path.with_name(path.name + ".status.txt")
        self._last_status = ""

    def reload(self) -> None:
        """Re-read the queue file; keep the state of jobs already known. A bad edit keeps the old queue."""
        try:
            entries, workers = parse_queue(self.path)
        except (OSError, ValueError) as exc:
            print(f"[queue] couldn't re-read {self.path} ({exc}); keeping the previous queue", flush=True)
            return

        self.workers = workers or self.default_workers
        old, self.jobs = self.jobs, {}
        for out_dir, args in entries:
            job = old.pop(out_dir, None) or Job(out_dir, args)
            if job.state in ("pending", "waiting", "blocked"):
                job.args = args  # a pending job picks up edits
            if job.state == "pending" and is_done(out_dir):
                job.state, job.note = "done", "done before this start"
            self.jobs[out_dir] = job

        for job in old.values():  # a running job removed from the file still runs to the end
            if job.state == "running":
                self.jobs[job.out_dir] = job

    def producer(self, checkpoint: str) -> Job | None:
        """The queued job whose out_dir holds ``checkpoint``, if any."""
        ckpt = Path(checkpoint.replace("\\", "/"))
        for job in self.jobs.values():
            if Path(job.out_dir) in ckpt.parents:
                return job

        return None

    def readiness(self, job: Job) -> str:
        """'ready', 'waiting' or 'blocked' for a pending job, from its --init-from dependency."""
        ckpt = job.init_from
        if ckpt is None:
            return "ready"

        parent = self.producer(ckpt)
        if parent is not None:
            if parent.state == "done" and (REPO / ckpt).exists():
                return "ready"

            if parent.state in ("failed", "blocked"):
                job.note = f"needs {parent.name}, which {parent.state}"
                return "blocked"

            job.note = f"waiting for {parent.name}"
            return "waiting"

        if (REPO / ckpt).exists():
            return "ready"

        job.note = f"{ckpt} doesn't exist and no queued job makes it"
        return "blocked"

    def start(self, job: Job) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        job.log = log = open(self.log_dir / f"{job.name}.log", "w", encoding="utf-8")  # pylint: disable=consider-using-with
        cmd = [sys.executable, "scripts/train.py", *job.args, "--out-dir", job.out_dir, "--overwrite"]
        log.write(f"[queue] {datetime.now():%Y-%m-%d %H:%M:%S} {shlex.join(cmd)}\n")
        log.flush()
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        # The pool keeps the process (and its log) open across polls; reap() and stop() close them.
        job.proc = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,  # pylint: disable=consider-using-with
                                    env=env)
        job.state, job.note, job.started = "running", "", time.time()
        print(f"[queue] start  {job.name}", flush=True)

    def reap(self, job: Job) -> bool:
        """Finish a running job whose process has exited; True if it had."""
        rc = job.proc.poll()
        if rc is None:
            return False

        job.log.close()
        job.wall = time.time() - job.started
        problem = f"exit code {rc}" if rc != 0 else outputs_ok(job.out_dir)
        if problem:
            job.state, job.note = "failed", f"{problem}; log {self.log_dir / (job.name + '.log')}"
        else:
            (REPO / job.out_dir / DONE_MARKER).write_text(f"{datetime.now().isoformat()}\n", encoding="utf-8")
            job.state = "done"

        job.proc = None
        print(f"[queue] {job.state:6s} {job.name} ({job.wall / 60:.1f} min){'  ' + job.note if problem else ''}",
              flush=True)
        return True

    def step(self) -> None:
        """Reap finished jobs, re-read the queue, settle waiting jobs and fill free slots."""
        for job in self.jobs.values():
            if job.state == "running":
                self.reap(job)

        self.reload()
        running = sum(j.state == "running" for j in self.jobs.values())
        for job in self.jobs.values():
            if job.state not in ("pending", "waiting", "blocked"):
                continue

            job.state = "pending"
            ready = self.readiness(job)
            if ready != "ready":
                job.state = ready
                continue

            if running < self.workers:
                self.start(job)
                running += 1

        self.write_status()

    def active(self) -> bool:
        return any(j.state in ("pending", "waiting", "running") for j in self.jobs.values())

    def write_status(self) -> None:
        counts: dict[str, int] = {}
        for job in self.jobs.values():
            counts[job.state] = counts.get(job.state, 0) + 1

        lines = [f"queue {self.path}  workers {self.workers}  "
                 + "  ".join(f"{k} {v}" for k, v in sorted(counts.items()))]
        for job in self.jobs.values():
            if job.state == "running":
                when = f"{(time.time() - job.started) / 60:5.1f} min so far"
            elif job.wall:
                when = f"{job.wall / 60:5.1f} min"
            else:
                when = ""
            lines.append(f"{job.state:8s} {when:>18s}  {job.out_dir}  {job.note}".rstrip())

        body = "\n".join(lines)
        if body != self._last_status:  # the clock line below would always differ
            self._last_status = body
            self.status_path.write_text(f"updated {datetime.now():%Y-%m-%d %H:%M:%S}\n{body}\n", encoding="utf-8")

    def stop(self) -> None:
        """Ctrl-C: end the running jobs (they restart from scratch next time)."""
        for job in self.jobs.values():
            if job.state == "running":
                job.proc.terminate()
                job.proc.wait()
                job.log.close()
                job.state, job.note, job.proc = "pending", "stopped by Ctrl-C", None

        self.write_status()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run a queue of train.py jobs a few at a time (see the docstring).")
    ap.add_argument("queue", type=Path, help="the queue file: '<out_dir> | <train.py arguments>' per line")
    ap.add_argument("--workers", type=int, default=10, help="jobs at once unless the file says @workers N")
    ap.add_argument("--poll", type=float, default=15.0, help="seconds between checks")
    ap.add_argument("--dry-run", action="store_true", help="parse the queue, show each job's state, run nothing")
    args = ap.parse_args(argv)

    queue = Queue(args.queue.resolve(), args.workers, args.poll)
    queue.reload()
    if args.dry_run:
        for job in queue.jobs.values():
            if job.state == "pending":
                job.state = queue.readiness(job)
            print(f"{job.state:8s} {job.out_dir}  {job.note}".rstrip())
            print(f"         train.py {shlex.join(job.args)}")

        print(f"[queue] {len(queue.jobs)} jobs, workers {queue.workers}")
        return 0

    print(f"[queue] {len(queue.jobs)} jobs in {args.queue}, workers {queue.workers}; "
          f"status in {queue.status_path}", flush=True)
    try:
        while True:
            queue.step()
            if not queue.active():
                break

            time.sleep(args.poll)

    except KeyboardInterrupt:
        print("[queue] Ctrl-C: stopping the running jobs", flush=True)
        queue.stop()
        return 130

    failed = [j for j in queue.jobs.values() if j.state in ("failed", "blocked")]
    print(f"[queue] finished: {sum(j.state == 'done' for j in queue.jobs.values())} done, "
          f"{len(failed)} failed or blocked; see {queue.status_path}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
