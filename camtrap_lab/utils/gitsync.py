"""Commit + push results to GitHub in real time (per video)."""
from __future__ import annotations
import os, subprocess

class GitSync:
    def __init__(self, repo_dir, branch="main", remote="origin",
                 author_name="colab-bot", author_email="colab@example.com", enabled=True):
        self.repo_dir = repo_dir
        self.branch = branch
        self.remote = remote
        self.enabled = bool(enabled) and self._is_repo()
        if self.enabled:
            self._run(["git", "config", "user.name", author_name])
            self._run(["git", "config", "user.email", author_email])

    def _is_repo(self):
        return os.path.isdir(os.path.join(self.repo_dir, ".git"))

    def _run(self, args):
        return subprocess.run(args, cwd=self.repo_dir, capture_output=True, text=True)

    def commit_push(self, message, paths=None):
        if not self.enabled:
            return False
        self._run(["git", "add"] + (paths or ["-A"]))
        r = self._run(["git", "commit", "-m", message])
        if "nothing to commit" in (r.stdout + r.stderr).lower():
            return False
        p = self._run(["git", "push", self.remote, self.branch])
        return p.returncode == 0
