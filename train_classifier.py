"""Thin root-level wrapper so group members can run training from the repo root."""

import sys

from vi2026_pythonpackage.train_classifier import app


if __name__ == "__main__":
    sys.exit(app())