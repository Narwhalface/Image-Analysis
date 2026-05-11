"""Thin root-level wrapper so group members can run testing from the repo root."""

import sys

from vi2026_pythonpackage.test_classifier import app


if __name__ == "__main__":
    sys.exit(app())