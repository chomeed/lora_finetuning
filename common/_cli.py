"""Tiny CLI helpers shared by the console-script entrypoints.

draccus derives option names straight from dataclass field names, so it only
accepts the underscore form (``--num_demos``). The console-script shortcuts in
``pyproject.toml`` are documented with hyphens (``--num-demos``), which is the
more natural CLI convention, so we normalize hyphenated long options to their
underscore form before draccus parses argv. Both spellings then work.
"""

from __future__ import annotations

import sys


def normalize_hyphen_flags(argv: list[str] | None = None) -> list[str]:
    """Rewrite ``--foo-bar`` long options to ``--foo_bar`` in place.

    Only the option *name* is touched (the part before an ``=``); values keep
    any hyphens they carry, and single-dash tokens (e.g. negative numbers) are
    left alone. Mutates and returns ``argv`` (defaults to ``sys.argv``).
    """
    argv = sys.argv if argv is None else argv
    for i, tok in enumerate(argv):
        if not tok.startswith("--"):
            continue
        # Only rewrite the option NAME, keeping the leading "--" intact.
        if "=" in tok:
            name, val = tok.split("=", 1)
            argv[i] = "--" + name[2:].replace("-", "_") + "=" + val
        else:
            argv[i] = "--" + tok[2:].replace("-", "_")
    return argv
