"""Unified launcher for the three cosmological-model backends.

Examples
--------

    python cosmic.py statistical --n_bits 184
    python cosmic.py dicke --n_bits 184 --parallel
    python cosmic.py history --n_bits 184 --observer gaussian:9

The launcher intentionally executes the existing drivers as separate scripts,
so their command-line interfaces and plots remain unchanged.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BACKENDS = {
    "statistical": "cosmic_d.py",
    "d": "cosmic_d.py",
    "dicke": "cosmic_psi.py",
    "psi": "cosmic_psi.py",
    "history": "cosmic_history.py",
    "quantum-history": "cosmic_history.py",
    "q": "cosmic_history.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the statistical, Dicke, or observer-conditioned quantum-history cosmology backend.",
        add_help=True,
    )
    parser.add_argument("model", choices=tuple(BACKENDS))
    parser.add_argument(
        "backend_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed unchanged to the selected backend.",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    script = base / BACKENDS[args.model]
    if not script.exists():
        raise SystemExit(
            f"Backend script not found: {script}. Place cosmic.py and cosmic_history.py "
            "beside the existing cosmic_d.py and cosmic_psi.py files."
        )

    completed = subprocess.run(
        [sys.executable, str(script), *args.backend_args],
        cwd=base,
        check=False,
    )
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
