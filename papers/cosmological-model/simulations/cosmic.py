"""Unified launcher for the three cosmological simulation backends.

Examples
--------
python cosmic.py statistical --scales 6,12,20
python cosmic.py dicke --scales 6,12,20
python cosmic.py wavefunction --scales 6,12,20
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


MODES = {
    "statistical": "cosmic_d.py",
    "dicke": "cosmic_psi.py",
    "wavefunction": "cosmic_wavefunction.py",
    "general": "cosmic_wavefunction.py",
}


def _usage() -> str:
    return (
        "Usage: python cosmic.py {statistical|dicke|wavefunction} [backend options]\n\n"
        "  statistical  Existing classical relaxing-bitstring backend\n"
        "  dicke        Existing fixed-sector Dicke backend\n"
        "  wavefunction New general complex state with exact Born statistical shadow\n"
    )


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(_usage())
        return
    mode = sys.argv[1].lower()
    if mode not in MODES:
        print(f"Unknown mode: {mode}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        raise SystemExit(2)
    directory = Path(__file__).resolve().parent
    target = directory / MODES[mode]
    if not target.exists():
        raise SystemExit(f"Backend script is missing: {target}")
    os.execv(sys.executable, [sys.executable, str(target), *sys.argv[2:]])


if __name__ == "__main__":
    main()
