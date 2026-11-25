"""Reusable Fail-Closed Non-NDA Zero-IP-Leak Verifier (`scripts/verify_non_nda_ip_clean.py`).

Scans text, HTML/DOM, JSON responses, and exported compliance certificates against 56
forbidden internal math, algorithm, model-architecture, and code-identifier patterns.

Usage:
  python scripts/verify_non_nda_ip_clean.py [file1 file2 ...]
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import List, Sequence, Tuple

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_NON_NDA_FILES: Tuple[pathlib.Path, ...] = (
    REPO_ROOT / "docs" / "research" / "external-investor-teaser-non-nda.md",
    REPO_ROOT / "docs" / "research" / "external-investor-teaser-deck-non-nda.html",
)

FORBIDDEN_METHOD_PATTERNS: Tuple[str, ...] = (
    r"\bstiefel\b",
    r"\bprocrustes\b",
    r"zero[-_]?mu",
    r"\bsubtractive\b",
    r"\bcentroid\b",
    r"anisotrop",
    r"\bcsls\b",
    r"\bhubness\b",
    r"\bgeodesic\b",
    r"\bpolar\b",
    r"whiten",
    r"\bsvd\b",
    r"\beigen",
    r"\bcumulant",
    r"\bmarchenko\b",
    r"\bhouseholder\b",
    r"\bumeyama\b",
    r"\bek5\b",
    r"\bwfs\b",
    r"well[-_ ]founded",
    r"\bstratif",
    r"\bstratum\b",
    r"cites_audit",
    r"derives_current",
    r"derives_snapshot",
    r"copy_of",
    r"\bfixpoint\b",
    r"\bgrpo\b",
    r"\brlvr\b",
    r"\bsft\b",
    r"\blora\b",
    r"\bpeft\b",
    r"bge[-_]?m3",
    r"\bminilm\b",
    r"\bnemotron\b",
    r"\bqwen\b",
    r"cross[-_ ]encoder",
    r"\brerank",
    r"\bbm25\b",
    r"tra[-_]?1",
    r"\ba2c\b",
    r"\ba2g\b",
    r"\ba3\b",
    r"\bem-[1-5]\b",
    r"\bbitset",
    r"8\s*b/doc",
    r"8-byte",
    r"4-byte",
    r"16\s*kb",
    r"\bshortlist\b",
    r"oversampl",
    r"\befsearch\b",
    r"\bef_eff\b",
    r"\bk_os\b",
    r"spherical-migration",
    r"epistemic_grpo",
)


def scan_text_for_ip_leaks(text: str, source_label: str = "<memory>") -> List[str]:
    """Return list of human-readable leak descriptions (`[]` when 100% clean)."""
    leaks: List[str] = []
    for pat in FORBIDDEN_METHOD_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            leaks.append(f"{source_label}: matched '{pat}' -> '{m.group(0)}'")
    return leaks


def assert_non_nda_clean(text: str, source_label: str = "<memory>") -> None:
    """Fail-closed guard raising `ValueError` if any forbidden internal term appears."""
    leaks = scan_text_for_ip_leaks(text, source_label=source_label)
    if leaks:
        raise ValueError(f"Non-NDA IP leak detected ({len(leaks)} matches): {leaks[:5]}")


def verify_files(paths: Sequence[pathlib.Path]) -> Tuple[int, List[str]]:
    all_leaks: List[str] = []
    for p in paths:
        txt = p.read_text(encoding="utf-8")
        all_leaks.extend(scan_text_for_ip_leaks(txt, source_label=p.name))
    return len(FORBIDDEN_METHOD_PATTERNS), all_leaks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify Non-NDA files contain zero internal IP method hints.")
    parser.add_argument("files", nargs="*", help="Paths to check (defaults to Non-NDA teaser MD + HTML)")
    args = parser.parse_args(argv)
    targets = [pathlib.Path(f) for f in args.files] if args.files else list(DEFAULT_NON_NDA_FILES)
    n_pats, leaks = verify_files(targets)
    if leaks:
        print(f"IP_LEAK_DETECTED ({len(leaks)} matches):", leaks, file=sys.stderr)
        return 1
    print(f"ZERO_IP_LEAK_AUDIT: PASS (0/{n_pats} forbidden method patterns matched across {len(targets)} file(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
