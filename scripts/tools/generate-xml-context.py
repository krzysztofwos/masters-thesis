#!/usr/bin/env python3
"""
Generate a single XML document containing text files from a repository.

This script creates a shareable "context bundle" for LLM-assisted reviews.
All repository-specific configuration is done via command-line flags.

Features:
- Multiple language/file-type support (python, rust, markdown, config, tex, etc.)
- Optional profiles for convenient language groupings
- Include/exclude directory names and glob patterns
- Git-tracked mode (respects .gitignore)
- Safe handling of large files via truncation
- Optional JSON manifest output
- Optional splitting into multiple XML parts

XML structure:

  <codebase root="..." generated_at="..." languages="...">
    <summary file_count="..." included_bytes="..." truncated_files="..." skipped_files="..."/>
    <files>
      <file path="relative/path" bytes_disk="..." sha256_raw="..." sha256_included="..."
            truncated="false" included_bytes="...">
        <code><![CDATA[ ... ]]></code>
      </file>
      ...
    </files>
  </codebase>

Usage examples:

  # Default: Python files only
  python generate-xml-context.py -o context.xml

  # Multiple languages
  python generate-xml-context.py --language rust --language python -o code.xml

  # Use a profile (preset language grouping)
  python generate-xml-context.py --profile code -o code.xml
  python generate-xml-context.py --profile docs -o docs.xml
  python generate-xml-context.py --profile full -o full.xml

  # Exclude specific directories
  python generate-xml-context.py --profile full --exclude-dir node_modules --exclude-dir data -o context.xml

  # Use git to respect .gitignore
  python generate-xml-context.py --profile full --git-tracked -o context.xml

  # Truncate large files and split output
  python generate-xml-context.py --profile full --max-file-bytes 100000 --split-bytes 5000000 -o context.xml
"""

from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import hashlib
import json
import os
import platform
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from xml.sax.saxutils import quoteattr

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Common directories to exclude by default (build artifacts, caches, VCS)
DEFAULT_EXCLUDED_DIRS: Set[str] = {
    ".eggs",
    ".git",
    ".hg",
    ".idea",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".trunk",
    ".venv",
    ".vscode",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "site-packages",
    "target",
    "venv",
}

# If a file starts with NUL bytes, treat it as binary and do not inline it.
BINARY_SNIFF_BYTES = 4096


# ---------------------------------------------------------------------------
# File-type configuration
# ---------------------------------------------------------------------------

ReaderFn = Callable[[Path], str]


def _read_utf8_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_python_text(path: Path) -> str:
    """Read Python source honoring PEP 263 encoding declarations."""
    try:
        with tokenize.open(str(path)) as f:  # type: ignore[arg-type]
            return f.read()
    except Exception:
        return _read_utf8_text(path)


@dataclass(frozen=True)
class LanguageSpec:
    suffixes: Tuple[str, ...]
    basenames: Tuple[str, ...]  # Files matched by exact name (e.g., Makefile)
    reader: ReaderFn


LANGUAGE_CONFIGS: Dict[str, LanguageSpec] = {
    "python": LanguageSpec(suffixes=(".py",), basenames=(), reader=_read_python_text),
    "rust": LanguageSpec(suffixes=(".rs",), basenames=(), reader=_read_utf8_text),
    "quarto": LanguageSpec(suffixes=(".qmd",), basenames=(), reader=_read_utf8_text),
    "markdown": LanguageSpec(
        suffixes=(".md", ".markdown"),
        basenames=("README", "LICENSE"),
        reader=_read_utf8_text,
    ),
    "config": LanguageSpec(
        suffixes=(".yml", ".yaml", ".toml", ".lock", ".ini", ".cfg", ".json", ".jsonl"),
        basenames=(
            "Makefile",
            "Justfile",
            "Cargo.toml",
            "Cargo.lock",
            "pyproject.toml",
            "requirements.txt",
            "Pipfile",
            "Pipfile.lock",
            "poetry.lock",
        ),
        reader=_read_utf8_text,
    ),
    "bib": LanguageSpec(suffixes=(".bib",), basenames=(), reader=_read_utf8_text),
    "tex": LanguageSpec(
        suffixes=(".tex", ".cls", ".sty", ".bst"), basenames=(), reader=_read_utf8_text
    ),
    "shell": LanguageSpec(
        suffixes=(".sh", ".bash", ".zsh"), basenames=(), reader=_read_utf8_text
    ),
    "text": LanguageSpec(suffixes=(".txt",), basenames=(), reader=_read_utf8_text),
}

# Profiles are convenience presets for common use cases
PROFILES: Dict[str, List[str]] = {
    "code": ["python", "rust", "quarto"],
    "docs": ["markdown", "config", "bib", "tex"],
    "full": sorted(LANGUAGE_CONFIGS.keys()),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_excluded_dir(name: str, excluded: Set[str], included: Set[str]) -> bool:
    if name in included:
        return False
    return name in excluded


def _posix_relpath(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _cdata_wrap(text: str) -> str:
    """Wrap text in CDATA, safely handling embedded ']]>'."""
    if not text:
        return "<![CDATA[]]>"
    safe = text.replace("]]>", "]]]]><![CDATA[>")
    return f"<![CDATA[{safe}]]>"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sniff_binary(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(BINARY_SNIFF_BYTES)
        return b"\x00" in head
    except Exception:
        return True


def _truncate_bytes(data: bytes, max_bytes: int) -> Tuple[bytes, bool, int]:
    """Return (bytes_to_include, truncated?, omitted_bytes)."""
    if max_bytes <= 0 or len(data) <= max_bytes:
        return data, False, 0
    head_n = max_bytes // 2
    tail_n = max_bytes - head_n
    head = data[:head_n]
    tail = data[-tail_n:] if tail_n > 0 else b""
    omitted = max(0, len(data) - len(head) - len(tail))
    marker = f"\n\n<<<TRUNCATED: omitted {omitted} bytes>>>\n\n".encode("utf-8")
    return head + marker + tail, True, omitted


def _read_included_text(
    path: Path, max_file_bytes: int, reader: ReaderFn
) -> Tuple[str, bool]:
    """Read file content, truncating by raw byte size if requested."""
    if max_file_bytes <= 0:
        return reader(path), False

    raw = path.read_bytes()
    clipped, truncated, _ = _truncate_bytes(raw, max_file_bytes)

    # Python: attempt to respect encoding when truncating
    if path.suffix == ".py":
        try:
            from io import BytesIO

            bio = BytesIO(clipped)
            encoding, _ = tokenize.detect_encoding(bio.readline)
            return clipped.decode(encoding, errors="replace"), truncated
        except Exception:
            return clipped.decode("utf-8", errors="replace"), truncated

    return clipped.decode("utf-8", errors="replace"), truncated


def _matches_any_glob(rel_posix: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(rel_posix, pat) for pat in patterns)


def _git_ls_files(root: Path, include_untracked: bool = False) -> List[Path]:
    """Return a list of Paths (relative to root) using git ls-files."""
    args = ["git", "-C", str(root), "ls-files", "-z"]
    if include_untracked:
        args = [
            "git",
            "-C",
            str(root),
            "ls-files",
            "-z",
            "--others",
            "--cached",
            "--exclude-standard",
        ]
    try:
        res = subprocess.run(
            args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except Exception:
        return []
    data = res.stdout
    rels = [p for p in data.split(b"\x00") if p]
    return [Path(p.decode("utf-8", errors="replace")) for p in rels]


# ---------------------------------------------------------------------------
# Discovery + inclusion
# ---------------------------------------------------------------------------


@dataclass
class FileRecord:
    rel_posix: str
    abs_path: Path
    bytes_disk: int
    sha256_raw: str
    sha256_included: Optional[str]
    truncated: bool
    skipped: bool
    skip_reason: Optional[str]


def iter_candidate_files_walk(
    root: Path, excluded_dirs: Set[str], included_dirs: Set[str]
) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames if not _is_excluded_dir(d, excluded_dirs, included_dirs)
        ]
        for fn in filenames:
            yield Path(dirpath) / fn


def iter_candidate_files_git(
    root: Path,
    excluded_dirs: Set[str],
    included_dirs: Set[str],
    include_untracked: bool,
) -> Iterable[Path]:
    rels = _git_ls_files(root, include_untracked=include_untracked)
    if not rels:
        return []
    out: List[Path] = []
    for rel in rels:
        parts = rel.parts[:-1]
        if any(_is_excluded_dir(p, excluded_dirs, included_dirs) for p in parts):
            continue
        out.append(root / rel)
    return out


def select_files(
    root: Path,
    excluded_dirs: Set[str],
    included_dirs: Set[str],
    suffixes: Set[str],
    basenames: Set[str],
    include_globs: Sequence[str],
    exclude_globs: Sequence[str],
    use_git: bool,
    include_untracked: bool,
) -> List[Path]:
    if use_git:
        candidates = list(
            iter_candidate_files_git(
                root, excluded_dirs, included_dirs, include_untracked
            )
        )
        if not candidates:
            candidates = list(
                iter_candidate_files_walk(root, excluded_dirs, included_dirs)
            )
    else:
        candidates = list(iter_candidate_files_walk(root, excluded_dirs, included_dirs))

    selected: List[Path] = []
    for p in candidates:
        if not p.is_file():
            continue
        rel_posix = _posix_relpath(p, root)

        if exclude_globs and _matches_any_glob(rel_posix, exclude_globs):
            continue

        suffix_ok = p.suffix in suffixes if p.suffix else False
        basename_ok = p.name in basenames
        glob_ok = (
            _matches_any_glob(rel_posix, include_globs) if include_globs else False
        )

        if include_globs:
            if not (glob_ok or suffix_ok or basename_ok):
                continue
        else:
            if not (suffix_ok or basename_ok):
                continue

        selected.append(p)

    selected.sort(key=lambda x: _posix_relpath(x, root).lower())
    return selected


# ---------------------------------------------------------------------------
# XML generation
# ---------------------------------------------------------------------------


def file_to_xml_element(rec: FileRecord, included_text: Optional[str]) -> str:
    path_attr = quoteattr(rec.rel_posix)
    bytes_disk_attr = quoteattr(str(rec.bytes_disk))
    sha_raw_attr = quoteattr(rec.sha256_raw)
    truncated_attr = quoteattr("true" if rec.truncated else "false")
    skipped_attr = quoteattr("true" if rec.skipped else "false")
    sha_included_attr = quoteattr(rec.sha256_included or "")

    if rec.skipped or included_text is None:
        reason_attr = quoteattr(rec.skip_reason or "")
        return (
            f"    <file path={path_attr} bytes_disk={bytes_disk_attr} sha256_raw={sha_raw_attr} "
            f"sha256_included={sha_included_attr} truncated={truncated_attr} skipped={skipped_attr} "
            f'skip_reason={reason_attr} included_bytes="0">\n'
            f"    </file>\n"
        )

    included_bytes = len(included_text.encode("utf-8", errors="replace"))
    included_bytes_attr = quoteattr(str(included_bytes))
    cdata = _cdata_wrap(included_text)

    return (
        f"    <file path={path_attr} bytes_disk={bytes_disk_attr} sha256_raw={sha_raw_attr} "
        f"sha256_included={sha_included_attr} truncated={truncated_attr} skipped={skipped_attr} "
        f"included_bytes={included_bytes_attr}>\n"
        f"      <code>\n"
        f"        {cdata}\n"
        f"      </code>\n"
        f"    </file>\n"
    )


def write_xml_part(
    out_path: Optional[Path],
    root_attr_value: str,
    generated_at: str,
    languages: Sequence[str],
    file_elements: Sequence[str],
    summary: Dict[str, int],
) -> None:
    xml = []
    xml.append('<?xml version="1.0" encoding="UTF-8"?>\n')
    xml.append(
        f"<codebase root={quoteattr(root_attr_value)} generated_at={quoteattr(generated_at)} "
        f"languages={quoteattr(','.join(sorted(languages)))}>\n"
    )
    xml.append(
        "  <summary " + " ".join(f'{k}="{v}"' for k, v in summary.items()) + "/>\n"
    )
    xml.append("  <files>\n")
    xml.extend(file_elements)
    xml.append("  </files>\n")
    xml.append("</codebase>\n")

    if out_path is None:
        sys.stdout.write("".join(xml))
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("".join(xml), encoding="utf-8", newline="\n")


def generate(
    root: Path,
    output: Optional[Path],
    languages: Sequence[str],
    excluded_dirs: Set[str],
    included_dirs: Set[str],
    include_globs: Sequence[str],
    exclude_globs: Sequence[str],
    use_git: bool,
    include_untracked: bool,
    max_file_bytes: int,
    split_bytes: int,
    manifest_path: Optional[Path],
    redact_root: bool,
) -> int:
    suffixes: Set[str] = set()
    basenames: Set[str] = set()
    suffix_to_reader: Dict[str, ReaderFn] = {}
    basename_to_reader: Dict[str, ReaderFn] = {}

    for lang in languages:
        spec = LANGUAGE_CONFIGS.get(lang)
        if spec is None:
            print(f"error: unsupported language '{lang}'", file=sys.stderr)
            return 2
        suffixes.update(spec.suffixes)
        basenames.update(spec.basenames)
        for s in spec.suffixes:
            suffix_to_reader[s] = spec.reader
        for n in spec.basenames:
            basename_to_reader[n] = spec.reader

    selected = select_files(
        root=root,
        excluded_dirs=excluded_dirs,
        included_dirs=included_dirs,
        suffixes=suffixes,
        basenames=basenames,
        include_globs=include_globs,
        exclude_globs=exclude_globs,
        use_git=use_git,
        include_untracked=include_untracked,
    )

    generated_at = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    if generated_at.endswith("+00:00"):
        generated_at = generated_at[:-6] + "Z"

    root_attr_value = "REDACTED" if redact_root else str(root.resolve())

    manifest: Dict[str, object] = {
        "generated_at": generated_at,
        "root": root_attr_value,
        "languages": list(sorted(languages)),
        "python": sys.version,
        "platform": platform.platform(),
        "files": [],
        "skipped_files": [],
    }

    def choose_reader(p: Path) -> ReaderFn:
        if p.name in basename_to_reader:
            return basename_to_reader[p.name]
        if p.suffix in suffix_to_reader:
            return suffix_to_reader[p.suffix]
        return _read_utf8_text

    parts: List[Dict[str, object]] = []
    current_elements: List[str] = []
    current_bytes = 0
    part_file_count = 0
    part_included_bytes = 0
    part_truncated_files = 0
    part_skipped_files = 0

    def flush_part(part_index: int) -> None:
        nonlocal current_elements, current_bytes
        nonlocal part_file_count, part_included_bytes, part_truncated_files, part_skipped_files

        if not current_elements:
            return

        summary = {
            "file_count": part_file_count,
            "included_bytes": part_included_bytes,
            "truncated_files": part_truncated_files,
            "skipped_files": part_skipped_files,
        }

        out_path = output
        if split_bytes > 0:
            if output is None:
                raise ValueError("split-bytes requires --output")
            base = output.with_suffix("")
            ext = output.suffix or ".xml"
            out_path = Path(f"{base}.part{part_index:03d}{ext}")

        write_xml_part(
            out_path=out_path,
            root_attr_value=root_attr_value,
            generated_at=generated_at,
            languages=languages,
            file_elements=current_elements,
            summary=summary,
        )

        parts.append(
            {
                "part": part_index,
                "output": str(out_path) if out_path is not None else "<stdout>",
                "summary": summary,
            }
        )

        current_elements = []
        current_bytes = 0
        part_file_count = 0
        part_included_bytes = 0
        part_truncated_files = 0
        part_skipped_files = 0

    part_index = 1

    for p in selected:
        rel_posix = _posix_relpath(p, root)
        try:
            st = p.stat()
            bytes_disk = int(st.st_size)
        except Exception:
            bytes_disk = -1

        sha_raw = _sha256_file(p)

        rec = FileRecord(
            rel_posix=rel_posix,
            abs_path=p,
            bytes_disk=bytes_disk,
            sha256_raw=sha_raw,
            sha256_included=None,
            truncated=False,
            skipped=False,
            skip_reason=None,
        )

        if _sniff_binary(p):
            rec.skipped = True
            rec.skip_reason = "binary_or_unreadable"
            part_skipped_files += 1
            manifest["skipped_files"].append(
                {
                    "path": rel_posix,
                    "bytes_disk": bytes_disk,
                    "sha256_raw": sha_raw,
                    "reason": rec.skip_reason,
                }
            )
            element = file_to_xml_element(rec, included_text=None)
        else:
            reader = choose_reader(p)
            included_text, truncated = _read_included_text(p, max_file_bytes, reader)
            rec.truncated = truncated
            if truncated:
                part_truncated_files += 1

            sha_included = hashlib.sha256(
                included_text.encode("utf-8", errors="replace")
            ).hexdigest()
            rec.sha256_included = sha_included

            included_len = len(included_text.encode("utf-8", errors="replace"))
            part_included_bytes += included_len

            manifest["files"].append(
                {
                    "path": rel_posix,
                    "bytes_disk": bytes_disk,
                    "sha256_raw": sha_raw,
                    "sha256_included": sha_included,
                    "truncated": bool(truncated),
                    "included_bytes": included_len,
                }
            )

            element = file_to_xml_element(rec, included_text=included_text)

        element_bytes = len(element.encode("utf-8", errors="replace"))

        if (
            split_bytes > 0
            and current_elements
            and (current_bytes + element_bytes) > split_bytes
        ):
            flush_part(part_index)
            part_index += 1

        current_elements.append(element)
        current_bytes += element_bytes
        part_file_count += 1

    flush_part(part_index)

    manifest["parts"] = parts

    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8", newline="\n"
        )

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate XML context of repository files for LLM review",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(), help="Root directory to scan"
    )
    parser.add_argument(
        "--output", "-o", type=Path, help="Output XML path (omit to write to stdout)"
    )

    parser.add_argument(
        "--language",
        choices=sorted(LANGUAGE_CONFIGS.keys()),
        action="append",
        dest="languages",
        metavar="LANG",
        help=f"File-type group(s) to include (repeatable). Choices: {', '.join(sorted(LANGUAGE_CONFIGS.keys()))}",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES.keys()),
        help=f"Convenience preset for languages. Choices: {', '.join(f'{k}={v}' for k, v in PROFILES.items())}",
    )

    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        metavar="NAME",
        help="Directory name to exclude (repeatable). Added to default exclusions.",
    )
    parser.add_argument(
        "--include-dir",
        action="append",
        default=[],
        metavar="NAME",
        help="Directory name to force-include even if in default exclusions (repeatable).",
    )

    parser.add_argument(
        "--include-glob",
        action="append",
        default=[],
        metavar="GLOB",
        help="Include files matching glob pattern (POSIX paths, relative to root). Repeatable.",
    )
    parser.add_argument(
        "--exclude-glob",
        action="append",
        default=[],
        metavar="GLOB",
        help="Exclude files matching glob pattern (POSIX paths, relative to root). Repeatable.",
    )

    parser.add_argument(
        "--git-tracked",
        action="store_true",
        help="Use git ls-files for discovery (respects .gitignore).",
    )
    parser.add_argument(
        "--git-include-untracked",
        action="store_true",
        help="With --git-tracked, also include untracked files.",
    )

    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=200_000,
        help="Max bytes per file (head/tail truncation). Set 0 to disable.",
    )
    parser.add_argument(
        "--split-bytes",
        type=int,
        default=0,
        help="Split output into parts of ~N bytes (requires --output).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Write JSON manifest of included/skipped files.",
    )
    parser.add_argument(
        "--redact-root",
        action="store_true",
        help="Redact absolute root path in output (sets root='REDACTED').",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    ns = parse_args(argv or sys.argv[1:])
    root = ns.root.resolve()
    if not root.exists() or not root.is_dir():
        print(
            f"error: root '{root}' does not exist or is not a directory",
            file=sys.stderr,
        )
        return 2

    # Determine languages from --language flags or --profile
    languages = ns.languages
    if not languages:
        if ns.profile:
            languages = PROFILES[ns.profile]
        else:
            languages = ["python"]

    # Build exclusion set
    excluded = set(DEFAULT_EXCLUDED_DIRS)
    excluded.update(ns.exclude_dir or [])

    # Force-include overrides exclusions
    included = set(ns.include_dir or [])
    excluded.difference_update(included)

    if ns.split_bytes > 0 and ns.output is None:
        print("error: --split-bytes requires --output", file=sys.stderr)
        return 2

    return generate(
        root=root,
        output=ns.output,
        languages=languages,
        excluded_dirs=excluded,
        included_dirs=included,
        include_globs=ns.include_glob or [],
        exclude_globs=ns.exclude_glob or [],
        use_git=bool(ns.git_tracked),
        include_untracked=bool(ns.git_include_untracked),
        max_file_bytes=int(ns.max_file_bytes),
        split_bytes=int(ns.split_bytes),
        manifest_path=ns.manifest,
        redact_root=bool(ns.redact_root),
    )


if __name__ == "__main__":
    raise SystemExit(main())
