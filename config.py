"""Settings: loads .env (no extra package needed) and finds the dataset folder."""
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TODAY_ISO = "2026-10-20"          # the dataset's fixed "today" (README, KB-00.2)
GOOGLE_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

_loaded = False
warnings = []                     # shown by `python run.py doctor`


def _parse_line(line):
    line = line.strip().lstrip("﻿")
    if not line or line.startswith("#") or "=" not in line:
        return None
    key, _, val = line.partition("=")
    key = key.strip()
    val = val.strip()
    if val[:1] in ("'", '"') and val.count(val[0]) >= 2:      # quoted value
        q = val[0]
        val = val[1:val.index(q, 1)]
    else:
        val = re.split(r"\s+#", val, maxsplit=1)[0].strip()   # inline comment
    return key, val


def load_env(force=False):
    """Read .env from the project folder (or the current folder). .env wins."""
    global _loaded
    if _loaded and not force:
        return
    _loaded = True
    for path in (ROOT / ".env", Path.cwd() / ".env"):
        if path.is_file():
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                kv = _parse_line(line)
                if kv and kv[1] != "":
                    os.environ[kv[0]] = kv[1]
            break
    _fix_base_url()


def _fix_base_url():
    url = os.environ.get("LLM_BASE_URL", "")
    if "generativelanguage.googleapis.com" in url:
        if not re.search(r"/v1beta/openai/?$", url):
            warnings.append(f"LLM_BASE_URL looked wrong ({url!r}); using {GOOGLE_OPENAI_URL}")
            os.environ["LLM_BASE_URL"] = GOOGLE_OPENAI_URL
        elif not url.endswith("/"):
            os.environ["LLM_BASE_URL"] = url + "/"


def env(name, default=None):
    load_env()
    return os.environ.get(name, default)


def _is_dataset(p):
    return (p / "kb").is_dir() and (p / "data").is_dir() and (p / "tools").is_dir()


def dataset_dir():
    """Return the extracted Capstone_Project_Dataset_FTTT_2026 folder."""
    load_env()
    candidates = []
    raw = os.environ.get("DATASET_DIR", "").strip().strip('"')
    if raw:
        candidates.append(Path(raw))
    name = "Capstone_Project_Dataset_FTTT_2026"
    candidates += [ROOT / name, ROOT.parent / name, Path.cwd() / name]
    for base in list(candidates):
        if base.is_dir():
            candidates.append(base / name)          # one level of nesting
    for c in candidates:
        if c.is_dir() and _is_dataset(c):
            return c
    raise RuntimeError(
        "Dataset folder not found. Extract Capstone_Project_Dataset_FTTT_2026.zip and put the "
        "folder that contains kb/, data/, tools/ in DATASET_DIR inside your .env file.\n"
        f"Tried: {', '.join(str(c) for c in candidates[:4])}")


def ensure_tools_importable():
    ds = dataset_dir()
    if str(ds) not in sys.path:
        sys.path.insert(0, str(ds))
    return ds
