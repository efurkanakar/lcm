import json
import os
from pathlib import Path


_DEFAULT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_FILE = _DEFAULT_ROOT / "download_settings.json"  # Example (Linux/macOS): {"download_dir": "~/Dropbox/PCEB_downloads"}, (Windows): {"download_dir": "%USERPROFILE%\\Dropbox\\PCEB_downloads"}
_USER_CONFIG = Path.home() / ".lcm_download_settings.json"  # User-specific config (overrides repo file, keeps Dropbox clean)


def load_download_dir() -> Path:
    """
    Return the target directory for downloads.
    Precedence:
    1) Env var LCM_DOWNLOAD_DIR (expanded)
    2) User config at ~/.lcm_download_settings.json
    3) Repo config download_settings.json
    4) None → let astroquery use its default (mastDownload)
    """
    def resolve_and_create(val: str | None) -> Path | None:
        if not val:
            return None
        try:
            candidate = Path(os.path.expandvars(os.path.expanduser(val))).resolve()
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except Exception:
            return None

    env_dir = resolve_and_create(os.environ.get("LCM_DOWNLOAD_DIR"))
    if env_dir:
        return env_dir

    for cfg in (_USER_CONFIG, _CONFIG_FILE):
        if cfg.exists():
            try:
                data = json.loads(cfg.read_text(encoding="utf-8"))
                cfg_dir = resolve_and_create(data.get("download_dir", ""))
                if cfg_dir:
                    return cfg_dir
            except Exception:
                pass

    return None


def save_download_dir(path: str | None, user: bool = True) -> None:
    """
    Persist the chosen download directory.
    - If path is None: remove user-level config so defaults apply.
    - If path is a valid directory: write to user config (or repo config if user=False).
    """
    target = _USER_CONFIG if user else _CONFIG_FILE

    if path is None:
        if target.exists():
            try:
                target.unlink()
            except OSError:
                pass
        return

    candidate = Path(os.path.expandvars(os.path.expanduser(path))).resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Download directory does not exist: {candidate}")

    payload = {"download_dir": str(candidate)}
    target.write_text(json.dumps(payload), encoding="utf-8")
