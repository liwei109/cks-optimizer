"""HF dataset hub "local cache priority" parsing
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# load_dataset supported data file extensions
# used to confirm snapshot has real data
_DATA_EXTENSIONS = (
    ".parquet", ".json", ".jsonl", ".csv", ".tsv", ".arrow", ".txt", ".xml",
)


def _hub_root() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _has_data_files(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    return any(
        entry.is_file() and entry.suffix.lower() in _DATA_EXTENSIONS
        for entry in directory.iterdir()
    )


def set_hub_offline(flag: bool) -> Callable[[], None]:
    """Set the offline mode of the HF hub cache.
    """
    prev_env = os.environ.get("HF_HUB_OFFLINE")
    prev_consts: dict[str, bool] = {}
    os.environ["HF_HUB_OFFLINE"] = "1" if flag else "0"
    for mod_name, mod in list(sys.modules.items()):
        if isinstance(getattr(mod, "HF_HUB_OFFLINE", None), bool):
            prev_consts[mod_name] = mod.HF_HUB_OFFLINE
            try:
                setattr(mod, "HF_HUB_OFFLINE", flag)
            except Exception:  # pragma: no cover
                pass

    def restore() -> None:
        if prev_env is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev_env
        for mod_name, value in prev_consts.items():
            mod = sys.modules.get(mod_name)
            if mod is not None:
                try:
                    setattr(mod, "HF_HUB_OFFLINE", value)
                except Exception:  # pragma: no cover
                    pass

    return restore


def resolve_hf_dataset_local(name: str, config: str | None = None) -> Path | None:
    """Parse the dataset snapshot in the local hub cache and return a directory that can be loaded directly with load_dataset.

    :param name: hub dataset name('wikitext' bare name or 'org/name'); 
        Local paths just return None (the caller is supposed to use the local path anyway, no need to parse)
    :param config: dataset config name(like 'wikitext-2-raw-v1');
        None to match any subdirectory in the snapshot
    :return: directory like this: ``.../datasets--org--name/snapshots/<rev>/<config>``;
        None if not found
    """
    if not name or not name.strip():
        return None
    name = name.strip()
    if os.path.exists(name):
        return None

    root = _hub_root()
    if "/" in name:
        repo_dirs = [root / f"datasets--{name.replace('/', '--')}"]
    else:
        # bare name redirect to canonical repo
        repo_dirs = sorted(root.glob(f"datasets--*--{name}"))

    for repo_dir in repo_dirs:
        snapshots_dir = repo_dir / "snapshots"
        if not snapshots_dir.is_dir():
            continue
        # latest snapshot first
        for snapshot in sorted(
            snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            if not snapshot.is_dir():
                continue
            if config:
                candidate = snapshot / config
                if _has_data_files(candidate):
                    logger.debug("dataset '%s' local snapshot hit: %s", name, candidate)
                    return candidate
            else:
                for child in sorted(snapshot.iterdir()):
                    if _has_data_files(child):
                        logger.debug("dataset '%s' local snapshot hit: %s", name, child)
                        return child
    return None


def resolve_hf_dataset_revision(name: str, config: str | None = None) -> tuple[str, str] | None:
    """ resolve_hf_dataset_revision: cache hit local snapshot first, fallback to online loading

    :return: (canonical_repo_id, revision_sha); fail to resolve return None
    """
    local = resolve_hf_dataset_local(name, config)
    if local is None:
        return None
    # local = .../datasets--{org}--{repo}/snapshots/<rev>/<config>
    rev = local.parent.name
    repo_dir_name = local.parents[2].name  # datasets--{org}--{repo}
    if not repo_dir_name.startswith("datasets--"):
        return None
    canonical = repo_dir_name[len("datasets--"):].replace("--", "/")
    return canonical, rev


def load_dataset_local_first(name: str, config: str | None = None, **kwargs):
    """load_dataset: cache hit local snapshot first, fallback to online loading

    :param kwargs: passed to load_dataset
    """
    from datasets import load_dataset

    local = resolve_hf_dataset_local(name, config)
    if local is not None:
        try:
            return load_dataset(str(local), **kwargs)
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "local snapshot %s failed to load, fallback to online loading", local, exc
            )
    return load_dataset(name, config or None, **kwargs)
