#!/usr/bin/env python3
"""Upload a trained AEGIS checkpoint to the Hugging Face model repo.

Hosts the model at https://huggingface.co/bragee/AEGIS. Auth comes from the local HF token cache
(`huggingface-cli login`) or the HF_TOKEN env var — the token is NEVER hardcoded or committed.

    python scripts/upload_to_hf.py --run-dir outputs/gcn_baseline_12345
    python scripts/upload_to_hf.py --run-dir outputs/gcn_baseline_12345 --path-in-repo gcn-li-small

Notes for this cluster:
- Outbound HTTPS is TLS-intercepted, so we point Python at the system CA bundle (otherwise
  certificate verification fails). Override with AEGIS_CA_BUNDLE if your bundle lives elsewhere.
- There is no model to upload until a real training run has produced best.pt (see ml/train.py).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# TLS-interception fix: trust the system CA bundle for all outbound HTTPS (see module docstring).
_CA = os.environ.get("AEGIS_CA_BUNDLE", "/etc/ssl/certs/ca-certificates.crt")
if Path(_CA).exists():
    os.environ.setdefault("SSL_CERT_FILE", _CA)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", _CA)

from huggingface_hub import HfApi  # noqa: E402

DEFAULT_REPO = "bragee/AEGIS"
# Files worth publishing from a run dir (skip bulky per-epoch logs by default).
DEFAULT_INCLUDE = ["best.pt", "metrics.json", "run_context.json",
                   "eval_metrics_test.json", "eval_metrics_val.json"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload an AEGIS checkpoint to Hugging Face")
    p.add_argument("--run-dir", required=True, type=Path,
                   help="A train.py --out-dir containing best.pt and metrics")
    p.add_argument("--repo", default=DEFAULT_REPO, help="HF model repo id")
    p.add_argument("--path-in-repo", default=None,
                   help="Subfolder in the repo (default: the run-dir name)")
    p.add_argument("--all-files", action="store_true",
                   help="Upload the entire run dir (default: only the curated artifact set)")
    p.add_argument("--token", default=None,
                   help="HF token (default: HF_TOKEN env or the huggingface-cli login cache)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not (args.run_dir / "best.pt").exists():
        raise SystemExit(f"no best.pt in {args.run_dir} — nothing trained to upload yet")

    token = args.token or os.environ.get("HF_TOKEN")  # else falls back to the cached login
    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    dest = args.path_in_repo or args.run_dir.name

    if args.all_files:
        api.upload_folder(folder_path=str(args.run_dir), repo_id=args.repo,
                          repo_type="model", path_in_repo=dest)
        print(f"uploaded all of {args.run_dir} -> {args.repo}/{dest}")
    else:
        uploaded = []
        for name in DEFAULT_INCLUDE:
            f = args.run_dir / name
            if f.exists():
                api.upload_file(path_or_fileobj=str(f), path_in_repo=f"{dest}/{name}",
                                repo_id=args.repo, repo_type="model")
                uploaded.append(name)
        print(f"uploaded {uploaded} -> {args.repo}/{dest}")
    print(f"done: https://huggingface.co/{args.repo}/tree/main/{dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
