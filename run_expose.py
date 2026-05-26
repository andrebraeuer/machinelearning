#!/usr/bin/env python3
"""
Run the full exposé pipeline (VAE design study + model risk) in PyTorch.

Usage
-----
    python run_expose.py --fast          # smoke test (~minutes)
    python run_expose.py                 # full OAT + rolling (hours)
    python run_expose.py --stage1-only   # Part 1 Stage 1 only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import (
    COPULA_FAMILIES,
    DATA_PATH,
    MAX_EPOCHS,
    MIN_EPOCHS,
    SEED,
    VAEConfig,
    oat_experiments,
    oat_experiments_fast,
)
from pipeline import (
    load_returns,
    run_stage2_and_modelrisk,
    select_stage2_configs,
    stage1_oat_study,
)
from config import oat_experiments as all_oat


def parse_args():
    p = argparse.ArgumentParser(description="Exposé VAE-GARCH pipeline (PyTorch)")
    p.add_argument("--data", default=DATA_PATH, help="CSV path for returns")
    p.add_argument("--output", default="results", help="Output directory")
    p.add_argument("--fast", action="store_true", help="Reduced experiments for testing")
    p.add_argument("--stage1-only", action="store_true", help="Run only OAT Stage 1")
    p.add_argument("--n-forecast", type=int, default=None, help="Rolling forecast horizon")
    p.add_argument("--max-epochs", type=int, default=None, help="VAE training cap")
    p.add_argument("--top-k", type=int, default=5, help="Configs promoted to Stage 2")
    return p.parse_args()


def configs_for_stage2(
    stage1: pd.DataFrame,
    all_cfgs: list[VAEConfig],
    top_k: int,
) -> list[VAEConfig]:
    names = select_stage2_configs(stage1, top_k=top_k)
    if not names:
        names = [c.name for c in all_cfgs[: min(3, len(all_cfgs))]]
    by_name = {c.name: c for c in all_cfgs}
    return [by_name[n] for n in names if n in by_name]


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    returns = load_returns(args.data)
    print(f"Loaded returns: {returns.shape[0]} days × {returns.shape[1]} assets")
    print(f"Device: {torch.cuda.is_available() and 'cuda' or 'cpu'}")

    if args.fast:
        cfgs = oat_experiments_fast()
        max_epochs = args.max_epochs or 50
        n_forecast = args.n_forecast or 30
        copulas = ("gaussian", "t", "clayton")
        import config as cfg_mod
        import vae as vae_mod

        cfg_mod.MIN_EPOCHS = 10
        vae_mod.MIN_EPOCHS = 10
    else:
        cfgs = all_oat()
        max_epochs = args.max_epochs or MAX_EPOCHS
        n_forecast = args.n_forecast
        copulas = COPULA_FAMILIES

    min_epochs = 10 if args.fast else MIN_EPOCHS

    print(f"\n=== Part 1, Stage 1: OAT design study ({len(cfgs)} configurations) ===")
    stage1 = stage1_oat_study(
        returns, cfgs, max_epochs=max_epochs, min_epochs=min_epochs
    )
    stage1.to_csv(out / "stage1_oat.csv", index=False)
    print(stage1.to_string(index=False))

    if args.stage1_only:
        print(f"\nStage 1 results saved to {out / 'stage1_oat.csv'}")
        return

    stage2_cfgs = configs_for_stage2(stage1, cfgs, top_k=args.top_k)
    print(f"\n=== Part 1 Stage 2 + Part 2: rolling & model risk ===")
    print(f"Promoted VAE configs: {[c.name for c in stage2_cfgs]}")
    print(f"Copula families: {copulas}")

    summary = run_stage2_and_modelrisk(
        returns,
        stage2_cfgs,
        list(copulas),
        n_forecast=n_forecast,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        output_dir=str(out),
    )

    print("\n--- Model risk (MAD) ---")
    print(f"Mean MAD VAE:    {summary['mean_mad_vae']:.6f}")
    print(f"Mean MAD Copula: {summary['mean_mad_cop']:.6f}")
    print(f"Valid VAE models: {summary['n_valid_vae']}")
    print(f"Valid copula models: {summary['n_valid_cop']}")
    print("\nMAD summary (VAE):")
    print(summary["mad_vae_summary"])
    print("\nBacktests (VAE):")
    print(summary["backtest_vae"])
    print(f"\nAll outputs written to {out.resolve()}")


if __name__ == "__main__":
    main()
