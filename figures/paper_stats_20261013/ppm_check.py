# -*- coding: utf-8 -*-
"""P0-3 复核：MiPD513 数据集 pep_mass（观测 PEPMASS[0]）与理论前体 m/z 的 ppm 偏差。

复刻 score_presynthesis.py::reference_ppm_check 的口径：
  - 按 (seq, charge) 取观测 pep_mass 均值
  - 理论 m/z = pyteomics fast_mass(seq, charge)（经 PBCLA/utils.py 补 B/O/X/Z 表）
  - ppm = (theoretical - observed) / observed * 1e6
"""
import importlib.util
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
from pyteomics import mass  # noqa: E402

ROOT = r"D:\UserData\Desktop\Github-project\Dbond2\DBond"
spec = importlib.util.spec_from_file_location("pbcla_utils", os.path.join(ROOT, "PBCLA", "utils.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules["pbcla_utils"] = mod
spec.loader.exec_module(mod)

FOLDS = ["1222", "2252", "3514", "6072", "9075"]
base = os.path.join(ROOT, "dataset", "5fold")
files = []
for f in FOLDS:
    files.append((f + " train", os.path.join(base, f + ".train.fbr.shuffle.multi.csv")))
    files.append((f + " test", os.path.join(base, f + ".test.fbr.multi.csv")))


def ppm_stats(tag, df):
    obs = df.groupby(["seq", "charge"])["pep_mass"].mean().reset_index()
    if obs.empty:
        print(f"{tag}: empty")
        return
    ppm = np.array([
        (float(mass.fast_mass(sequence=s, charge=int(z))) - pm) / pm * 1e6
        for s, z, pm in zip(obs["seq"], obs["charge"], obs["pep_mass"])
    ])
    n10 = int((np.abs(ppm) > 10).sum())
    n20 = int((np.abs(ppm) > 20).sum())
    print(f"{tag:10s}: n={len(ppm):4d}  median={np.median(ppm):+8.3f}  "
          f"p5={np.percentile(ppm, 5):+8.3f}  p95={np.percentile(ppm, 95):+8.3f}  "
          f"max|ppm|={np.abs(ppm).max():8.3f}  n|>10|={n10}  n|>20|={n20}")


allobs = []
for tag, path in files:
    if not os.path.exists(path):
        print("missing:", path)
        continue
    df = pd.read_csv(path, usecols=["seq", "charge", "pep_mass"])
    ppm_stats(tag, df)
    allobs.append(df)

if allobs:
    big = pd.concat(allobs, ignore_index=True)
    ppm_stats("UNION", big)
    big["charge"] = big["charge"].astype(int)
    print("\nper-charge (union):")
    for z, sub in big.groupby("charge"):
        obs = sub.groupby(["seq", "charge"])["pep_mass"].mean().reset_index()
        ppm = np.array([
            (float(mass.fast_mass(sequence=s, charge=int(zz))) - pm) / pm * 1e6
            for s, zz, pm in zip(obs["seq"], obs["charge"], obs["pep_mass"])
        ])
        print(f"  z={z}: n={len(ppm):4d} median={np.median(ppm):+8.3f} max|ppm|={np.abs(ppm).max():8.3f}")