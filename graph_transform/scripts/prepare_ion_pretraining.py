#!/usr/bin/env python3
"""Build the frozen ion4 auxiliary corpus once into a new directory.

Only seq columns of existing NIST external / all ten original MiPD split CSVs
are read. Known sequence-set hashes and original MSP SHA fail closed. This
script never modifies the existing external dataset or its converter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from graph_transform.pretraining import msp_dataset as data

EXPECTED_NIST = (238178, "aad4fe6b2cca66bdaab09fca6aad49db5e3923bb27aa9bd9f8de4c86de596e84")
EXPECTED_MIPD = (513, "ee70d1ede41b7359b74eec10a3c501ddf78388055d23ae8de972bb868bd44e64")
EXPECTED_MSP_SHA256 = "1c1a8c7ba5299a0312b26f5eb5f8e4974c5a3339cdb4253869067b8e9b4102f1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--msp", type=Path, default=Path(r"G:\msp-demo\FTMS_HCD_20_annotated_2019-11-12.msp"))
    parser.add_argument("--nist_csv", type=Path, default=REPO / "dataset/nist_external/nist_hcd20_ext_test.csv")
    parser.add_argument("--mipd_fold_dir", type=Path, default=REPO / "dataset/5fold")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output_dir must be NEW; existing paths are never overwritten")
    paths = sorted(args.mipd_fold_dir.glob("*.csv"))
    expected_names = {f"{seed}.{suffix}" for seed in (1222, 2252, 3514, 6072, 9075)
                      for suffix in ("test.fbr.multi.csv", "train.fbr.shuffle.multi.csv")}
    if {path.name for path in paths} != expected_names:
        parser.error("Expected exactly all ten original MiPD split CSVs")
    nist, nist_sources = data.read_sequence_columns([args.nist_csv])
    mipd, mipd_sources = data.read_sequence_columns(paths)
    for name, sequences, expected in (("NIST", nist, EXPECTED_NIST), ("MiPD", mipd, EXPECTED_MIPD)):
        actual = (len(sequences), data.sequence_set_sha256(sequences))
        if actual != expected:
            raise ValueError(f"{name} complete exclusion identity changed: {actual}, expected {expected}")
    manifest = data.build_dataset(
        args.msp, args.output_dir, nist, mipd,
        exclusion_sources={"nist": nist_sources, "mipd": mipd_sources},
        certification={"certified_full_exclusion_sets": True,
                       "method": "all ten original MiPD CSV seq columns and complete NIST external CSV seq column; fixed counts and set hashes",
                       "nist_expected_count": EXPECTED_NIST[0], "nist_expected_sequence_set_sha256": EXPECTED_NIST[1],
                       "mipd_expected_count": EXPECTED_MIPD[0], "mipd_expected_sequence_set_sha256": EXPECTED_MIPD[1]},
        expected_msp_sha256=EXPECTED_MSP_SHA256,
        source_files=[Path(data.__file__), Path(__file__)],
        invocation={"executable": sys.executable, "argv": [str(Path(__file__).resolve()), *sys.argv[1:]],
                    "cwd": str(Path.cwd()), "effective_arguments": {key: str(value.resolve()) for key, value in vars(args).items()}},
    )
    report = {"status": manifest["status"], "output_dir": str(args.output_dir.resolve()),
              "source_sha256": manifest["source"]["sha256"],
              "outputs": {split: {key: value[key] for key in ("csv", "sha256", "rows", "unique_sequences", "valid_slots")}
                          for split, value in manifest["outputs"].items()},
              "statistics": manifest["statistics"]}
    data.atomic_json(args.output_dir / "build_report.json", report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
