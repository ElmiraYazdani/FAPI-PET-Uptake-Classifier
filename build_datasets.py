"""
Build the four experiment datasets from the Target_Assigned LIFEx exports (all foci of uptake).

Inputs (read-only):  <src>/CT.csv, <src>/PET_64.csv, <src>/PET_128.csv
    columns: INFO_PatientName (CTn / PTn), INFO_NameOfRoi (Segn-segmentation_#k), Target (0/1),
             145 radiomic features with suffix _CT or _PT, LIFEx warning column (PET files, empty header)
Outputs (<out>):     exp1_pet64.csv, exp2_pet128.csv, exp3_pet64_ct.csv, exp4_pet128_ct.csv, exp5_ct.csv,
                     data_check_report.json, data_check_tables.csv

Records are matched by the key (patient number, focus number) - never by row position. Target must agree for the
same key in all files. Feature names: suffix _PT -> prefix 'PET_', suffix _CT -> prefix 'CT_' (the prefixes are used
by the modality-balanced mRMR stage). Metadata columns (IDs, Target, LIFEx warnings) are not features.
Center: P1-P12 = 1, P13-P24 = 2.

The pre-harmonized *_combat.csv / *_combat_yj.csv files are deliberately NOT used: they were harmonized on all foci
with Target as a ComBat covariate, so every focus's features depend on its own label (leakage in cross-validation).
The equivalent Yeo-Johnson + ComBat (batch only) preprocessing is fitted inside each training fold instead.

Usage:
    python build_datasets.py --src "D:/FAPI-Seg/Radiomics/Modeling/Final_Excels/Target_Assigned" --out ../01_data
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

FILES = {"CT": "CT.csv", "PET64": "PET_64.csv", "PET128": "PET_128.csv"}
EXPERIMENTS = {"exp1_pet64": ["PET64"], "exp2_pet128": ["PET128"],
               "exp3_pet64_ct": ["PET64", "CT"], "exp4_pet128_ct": ["PET128", "CT"],
               "exp5_ct": ["CT"]}  # CT-only: sensitivity analysis as in the previous manuscript


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    pid = df.INFO_PatientName.str.fullmatch(r"(?:CT|PT)(\d+)")
    assert pid.all(), f"{path}: unparseable patient IDs"
    m = df.INFO_NameOfRoi.str.extract(r"^Seg(\d+)-segmentation_#(\d+)$")
    assert m.notna().all().all(), f"{path}: unparseable focus IDs"
    key = pd.DataFrame({"patient_id": df.INFO_PatientName.str.extract(r"(\d+)$")[0].astype(int),
                        "focus": m[1].astype(int)})
    assert (m[0].astype(int) == key.patient_id).all(), f"{path}: focus ID belongs to another patient"
    assert set(df.Target) <= {"0", "1"}, f"{path}: Target not coded 0/1: {sorted(set(df.Target))}"
    feats = [c for c in df.columns if c.endswith(("_CT", "_PT"))]
    other = [c for c in df.columns if c not in feats]
    X = df[feats].apply(lambda s: pd.to_numeric(s.replace({"": np.nan}), errors="coerce"))
    nonnum = [c for c in feats if (X[c].isna() & ~df[c].isin(["", "NaN", "nan"])).any()]
    assert not nonnum, f"{path}: non-numeric feature values in {nonnum}"
    return key, df.Target.astype(int), X, feats, other


def rename(c):
    return ("PET_" + c[:-3]) if c.endswith("_PT") else ("CT_" + c[:-3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    blocks, report, rows = {}, {"source_folder": str(src), "files": {}}, []
    for mod, fn in FILES.items():
        key, y, X, feats, other = load(src / fn)
        assert not key.duplicated().any(), f"{fn}: duplicate (patient, focus) keys"
        X.columns = [rename(c) for c in feats]
        blocks[mod] = (key, y, X)
        report["files"][fn] = {
            "sha256": sha256(src / fn), "format": "CSV (no sheets)", "rows": len(key), "radiomic_features": len(feats),
            "non_feature_columns": [c if c else "(empty header: LIFEx warning text)" for c in other],
            "feature_dtype": "float64 (all numeric after parsing)", "patients": int(key.patient_id.nunique()),
            "Target_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
            "missing_feature_cells": int(X.isna().sum().sum()),
            "features_with_missing": int(X.isna().any().sum())}
    # cross-file key and label agreement
    ref_key, ref_y, _ = blocks["CT"]
    ref = ref_key.assign(label=ref_y.values)
    for mod, (key, y, _) in blocks.items():
        k = key.assign(label=y.values)
        mrg = ref.merge(k, on=["patient_id", "focus"], how="outer", suffixes=("_ct", "_x"), indicator=True)
        assert (mrg._merge == "both").all(), f"{mod}: keys differ from CT"
        assert (mrg.label_ct == mrg.label_x).all(), f"{mod}: Target disagrees with CT"
    report["cross_file"] = "identical (patient, focus) key sets in all three files; Target identical for every key"

    for exp, mods in EXPERIMENTS.items():
        key, y, X = blocks[mods[0]]
        df = key.assign(label=y.values).join(X)
        for m in mods[1:]:
            k2, y2, X2 = blocks[m]
            df2 = k2.assign(label_check=y2.values).join(X2)
            df = df.merge(df2, on=["patient_id", "focus"], how="inner", validate="one_to_one")
            assert len(df) == len(key) and (df.label == df.label_check).all()
            df = df.drop(columns="label_check")
        df.insert(1, "region_id", [f"P{p:02d}_F{f:03d}" for p, f in zip(df.patient_id, df.focus)])
        df.insert(3, "center", np.where(df.patient_id <= 12, 1, 2))
        df = df.sort_values(["patient_id", "focus"]).reset_index(drop=True)
        feats = [c for c in df.columns if c.startswith(("PET_", "CT_"))]
        df.to_csv(out / f"{exp}.csv", index=False)
        report.setdefault("experiments", {})[exp] = {
            "modalities": mods, "observations": len(df), "patients": int(df.patient_id.nunique()),
            "class_counts": {str(k): int(v) for k, v in df.label.value_counts().sort_index().items()},
            "candidate_features": len(feats),
            "PET_features": sum(c.startswith("PET_") for c in feats), "CT_features": sum(c.startswith("CT_") for c in feats),
            "missing_cells": int(df[feats].isna().sum().sum())}
        for (pt, c), g in df.groupby(["patient_id", "center"]):
            rows.append({"experiment": exp, "patient_id": pt, "center": c, "foci": len(g),
                         "label_0": int((g.label == 0).sum()), "label_1": int((g.label == 1).sum())})
    with open(out / "data_check_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    pd.DataFrame(rows).to_csv(out / "data_check_tables.csv", index=False)
    print(json.dumps(report["experiments"], indent=1))


if __name__ == "__main__":
    main()
