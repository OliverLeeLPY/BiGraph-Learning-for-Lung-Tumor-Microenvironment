"""
LUAD → BiGraph inputs: single-cell (SC_d, SC_v) and survival (survival_d, survival_v).

See LLM.md for required columns. Core entry point: export_bigraph_csvs().
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_CELLTYPE_DIR = PROJECT_ROOT / "LungData" / "LUAD_IMC_CellType"
DEFAULT_CLINICAL_XLSX = PROJECT_ROOT / "LungData" / "LUAD Clinical Data.xlsx"

SC_COLS = ["patientID", "imageID", "celltypeID", "coorX", "coorY"]
SURV_COLS = ["patientID", "status", "length"]

# Clinical column names (LUAD Clinical Data.xlsx)
COL_KEY = "Key"
COL_DEATH = "Death (No: 0, Yes: 1)"
COL_SURV_YEARS = "Survival or loss to follow-up (years)"
COL_PROGRESSION = "Progression (No: 0, Yes: 1) "  # trailing space as in Excel


def _wrapped_mean(arr: np.ndarray, modulus: int = 1000) -> float:
    arr = np.asarray(arr, dtype=float).copy()
    if arr.max() - arr.min() > modulus / 2:
        arr[arr < modulus / 2] += modulus
    return float(arr.mean() % modulus)


def _cells_from_mat(mat_path: Path) -> pd.DataFrame:
    mat = loadmat(mat_path)
    cell_types = mat["cellTypes"]
    boundaries = mat["Boundaries"]

    names = [str(np.array(cell_types[i, 0]).squeeze()) for i in range(cell_types.shape[0])]
    coor_x: list[float] = []
    coor_y: list[float] = []
    for i in range(boundaries.shape[1]):
        b = boundaries[0, i].squeeze()
        x = b // 1000
        y = b % 1000
        coor_x.append(_wrapped_mean(x, 1000))
        coor_y.append(_wrapped_mean(y, 1000))

    image_id = mat_path.stem
    n = len(names)
    return pd.DataFrame(
        {
            "patientID": np.repeat(image_id, n),
            "imageID": np.repeat(image_id, n),
            "celltypeName": names,
            "coorX": np.asarray(coor_x, dtype=float),
            "coorY": np.asarray(coor_y, dtype=float),
        }
    )


def build_single_cell_tables(
    celltype_dir: str | Path | None = None,
    *,
    discovery_token: str = "LUAD_D",
    validation_token: str = "LUAD_V",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Step 1–2 (single-cell): load all *.mat, global celltype IDs 0..K-1, split D/V.

    Returns (SC_d, SC_v, celltype_lookup).
    """
    root = Path(celltype_dir) if celltype_dir is not None else DEFAULT_CELLTYPE_DIR
    parts = [_cells_from_mat(p) for p in sorted(root.glob("*.mat"))]
    raw = pd.concat(parts, ignore_index=True)
    raw = raw.dropna(subset=["patientID", "imageID", "celltypeName", "coorX", "coorY"])
    for col, typ in (("patientID", str), ("imageID", str), ("celltypeName", str)):
        raw[col] = raw[col].astype(typ)
    raw["coorX"] = raw["coorX"].astype(float)
    raw["coorY"] = raw["coorY"].astype(float)

    ordered_names = sorted(raw["celltypeName"].unique())
    name_to_id = {n: i for i, n in enumerate(ordered_names)}
    celltype_lookup = pd.DataFrame(
        {"celltypeName": ordered_names, "celltypeID": range(len(ordered_names))}
    )

    sc = raw.assign(celltypeID=raw["celltypeName"].map(name_to_id).astype(int))[
        SC_COLS
    ].reset_index(drop=True)

    mask_d = sc["imageID"].str.contains(discovery_token, regex=False)
    mask_v = sc["imageID"].str.contains(validation_token, regex=False)
    sc_d = sc.loc[mask_d].reset_index(drop=True)
    sc_v = sc.loc[mask_v].reset_index(drop=True)
    return sc_d, sc_v, celltype_lookup


def build_survival_tables(clinical_xlsx: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Discovery: overall survival — status 0 alive, 1 death; length in months.
    Validation sheet only has progression; length is NaN unless you merge external follow-up.
    """
    path = Path(clinical_xlsx) if clinical_xlsx is not None else DEFAULT_CLINICAL_XLSX

    disc = pd.read_excel(path, sheet_name="LUAD_416_Discovery", engine="openpyxl")
    survival_d = pd.DataFrame(
        {
            "patientID": disc[COL_KEY].astype(str),
            "status": pd.to_numeric(disc[COL_DEATH], errors="coerce").astype(float).astype(int),
            "length": pd.to_numeric(disc[COL_SURV_YEARS], errors="coerce").astype(float) * 12.0,
        }
    )
    survival_d = survival_d.dropna(subset=["patientID", "status", "length"]).reset_index(drop=True)

    val = pd.read_excel(path, sheet_name="LUAD_120_Validation", engine="openpyxl")
    survival_v = pd.DataFrame(
        {
            "patientID": val[COL_KEY].astype(str),
            "status": pd.to_numeric(val[COL_PROGRESSION], errors="coerce").astype(float).astype(int),
            "length": np.nan,
        }
    )
    survival_v = survival_v.dropna(subset=["patientID", "status"]).reset_index(drop=True)

    return survival_d, survival_v


def export_bigraph_csvs(
    out_dir: str | Path,
    *,
    celltype_dir: str | Path | None = None,
    clinical_xlsx: str | Path | None = None,
    write_survival: bool = True,
) -> dict[str, Path]:
    """
    Write discovery/validation single-cell and survival CSVs for BiGraph load step.

    Filenames: SC_d.csv, SC_v.csv, survival_d.csv, survival_v.csv, celltype_lookup.csv
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sc_d, sc_v, lookup = build_single_cell_tables(celltype_dir=celltype_dir)
    paths: dict[str, Path] = {}
    paths["SC_d"] = out / "SC_d.csv"
    paths["SC_v"] = out / "SC_v.csv"
    paths["celltype_lookup"] = out / "celltype_lookup.csv"

    sc_d.to_csv(paths["SC_d"], index=False)
    sc_v.to_csv(paths["SC_v"], index=False)
    lookup.to_csv(paths["celltype_lookup"], index=False)

    if write_survival:
        survival_d, survival_v = build_survival_tables(clinical_xlsx=clinical_xlsx)
        paths["survival_d"] = out / "survival_d.csv"
        paths["survival_v"] = out / "survival_v.csv"
        survival_d.to_csv(paths["survival_d"], index=False)
        survival_v.to_csv(paths["survival_v"], index=False)

    return paths
