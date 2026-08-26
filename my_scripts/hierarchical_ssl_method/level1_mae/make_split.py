import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def main(args):
    df = pd.read_csv(args.csv)

    if args.patient_col not in df.columns:
        raise ValueError(f"patient_col='{args.patient_col}' not in columns: {list(df.columns)[:30]}...")

    if args.image_col not in df.columns:
        raise ValueError(f"image_col='{args.image_col}' not in columns: {list(df.columns)[:30]}...")

    # unique patients
    patients = df[args.patient_col].astype(str).unique()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(patients)

    n_val = int(len(patients) * args.val_frac)
    val_p = set(patients[:n_val])
    df["split"] = df[args.patient_col].astype(str).apply(lambda x: "val" if x in val_p else "train")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df[[args.patient_col, args.image_col, "split"]].drop_duplicates().to_csv(out, index=False)
    print("Wrote:", out)
    print(df["split"].value_counts())

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--patient_col", default="patient_id")
    p.add_argument("--image_col", default="image_name")
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_csv", default="splits/split.csv")
    args = p.parse_args()
    main(args)