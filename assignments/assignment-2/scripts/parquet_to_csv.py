import pandas as pd

in_path = "outputs/detections_50ep_framelevel.parquet"
out_path = "outputs/detections_50ep_framelevel.csv"

df = pd.read_parquet(in_path)
df.to_csv(out_path, index=False)

print("Wrote:", out_path)
print("Rows:", len(df), "Cols:", len(df.columns))
print("Columns:", df.columns.tolist())