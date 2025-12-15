import os
import pandas as pd

IN_PATH  = r"C:\Users\afaan\Downloads\buoybay-reports-52182\SN_OCEAN_2010-2025.csv"
OUT_PATH = r"C:\Users\afaan\noon_only_2010-2025.csv"

TIME_COL = "Time (UTC)"
TIME_FMT = "%d-%b-%y %H:%M:%S"
CHUNKSIZE = 250_000

# Make sure the output folder exists
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

first_write = True

for chunk in pd.read_csv(
    IN_PATH,
    chunksize=CHUNKSIZE,
    low_memory=False,   # avoids mixed-type warning spam
):
    dt = pd.to_datetime(chunk[TIME_COL], format=TIME_FMT, errors="coerce")

    # Strict noon = exactly 12:00:00
    mask = (dt.dt.hour == 12) & (dt.dt.minute == 0) & (dt.dt.second == 0)

    noon_chunk = chunk.loc[mask].copy()
    noon_chunk["datetime"] = dt.loc[mask].values

    noon_chunk.to_csv(
        OUT_PATH,
        index=False,
        mode="w" if first_write else "a",
        header=first_write,
    )
    first_write = False

print(f"Saved strict-noon-only data to: {OUT_PATH}")
