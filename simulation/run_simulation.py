from pathlib import Path
import time

import polars as pl
import requests

SIMULATION_DATA_PATH = Path("data") / "split" / "simul_transaction.parquet"
PREDICT_URL = "http://127.0.0.1:8000/predict"

if __name__ == "__main__":
    simul_df = pl.read_parquet(SIMULATION_DATA_PATH)
    simul_features = simul_df.select(["amount", "month"]).to_dicts()

    for idx in range(len(simul_features)):
        response = requests.post(PREDICT_URL, json=simul_features[idx])
        if response.ok:
            result = response.json()
            print(result)
            time.sleep(1)