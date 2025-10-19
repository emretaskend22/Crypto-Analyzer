from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine, text
import sys
from pathlib import Path

# -----------------------------
# Add backend to sys.path
# -----------------------------
PROJECT_ROOT = Path(__file__).parents[1] / "backend"
sys.path.append(str(PROJECT_ROOT))

from backend.db.data_fetcher import fetch_binance_ohlcv
from backend.db.db_data_utils import add_indicators, filter_new_rows

# -----------------------------
# DB Connection
# -----------------------------
DB_USER = "crypto_user"
DB_PASS = "Whitewolf2206"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "crypto_db"
DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DB_URL, echo=False)

# -----------------------------
# Coins
# -----------------------------
COINS = ["BTCUSDT", "ETHUSDT"]

# -----------------------------
# ETL function
# -----------------------------
def etl_for_coin(coin: str):
    df = fetch_binance_ohlcv(symbol=coin, interval="1h", limit=500)
    df = add_indicators(df)

    with engine.connect() as conn:
        try:
            query = text("SELECT coin, open_time FROM crypto_ohlcv WHERE coin=:coin")
            existing_df = pd.read_sql(query, conn, params={"coin": coin})
        except Exception:
            existing_df = pd.DataFrame(columns=["coin", "open_time"])

        df_new = filter_new_rows(df, existing_df)

        if not df_new.empty:
            df_new.to_sql(
                "crypto_ohlcv",
                con=conn,
                if_exists="append",
                index=False,
                method="multi"
            )
            print(f"✅ Inserted {len(df_new)} rows for {coin}")
        else:
            print(f"ℹ️ No new rows for {coin}")

# -----------------------------
# DAG definition
# -----------------------------
default_args = {
    "owner": "emre",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="crypto_etl",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule="@hourly",   # ✅ use this
    catchup=False,
    tags=["crypto", "etl"],
) as dag:

    tasks = []
    for coin in COINS:
        task = PythonOperator(
            task_id=f"etl_{coin.lower()}",
            python_callable=etl_for_coin,
            op_args=[coin],
        )
        tasks.append(task)

    # Sequential execution
    for i in range(len(tasks) - 1):
        tasks[i] >> tasks[i + 1]
