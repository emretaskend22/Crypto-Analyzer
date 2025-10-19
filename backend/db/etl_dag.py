from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine
from data_fetcher import fetch_binance_ohlcv
from db_data_utils import add_indicators, filter_new_rows
from db_connection import engine


# -----------------------------
# Coins
# -----------------------------
COINS = ["BTCUSDT", "ETHUSDT"]

# -----------------------------
# ETL function
# -----------------------------
def etl_for_coin(coin):
    df = fetch_binance_ohlcv(symbol=coin, interval="1h", limit=500)
    df = add_indicators(df)

    try:
        existing_df = pd.read_sql(f"SELECT coin, open_time FROM crypto_ohlcv WHERE coin='{coin}'", engine)
    except Exception:
        existing_df = pd.DataFrame(columns=["coin", "Open time"])

    df_new = filter_new_rows(df, existing_df)
    if not df_new.empty:
        df_new.to_sql("crypto_ohlcv", engine, if_exists="append", index=False, method="multi")
        print(f"Inserted {len(df_new)} rows for {coin}")
    else:
        print(f"No new rows for {coin}")

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
    schedule_interval="@hourly",
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

    # If you want coins sequentially (or you can remove dependency for parallel)
    for i in range(len(tasks)-1):
        tasks[i] >> tasks[i+1]
