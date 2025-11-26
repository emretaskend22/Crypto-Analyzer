from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
from pathlib import Path

# -----------------------------
# Add backend to sys.path
# -----------------------------
PROJECT_ROOT = Path(__file__).parents[1] / "backend"
sys.path.append(str(PROJECT_ROOT))

from backend.db.etl_utils import etl_for_coin
from backend.db.enums import Coin

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
    for coin_enum in Coin:
        coin = coin_enum.value
        task = PythonOperator(
            task_id=f"etl_{coin.lower()}",
            python_callable=etl_for_coin,
            op_args=[coin],
        )
        tasks.append(task)

    # Sequential execution
    for i in range(len(tasks) - 1):
        tasks[i] >> tasks[i + 1]
