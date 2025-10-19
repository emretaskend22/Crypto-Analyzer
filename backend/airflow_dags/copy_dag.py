from pathlib import Path
import shutil
import os

# Source is in the same folder as this script
src = Path(__file__).parent / "crypto_etl_dag.py"

# Destination is the Airflow DAGs folder
dst = Path(os.path.expanduser("~/airflow/dags/crypto_etl_dag.py"))

# Ensure destination folder exists
dst.parent.mkdir(parents=True, exist_ok=True)

shutil.copy(src, dst)
print(f"DAG copied to Airflow folder: {dst}")
