# create_db.py
from sqlalchemy import create_engine, Column, String, Float, DateTime, MetaData, Table, Integer
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION
import os

# ---------- Configuration ----------
DB_USER = ""
DB_PASS = ""
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "crypto_db"

DB_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create SQLAlchemy engine
engine = create_engine(DB_URL)

# ---------- Define Table Schema ----------
metadata = MetaData()

crypto_ohlcv = Table(
    "crypto_ohlcv",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("coin", String(20), nullable=False),
    Column("open_time", DateTime, nullable=False),
    Column("open", DOUBLE_PRECISION, nullable=False),
    Column("high", DOUBLE_PRECISION, nullable=False),
    Column("low", DOUBLE_PRECISION, nullable=False),
    Column("close", DOUBLE_PRECISION, nullable=False),
    Column("volume", DOUBLE_PRECISION, nullable=False),
    Column("sma_20", DOUBLE_PRECISION, nullable=True),
    Column("ema_20", DOUBLE_PRECISION, nullable=True),
    Column("rsi_14", DOUBLE_PRECISION, nullable=True),
    Column("macd", DOUBLE_PRECISION, nullable=True),
    Column("macd_signal", DOUBLE_PRECISION, nullable=True),
    Column("macd_hist", DOUBLE_PRECISION, nullable=True),
    Column("bb_lower", DOUBLE_PRECISION, nullable=True),
    Column("bb_middle", DOUBLE_PRECISION, nullable=True),
    Column("bb_upper", DOUBLE_PRECISION, nullable=True),
    # Add a unique constraint to prevent duplicates
    extend_existing=True
)

# ---------- Create Table ----------
def create_table():
    print("Creating table 'crypto_ohlcv'...")
    metadata.create_all(engine)
    print("Table created successfully!")

if __name__ == "__main__":
    create_table()
