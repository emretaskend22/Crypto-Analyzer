from sqlalchemy import text
from db_connection import engine  # make sure you import your SQLAlchemy engine

def get_row_count(coin: str = "BTCUSDT"):
    query = text("""
        SELECT COUNT(*) AS cnt
        FROM crypto_ohlcv
        WHERE coin = :coin
    """)
    with engine.connect() as conn:
        result = conn.execute(query, {"coin": coin.upper()})
        count = result.fetchone()[0]  # access by index
    return count

def get_time_span(coin: str = "BTCUSDT"):
    query = text("""
        SELECT MIN(open_time) AS first, MAX(open_time) AS last
        FROM crypto_ohlcv
        WHERE coin = :coin
    """)
    with engine.connect() as conn:
        result = conn.execute(query, {"coin": coin.upper()})
        row = result.fetchone()
    first, last = row[0], row[1]  # access by index
    return first, last

if __name__ == "__main__":
    coin = "BTCUSDT"
    total_rows = get_row_count(coin)
    print(f"Total rows for {coin}: {total_rows}")

    first, last = get_time_span(coin)
    print(f"Data time span for {coin}: {first} → {last}")
