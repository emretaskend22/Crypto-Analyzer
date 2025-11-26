import pandas as pd
from sqlalchemy import text
from .data_fetcher import fetch_binance_ohlcv
from .db_data_utils import add_indicators, filter_new_rows
from .db_connection import engine

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
