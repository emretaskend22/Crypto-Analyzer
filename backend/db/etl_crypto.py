from .enums import Coin
from .etl_utils import etl_for_coin

def etl_crypto():
    for coin_enum in Coin:
        etl_for_coin(coin_enum.value)

if __name__ == "__main__":
    etl_crypto()
    print("ETL completed successfully!")
