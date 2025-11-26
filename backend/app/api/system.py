from fastapi import APIRouter, Query, Body
from ..business.system_service import (
    get_coin_analytics,
    run_backtest,
    run_prediction
)

system_router = APIRouter()


# -------------------------------
# Endpoint: /analytics
# -------------------------------
@system_router.get("/analytics")
def analytics(
    coin: str = Query("BTCUSDT", description="Coin symbol, e.g., BTCUSDT")
):
    """Get analytics data with technical indicators."""
    try:
        data = get_coin_analytics(coin_symbol=coin)

        if isinstance(data, dict) and "error" in data:
            print(f"[API] Analytics error: {data['error']}")
            return {"coin": coin, "data": [], "error": data["error"]}

        print(f"[API] Analytics success: {len(data) if isinstance(data, list) else 0} records")
        return {"coin": coin, "data": data}

    except Exception as e:
        print(f"[API] Analytics exception: {e}")
        return {"coin": coin, "data": [], "error": str(e)}




@system_router.post("/backtest")
def backtest(
    coin: str = Query(..., description="Coin symbol"),
    start_date: str = Query(..., description="Start date YYYY-MM-DD"),
    end_date: str = Query(..., description="End date YYYY-MM-DD"),
    strategy: str = Query(..., description="Backtesting strategy name"),
    initial_balance: float = Query(10000, description="Initial portfolio balance")
):
    """
    Run a backtest for the given coin and strategy.
    """
    result = run_backtest(coin, start_date, end_date, strategy, initial_balance)
    return {"coin": coin, "strategy": strategy, "data": result}


# -------------------------------
# Endpoint: /predict
# -------------------------------
@system_router.get("/predict")
def predict(
    coin: str = Query(..., description="Coin symbol, e.g., BTCUSDT or ETHUSDT")
):
    result = run_prediction(coin)
    if "error" in result:
        return {"coin": coin, "error": result["error"]}
    return {"coin": coin, "data": result}
