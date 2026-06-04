import itertools
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf

app = FastAPI()

# Enable CORS so your CodePen frontend can fetch data from Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/discover")
def discover_strategy(ticker: str, tier: str = "light"):
    ticker = ticker.upper().strip()

    # 1. Monetization Gate: Block Crypto on Free Tier
    if tier == "light" and ("-USD" in ticker or ticker in ["BTC", "ETH"]):
        raise HTTPException(
            status_code=403,
            detail="Crypto tracking is a Premium Feature. Please upgrade your tier.",
        )

    # 2. Gated Lookback Window Fetching
    period = "1y" if tier == "light" else "5y"
    try:
        df = yf.download(ticker, period=period, progress=False)
    except Exception:
        raise HTTPException(
            status_code=500, detail="Error fetching data from market feeds."
        )

    if df.empty or len(df) < 50:
        raise HTTPException(
            status_code=404,
            detail="Asset not found or insufficient trading data history.",
        )

    # Flatten columns if multi-indexed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    data = df[["Close"]].copy()
    data["Baseline_Return"] = data["Close"].pct_change()

    # 3. Gated Indicator Depth Generation
    windows = [10, 20, 50] if tier == "light" else [5, 10, 20, 30, 50, 100, 200]
    for w in windows:
        data[f"SMA_{w}"] = data["Close"].rolling(window=w).mean()

    # 4. Combinatorial Optimization Loop
    sma_cols = [col for col in data.columns if "SMA_" in col]
    permutations = list(itertools.combinations(sma_cols, 2))
    results = []

    for fast_sma, slow_sma in permutations:
        fast_w = int(fast_sma.split("_")[1])
        slow_w = int(slow_sma.split("_")[1])
        if fast_w >= slow_w:
            continue

        df_strat = data[["Baseline_Return", fast_sma, slow_sma]].copy()
        df_strat["Signal"] = np.where(
            df_strat[fast_sma] > df_strat[slow_sma], 1, -1
        )

        # Shift signals forward 1 day to prevent forward-looking bias
        df_strat["Strat_Return"] = (
            df_strat["Signal"].shift(1) * df_strat["Baseline_Return"]
        )

        cum_bh = (1 + df_strat["Baseline_Return"].dropna()).prod() - 1
        cum_strat = (1 + df_strat["Strat_Return"].dropna()).prod() - 1

        metrics = {
            "strategy": f"{fast_sma} x {slow_sma} Crossover",
            "return_pct": f"{round(cum_strat * 100, 1)}%",
            "outperformance": f"{'+' if cum_strat - cum_bh >= 0 else ''}{round((cum_strat - cum_bh) * 100, 1)}%",
        }

        # Premium Metric Layer (Sharpe Ratio calculation)
        if tier == "premium":
            daily_std = df_strat["Strat_Return"].std()
            avg_daily = df_strat["Strat_Return"].mean()
            sharpe = (
                (avg_daily / daily_std) * np.sqrt(252) if daily_std != 0 else 0
            )
            metrics["strategy"] += " 💎"
            metrics["sharpe"] = f"Sharpe: {round(sharpe, 2)}"

        results.append(metrics)

    # 5. Dynamic Sorting & Output Filter
    if tier == "premium":
        results.sort(
            key=lambda x: float(x["outperformance"].replace("%", "")),
            reverse=True,
        )
    else:
        results.sort(
            key=lambda x: float(x["return_pct"].replace("%", "")), reverse=True
        )

    return results[:3]
