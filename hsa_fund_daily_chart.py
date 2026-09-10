#!/usr/bin/env python3
"""Download and chart one year of daily performance for the HSA fund lineup.

Dependencies:
    python -m pip install requests pandas matplotlib

Examples:
    python hsa_fund_daily_chart.py
    python hsa_fund_daily_chart.py --fund-set equity
    python hsa_fund_daily_chart.py --period 6mo --output hsa_6_months.png
    python hsa_fund_daily_chart.py --mode price --show
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import requests


FUNDS: dict[str, str] = {
    "VBIRX": "Short-Term Bond",
    "VBMPX": "Total Bond Market",
    "VEMIX": "Emerging Markets Stock",
    "VGSNX": "Real Estate",
    "VIGIX": "U.S. Large Growth",
    "VIIIX": "S&P 500",
    "VIMAX": "U.S. Mid Cap",
    "VSMAX": "U.S. Small Cap",
    "VTAPX": "Short-Term TIPS",
    "VTPSX": "Total International Stock",
    "VUSFX": "Ultra-Short-Term Bond",
}

FUND_SETS: dict[str, tuple[str, ...]] = {
    "all": tuple(FUNDS),
    "equity": ("VEMIX", "VIGIX", "VIIIX", "VIMAX", "VSMAX", "VTPSX"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot daily prices or normalized performance for the HSA funds."
    )
    parser.add_argument(
        "--period",
        default="1y",
        help="Yahoo Finance lookback period, such as 6mo, 1y, 2y, or 5y (default: 1y).",
    )
    parser.add_argument(
        "--mode",
        choices=("normalized", "price"),
        default="normalized",
        help="Index each fund to 100 or display its adjusted price (default: normalized).",
    )
    parser.add_argument(
        "--fund-set",
        choices=tuple(FUND_SETS),
        default="all",
        help="Chart all funds or only stock/equity funds (default: all).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hsa_funds_1y_daily.png"),
        help="Output PNG path (default: hsa_funds_1y_daily.png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the interactive chart window after saving the PNG.",
    )
    return parser.parse_args()


def _download_one_fund(
    session: requests.Session, ticker: str, period: str, retries: int = 4
) -> pd.Series:
    """Download one adjusted-close series from Yahoo's public chart endpoint."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "range": period,
        "interval": "1d",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }

    for attempt in range(retries):
        response = session.get(url, params=params, timeout=30)
        if response.status_code == 429 and attempt < retries - 1:
            time.sleep(2**attempt)
            continue
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        chart = payload.get("chart", {})
        if chart.get("error"):
            raise RuntimeError(str(chart["error"]))

        results = chart.get("result") or []
        if not results:
            return pd.Series(name=ticker, dtype="float64")

        result = results[0]
        timestamps = result.get("timestamp") or []
        indicators = result.get("indicators") or {}
        adjusted_blocks = indicators.get("adjclose") or []
        quote_blocks = indicators.get("quote") or []
        if adjusted_blocks:
            values = adjusted_blocks[0].get("adjclose") or []
        elif quote_blocks:
            values = quote_blocks[0].get("close") or []
        else:
            values = []

        if len(timestamps) != len(values):
            raise RuntimeError("Yahoo returned mismatched dates and prices")

        # Convert market timestamps to date-only values for clean alignment.
        dates = pd.DatetimeIndex(pd.to_datetime(timestamps, unit="s", utc=True).date)
        return pd.Series(values, index=dates, name=ticker, dtype="float64")

    raise RuntimeError("Yahoo Finance rate limit persisted after all retries")


def download_adjusted_closes(tickers: list[str], period: str) -> pd.DataFrame:
    """Return adjusted daily closes with one ticker per column."""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/124 Safari/537.36"
            )
        }
    )

    series: list[pd.Series] = []
    errors: list[str] = []
    for ticker in tickers:
        try:
            fund_data = _download_one_fund(session, ticker, period)
            series.append(fund_data)
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            errors.append(f"{ticker}: {exc}")
            series.append(pd.Series(name=ticker, dtype="float64"))
        time.sleep(0.15)

    closes = pd.concat(series, axis=1).reindex(columns=tickers).sort_index()
    closes = closes.dropna(how="all")
    if closes.empty:
        detail = "; ".join(errors) if errors else "no price observations"
        raise RuntimeError(f"Yahoo Finance returned no data ({detail}).")
    if errors:
        print("Yahoo warnings: " + "; ".join(errors), file=sys.stderr)
    return closes


def normalize_to_100(closes: pd.DataFrame) -> pd.DataFrame:
    """Index each series to 100 using that fund's first available observation."""
    normalized = closes.copy()
    for ticker in normalized.columns:
        valid = normalized[ticker].dropna()
        if not valid.empty:
            normalized[ticker] = normalized[ticker] / valid.iloc[0] * 100.0
    return normalized


def print_summary(closes: pd.DataFrame) -> None:
    rows: list[tuple[str, str, float, int]] = []
    for ticker in closes.columns:
        valid = closes[ticker].dropna()
        if len(valid) >= 2:
            total_return = (valid.iloc[-1] / valid.iloc[0] - 1.0) * 100.0
            rows.append((ticker, FUNDS[ticker], total_return, len(valid)))

    print("\nPerformance over each fund's available observations:")
    for ticker, name, total_return, observations in sorted(
        rows, key=lambda row: row[2], reverse=True
    ):
        print(f"  {ticker:<5} {total_return:>8.2f}%  {observations:>3} days  {name}")


def plot_funds(
    closes: pd.DataFrame,
    mode: str,
    period: str,
    fund_set: str,
    output_path: Path,
) -> None:
    plot_data = normalize_to_100(closes) if mode == "normalized" else closes

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 8.5))

    colors = plt.colormaps["tab20"].resampled(len(plot_data.columns))
    plotted = 0
    for index, ticker in enumerate(plot_data.columns):
        series = plot_data[ticker].dropna()
        if series.empty:
            continue
        ax.plot(
            series.index,
            series,
            linewidth=2.0,
            color=colors(index),
            label=f"{ticker} — {FUNDS[ticker]}",
        )
        plotted += 1

    if plotted == 0:
        raise RuntimeError("No valid closing-price series were available to plot.")

    if mode == "normalized":
        ax.axhline(100, color="#555555", linewidth=1, linestyle="--", alpha=0.7)
        ax.set_ylabel("Growth of $100 (start = 100)")
        subtitle = "Adjusted daily close, each fund indexed to 100 on its first available date"
    else:
        ax.set_ylabel("Adjusted closing price ($)")
        subtitle = "Adjusted daily closing price; raw prices are not directly comparable"

    scope = "Equity Funds" if fund_set == "equity" else "All Funds"
    ax.set_title(
        f"HSA {scope} — {period.upper()} Daily Performance\n{subtitle}",
        fontsize=16,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("Date")
    ax.grid(True, which="major", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"\nSaved chart to: {output_path.resolve()}")


def main() -> int:
    args = parse_args()
    tickers = list(FUND_SETS[args.fund_set])

    try:
        closes = download_adjusted_closes(tickers, args.period)
        missing = [ticker for ticker in tickers if closes[ticker].dropna().empty]
        if missing:
            print(
                "Warning: no data returned for: " + ", ".join(missing),
                file=sys.stderr,
            )

        print_summary(closes)
        plot_funds(closes, args.mode, args.period, args.fund_set, args.output)
        if args.show:
            plt.show()
        else:
            plt.close("all")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
