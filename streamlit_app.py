import math
import re
import time
import json
import threading
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from concurrent.futures import ThreadPoolExecutor
from statistics import median

import ccxt
import numpy as np
import pandas as pd
import requests
import streamlit as st


WATCH = (
    "HYPEUSDT BTCUSDC KASUSDT NEARUSDC ETHUSDC FETUSDC "
    "XRPUSDC SOLUSDC BNBUSDC BCHUSDC LINKUSDC AAVEUSDC "
    "ZECUSDC RENDERUSDC TAOUSDC AKTUSDT ONDOUSDC SUIUSDC "
    "WLDUSDC INJUSDC ENAUSDC UNIUSDC ARBUSDC"
).split()

PERPS = {"HYPEUSDT", "KASUSDT", "AKTUSDT"}

TF = {
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

VENUES = [
    "binance",
    "bybit",
    "okx",
    "coinbaseexchange",
    "kraken",
    "bitget",
    "kucoin",
]

NAMES = dict(zip(
    VENUES,
    [
        "Binance", "Bybit", "OKX",
        "Coinbase", "Kraken", "Bitget", "KuCoin",
    ],
))

FEEDS = {
    "CoinDesk":
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph":
        "https://cointelegraph.com/rss",
}

ALIASES = dict(zip(
    (
        "BTC ETH HYPE KAS NEAR FET XRP SOL BNB BCH "
        "LINK AAVE ZEC RENDER TAO AKT ONDO SUI WLD "
        "INJ ENA UNI ARB"
    ).split(),
    [
        "bitcoin", "ethereum", "hyperliquid", "kaspa",
        "near protocol", "fetch.ai", "ripple", "solana",
        "bnb", "bitcoin cash", "chainlink", "aave",
        "zcash", "render network", "bittensor", "akash",
        "ondo", "sui", "worldcoin", "injective",
        "ethena", "uniswap", "arbitrum",
    ],
))

ALERT = re.compile(
    r"\b(hack\w*|exploit\w*|breach\w*|delist\w*|"
    r"insolven\w*|bankrupt\w*|depeg\w*|outage\w*|"
    r"lawsuit\w*|unlock\w*|fomc|cpi)\b|"
    r"(withdrawals?|trading|network).{0,35}"
    r"(halt|suspend|pause)",
    re.I,
)


def num(value, default=None):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (ValueError, TypeError):
        return default


def fmt(value):
    return "—" if num(value) is None else f"{value:,.8g}"


def split(symbol):
    for quote in ("USDC", "USDT"):
        if symbol.endswith(quote):
            return symbol[:-len(quote)], quote
    raise ValueError("Usare simboli USDC o USDT")


class Adapter:
    """Una connessione riutilizzata e protetta per exchange/mercato."""

    def __init__(self, venue, kind):
        exchange_id = (
            "binanceusdm"
            if venue == "binance" and kind == "perp"
            else venue
        )

        options = {
            "defaultType": "spot" if kind == "spot" else "swap"
        }

        if venue == "binance":
            options["fetchMarkets"] = {
                "types": ["spot" if kind == "spot" else "linear"]
            }

        if venue == "bybit" and kind == "perp":
            options["defaultType"] = "linear"

        self.ex = getattr(ccxt, exchange_id)({
            "timeout": 4500,
            "enableRateLimit": True,
            "options": options,
        })

        if venue == "binance" and kind == "spot":
            self.ex.urls["api"]["public"] = (
                "https://data-api.binance.vision/api/v3"
            )

        self.lock = threading.Lock()
        self.blocked = 0
        self.loaded = 0

    def call(self, action):
        if not self.lock.acquire(timeout=2):
            raise RuntimeError(
                "API occupata; nuovo tentativo automatico"
            )

        try:
            if time.time() < self.blocked:
                raise RuntimeError(
                    "API in pausa dopo errore/restrizione"
                )

            if (
                not self.ex.markets
                or time.time() - self.loaded > 3600
            ):
                self.ex.load_markets(
                    reload=bool(self.ex.markets)
                )
                self.loaded = time.time()

            return action(self.ex)

        except (
            ccxt.NetworkError,
            ccxt.PermissionDenied,
        ) as exc:
            self.blocked = time.time() + 60
            raise RuntimeError(
                type(exc).__name__
            ) from exc

        finally:
            self.lock.release()


class Engine:
    """Richieste in background: l'interfaccia resta utilizzabile."""

    def __init__(self):
        self.guard = threading.RLock()
        self.adapters = {}
        self.jobs = {}

        self.pools = {
            "ta": ThreadPoolExecutor(4),
            "flow": ThreadPoolExecutor(7),
            "news": ThreadPoolExecutor(2),
        }

    def adapter(self, venue, kind):
        with self.guard:
            key = (venue, kind)

            if key not in self.adapters:
                self.adapters[key] = Adapter(venue, kind)

            return self.adapters[key]

    def poll(self, key, group, function, args=(), ttl=60):
        with self.guard:
            slot = self.jobs.setdefault(
                key,
                {
                    "value": None,
                    "error": "",
                    "next": 0,
                    "future": None,
                },
            )

            future = slot["future"]

            if future and future.done():
                try:
                    slot["value"] = future.result()
                    slot["error"] = ""
                    slot["next"] = time.time() + ttl

                except Exception as exc:
                    slot["error"] = (
                        f"{type(exc).__name__}: "
                        f"{str(exc)[:140]}"
                    )
                    slot["next"] = time.time() + 30

                slot["future"] = None

            if (
                slot["future"] is None
                and time.time() >= slot["next"]
            ):
                slot["future"] = self.pools[group].submit(
                    function, *args
                )

            return (
                slot["value"],
                slot["error"],
                slot["future"] is not None,
            )


def choose_market(ex, base, quote, kind, exact=True):
    quotes = (
        [quote]
        if exact
        else list(dict.fromkeys(
            [quote, "USDT", "USDC", "USD"]
        ))
    )

    for q in quotes:
        symbol = (
            f"{base}/{q}"
            if kind == "spot"
            else f"{base}/{q}:{q}"
        )

        market = ex.markets.get(symbol)

        if not market or market.get("active") is False:
            continue

        valid_type = (
            market.get("spot")
            if kind == "spot"
            else (
                market.get("swap")
                and market.get("linear")
            )
        )

        if valid_type:
            return market

    raise ValueError(
        "Coppia non quotata nel mercato selezionato"
    )


def closed_frame(raw, timeframe, now):
    frame = pd.DataFrame(
        [row[:6] for row in raw],
        columns=[
            "ts", "open", "high",
            "low", "close", "volume",
        ],
    )

    frame = (
        frame.apply(pd.to_numeric, errors="coerce")
        .sort_values("ts")
        .drop_duplicates("ts")
    )

    step = TF[timeframe] * 1000

    # Esclude la candela ancora aperta, con margine di 2 secondi.
    frame = frame[
        frame.ts + step <= now * 1000 - 2000
    ].reset_index(drop=True)

    if (
        len(frame) < 210
        or not np.isfinite(frame.to_numpy()).all()
    ):
        raise ValueError(
            f"{timeframe}: storico insufficiente "
            "o valori non validi"
        )

    if (
        (frame[["open", "high", "low", "close"]] <= 0)
        .any().any()
        or (frame.volume < 0).any()
    ):
        raise ValueError(
            f"{timeframe}: prezzi/volumi non validi"
        )

    if (
        (
            frame.high
            < frame[["open", "low", "close"]].max(axis=1)
        ).any()
        or (
            frame.low
            > frame[["open", "high", "close"]].min(axis=1)
        ).any()
    ):
        raise ValueError(
            f"{timeframe}: OHLC incoerenti"
        )

    gaps = (
        frame.ts.tail(210).diff().dropna() != step
    ).any()

    stale = (
        now * 1000
        - frame.ts.iloc[-1]
        - step
        > step + 10000
    )

    if gaps or stale:
        raise ValueError(
            f"{timeframe}: candele mancanti o vecchie"
        )

    return frame


def wilder(series, period):
    result = pd.Series(np.nan, index=series.index)
    valid = series.dropna()

    if len(valid) >= period:
        start = valid.index[period - 1]
        result.loc[start] = valid.iloc[:period].mean()

        for index in range(
            series.index.get_loc(start) + 1,
            len(series),
        ):
            result.iloc[index] = (
                result.iloc[index - 1] * (period - 1)
                + series.iloc[index]
            ) / period

    return result


def indicators(frame):
    close, high, low, volume = (
        frame[column]
        for column in ("close", "high", "low", "volume")
    )

    out = frame.copy()

    for period in (7, 25, 50, 99, 200):
        out[f"EMA{period}"] = close.ewm(
            span=period,
            adjust=False,
        ).mean()

    true_range = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    out["ATR"] = wilder(true_range, 14)

    up = wilder(
        close.diff().clip(lower=0),
        14,
    )

    down = wilder(
        -close.diff().clip(upper=0),
        14,
    )

    out["RSI"] = (
        100 * up / (up + down)
    ).where(
        (up + down).ne(0),
        50,
    )

    rsi = out.RSI
    span = (
        rsi.rolling(14).max()
        - rsi.rolling(14).min()
    )

    out["K"] = (
        100
        * (rsi - rsi.rolling(14).min())
        / span.replace(0, np.nan)
    ).rolling(3).mean()

    out["D"] = out.K.rolling(3).mean()

    macd = (
        close.ewm(span=12, adjust=False).mean()
        - close.ewm(span=26, adjust=False).mean()
    )

    out["MACD"] = (
        macd - macd.ewm(span=9, adjust=False).mean()
    )

    move_up = high.diff()
    move_down = -low.diff()

    positive_dm = move_up.where(
        (move_up > move_down) & (move_up > 0),
        0.0,
    )

    negative_dm = move_down.where(
        (move_down > move_up) & (move_down > 0),
        0.0,
    )

    plus = (
        100 * wilder(positive_dm, 14)
        / out.ATR.replace(0, np.nan)
    )

    minus = (
        100 * wilder(negative_dm, 14)
        / out.ATR.replace(0, np.nan)
    )

    dx = (
        100 * (plus - minus).abs() / (plus + minus)
    ).where(
        (plus + minus).ne(0),
        0,
    )

    out["ADX"] = wilder(dx, 14)

    out["BB_low"] = (
        close.rolling(20).mean()
        - 2 * close.rolling(20).std(ddof=0)
    )

    out["BB_high"] = (
        close.rolling(20).mean()
        + 2 * close.rolling(20).std(ddof=0)
    )

    out["RVOL"] = (
        volume
        / volume.shift(1).rolling(20).mean().replace(
            0, np.nan
        )
    )

    atr10 = wilder(true_range, 10)
    upper = (high + low) / 2 + 3 * atr10
    lower = (high + low) / 2 - 3 * atr10

    bullish = pd.Series(False, index=frame.index)

    for index in range(10, len(frame)):
        upper.iloc[index] = (
            upper.iloc[index]
            if (
                upper.iloc[index] < upper.iloc[index - 1]
                or close.iloc[index - 1]
                > upper.iloc[index - 1]
            )
            else upper.iloc[index - 1]
        )

        lower.iloc[index] = (
            lower.iloc[index]
            if (
                lower.iloc[index] > lower.iloc[index - 1]
                or close.iloc[index - 1]
                < lower.iloc[index - 1]
            )
            else lower.iloc[index - 1]
        )

        bullish.iloc[index] = (
            close.iloc[index] >= lower.iloc[index - 1]
            if bullish.iloc[index - 1]
            else close.iloc[index] > upper.iloc[index - 1]
        )

    out["Supertrend"] = lower.where(bullish, upper)
    out["ST_bull"] = bullish

    out["Delta_taker"] = frame.get(
        "Delta_taker",
        pd.Series(np.nan, index=frame.index),
    )

    return out


def technical(engine, symbol, kind, pinned=None):
    base, quote = split(symbol)
    errors = []

    venues = [pinned] if pinned else ["binance", "bybit"]

    for venue in venues:
        try:
            adapter = engine.adapter(venue, kind)

            market = adapter.call(
                lambda ex: choose_market(
                    ex, base, quote, kind
                )
            )

            frames = {}

            for timeframe in TF:
                if venue == "binance":
                    method = (
                        "public_get_klines"
                        if kind == "spot"
                        else "fapiPublicGetKlines"
                    )

                    raw = adapter.call(
                        lambda ex: getattr(ex, method)({
                            "symbol": market["id"],
                            "interval": timeframe,
                            "limit": 320,
                        })
                    )
                else:
                    raw = adapter.call(
                        lambda ex: ex.fetch_ohlcv(
                            market["symbol"],
                            timeframe,
                            limit=320,
                        )
                    )

                frame = closed_frame(
                    raw,
                    timeframe,
                    time.time(),
                )

                if venue == "binance":
                    delta = {
                        int(row[0]): (
                            100
                            * (
                                2 * float(row[9])
                                - float(row[5])
                            )
                            / float(row[5])
                            if float(row[5]) > 0
                            else np.nan
                        )
                        for row in raw
                    }

                    frame["Delta_taker"] = frame.ts.map(delta)

                frames[timeframe] = indicators(frame)

                atr = frames[timeframe].ATR.iloc[-1]

                if not num(atr) or atr <= 0:
                    raise ValueError(
                        f"{timeframe}: volatilità non calcolabile"
                    )

            return {
                "frames": frames,
                "venue": venue,
                "kind": kind,
                "quote": quote,
                "stamp": time.time(),
            }

        except Exception as exc:
            errors.append(
                f"{NAMES[venue]}: {str(exc)[:90]}"
            )

    raise RuntimeError(" | ".join(errors))


def analyse(bundle, btc_ok, fee, slip):
    frames = bundle["frames"]

    entry_bar, hour, four, day = [
        frames[timeframe].iloc[-1]
        for timeframe in TF
    ]

    breakout = (
        frames["15m"].high.iloc[-21:-1].max()
    )

    checks = {
        "Trend 1D": (
            day.close > day.EMA200
            and day.EMA50 > day.EMA200,
            15,
        ),
        "Trend 4H": (
            four.close > four.EMA25
            and four.ST_bull,
            20,
        ),
        "Trend 1H": (
            hour.EMA7 > hour.EMA25 > hour.EMA99,
            15,
        ),
        "Momentum": (
            hour.MACD > 0
            and 45 <= hour.RSI <= 70
            and entry_bar.K > entry_bar.D,
            15,
        ),
        "ADX 1H": (
            hour.ADX >= 18,
            10,
        ),
        "RVOL 15m": (
            entry_bar.RVOL >= 1.2,
            10,
        ),
        "Breakout chiuso": (
            entry_bar.close > breakout,
            10,
        ),
        "Regime BTC": (
            btc_ok,
            5,
        ),
    }

    score = sum(
        weight
        for passed, weight in checks.values()
        if passed
    )

    entry = (
        max(entry_bar.high, breakout)
        + 0.05 * entry_bar.ATR
    )

    stop = min(
        frames["15m"].low.tail(6).min()
        - 0.2 * entry_bar.ATR,
        entry - 1.5 * entry_bar.ATR,
    )

    risk = entry - stop

    targets = [
        entry + risk * multiple
        for multiple in (1, 2, 3)
    ]

    cost = entry * (1 + slip) * (1 + fee)

    loss = (
        cost
        - stop * (1 - slip) * (1 - fee)
    )

    net = [
        (
            target * (1 - slip) * (1 - fee)
            - cost
        )
        / cost
        * 100
        for target in targets
    ]

    rr = [
        (
            target * (1 - slip) * (1 - fee)
            - cost
        )
        / loss
        for target in targets
    ]

    high = frames["1h"].high

    pivots = high[
        (high > high.shift(1))
        & (high > high.shift(2))
        & (high > high.shift(-1))
        & (high > high.shift(-2))
    ].tail(60)

    above = pivots[pivots > entry]

    resistance = (
        float(above.min())
        if len(above)
        else None
    )

    guards = [
        score >= 85,
        all(
            checks[name][0]
            for name in (
                "Trend 4H",
                "Trend 1H",
                "Momentum",
                "Breakout chiuso",
                "Regime BTC",
            )
        ),
        0.003 <= risk / entry <= 0.05,
        rr[1] >= 1.5,
        entry_bar.RVOL >= 1.2,
        hour.ADX >= 18,
        (
            (entry_bar.high - entry_bar.close)
            / max(
                entry_bar.high - entry_bar.low,
                1e-12,
            )
            <= 0.45
        ),
        (
            resistance is None
            or resistance >= targets[1]
        ),
    ]

    return {
        "score": score,
        "checks": checks,
        "ready": all(guards),
        "entry": entry,
        "stop": stop,
        "targets": targets,
        "net": net,
        "rr": rr,
        "risk": risk / entry * 100,
        "unit_loss": loss,
        "cost": cost,
        "atr": entry_bar.ATR,
        "bar": int(entry_bar.ts),
        "resistance": resistance,
    }


def book_job(engine, venue, base, quote, kind, exact):
    if (
        kind == "perp"
        and venue in (
            "coinbaseexchange",
            "kraken",
            "kucoin",
        )
    ):
        raise ValueError(
            "Adapter derivati non abilitato per questa sede"
        )

    adapter = engine.adapter(venue, kind)

    market = adapter.call(
        lambda ex: choose_market(
            ex, base, quote, kind, exact
        )
    )

    book = adapter.call(
        lambda ex: ex.fetch_order_book(
            market["symbol"],
            limit=100,
        )
    )

    received = time.time()

    if book.get("timestamp"):
        age = received - book["timestamp"] / 1000

        if not -5 <= age <= 25:
            raise ValueError("Timestamp book scaduto")

    scale = (
        float(market.get("contractSize") or 1)
        if kind == "perp"
        else 1
    )

    bids = sorted(
        [
            (float(price), float(amount) * scale)
            for price, amount, *_ in book["bids"]
            if float(amount) > 0
        ],
        reverse=True,
    )

    asks = sorted([
        (float(price), float(amount) * scale)
        for price, amount, *_ in book["asks"]
        if float(amount) > 0
    ])

    if any(
        not math.isfinite(price)
        or not math.isfinite(amount)
        or price <= 0
        for price, amount in bids + asks
    ):
        raise ValueError("Livelli book non validi")

    if (
        not bids
        or not asks
        or bids[0][0] <= 0
        or asks[0][0] <= bids[0][0]
    ):
        raise ValueError("Book vuoto o incrociato")

    mid = (bids[0][0] + asks[0][0]) / 2

    trades = []
    trade_error = ""

    try:
        trades = adapter.call(
            lambda ex: ex.fetch_trades(
                market["symbol"],
                limit=100,
            )
        )
    except Exception as exc:
        trade_error = str(exc)[:80]

    recent = [
        trade
        for trade in trades
        if (
            trade.get("timestamp")
            and 0 <= (
                time.time()
                - trade["timestamp"] / 1000
            ) <= 60
        )
    ]

    buy = 0.0
    sell = 0.0
    known = 0

    for trade in recent:
        # CCXT normalizza già il lato maker di Coinbase.
        side = trade.get("side")
        price = num(trade.get("price"))
        amount = num(trade.get("amount"))

        if (
            side not in ("buy", "sell")
            or not price
            or not amount
        ):
            continue

        value = price * amount * scale

        if side == "buy":
            buy += value
        else:
            sell += value

        known += 1

    times = [
        trade["timestamp"] / 1000
        for trade in recent
    ]

    delta = (
        (buy - sell) / (buy + sell)
        if (
            buy + sell
            and known >= 20
            and known >= 0.8 * len(recent)
            and time.time() - max(times) <= 20
        )
        else None
    )

    coverage = min(
        0.005,
        (mid - bids[-1][0]) / mid,
        (asks[-1][0] - mid) / mid,
    )

    return {
        "venue": venue,
        "pair": market["symbol"],
        "quote": market["quote"],
        "kind": kind,
        "market": market,
        "precision_mode": adapter.ex.precisionMode,
        "scale": scale,
        "bids": bids,
        "asks": asks,
        "mid": mid,
        "stamp": received,
        "coverage": coverage,
        "spread": (asks[0][0] - bids[0][0]) / mid,
        "delta": delta,
        "trades": known,
        "span": max(times) - min(times) if times else 0,
        "trade_error": trade_error,
    }


def consensus(books):
    valid = [
        book
        for book in books
        if (
            time.time() - book["stamp"] <= 25
            and book["coverage"] >= 0.0005
            and book["spread"] <= 0.002
        )
    ]

    if len(valid) < 2:
        return None

    band = min(
        book["coverage"]
        for book in valid
    )

    imbalances = []

    for book in valid:
        buy = sum(
            price * amount
            for price, amount in book["bids"]
            if price >= book["mid"] * (1 - band)
        )

        sell = sum(
            price * amount
            for price, amount in book["asks"]
            if price <= book["mid"] * (1 + band)
        )

        imbalances.append(
            (buy - sell) / (buy + sell)
            if buy + sell
            else 0
        )

    deltas = [
        book["delta"]
        for book in valid
        if book["delta"] is not None
    ]

    return {
        "band": band,
        "n": len(valid),
        "book": median(imbalances),
        "agree": (
            sum(value > 0.05 for value in imbalances)
            / len(valid)
        ),
        "trade": (
            median(deltas)
            if len(deltas) >= 2
            else None
        ),
    }


def vwap(levels, quantity):
    remaining = quantity
    cost = 0.0

    for price, available in levels:
        used = min(remaining, available)
        cost += used * price
        remaining -= used

        if remaining <= quantity * 1e-10:
            return cost / quantity

    return None


def tradable_qty(book, quantity):
    market = book["market"]
    scale = book["scale"]

    if market["precision"].get("amount") is not None:
        quantity = float(ccxt.decimal_to_precision(
            quantity / scale,
            ccxt.TRUNCATE,
            market["precision"]["amount"],
            book["precision_mode"],
        )) * scale

    return quantity


def read_feed(name, url):
    response = requests.get(
        url,
        timeout=(3, 5),
        headers={"User-Agent": "CryptoMonitor/4.0"},
    )

    response.raise_for_status()

    if len(response.content) > 3_000_000:
        raise ValueError("Feed troppo grande")

    root = ET.fromstring(response.content)
    articles = []

    for item in root.findall(".//item"):
        try:
            date = parsedate_to_datetime(
                item.findtext("pubDate", "")
            ).astimezone(timezone.utc).timestamp()

            title = item.findtext("title", "").strip()
            link = item.findtext("link", "").strip()

            if title and link.startswith("https://"):
                articles.append({
                    "source": name,
                    "title": title,
                    "url": link,
                    "ts": date,
                })

        except (ValueError, TypeError):
            continue

    if (
        not articles
        or time.time() - max(
            article["ts"]
            for article in articles
        ) > 172800
    ):
        raise ValueError(
            "Feed vuoto, senza date o non aggiornato"
        )

    return {
        "articles": articles,
        "stamp": time.time(),
    }


def news_filter(feeds, base, hours):
    good = [
        feed
        for feed in feeds
        if feed and time.time() - feed["stamp"] < 600
    ]

    relevant = []
    seen = set()

    for feed in good:
        for article in feed["articles"]:
            title = article["title"]
            lower = title.lower()

            coin = (
                ALIASES.get(base, base.lower()) in lower
                or re.search(
                    r"\$" + re.escape(base) + r"\b",
                    title,
                )
            )

            if base not in (
                "NEAR", "LINK", "INJ", "RENDER"
            ):
                coin = coin or re.search(
                    r"\b" + re.escape(base) + r"\b",
                    title,
                )

            macro = re.search(
                r"bitcoin|\bBTC\b|\bfed\b|fomc|"
                r"inflation|\bcpi\b|binance|coinbase|"
                r"bybit|tether|circle",
                title,
                re.I,
            )

            if (
                (coin or macro)
                and 0 <= time.time() - article["ts"]
                <= hours * 3600
                and article["url"] not in seen
            ):
                relevant.append(dict(
                    article,
                    alert=bool(ALERT.search(title)),
                ))
                seen.add(article["url"])

    relevant.sort(
        key=lambda article: article["ts"],
        reverse=True,
    )

    return (
        len(good),
        relevant,
        any(article["alert"] for article in relevant),
    )


def entry_reasons(
    bundle,
    plan,
    book,
    flow,
    news_ok,
    news_alert,
    manual,
    quantity,
):
    reasons = []

    if time.time() - bundle["stamp"] > 180:
        reasons.append("Analisi tecnica scaduta")

    if not plan["ready"]:
        reasons.append(
            "Setup tecnico incompleto / resistenza / R:R"
        )

    if not news_ok:
        reasons.append(
            "Copertura notizie insufficiente"
        )

    if news_alert or manual:
        reasons.append(
            "Allerta notizie / blocco manuale"
        )

    if (
        not book
        or time.time() - book["stamp"] > 25
    ):
        return reasons + [
            "Book della sede operativa non aggiornato"
        ]

    if book["spread"] > 0.0015:
        reasons.append("Spread >0,15%")

    if quantity <= 0:
        reasons.append("Quantità non valida")

    limits = book["market"].get("limits", {})

    for field, value in [
        ("amount", quantity / book["scale"]),
        ("cost", quantity * book["asks"][0][0]),
    ]:
        bounds = limits.get(field) or {}

        if (
            bounds.get("min") is not None
            and value < bounds["min"]
        ):
            reasons.append(
                "Ordine inferiore al minimo exchange"
            )

        if (
            bounds.get("max") is not None
            and value > bounds["max"]
        ):
            reasons.append(
                "Ordine superiore al massimo exchange"
            )

    price = (
        vwap(book["asks"], quantity)
        if quantity > 0
        else None
    )

    if price is None:
        reasons.append(
            "Profondità insufficiente per la quantità"
        )
    else:
        if price / book["asks"][0][0] - 1 > 0.001:
            reasons.append(
                "Impatto stimato >0,10%"
            )

        if not (
            plan["entry"]
            <= price
            <= plan["entry"] + 0.25 * plan["atr"]
        ):
            reasons.append(
                "Prezzo fuori dalla fascia di ingresso"
            )

    if (
        not flow
        or flow["book"] < 0.08
        or flow["agree"] < 0.6
        or flow["trade"] is None
        or flow["trade"] <= 0
    ):
        reasons.append(
            "Consenso book/trade insufficiente"
        )

    return reasons


def position_event(position, bid, now):
    # Sono soglie osservate, non esecuzioni.
    # Stop e target non vengono ricalcolati.
    event = (
        "STOP"
        if bid <= position["stop"]
        else next(
            (
                f"TP{index + 1}"
                for index in (2, 1, 0)
                if bid >= position["targets"][index]
            ),
            None,
        )
    )

    if event and not any(
        item["event"] == event
        for item in position["events"]
    ):
        position["events"].append({
            "time": now,
            "event": event,
            "bid": bid,
        })

    return event


def monitor_positions(engine, positions):
    if not positions:
        return

    st.subheader(
        "Posizioni registrate · controllo uscite"
    )

    for key, position in positions.items():
        symbol, kind = key.split(":")
        base, quote = split(symbol)
        venue = position["venue"]

        reference, error, _ = engine.poll(
            ("book", venue, base, quote, kind, True),
            "flow",
            book_job,
            (engine, venue, base, quote, kind, True),
            10,
        )

        if (
            error
            or not reference
            or time.time() - reference["stamp"] > 25
        ):
            st.error(
                f"{key}: prezzo non disponibile. "
                "Verifica la posizione sull’exchange."
            )
            continue

        event = position_event(
            position,
            reference["bids"][0][0],
            datetime.now(timezone.utc).isoformat(),
        )

        if any(
            item["event"] == "STOP"
            for item in position["events"]
        ):
            st.error(
                f"{key}: STOP OSSERVATO · "
                "non implica che il tuo ordine "
                "sia stato eseguito."
            )
        elif event:
            st.warning(
                f"{key}: {event} OSSERVATO · "
                "verifica la tua gestione dell’uscita."
            )
        else:
            st.info(
                f"{key}: soglie attuali non raggiunte."
            )

        exit_price = vwap(
            reference["bids"],
            position["qty"],
        )

        if exit_price:
            pnl = position["qty"] * (
                exit_price * (1 - position["fee"])
                - position["entry"] * (1 + position["fee"])
            )

            st.write(
                f"P&L stimato su bid disponibili: "
                f"{pnl:+.2f} {quote}, commissioni incluse."
            )
        else:
            st.warning(
                "Profondità insufficiente "
                "per stimare l’intera uscita."
            )


@st.cache_resource
def engine_resource():
    return Engine()


def app():
    st.set_page_config(
        page_title="Crypto Screener · Risk & News",
        page_icon="⚡",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1150px;
            padding-top: 1.4rem;
        }
        [data-testid="stMetric"] {
            border: 1px solid #64748b55;
            border-radius: 12px;
            padding: 12px;
        }
        @media(max-width:600px) {
            .block-container {padding:1rem}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("⚡ Crypto Screener · Risk & News")

    st.caption(
        "Classifica tecnica, liquidità, notizie e gestione "
        "del rischio · Solo analisi long · Nessun ordine inviato"
    )

    engine = engine_resource()

    positions = st.session_state.setdefault(
        "positions", {}
    )

    with st.sidebar:
        raw = st.text_area(
            "Watchlist",
            ", ".join(WATCH),
        )

        watch = list(dict.fromkeys(
            re.findall(
                r"\b[A-Z0-9]+(?:USDC|USDT)\b",
                raw.upper(),
            )
        ))[:40]

        mode = st.selectbox(
            "Mercato",
            [
                "Misto come prima",
                "Solo spot",
                "Solo perpetual",
            ],
        )

        capital = st.number_input(
            "Capitale disponibile (valuta della coppia)",
            min_value=0.0,
            value=10000.0,
        )

        risk_pct = st.number_input(
            "Rischio per scenario %",
            min_value=0.1,
            max_value=5.0,
            value=0.5,
            step=0.1,
        )

        fee = st.number_input(
            "Commissione per lato %",
            min_value=0.0,
            max_value=2.0,
            value=0.10,
            step=0.01,
        ) / 100

        slip = st.number_input(
            "Slippage prudenziale per lato %",
            min_value=0.0,
            max_value=2.0,
            value=0.10,
            step=0.01,
        ) / 100

        news_hours = st.selectbox(
            "Finestra allerta notizie (ore)",
            [3, 6, 12, 24],
            index=1,
        )

        manual = st.toggle(
            "Blocca nuovi ingressi: evento noto",
            value=False,
        )

        st.caption(
            "Notizie: controllo dei titoli RSS, "
            "non verifica dei fatti. Due feed richiesti "
            "per sbloccare gli ingressi. "
            "Posizioni salvate solo nella sessione."
        )

    @st.fragment(run_every="5s")
    def screen():
        # Il controllo uscite precede notizie e indicatori.
        monitor_positions(engine, positions)

        if not watch:
            st.info(
                "Inserisci almeno una coppia USDC/USDT."
            )
            return

        bundles = {}
        errors = {}
        rows = []
        plans = {}

        for symbol in watch:
            kind = (
                "spot"
                if mode == "Solo spot"
                else (
                    "perp"
                    if mode == "Solo perpetual"
                    or symbol in PERPS
                    else "spot"
                )
            )

            key = f"{symbol}:{kind}"

            pinned = positions.get(
                key, {}
            ).get("venue")

            value, error, pending = engine.poll(
                ("ta", key),
                "ta",
                technical,
                (engine, symbol, kind, pinned),
                90,
            )

            if error:
                errors[key] = error

            if value:
                bundles[key] = value

        btc = next(
            (
                bundle
                for key, bundle in bundles.items()
                if key.startswith("BTCUSDC:")
                and time.time() - bundle["stamp"] < 180
            ),
            None,
        )

        btc_ok = bool(
            btc
            and btc["frames"]["4h"].close.iloc[-1]
            > btc["frames"]["4h"].EMA25.iloc[-1]
        )

        for key, bundle in bundles.items():
            plan = analyse(
                bundle, btc_ok, fee, slip
            )

            plans[key] = plan

            last = bundle["frames"]["15m"].iloc[-1]
            hour = bundle["frames"]["1h"]

            stale = (
                time.time() - bundle["stamp"] > 180
                or key in errors
            )

            row = {
                "Coppia": key,
                "Fonte": NAMES[bundle["venue"]],
                "Score /100": plan["score"],
                "Stato": (
                    "DATI NON VALIDI"
                    if stale
                    else (
                        "SETUP"
                        if plan["ready"]
                        else "ATTENDI"
                    )
                ),
                "Chiusura 15m": last.close,
                "24h chiuse %": (
                    hour.close.iloc[-1]
                    / hour.close.iloc[-25]
                    - 1
                ) * 100,
                "Entry": plan["entry"],
                "Stop": plan["stop"],
                "Rischio %": plan["risk"],
                "R/R TP2 netto": plan["rr"][1],
                "RSI 1h": hour.RSI.iloc[-1],
                "ADX 1h": hour.ADX.iloc[-1],
                "RVOL 15m": last.RVOL,
            }

            for index in range(3):
                row.update({
                    f"TP{index + 1}":
                        plan["targets"][index],
                    f"TP{index + 1} netto %":
                        plan["net"][index],
                })

            rows.append(row)

        st.caption(
            f"Analisi disponibili {len(bundles)}/{len(watch)} · "
            "caricamento progressivo · dati tecnici ogni ~90 s"
        )

        if rows:
            table = pd.DataFrame(rows)

            table["_priorita"] = table.Stato.map({
                "SETUP": 2,
                "ATTENDI": 1,
            }).fillna(0)

            table = (
                table.sort_values(
                    ["_priorita", "Score /100"],
                    ascending=False,
                )
                .drop(columns="_priorita")
            )

            st.dataframe(
                table,
                hide_index=True,
                use_container_width=True,
            )

        if errors:
            with st.expander(
                "Connessioni e coppie non disponibili"
            ):
                st.json(errors)

        symbol = st.selectbox(
            "Analisi moneta",
            watch,
            key="coin",
        )

        kind = (
            "spot"
            if mode == "Solo spot"
            else (
                "perp"
                if mode == "Solo perpetual"
                or symbol in PERPS
                else "spot"
            )
        )

        key = f"{symbol}:{kind}"
        base, quote = split(symbol)

        feed_values = []
        feed_status = []

        for name, url in FEEDS.items():
            value, error, pending = engine.poll(
                ("news", name),
                "news",
                read_feed,
                (name, url),
                120,
            )

            feed_values.append(
                None if error else value
            )

            feed_status.append({
                "Fonte": name,
                "Stato": (
                    error
                    or (
                        "OK"
                        if value
                        else "Caricamento"
                    )
                ),
            })

        nfeeds, articles, news_alert = news_filter(
            feed_values,
            base,
            news_hours,
        )

        st.subheader(
            "Notizie e blocco ingressi"
        )

        st.write(
            f"Feed validi: {nfeeds}/{len(FEEDS)} · "
            + (
                "ALLERTA DA VERIFICARE"
                if news_alert
                else (
                    "Nessuna parola di allerta rilevata "
                    "nei titoli pertinenti"
                )
            )
        )

        for article in articles[:8]:
            stamp = datetime.fromtimestamp(
                article["ts"],
                timezone.utc,
            )

            st.write(
                ("⚠️ " if article["alert"] else "")
                + f"{article['source']} · "
                + f"{stamp:%d/%m %H:%M} UTC"
            )

            st.link_button(
                article["title"][:180],
                article["url"],
            )

        with st.expander(
            "Copertura notizie"
        ):
            st.dataframe(
                pd.DataFrame(feed_status),
                hide_index=True,
            )

            st.caption(
                "Nessuna allerta non significa assenza "
                "di rischio. Titoli ambigui o smentite "
                "possono attivare il filtro; verificare "
                "l’articolo. Unlock programmati e calendario "
                "macro non sono coperti integralmente."
            )

        if key not in bundles:
            st.info(
                "Analisi della moneta in caricamento "
                "o non disponibile."
            )
            return

        bundle = bundles[key]
        plan = plans[key]

        st.subheader(
            f"{symbol} · {kind} · "
            f"{NAMES[bundle['venue']]}"
        )

        if kind == "perp":
            st.caption(
                "Scenario senza leva; funding e liquidazioni "
                "non stimati. Quantità espresse in unità "
                "della crypto."
            )

        first, second, third = st.columns(3)

        first.metric(
            "Score euristico",
            f"{plan['score']}/100",
        )

        second.metric(
            "Entry scenario",
            fmt(plan["entry"]),
        )

        third.metric(
            "Stop scenario",
            fmt(plan["stop"]),
            f"-{plan['risk']:.2f}%",
        )

        targets = [
            {
                "Target": f"TP{index + 1}",
                "Prezzo": plan["targets"][index],
                "Netto stimato %": plan["net"][index],
                "R/R netto": plan["rr"][index],
            }
            for index in range(3)
        ]

        st.dataframe(
            pd.DataFrame(targets),
            hide_index=True,
            use_container_width=True,
        )

        if plan["resistance"]:
            st.caption(
                "Prima resistenza pivot 1H rilevata: "
                + fmt(plan["resistance"])
            )

        with st.expander(
            "Indicatori e regole della classifica"
        ):
            columns = [
                "close", "RSI", "K", "D",
                "EMA7", "EMA25", "EMA50",
                "EMA99", "EMA200", "MACD",
                "ADX", "ATR", "Supertrend",
                "BB_low", "BB_high",
                "RVOL", "Delta_taker",
            ]

            st.dataframe(
                pd.DataFrame({
                    timeframe: frame.iloc[-1][columns]
                    for timeframe, frame
                    in bundle["frames"].items()
                }).T,
                use_container_width=True,
            )

            st.dataframe(
                pd.DataFrame([
                    {
                        "Criterio": name,
                        "Superato": bool(passed),
                        "Peso": weight,
                    }
                    for name, (passed, weight)
                    in plan["checks"].items()
                ]),
                hide_index=True,
            )

            st.caption(
                "Score di condizioni tecniche, "
                "non probabilità. Candela 15m chiusa: "
                + str(pd.to_datetime(
                    plan["bar"],
                    unit="ms",
                    utc=True,
                ))
            )

        open_risk = sum(
            max(
                0,
                position["qty"] * (
                    position["entry"]
                    * (1 + position["fee"])
                    - position["stop"]
                    * (1 - position["fee"])
                    * (1 - slip)
                ),
            )
            for position_key, position
            in positions.items()
            if split(
                position_key.split(":")[0]
            )[1] == quote
        )

        budget = min(
            capital * risk_pct / 100,
            max(
                0,
                capital * 0.02 - open_risk,
            ),
        )

        quantity = (
            min(
                capital / plan["cost"],
                budget / plan["unit_loss"],
            )
            if plan["unit_loss"] > 0
            else 0
        )

        st.info(
            "Calcolatore: quantità massima indicativa "
            f"{quantity:.8g} {base} · "
            "impiego inclusi costi "
            f"{quantity * plan['cost']:.2f} {quote} · "
            "perdita allo stop stimata "
            f"{quantity * plan['unit_loss']:.2f} {quote}. "
            "Limite rischio aggregato 2% sul capitale, "
            "per la stessa valuta. Gaps possono superarlo."
        )

        books = []
        reference = None
        book_errors = {}

        operational_venue = positions.get(
            key, {}
        ).get(
            "venue",
            bundle["venue"],
        )

        for venue in VENUES:
            exact = venue == operational_venue

            value, error, pending = engine.poll(
                (
                    "book", venue, base,
                    quote, kind, exact,
                ),
                "flow",
                book_job,
                (
                    engine, venue, base,
                    quote, kind, exact,
                ),
                10,
            )

            if error:
                book_errors[NAMES[venue]] = error

            if value and not error:
                books.append(value)

                if exact:
                    reference = value

        flow = consensus(books)

        st.subheader(
            "Book e trade multi-exchange"
        )

        if flow:
            st.write(
                f"{flow['n']} sedi · "
                "fascia comune "
                f"±{flow['band'] * 100:.3f}% · "
                "mediana imbalance "
                f"{flow['book']:+.1%}"
            )

        st.caption(
            "Mercati spot e perpetual separati. "
            "Consenso di percentuali per sede; "
            "nessuna somma artificiale di USD/USDC/USDT. "
            "Campioni REST, non CVD completo."
        )

        if books:
            st.dataframe(
                pd.DataFrame([
                    {
                        "Sede": NAMES[book["venue"]],
                        "Coppia": book["pair"],
                        "Età s": round(
                            time.time() - book["stamp"],
                            1,
                        ),
                        "Spread %": book["spread"] * 100,
                        "Delta campione %": (
                            None
                            if book["delta"] is None
                            else book["delta"] * 100
                        ),
                        "Trade noti": book["trades"],
                        "Intervallo campione s": round(
                            book["span"], 1
                        ),
                    }
                    for book in books
                ]),
                hide_index=True,
                use_container_width=True,
            )

        with st.expander(
            "Diagnostica book"
        ):
            st.json(book_errors)

        if reference:
            try:
                quantity = tradable_qty(
                    reference,
                    quantity,
                )
            except Exception:
                quantity = 0

        reasons = entry_reasons(
            bundle,
            plan,
            reference,
            flow,
            nfeeds == len(FEEDS),
            news_alert,
            manual,
            quantity,
        )

        if key in errors:
            reasons.append(
                "Ultima lettura tecnica fallita"
            )

        observation_key = (
            f"obs:{key}:{bundle['venue']}:{plan['bar']}"
        )

        observation = st.session_state.setdefault(
            observation_key,
            {"time": 0, "count": 0},
        )

        if reasons:
            observation["count"] = 0

        elif (
            reference
            and reference["stamp"]
            > observation["time"] + 5
        ):
            observation["count"] = (
                observation["count"] + 1
                if (
                    reference["stamp"]
                    - observation["time"]
                    < 45
                )
                else 1
            )

            observation["time"] = reference["stamp"]

        if key not in positions:
            if reasons:
                st.warning(
                    "ATTENDI · "
                    + "; ".join(dict.fromkeys(reasons))
                )

            elif observation["count"] < 3:
                st.info(
                    "Conferme consecutive "
                    f"{observation['count']}/3"
                )

            else:
                st.success(
                    "SETUP LONG CONFERMATO DALLE REGOLE · "
                    "verifica il prezzo prima di operare"
                )

        with st.expander(
            "Registra posizione reale e monitora le uscite"
        ):
            st.caption(
                "Inserisci eseguito medio e quantità reali. "
                "Registrazione manuale, nessun ordine. "
                "Salva il JSON sotto per ripristinare "
                "la posizione dopo la chiusura della sessione."
            )

            saved = positions.get(key, {})

            form_key = (
                key
                + ":"
                + str(st.session_state.get(
                    "form_version", 0
                ))
            )

            with st.form("posform:" + key):
                actual = st.number_input(
                    "Prezzo medio eseguito",
                    min_value=0.0,
                    value=float(saved.get(
                        "entry", plan["entry"]
                    )),
                    format="%.8f",
                    key="e:" + form_key,
                )

                amount = st.number_input(
                    "Quantità reale (unità crypto)",
                    min_value=0.0,
                    value=float(saved.get(
                        "qty", quantity
                    )),
                    format="%.8f",
                    key="q:" + form_key,
                )

                stop = st.number_input(
                    "Stop reale",
                    min_value=0.0,
                    value=float(saved.get(
                        "stop", plan["stop"]
                    )),
                    format="%.8f",
                    key="s:" + form_key,
                )

                tps = [
                    st.number_input(
                        f"TP{index + 1} reale",
                        min_value=0.0,
                        value=float(
                            saved.get(
                                "targets",
                                plan["targets"],
                            )[index]
                        ),
                        format="%.8f",
                        key=f"t{index}:" + form_key,
                    )
                    for index in range(3)
                ]

                save = st.form_submit_button(
                    "Registra / aggiorna livelli"
                )

            if save:
                valid_levels = (
                    0 < stop < actual
                    < tps[0] < tps[1] < tps[2]
                    and amount > 0
                )

                if not valid_levels:
                    st.error(
                        "Richiesti 0 < stop < ingresso "
                        "< TP1 < TP2 < TP3 e quantità >0."
                    )
                else:
                    positions[key] = {
                        "entry": actual,
                        "qty": amount,
                        "stop": stop,
                        "targets": tps,
                        "venue": bundle["venue"],
                        "fee": fee,
                        "events": [],
                    }

                    st.success(
                        "Posizione registrata; livelli fissi "
                        "fino a una tua modifica."
                    )

            if key in positions:
                st.json(positions[key])

                if st.button(
                    "Rimuovi: posizione chiusa manualmente",
                    key="remove:" + key,
                ):
                    del positions[key]
                    st.rerun(scope="fragment")

        with st.expander(
            "Copia / ripristina posizioni"
        ):
            st.code(
                json.dumps(
                    positions,
                    ensure_ascii=False,
                    indent=2,
                ),
                language="json",
            )

            text = st.text_area(
                "Incolla JSON precedentemente salvato",
                key="restore_text",
            )

            if st.button(
                "Ripristina",
                key="restore",
            ):
                try:
                    loaded = json.loads(text)

                    if (
                        not isinstance(loaded, dict)
                        or len(loaded) > 40
                    ):
                        raise ValueError(
                            "Formato non valido"
                        )

                    for position_key, position in loaded.items():
                        saved_symbol, saved_kind = (
                            position_key.split(":")
                        )

                        split(saved_symbol)

                        if (
                            saved_kind not in ("spot", "perp")
                            or position["venue"] not in VENUES
                        ):
                            raise ValueError(
                                "Mercato non valido"
                            )

                        values = [
                            position["entry"],
                            position["qty"],
                            position["stop"],
                            *position["targets"],
                            position["fee"],
                        ]

                        if (
                            not all(
                                num(value) is not None
                                for value in values
                            )
                            or len(position["targets"]) != 3
                        ):
                            raise ValueError(
                                "Numeri non validi"
                            )

                        if (
                            not (
                                0
                                < position["stop"]
                                < position["entry"]
                                < position["targets"][0]
                                < position["targets"][1]
                                < position["targets"][2]
                            )
                            or position["qty"] <= 0
                            or not 0 <= position["fee"] <= 0.02
                        ):
                            raise ValueError(
                                "Livelli non validi"
                            )

                        position["events"] = []

                    positions.update(loaded)

                    st.session_state["form_version"] = (
                        st.session_state.get(
                            "form_version", 0
                        )
                        + 1
                    )

                    st.rerun(scope="fragment")

                except (
                    ValueError,
                    TypeError,
                    KeyError,
                ):
                    st.error(
                        "JSON non valido; "
                        "nessuna posizione importata."
                    )

        st.caption(
            "Software sperimentale: soglie non ottimizzate "
            "né validate da un backtest. Aggiornamenti "
            "solo a pagina aperta; nessuno stop è eseguito "
            "dall’app."
        )

    screen()


if __name__ == "__main__":
    app()
