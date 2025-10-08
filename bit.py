#!/usr/bin/env python3
"""
futures_signals_ai.py — production-ready single-process bot, now with
online training on emulated trade outcomes.

When a signal opens:
- open pseudo-trade for $1000,
- close on TP/SL,
- on close compute PnL and label (1 if pnl>0 else 0),
- immediately partial_fit(StandardScaler, SGDClassifier) with that sample,
- save model/scaler after update.

Everything else: light->heavy pipeline, rate-limit backoff, Telethon/HTML fallback, etc.
"""
import os
import argparse
import asyncio
import aiohttp
import math
import time
import json
import re
from pathlib import Path
from datetime import datetime
from collections import deque, defaultdict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from joblib import dump, load

# ML
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

# Telethon (optional)
try:
    from telethon import TelegramClient, events
    TELETHON_AVAILABLE = True
except Exception:
    TELETHON_AVAILABLE = False

# ---------------- Config & env ----------------
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
TG_API_ID = os.getenv("TELEGRAM_API_ID")
TG_API_HASH = os.getenv("TELEGRAM_API_HASH")

BINANCE_API = "https://fapi.binance.com"
LIQ_CHANNEL = os.getenv("LIQ_CHANNEL", "BinanceLiquidations")

# CLI
parser = argparse.ArgumentParser()
parser.add_argument("--heavy", action="store_true", help="enable heavy computations")
args = parser.parse_args()
HEAVY = args.heavy

# Tunables (env overrides possible)
TOP_N = int(os.getenv("TOP_N", "20"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60")) if not HEAVY else int(os.getenv("POLL_INTERVAL_HEAVY", "30"))
MINUTE_OI_THRESHOLD_PCT = float(os.getenv("MINUTE_OI_THRESHOLD_PCT", "5.0"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "2.0"))
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", str(60*60*2)))
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "6"))  # controls semaphore
CANDIDATE_LIMIT = int(os.getenv("CANDIDATE_LIMIT", "6"))  # how many symbols get heavy scan per cycle

# indicators params
EMA_FAST = int(os.getenv("EMA_FAST", "9"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "21"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_SL_MULT = float(os.getenv("ATR_SL_MULT", "1.5"))
RR = float(os.getenv("RR", "2.0"))
VOLUME_SURGE_MULT = float(os.getenv("VOLUME_SURGE_MULT", "1.6"))
OI_1H_THRESHOLD_PCT = float(os.getenv("OI_1H_THRESHOLD_PCT", "3.0"))

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(exist_ok=True)
SENT_FILE = DATA_DIR / "sent_signals.json"
MODEL_FILE = DATA_DIR / "sgd_model.joblib"
SCALER_FILE = DATA_DIR / "scaler.joblib"
TRADE_LOG_FILE = DATA_DIR / "trade_history.json"
LOG_FILE = DATA_DIR / "bot.log"

# regex for parsing liq text
RE_LIQ = re.compile(r"\b([A-Z0-9]{3,8}USDT)\b.*?\b(LONG|SHORT)\b.*?\$?([\d,\,\.]+)", re.IGNORECASE)

# ---------------- State ----------------
LAST_OI = {}
LIQ_QUEUE = asyncio.Queue()
LIQ_FLOW = defaultdict(lambda: deque(maxlen=500))
FEATURE_HISTORY = defaultdict(lambda: deque(maxlen=2000))
TRADE_HISTORY = []
SENT = {}
SENT_LOCK = asyncio.Lock()
TRADE_LOCK = asyncio.Lock()

# ML state
MODEL = None
SCALER = None
MODEL_FITTED = False
RET_THR = float(os.getenv("RET_THR", "0.001"))
LABEL_HORIZON_MINUTES = int(os.getenv("LABEL_HORIZON_MINUTES", "5"))

# HTTP concurrency
SEM = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# ---------------- Logging helper ----------------
def now_iso():
    return datetime.utcnow().isoformat(sep=' ', timespec='seconds')

def log(*args, **kwargs):
    s = " ".join(str(a) for a in args)
    line = f"[{now_iso()}] {s}"
    print(line, **kwargs)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass

def save_json(path: Path, obj):
    try:
        path.write_text(json.dumps(obj, indent=2))
    except Exception as e:
        log("save_json error", e)

def load_json(path: Path, default=None):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return default

# ---------------- HTTP + backoff ----------------
async def fetch_json(session: aiohttp.ClientSession, url: str, params: dict = None, timeout=15, retries=4):
    backoff = 1.0
    for attempt in range(retries):
        try:
            async with SEM:
                async with session.get(url, params=params, timeout=timeout) as resp:
                    txt = await resp.text()
                    if resp.status == 200:
                        try:
                            return json.loads(txt)
                        except Exception:
                            return None
                    elif resp.status in (429, 418):
                        log("Rate limited", resp.status, url, "params", params)
                        await asyncio.sleep(backoff)
                        backoff *= 2
                    elif 500 <= resp.status < 600:
                        log("Server error", resp.status, "sleep", backoff)
                        await asyncio.sleep(backoff)
                        backoff *= 2
                    else:
                        try:
                            return json.loads(txt)
                        except Exception:
                            log("HTTP error", resp.status, url, txt[:200])
                            return None
        except asyncio.TimeoutError:
            log("Timeout", url, "attempt", attempt)
            await asyncio.sleep(backoff); backoff *= 2
        except Exception as e:
            log("Fetch exception", e, url)
            await asyncio.sleep(backoff); backoff *= 2
    return None

# ---------------- Technical indicators ----------------
def ema(series: pd.Series, period: int):
    return series.ewm(span=period, adjust=False).mean()

def atr(df: pd.DataFrame, period: int = 14):
    prev = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev).abs()
    tr3 = (df['low'] - prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))

def vwap_from_df(df: pd.DataFrame):
    pv = df['close'] * df['volume']
    return pv.cumsum() / df['volume'].cumsum()

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist

def bollinger_bands(series: pd.Series, n=20, k=2):
    ma = series.rolling(n).mean()
    std = series.rolling(n).std()
    upper = ma + k * std
    lower = ma - k * std
    width = (upper - lower) / ma.replace(0, np.nan)
    return ma, upper, lower, width

def obv(df: pd.DataFrame):
    direction = np.sign(df['close'].diff().fillna(0))
    return (direction * df['volume']).cumsum()

def adx(df: pd.DataFrame, n=14):
    up = df['high'].diff()
    down = -df['low'].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    atr_series = tr.rolling(n).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(n).sum() / (atr_series + 1e-12))
    minus_di = 100 * (pd.Series(minus_dm).rolling(n).sum() / (atr_series + 1e-12))
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12)) * 100
    adx_val = pd.Series(dx).rolling(n).mean()
    return adx_val

def keltner_channel(df: pd.DataFrame, ema_period=20, atr_period=10, atr_mult=1.5):
    ema_mid = df['close'].ewm(span=ema_period, adjust=False).mean()
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift()).abs()
    tr3 = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_series = tr.rolling(atr_period).mean()
    upper = ema_mid + atr_mult * atr_series
    lower = ema_mid - atr_mult * atr_series
    return ema_mid, upper, lower

# ---------------- Binance helpers (cached per-cycle) ----------------
async def fetch_top_symbols(session: aiohttp.ClientSession, limit=TOP_N):
    url = f"{BINANCE_API}/fapi/v1/ticker/24hr"
    data = await fetch_json(session, url)
    if not data:
        return []
    df = pd.DataFrame(data)
    df['quoteVolume'] = pd.to_numeric(df.get('quoteVolume', 0), errors='coerce').fillna(0)
    top = df.sort_values('quoteVolume', ascending=False).head(limit)
    return top['symbol'].tolist()

async def fetch_klines(session: aiohttp.ClientSession, symbol: str, interval: str, limit: int = 200):
    url = f"{BINANCE_API}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    data = await fetch_json(session, url, params=params)
    if not data or not isinstance(data, list):
        return None
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_av", "trades", "tb_base_av", "tb_quote_av", "ignore"
    ])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    return df

async def fetch_current_oi(session: aiohttp.ClientSession, symbol: str):
    url = f"{BINANCE_API}/fapi/v1/openInterest"
    params = {"symbol": symbol}
    d = await fetch_json(session, url, params=params)
    if d and "openInterest" in d:
        try:
            return float(d["openInterest"])
        except:
            return None
    return None

async def fetch_open_interest_hist(session: aiohttp.ClientSession, symbol: str, period: str = '1h', limit: int = 25):
    url = f"{BINANCE_API}/futures/data/openInterestHist"
    params = {"symbol": symbol, "period": period, "limit": limit}
    return await fetch_json(session, url, params=params)

async def fetch_funding(session: aiohttp.ClientSession, symbol: str, limit: int = 6):
    url = f"{BINANCE_API}/fapi/v1/fundingRate"
    params = {"symbol": symbol, "limit": limit}
    return await fetch_json(session, url, params=params)

async def orderbook_imbalance(session: aiohttp.ClientSession, symbol: str, limit: int = 100):
    url = f"{BINANCE_API}/fapi/v1/depth"
    params = {"symbol": symbol, "limit": min(limit, 100)}
    data = await fetch_json(session, url, params=params)
    if not data:
        return 0.0
    bids = data.get('bids', [])
    asks = data.get('asks', [])
    topN = 10
    bid_vol = sum(float(b[1]) for b in bids[:topN])
    ask_vol = sum(float(a[1]) for a in asks[:topN])
    if (bid_vol + ask_vol) == 0:
        return 0.0
    return (bid_vol - ask_vol) / (bid_vol + ask_vol)

async def whale_flow_from_aggtrades(session: aiohttp.ClientSession, symbol: str, lookback: int = 500, size_ratio: float = 0.02):
    url = f"{BINANCE_API}/fapi/v1/aggTrades"
    params = {"symbol": symbol, "limit": min(lookback, 1000)}
    data = await fetch_json(session, url, params=params)
    if not data or not isinstance(data, list):
        return {"net_whale": 0.0, "count_whale": 0}
    sizes = np.array([float(d['q']) for d in data])
    avg = sizes.mean() if sizes.size else 0.0
    thr = max(avg * (1 + size_ratio), np.percentile(sizes, 90)) if sizes.size else 0.0
    net = 0.0; cnt = 0
    for d in data:
        q = float(d['q']); is_buyer = not d.get('m', False)
        if q >= thr and thr > 0:
            net += q if is_buyer else -q
            cnt += 1
    return {"net_whale": float(net), "count_whale": int(cnt)}

# ---------------- Support/Resistance helper ----------------
def support_resistance_levels(highs: pd.Series, lows: pd.Series, lookback: int = 72, peaks: int = 3):
    highs_window = highs[-lookback:]
    lows_window = lows[-lookback:]
    try:
        res_levels = sorted(highs_window.sort_values(ascending=False).head(peaks).values.tolist())
        sup_levels = sorted(lows_window.sort_values().head(peaks).values.tolist())
    except Exception:
        res_levels = []; sup_levels = []
    return sup_levels, res_levels

# ---------------- ML helpers ----------------
def init_model():
    global MODEL, SCALER, MODEL_FITTED
    if MODEL is None:
        MODEL = SGDClassifier(loss='log')
    if SCALER is None:
        SCALER = StandardScaler()
    try:
        if MODEL_FILE.exists():
            MODEL = load(MODEL_FILE); MODEL_FITTED = True
        if SCALER_FILE.exists():
            sc = load(SCALER_FILE); SCALER = sc
    except Exception as e:
        log("model load error", e)

def save_model():
    try:
        if MODEL is not None: dump(MODEL, MODEL_FILE)
        if SCALER is not None: dump(SCALER, SCALER_FILE)
    except Exception as e:
        log("model save err", e)

def features_vec(feat: dict):
    """
    Feature vector shape used for ML.
    Ensure keys exist (fill with 0.0 if missing).
    """
    # keys chosen to be robust and available in heavy features
    v_whale = float(feat.get("whale", {}).get("net_whale", 0.0)) if isinstance(feat.get("whale"), dict) else float(feat.get("whale", 0.0) if feat.get("whale") is not None else 0.0)
    v_vwap_diff = 0.0
    if feat.get("price") is not None and feat.get("vwap") is not None:
        try: v_vwap_diff = float(feat["price"]) - float(feat["vwap"])
        except: v_vwap_diff = 0.0
    keys = [
        "oi_1m","price_ret_1m","price_ret_5m","price_ret_1h",
        "rsi","ema_diff","atr","vol_ratio",
        "orderbook_imb"
    ]
    vec = []
    for k in keys:
        vec.append(float(feat.get(k, 0.0) if feat.get(k) is not None else 0.0))
    vec.append(v_whale)     # whale_net
    vec.append(v_vwap_diff) # vwap_diff
    vec.append(float(feat.get("macd_hist", 0.0)))
    return np.array(vec, dtype=float)

# ---------------- Liquidation ingestion ----------------
async def start_telethon(loop):
    if not TELETHON_AVAILABLE:
        log("Telethon not installed")
        return
    if not TG_API_ID or not TG_API_HASH:
        log("No TELEGRAM_API_ID/API_HASH -> HTML fallback")
        return
    client = TelegramClient("telethon_session", int(TG_API_ID), TG_API_HASH, loop=loop)
    @client.on(events.NewMessage(chats=LIQ_CHANNEL))
    async def handler(event):
        text = (event.message.message or event.message.raw_text or "")
        m = RE_LIQ.search(text)
        parsed = {"raw": text, "time": now_iso()}
        if m:
            try:
                parsed.update({"symbol": m.group(1).upper(), "side": m.group(2).upper(), "size": float(m.group(3).replace(",",""))})
            except:
                pass
        await LIQ_QUEUE.put(parsed)
        log("telethon queued", parsed.get("symbol"))
    await client.start()
    log("Telethon started")
    await client.run_until_disconnected()

async def html_liq_poller(session: aiohttp.ClientSession):
    poll_interval = int(os.getenv("HTML_LIQ_POLL_INTERVAL", "30"))
    while True:
        try:
            url = f"https://t.me/s/{LIQ_CHANNEL}"
            async with SEM:
                async with session.get(url, timeout=15) as r:
                    html = await r.text()
            soup = BeautifulSoup(html, "lxml")
            msgs = soup.find_all("div", class_="tgme_widget_message_text")
            for m in msgs[-40:]:
                text = m.get_text(" ", strip=True)
                if "liquidation" in text.lower() or "long" in text.lower() or "short" in text.lower():
                    parsed = {"raw": text, "time": now_iso()}
                    mm = RE_LIQ.search(text)
                    if mm:
                        try:
                            parsed.update({"symbol": mm.group(1).upper(), "side": mm.group(2).upper(), "size": float(mm.group(3).replace(",", ""))})
                        except:
                            pass
                    await LIQ_QUEUE.put(parsed)
        except Exception as e:
            log("html_liq_poller err", e)
        await asyncio.sleep(poll_interval)

async def liq_consumer():
    while True:
        try:
            ev = await LIQ_QUEUE.get()
            if not ev:
                LIQ_QUEUE.task_done(); continue
            sym = ev.get("symbol"); side = ev.get("side"); size = ev.get("size", 0)
            if sym and side:
                val = size if side.upper() == "SHORT" else -size
                LIQ_FLOW[sym].append((time.time(), val))
                log("LIQ_FLOW add", sym, val)
            LIQ_QUEUE.task_done()
        except Exception as e:
            log("liq_consumer err", e)
            await asyncio.sleep(0.5)

# ---------------- Feature computation (light & heavy) ----------------
async def compute_light_features(session: aiohttp.ClientSession, symbol: str, cache: dict):
    df1h = await fetch_klines(session, symbol, "1h", limit=200)
    if df1h is None or df1h.empty:
        return None
    p1h = float(df1h['close'].iloc[-1]); p1h_prev = float(df1h['close'].iloc[-2])
    price_ret_1h = safe_div(p1h - p1h_prev, p1h_prev)
    ema_f = float(ema(df1h['close'], EMA_FAST).iloc[-1])
    ema_s = float(ema(df1h['close'], EMA_SLOW).iloc[-1])
    ema_diff = safe_div(ema_f - ema_s, p1h)
    atr_val = float(atr(df1h, ATR_PERIOD).iloc[-1]) if not df1h.empty else 0.0
    rsi_val = float(rsi(df1h['close'], RSI_PERIOD).iloc[-1])
    vol_ratio = safe_div(df1h['volume'].iloc[-1], df1h['volume'].rolling(24).mean().iloc[-1] if not math.isnan(df1h['volume'].rolling(24).mean().iloc[-1]) else 1.0)

    oi_now = await fetch_current_oi(session, symbol)
    oi_1h_pct = None
    oi_hist = await fetch_open_interest_hist(session, symbol, period='1h', limit=25)
    if oi_hist and isinstance(oi_hist, list) and len(oi_hist) >= 2:
        try:
            st = float(oi_hist[0].get('sumOpenInterestValue') or oi_hist[0].get('sumOpenInterest', 0))
            en = float(oi_hist[-1].get('sumOpenInterestValue') or oi_hist[-1].get('sumOpenInterest', 0))
            if st > 0:
                oi_1h_pct = safe_div(en - st, st) * 100.0
        except:
            oi_1h_pct = None

    cache[symbol] = {"oi_last": oi_now, "last_price": float(df1h['close'].iloc[-1])}
    feat = {
        "symbol": symbol,
        "time": now_iso(),
        "price": float(df1h['close'].iloc[-1]),
        "price_ret_1h": price_ret_1h,
        "ema_diff": ema_diff,
        "atr": atr_val,
        "rsi": rsi_val,
        "vol_ratio": vol_ratio,
        "oi_now": oi_now,
        "oi_1h": oi_1h_pct
    }
    return feat

async def compute_heavy_features(session: aiohttp.ClientSession, symbol: str, cache: dict):
    df1m = await fetch_klines(session, symbol, "1m", limit=120)
    df5m = await fetch_klines(session, symbol, "5m", limit=60)
    df1h = await fetch_klines(session, symbol, "1h", limit=240)
    if df1m is None or df1h is None:
        return None
    p1 = float(df1m['close'].iloc[-1]); p1_prev = float(df1m['close'].iloc[-2])
    p5 = float(df5m['close'].iloc[-1]) if df5m is not None else p1
    p1h = float(df1h['close'].iloc[-1]); p1h_prev = float(df1h['close'].iloc[-2])
    price_ret_1m = safe_div(p1 - p1_prev, p1_prev)
    price_ret_5m = safe_div(p5 - float(df5m['close'].iloc[-2]) if df5m is not None else 0.0, p5) if df5m is not None else 0.0
    price_ret_1h = safe_div(p1h - p1h_prev, p1h_prev)

    rsi_val = float(rsi(df1m['close'], RSI_PERIOD).iloc[-1])
    ema_f = float(ema(df1h['close'], EMA_FAST).iloc[-1]); ema_s = float(ema(df1h['close'], EMA_SLOW).iloc[-1])
    ema_diff = safe_div(ema_f - ema_s, p1h)
    atr_val = float(atr(df1h, ATR_PERIOD).iloc[-1]) if not df1h.empty else 0.0
    vol_ratio = safe_div(df1h['volume'].iloc[-1], df1h['volume'].rolling(24).mean().iloc[-1] if not math.isnan(df1h['volume'].rolling(24).mean().iloc[-1]) else 1.0)

    oi_now = await fetch_current_oi(session, symbol)
    oi_1m = 0.0
    prev = cache.get(symbol)
    if prev and prev.get('oi_last'):
        prev_oi = prev['oi_last']
        if prev_oi and prev_oi != 0:
            oi_1m = safe_div(oi_now - prev_oi, prev_oi) * 100.0

    fund = await fetch_funding(session, symbol, limit=6)
    funding_avg = None
    if fund and isinstance(fund, list):
        vals = []
        for it in fund:
            try: vals.append(float(it.get('fundingRate', 0)))
            except: pass
        if vals: funding_avg = sum(vals) / len(vals)

    # heavy additions
    ob_imb = await orderbook_imbalance(session, symbol, limit=100)
    whale = await whale_flow_from_aggtrades(session, symbol, lookback=500, size_ratio=0.02)
    vwap_val = None
    try:
        v = vwap_from_df(df1h)
        vwap_val = float(v.iloc[-1])
    except Exception:
        vwap_val = None
    macd_line = macd_signal = macd_hist = None
    try:
        macd_line, macd_signal, macd_hist = macd(df1h['close'])
        macd_line = float(macd_line.iloc[-1]); macd_signal = float(macd_signal.iloc[-1]); macd_hist = float(macd_hist.iloc[-1])
    except Exception:
        macd_line = macd_signal = macd_hist = None
    bb_mid = bb_up = bb_low = bb_width = None
    try:
        bb_mid, bb_up, bb_low, bb_width = bollinger_bands(df1h['close'], n=20, k=2)
        bb_mid = float(bb_mid.iloc[-1]); bb_up = float(bb_up.iloc[-1]); bb_low = float(bb_low.iloc[-1]); bb_width = float(bb_width.iloc[-1])
    except Exception:
        bb_mid = bb_up = bb_low = bb_width = None
    obv_series = None
    try:
        obv_series = float(obv(df1h).iloc[-1])
    except Exception:
        obv_series = None
    adx_series = None
    try:
        adx_series = float(adx(df1h, n=14).iloc[-1])
    except Exception:
        adx_series = None
    k_mid = k_up = k_low = None
    try:
        k_mid, k_up, k_low = keltner_channel(df1h, ema_period=20, atr_period=10, atr_mult=1.5)
        k_mid = float(k_mid.iloc[-1]); k_up = float(k_up.iloc[-1]); k_low = float(k_low.iloc[-1])
    except Exception:
        k_mid = k_up = k_low = None
    sup_levels, res_levels = support_resistance_levels(df1h['high'], df1h['low'], lookback=72, peaks=3)

    # vwap_diff
    vwap_diff = None
    if vwap_val is not None:
        try:
            vwap_diff = float(p1) - float(vwap_val)
        except:
            vwap_diff = None

    cache[symbol] = {"oi_last": oi_now, "last_price": float(p1)}
    feat = {
        "symbol": symbol, "time": now_iso(), "price": float(p1),
        "oi_now": oi_now, "oi_1m": oi_1m, "oi_1h": None,
        "price_ret_1m": price_ret_1m, "price_ret_5m": price_ret_5m, "price_ret_1h": price_ret_1h,
        "rsi": rsi_val, "ema_diff": ema_diff, "atr": atr_val, "vol_ratio": vol_ratio, "funding": funding_avg,
        "orderbook_imb": ob_imb, "whale": whale, "vwap": vwap_val, "vwap_diff": vwap_diff,
        "macd_line": macd_line, "macd_signal": macd_signal, "macd_hist": macd_hist,
        "bb_mid": bb_mid, "bb_up": bb_up, "bb_low": bb_low, "bb_width": bb_width,
        "obv": obv_series, "adx": adx_series, "k_mid": k_mid, "k_up": k_up, "k_low": k_low,
        "sup_levels": sup_levels, "res_levels": res_levels
    }
    FEATURE_HISTORY[symbol].append({"time": time.time(), "feat": feat})
    return feat

# ---------------- Signal generation (light/heavy) ----------------
async def generate_signal_light(feat: dict):
    score = 0.0
    reasons = []
    if feat.get("ema_diff") is not None:
        score += np.sign(feat["ema_diff"])
    if feat.get("rsi") is not None:
        if feat["rsi"] > 58:
            score += 1.0; reasons.append("rsi_hi")
        elif feat["rsi"] < 42:
            score -= 1.0; reasons.append("rsi_lo")
    if feat.get("oi_1h") is not None:
        if feat["oi_1h"] > OI_1H_THRESHOLD_PCT: score += 0.8; reasons.append("oi_1h+")
        elif feat["oi_1h"] < -OI_1H_THRESHOLD_PCT: score -= 0.8; reasons.append("oi_1h-")
    if feat.get("vol_ratio") and feat["vol_ratio"] > VOLUME_SURGE_MULT:
        score += 0.5; reasons.append("vol_surge")
    return {"symbol": feat["symbol"], "score": score, "reasons": reasons}

async def generate_signal_heavy(session: aiohttp.ClientSession, feat: dict):
    w = {"ema":1.0,"rsi":0.9,"macd":1.0,"oi":1.5,"vol":0.8,"ob":1.2,"whale":1.6,"adx":0.6,"bb":0.6,"vwap":0.5,"liq":1.2}
    score = 0.0; reasons = []

    if feat.get("ema_diff") is not None:
        score += w["ema"] * np.sign(feat["ema_diff"]); reasons.append(f"ema:{feat['ema_diff']:.4f}")
    if feat.get("rsi") is not None:
        if feat["rsi"] > 60:
            score += w["rsi"]; reasons.append(f"rsi{feat['rsi']:.0f}")
        elif feat["rsi"] < 40:
            score -= w["rsi"]; reasons.append(f"rsi{feat['rsi']:.0f}")
    if feat.get("macd_line") is not None and feat.get("macd_signal") is not None:
        macd_delta = feat["macd_line"] - feat["macd_signal"]
        score += w["macd"] * np.sign(macd_delta); reasons.append(f"macd{macd_delta:.4f}")
    if feat.get("bb_up") and feat.get("bb_low"):
        price = feat["price"]
        if price > feat["bb_up"]:
            score += w["bb"]; reasons.append("bb_break_up")
        elif price < feat["bb_low"]:
            score -= w["bb"]; reasons.append("bb_break_dn")
    if feat.get("vwap") is not None:
        if feat["price"] > feat["vwap"]:
            score += w["vwap"]; reasons.append("above_vwap")
        else:
            score -= w["vwap"]; reasons.append("below_vwap")
    if feat.get("obv") is not None and feat.get("adx") is not None:
        if feat["obv"] > 0 and feat["adx"] > 20:
            score += w["ob"]; reasons.append("obv_adx")
        elif feat["obv"] < 0 and feat["adx"] > 20:
            score -= w["ob"]; reasons.append("obv_adx_dn")
    ob = feat.get("orderbook_imb") or 0.0
    if ob > 0.2:
        score += w["ob"]; reasons.append(f"ob_imb+{ob:.2f}")
    elif ob < -0.2:
        score -= w["ob"]; reasons.append(f"ob_imb{ob:.2f}")
    wh = feat.get("whale", {}) or {}
    if wh and wh.get("net_whale"):
        if wh["net_whale"] > 0:
            score += w["whale"]; reasons.append(f"wh_buy{wh['net_whale']:.1f}")
        elif wh["net_whale"] < 0:
            score -= w["whale"]; reasons.append(f"wh_sell{wh['net_whale']:.1f}")
    if feat.get("oi_1m") is not None:
        if feat["oi_1m"] >= MINUTE_OI_THRESHOLD_PCT:
            score += w["oi"]; reasons.append(f"oi1m+{feat['oi_1m']:.1f}%")
        elif feat["oi_1m"] <= -MINUTE_OI_THRESHOLD_PCT:
            score -= w["oi"]; reasons.append(f"oi1m{feat['oi_1m']:.1f}%")
    liq_sum = 0.0; nowt = time.time()
    for (t, v) in list(LIQ_FLOW.get(feat["symbol"], [])):
        if nowt - t <= 600: liq_sum += v
    if abs(liq_sum) > 1_000_000:
        score += w["liq"] * np.sign(liq_sum); reasons.append(f"liq{liq_sum:.0f}")

    init_model()
    ml_prob = None
    try:
        vec = features_vec(feat).reshape(1, -1)
        if SCALER is not None:
            try:
                vec_s = SCALER.transform(vec)
            except Exception:
                vec_s = vec
        else:
            vec_s = vec
        if MODEL_FITTED and hasattr(MODEL, "predict_proba"):
            ml_prob = float(MODEL.predict_proba(vec_s)[0][1])
            score += 1.5 * ((ml_prob - 0.5) * 2)
            reasons.append(f"MLp{ml_prob:.2f}")
    except Exception:
        ml_prob = None

    side = None
    if score >= SCORE_THRESHOLD:
        side = "LONG"
    elif score <= -SCORE_THRESHOLD:
        side = "SHORT"
    else:
        return None

    entry = float(feat["price"])
    atrv = float(feat.get("atr") or 0.0)
    sup = feat.get("sup_levels") or []
    res = feat.get("res_levels") or []
    buffer = max(atrv * 0.5, entry * 0.002)
    if side == "LONG":
        cand_sup = [s for s in sup if s < entry]
        if cand_sup:
            sl_candidate = max(cand_sup)
            sl = sl_candidate - buffer
        else:
            sl = entry - 1.5 * atrv if atrv > 0 else entry * 0.99
        cand_res = [r for r in res if r > entry]
        if cand_res:
            tp_candidate = min(cand_res)
            tp = tp_candidate + buffer * 0.5
        else:
            tp = entry + (entry - sl) * RR
    else:
        cand_res = [r for r in res if r > entry]
        if cand_res:
            sl_candidate = min(cand_res)
            sl = sl_candidate + buffer
        else:
            sl = entry + 1.5 * atrv if atrv > 0 else entry * 1.01
        cand_sup = [s for s in sup if s < entry]
        if cand_sup:
            tp_candidate = max(cand_sup)
            tp = tp_candidate - buffer * 0.5
        else:
            tp = entry - (sl - entry) * RR

    if not tp or not sl or tp == sl:
        if side == "LONG":
            sl = entry - max(1.5 * atrv, entry * 0.005)
            tp = entry + (entry - sl) * RR
        else:
            sl = entry + max(1.5 * atrv, entry * 0.005)
            tp = entry - (sl - entry) * RR

    emoji = "🚀" if side == "LONG" else "🔥"
    txt = (
        f"{emoji} <b>{side}</b> <code>{feat['symbol']}</code>\n"
        f"💵 <b>Entry:</b> <code>{entry:.8f}</code>\n"
        f"🎯 <b>TP:</b> <code>{tp:.8f}</code>    ⚠️ <b>SL:</b> <code>{sl:.8f}</code>\n"
        f"⭐ <b>Score</b>: {score:.2f} | {' • '.join(reasons[:6])}\n"
        f"📊 ob:{feat.get('orderbook_imb'):.3f}  whales:{feat.get('whale',{}).get('net_whale',0):.1f}  ATR:{atrv:.6f}\n"
        f"⏱️ <b>Time:</b> {feat['time']}"
    )
    # include feat in returned dict for later training
    return {"symbol": feat["symbol"], "side": side, "entry": entry, "sl": sl, "tp": tp, "score": score, "txt": txt, "feat": feat, "ml_prob": ml_prob}

# ---------------- Emulation + training on close ----------------
async def open_trade_emulation(sig: dict):
    """
    Open pseudo trade with USD size (1000$).
    Save feat vector at open for training on close.
    """
    async with TRADE_LOCK:
        entry = float(sig["entry"])
        usd_size = float(os.getenv("EMU_USD_SIZE", "1000.0"))
        # compute base size in contracts (approx): qty = usd_size / entry
        size = usd_size / entry if entry > 0 else 1.0
        trade = {
            "symbol": sig["symbol"],
            "side": sig["side"],
            "entry": entry,
            "sl": sig["sl"],
            "tp": sig["tp"],
            "open_time": now_iso(),
            "status": "open",
            "size": float(size),
            "size_usd": usd_size,
            "feat_at_open": sig.get("feat", {})  # store feature snapshot
        }
        TRADE_HISTORY.append(trade)
        save_json(TRADE_LOG_FILE, TRADE_HISTORY)
        log("EMU OPEN", trade["symbol"], trade["side"], "@", f"{trade['entry']:.8f}", "size=", f"{trade['size']:.6f}")
        return trade

async def _train_on_closed_trade(trade: dict):
    """
    Called when a trade is closed; compute label and partial_fit model.
    label = 1 if pnl_usd > 0 else 0
    """
    global MODEL_FITTED, MODEL, SCALER
    try:
        feat = trade.get("feat_at_open") or {}
        if not feat:
            log("no feat_at_open, skip training for trade", trade.get("symbol"))
            return
        # create vector
        X = features_vec(feat).reshape(1, -1)
        # compute label from pnl
        pnl_usd = trade.get("pnl_usd")
        if pnl_usd is None:
            entry = trade.get("entry")
            exitp = trade.get("exit_price")
            size = trade.get("size", 1.0)
            if entry is None or exitp is None:
                log("trade missing prices, skip training")
                return
            if trade.get("side") == "LONG":
                pnl_usd = (exitp - entry) * size
            else:
                pnl_usd = (entry - exitp) * size
        label = 1 if pnl_usd > 0 else 0

        # online scaler update
        try:
            if SCALER is not None:
                # StandardScaler supports partial_fit
                try:
                    SCALER.partial_fit(X)
                except Exception:
                    # fallback to fit if not possible
                    SCALER.fit(X)
        except Exception as e:
            log("scaler update err", e)

        # transform
        try:
            Xs = SCALER.transform(X) if SCALER is not None else X
        except Exception:
            Xs = X

        # partial_fit on model
        try:
            if not MODEL_FITTED:
                MODEL.partial_fit(Xs, [label], classes=np.array([0, 1]))
                MODEL_FITTED = True
                log("MODEL initial partial_fit with label", label, "for", trade.get("symbol"))
            else:
                MODEL.partial_fit(Xs, [label])
                log("MODEL partial_fit +1 sample label", label, "for", trade.get("symbol"))
            # save model/scaler
            save_model()
        except Exception as e:
            log("model partial_fit err", e)
    except Exception as e:
        log("train_on_closed_trade err", e)

async def trade_monitor(session: aiohttp.ClientSession, price_cache: dict):
    interval = 30 if HEAVY else 60
    while True:
        try:
            async with TRADE_LOCK:
                for trade in list(TRADE_HISTORY):
                    if trade.get("status") != "open":
                        continue
                    sym = trade["symbol"]
                    last_price = price_cache.get(sym, {}).get("last_price")
                    if last_price is None:
                        df1m = await fetch_klines(session, sym, "1m", limit=2)
                        if df1m is not None:
                            last_price = float(df1m["close"].iloc[-1])
                    if last_price is None:
                        continue
                    if trade["side"] == "LONG":
                        if last_price >= trade["tp"]:
                            trade["status"] = "closed"
                            trade["exit_price"] = last_price
                            trade["close_time"] = now_iso()
                            trade["pnl"] = (last_price - trade["entry"]) * trade["size"]
                            trade["pnl_usd"] = trade["pnl"]
                            trade["pnl_pct"] = safe_div(trade["pnl"], trade["size_usd"])
                            log("EMU CLOSE TP", sym, "pnl_usd=", trade["pnl_usd"])
                            # train on closed trade
                            asyncio.create_task(_train_on_closed_trade(trade))
                        elif last_price <= trade["sl"]:
                            trade["status"] = "closed"
                            trade["exit_price"] = last_price
                            trade["close_time"] = now_iso()
                            trade["pnl"] = (last_price - trade["entry"]) * trade["size"]
                            trade["pnl_usd"] = trade["pnl"]
                            trade["pnl_pct"] = safe_div(trade["pnl"], trade["size_usd"])
                            log("EMU CLOSE SL", sym, "pnl_usd=", trade["pnl_usd"])
                            asyncio.create_task(_train_on_closed_trade(trade))
                    else:
                        if last_price <= trade["tp"]:
                            trade["status"] = "closed"
                            trade["exit_price"] = last_price
                            trade["close_time"] = now_iso()
                            trade["pnl"] = (trade["entry"] - last_price) * trade["size"]
                            trade["pnl_usd"] = trade["pnl"]
                            trade["pnl_pct"] = safe_div(trade["pnl"], trade["size_usd"])
                            log("EMU CLOSE TP", sym, "pnl_usd=", trade["pnl_usd"])
                            asyncio.create_task(_train_on_closed_trade(trade))
                        elif last_price >= trade["sl"]:
                            trade["status"] = "closed"
                            trade["exit_price"] = last_price
                            trade["close_time"] = now_iso()
                            trade["pnl"] = (trade["entry"] - last_price) * trade["size"]
                            trade["pnl_usd"] = trade["pnl"]
                            trade["pnl_pct"] = safe_div(trade["pnl"], trade["size_usd"])
                            log("EMU CLOSE SL", sym, "pnl_usd=", trade["pnl_usd"])
                            asyncio.create_task(_train_on_closed_trade(trade))
                save_json(TRADE_LOG_FILE, TRADE_HISTORY)
            await asyncio.sleep(interval)
        except Exception as e:
            log("trade_monitor err", e)
            await asyncio.sleep(5)

# ---------------- ML trainer (background) - keeps previous functionality) ----------------
async def ml_trainer(session: aiohttp.ClientSession):
    global MODEL, SCALER, MODEL_FITTED
    init_model()
    while True:
        try:
            X = []; Y = []
            for sym, dq in list(FEATURE_HISTORY.items()):
                for item in list(dq):
                    sample_time = item["time"]
                    if time.time() - sample_time < LABEL_HORIZON_MINUTES * 60:
                        continue
                    feat = item["feat"]
                    cur_price = None
                    try:
                        cur_price = FEATURE_HISTORY[sym][-1]["feat"]["price"]
                    except Exception:
                        cur_price = None
                    if cur_price is None:
                        df = await fetch_klines(session, sym, "1m", limit=2)
                        if df is None:
                            continue
                        cur_price = float(df["close"].iloc[-1])
                    base = feat.get("price")
                    if base is None or base == 0:
                        try: dq.remove(item)
                        except: pass
                        continue
                    ret = safe_div(cur_price - base, base)
                    label = 1 if ret >= RET_THR else 0
                    X.append(features_vec(feat)); Y.append(label)
                    try: dq.remove(item)
                    except: pass
            if X and len(Y) >= 8:
                X = np.vstack(X); Y = np.array(Y)
                if not MODEL_FITTED:
                    try: SCALER.fit(X)
                    except Exception: pass
                    Xs = SCALER.transform(X) if SCALER is not None else X
                    MODEL.partial_fit(Xs, Y, classes=np.array([0,1]))
                    MODEL_FITTED = True
                    log("ML initial partial_fit", len(Y))
                else:
                    try:
                        Xs = SCALER.transform(X)
                    except Exception:
                        try:
                            SCALER.fit(X); Xs = SCALER.transform(X)
                        except Exception:
                            Xs = X
                    MODEL.partial_fit(Xs, Y)
                    log("ML partial_fit +", len(Y))
                save_model()
            await asyncio.sleep(30 if HEAVY else 90)
        except Exception as e:
            log("ml_trainer err", e)
            await asyncio.sleep(10)

# ---------------- Minute OI worker ----------------
async def minute_oi_worker(session: aiohttp.ClientSession):
    global LAST_OI, SENT
    while True:
        try:
            syms = await fetch_top_symbols(session, limit=TOP_N)
            if not syms:
                await asyncio.sleep(5); continue
            syms = [s for s in syms if s.endswith("USDT")]
            for s in syms:
                oi_now = await fetch_current_oi(session, s)
                if oi_now is None:
                    continue
                prev = LAST_OI.get(s)
                LAST_OI[s] = oi_now
                if prev is None or prev == 0:
                    continue
                change_pct = (oi_now - prev) / prev * 100.0
                if abs(change_pct) >= MINUTE_OI_THRESHOLD_PCT:
                    side = "LONG" if change_pct > 0 else "SHORT"
                    key = f"{s}:MIN_OI:{side}"
                    async with SENT_LOCK:
                        last = SENT.get(key); now_ts = int(time.time())
                        if last and (now_ts - last) < COOLDOWN_SECONDS:
                            continue
                        emoji = "⚡"
                        txt = (f"{emoji} <b>Minute OI</b> <code>{s}</code> — <b>{side}</b>\n"
                               f"📊 <b>Δ1m OI:</b> {change_pct:.2f}%\n"
                               f"⏱ <b>Time:</b> {now_iso()}")
                        await send_telegram(session, txt)
                        SENT[key] = now_ts; save_json(SENT_FILE, SENT)
            await asyncio.sleep(60)
        except Exception as e:
            log("minute_oi err", e)
            await asyncio.sleep(5)

# ---------------- Telegram send ----------------
async def send_telegram(session: aiohttp.ClientSession, text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        log("Telegram not configured")
        return None
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        async with session.post(url, data=payload, timeout=10) as resp:
            txt = await resp.text(); status = resp.status
            log("tg status", status)
            return {"status": status, "text": txt}
    except Exception as e:
        log("tg send err", e)
        return None

# ---------------- Main loop (light -> heavy) ----------------
async def main():
    global SENT
    SENT = load_json(SENT_FILE, default={}) or {}
    init_model()
    async with aiohttp.ClientSession() as session:
        loop = asyncio.get_event_loop()
        # Liquidation input
        if TELETHON_AVAILABLE and TG_API_ID and TG_API_HASH:
            loop.create_task(start_telethon(loop))
        else:
            loop.create_task(html_liq_poller(session))
        # workers
        loop.create_task(liq_consumer())
        loop.create_task(minute_oi_worker(session))
        loop.create_task(ml_trainer(session))
        loop.create_task(trade_monitor(session, {}))

        price_cache = {}
        while True:
            try:
                symbols = await fetch_top_symbols(session, limit=TOP_N)
                if not symbols:
                    await asyncio.sleep(5); continue
                symbols = [s for s in symbols if s.endswith("USDT")]
                log("Scanning", len(symbols), "symbols (TOP_N =", TOP_N, ")")
                # Light phase
                light_tasks = [compute_light_features(session, s, price_cache) for s in symbols]
                light_results = await asyncio.gather(*light_tasks)
                scored = []
                for feat in light_results:
                    if not feat: continue
                    cand = await generate_signal_light(feat)
                    scored.append(cand)
                scored_sorted = sorted(scored, key=lambda x: x["score"], reverse=True)
                candidates = [c["symbol"] for c in scored_sorted if c["score"] > 0][:CANDIDATE_LIMIT]
                neg_candidates = [c["symbol"] for c in scored_sorted if c["score"] < -0.8][:CANDIDATE_LIMIT]
                for nc in neg_candidates:
                    if nc not in candidates:
                        candidates.append(nc)
                log("Candidates for heavy scan:", candidates)
                # Heavy phase
                heavy_tasks = [compute_heavy_features(session, s, price_cache) for s in candidates]
                heavy_results = await asyncio.gather(*heavy_tasks)
                for feat in heavy_results:
                    if not feat: continue
                    sig = await generate_signal_heavy(session, feat)
                    if not sig:
                        continue
                    key = f"{sig['symbol']}:{sig['side']}"
                    now_ts = int(time.time())
                    async with SENT_LOCK:
                        last = SENT.get(key)
                        if last and (now_ts - last) < COOLDOWN_SECONDS:
                            log("skip cooldown", key)
                            continue
                    await send_telegram(session, sig["txt"])
                    async with SENT_LOCK:
                        SENT[key] = now_ts; save_json(SENT_FILE, SENT)
                    # open emulation (and store feat_at_open inside trade)
                    await open_trade_emulation(sig)
                await asyncio.sleep(POLL_INTERVAL if not HEAVY else max(10, POLL_INTERVAL // 2))
            except Exception as e:
                log("main err", e)
                await asyncio.sleep(5)

# ---------------- Entrypoint ----------------
def safe_div(a,b):
    try:
        return a/b
    except Exception:
        return 0.0

if __name__ == "__main__":
    log("starting futures_signals_ai (HEAVY=", HEAVY, ")")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("stopped by user")
