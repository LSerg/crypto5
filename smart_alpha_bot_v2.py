#!/usr/bin/env python3
"""
smart_alpha_bot_v2.py
All-in-one signals + emulation + online-ML bot for Binance Futures (USDT pairs).

Usage:
  python3 smart_alpha_bot_v2.py [--heavy] [--ws] [--backtest]

.env should contain:
  TELEGRAM_TOKEN=<bot_token>
  CHAT_ID=<chat_id>
  OPTIONAL:
    TELEGRAM_API_ID, TELEGRAM_API_HASH (for Telethon liqui feed)
    GLASSNODE_API_KEY or CRYPTOQUANT_API_KEY (optional on-chain)
    TOP_N, POLL_INTERVAL, HEAVY=1/0 etc.

Read comments in code for optional features.
"""
import os
import argparse
import asyncio
import aiohttp
import math
import time
import json
import logging
import sys
import traceback
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

# Telethon optionally for liquidation feed
try:
    from telethon import TelegramClient, events
    TELETHON_AVAILABLE = True
except Exception:
    TELETHON_AVAILABLE = False

# ---------------- Config ----------------
load_dotenv()
BINANCE_API = os.getenv("BINANCE_API", "https://fapi.binance.com")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
TG_API_ID = os.getenv("TELEGRAM_API_ID")
TG_API_HASH = os.getenv("TELEGRAM_API_HASH")
GLASSNODE_API_KEY = os.getenv("GLASSNODE_API_KEY")
CRYPTOQUANT_API_KEY = os.getenv("CRYPTOQUANT_API_KEY")
WHALE_ALERT_API_KEY = os.getenv("WHALE_ALERT_API_KEY")  # optional

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(exist_ok=True, parents=True)

# files
SENT_FILE = DATA_DIR / "sent_signals.json"
MODEL_FILE = DATA_DIR / "sgd_model.joblib"
SCALER_FILE = DATA_DIR / "scaler.joblib"
TRADE_LOG_FILE = DATA_DIR / "trade_history.json"
SIGNALS_HISTORY_FILE = DATA_DIR / "signals_history.json"
LOG_FILE = DATA_DIR / "bot.log"

# CLI flags
parser = argparse.ArgumentParser()
parser.add_argument("--heavy", action="store_true", help="enable heavy computations")
parser.add_argument("--ws", action="store_true", help="use websockets for aggTrades/depth")
parser.add_argument("--backtest", action="store_true", help="run backtest mode (give historical files)")
args = parser.parse_args()
HEAVY = args.heavy
USE_WS = args.ws
BACKTEST = args.backtest

# Tunables
TOP_N = int(os.getenv("TOP_N", "20"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60")) if not HEAVY else int(os.getenv("POLL_INTERVAL_HEAVY", "20"))
MINUTE_OI_THRESHOLD_PCT = float(os.getenv("MINUTE_OI_THRESHOLD_PCT", "5.0"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "2.0"))
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", str(60 * 60)))
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "6"))
CANDIDATE_LIMIT = int(os.getenv("CANDIDATE_LIMIT", "6"))

# indicator params
EMA_FAST = int(os.getenv("EMA_FAST", "9"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "21"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))

# TP/SL bounds (user requested)
TP_MIN = float(os.getenv("TP_MIN", "0.015"))   # 1.5%
TP_MAX = float(os.getenv("TP_MAX", "0.02"))    # 2%
SL_MIN = float(os.getenv("SL_MIN", "0.005"))   # 0.5%
SL_MAX = float(os.getenv("SL_MAX", "0.01"))    # 1.0%
RR = float(os.getenv("RR", "2.0"))

# Emulation size & Kelly base
EMU_USD_SIZE = float(os.getenv("EMU_USD_SIZE", "1000.0"))
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))  # fraction of Kelly to be conservative

# ML
MODEL = None
SCALER = None
MODEL_FITTED = False
RET_THR = float(os.getenv("RET_THR", "0.001"))
LABEL_HORIZON_MINUTES = int(os.getenv("LABEL_HORIZON_MINUTES", "5"))

# Internal
LAST_OI = {}
LIQ_QUEUE = asyncio.Queue()
LIQ_FLOW = defaultdict(lambda: deque(maxlen=1000))
FEATURE_HISTORY = defaultdict(lambda: deque(maxlen=2000))
TRADE_HISTORY = []
SENT = {}
SENT_LOCK = asyncio.Lock()
TRADE_LOCK = asyncio.Lock()
SEM = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# logging
logging.basicConfig(level=logging.INFO, filename=str(LOG_FILE), filemode="a",
                    format="%(asctime)s %(levelname)s %(message)s")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

RE_LIQ = re.compile(r"\b([A-Z0-9]{3,8}USDT)\b.*?\b(LONG|SHORT)\b.*?\$?([\d,\,\.]+)", re.IGNORECASE)

# ---------------- Helpers ----------------
def now_iso():
    return datetime.utcnow().isoformat(sep=' ', timespec='seconds')

def save_json(path: Path, obj):
    try:
        path.write_text(json.dumps(obj, indent=2, default=str))
    except Exception as e:
        logging.exception("save_json error")

def load_json(path: Path, default=None):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return default

def safe_div(a,b):
    try:
        return a/b
    except Exception:
        return 0.0

# ---------------- HTTP with backoff ----------------
async def fetch_json(session, url, params=None, timeout=15, retries=4):
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
                        logging.warning("Rate limited %s %s", resp.status, url)
                        await asyncio.sleep(backoff); backoff *= 2
                    elif 500 <= resp.status < 600:
                        logging.warning("Server error %s %s", resp.status, url)
                        await asyncio.sleep(backoff); backoff *= 2
                    else:
                        try:
                            return json.loads(txt)
                        except Exception:
                            logging.warning("HTTP error %s %s", resp.status, txt[:200])
                            return None
        except asyncio.TimeoutError:
            logging.warning("Timeout %s attempt %d", url, attempt)
            await asyncio.sleep(backoff); backoff *= 2
        except Exception as e:
            logging.exception("Fetch exception %s", url)
            await asyncio.sleep(backoff); backoff *= 2
    return None

# ---------------- Indicators ----------------
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def atr(df, period=14):
    prev = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev).abs()
    tr3 = (df['low'] - prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up/(ma_down + 1e-12)
    return 100 - (100/(1+rs))

def macd(series, fast=12, slow=26, sig=9):
    ef = series.ewm(span=fast, adjust=False).mean()
    es = series.ewm(span=slow, adjust=False).mean()
    line = ef - es
    signal = line.ewm(span=sig, adjust=False).mean()
    hist = line - signal
    return line, signal, hist

def bollinger(series, n=20, k=2):
    ma = series.rolling(n).mean()
    std = series.rolling(n).std()
    upper = ma + k*std
    lower = ma - k*std
    width = (upper-lower)/ma.replace(0, np.nan)
    return ma, upper, lower, width

def vwap_from_df(df):
    pv = df['close'] * df['volume']
    return pv.cumsum() / df['volume'].cumsum()

# ---------------- Binance helpers ----------------
async def fetch_top_symbols(session, limit=TOP_N):
    url = f"{BINANCE_API}/fapi/v1/ticker/24hr"
    data = await fetch_json(session, url)
    if not data: return []
    df = pd.DataFrame(data)
    df['quoteVolume'] = pd.to_numeric(df.get('quoteVolume', 0), errors='coerce').fillna(0)
    top = df.sort_values('quoteVolume', ascending=False).head(limit)
    return top['symbol'].tolist()

async def fetch_klines(session, symbol, interval, limit=200):
    url = f"{BINANCE_API}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    data = await fetch_json(session, url, params=params)
    if not data or not isinstance(data, list):
        return None
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume","close_time","quote_av","trades","tb1","tb2","ignore"
    ])
    for c in ("open","high","low","close","volume"):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    return df

async def fetch_current_oi(session, symbol):
    url = f"{BINANCE_API}/fapi/v1/openInterest"
    params = {"symbol": symbol}
    d = await fetch_json(session, url, params=params)
    if d and "openInterest" in d:
        try: return float(d["openInterest"])
        except: return None
    return None

async def fetch_open_interest_hist(session, symbol, period='1h', limit=25):
    url = f"{BINANCE_API}/futures/data/openInterestHist"
    params = {"symbol": symbol, "period": period, "limit": limit}
    return await fetch_json(session, url, params=params)

async def fetch_funding(session, symbol, limit=6):
    url = f"{BINANCE_API}/fapi/v1/fundingRate"
    params = {"symbol": symbol, "limit": limit}
    return await fetch_json(session, url, params=params)

async def orderbook_imbalance(session, symbol, limit=100):
    url = f"{BINANCE_API}/fapi/v1/depth"
    params = {"symbol": symbol, "limit": min(limit,100)}
    d = await fetch_json(session, url, params=params)
    if not d: return 0.0
    bids = d.get('bids',[]); asks = d.get('asks',[])
    topN = 10
    bid_vol = sum(float(b[1]) for b in bids[:topN])
    ask_vol = sum(float(a[1]) for a in asks[:topN])
    if (bid_vol + ask_vol) == 0: return 0.0
    return (bid_vol - ask_vol)/(bid_vol + ask_vol)

async def whale_flow_from_aggtrades(session, symbol, limit=1000):
    url = f"{BINANCE_API}/fapi/v1/aggTrades"
    params = {"symbol": symbol, "limit": min(limit,1000)}
    data = await fetch_json(session, url, params=params)
    if not data or not isinstance(data, list): return {"net_whale":0.0,"count_whale":0}
    sizes = np.array([float(d['q']) for d in data])
    if sizes.size == 0: return {"net_whale":0.0,"count_whale":0}
    thr = max(np.percentile(sizes,90), sizes.mean()*2)
    net = 0.0; cnt=0
    for d in data:
        q = float(d['q']); is_buyer = not d.get('m', False)
        if q >= thr:
            net += q if is_buyer else -q
            cnt += 1
    return {"net_whale": float(net), "count_whale": int(cnt)}

# ---------------- Liquidation ingestion ----------------
async def telethon_listener(loop):
    if not TELETHON_AVAILABLE:
        logging.info("Telethon not installed - skip")
        return
    if not TG_API_ID or not TG_API_HASH:
        logging.info("No TELEGRAM_API_ID/API_HASH - skip telethon")
        return
    client = TelegramClient("liquidation_session", int(TG_API_ID), TG_API_HASH, loop=loop)
    @client.on(events.NewMessage(chats='BinanceLiquidations'))
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
        logging.info("Telethon queued liq %s", parsed.get("symbol"))
    await client.start()
    logging.info("Telethon started")
    await client.run_until_disconnected()

async def html_liq_poller(session):
    poll = int(os.getenv("HTML_LIQ_POLL_INTERVAL","30"))
    channel = os.getenv("LIQ_CHANNEL","BinanceLiquidations")
    while True:
        try:
            url = f"https://t.me/s/{channel}"
            async with SEM:
                async with session.get(url, timeout=15) as r:
                    html = await r.text()
            soup = BeautifulSoup(html, "lxml")
            msgs = soup.find_all("div", class_="tgme_widget_message_text")
            for m in msgs[-80:]:
                text = m.get_text(" ", strip=True)
                if "liquidation" in text.lower() or "long" in text.lower() or "short" in text.lower():
                    parsed = {"raw": text, "time": now_iso()}
                    mm = RE_LIQ.search(text)
                    if mm:
                        try:
                            parsed.update({"symbol": mm.group(1).upper(), "side": mm.group(2).upper(), "size": float(mm.group(3).replace(",",""))})
                        except: pass
                    await LIQ_QUEUE.put(parsed)
        except Exception as e:
            logging.exception("html_liq_poller")
        await asyncio.sleep(poll)

async def liq_consumer():
    while True:
        try:
            ev = await LIQ_QUEUE.get()
            if not ev:
                LIQ_QUEUE.task_done(); continue
            sym = ev.get("symbol"); side = ev.get("side"); size = ev.get("size",0)
            if sym and side:
                val = size if side.upper()=="SHORT" else -size
                LIQ_FLOW[sym].append((time.time(), val))
                logging.info("LIQ_FLOW add %s %s", sym, val)
            LIQ_QUEUE.task_done()
        except Exception:
            logging.exception("liq_consumer")
            await asyncio.sleep(0.5)

# ---------------- Feature builders ----------------
async def compute_light_features(session, symbol, cache):
    df1h = await fetch_klines(session, symbol, "1h", limit=200)
    if df1h is None or df1h.empty: return None
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
    if oi_hist and isinstance(oi_hist, list) and len(oi_hist)>=2:
        try:
            st = float(oi_hist[0].get('sumOpenInterestValue') or oi_hist[0].get('sumOpenInterest',0))
            en = float(oi_hist[-1].get('sumOpenInterestValue') or oi_hist[-1].get('sumOpenInterest',0))
            if st>0: oi_1h_pct = safe_div(en-st,st)*100.0
        except: oi_1h_pct = None
    cache[symbol] = {"oi_last": oi_now, "last_price": float(df1h['close'].iloc[-1])}
    feat = {"symbol": symbol, "time": now_iso(), "price": float(df1h['close'].iloc[-1]), "price_ret_1h": price_ret_1h,
            "ema_diff": ema_diff, "atr": atr_val, "rsi": rsi_val, "vol_ratio": vol_ratio, "oi_now": oi_now, "oi_1h": oi_1h_pct}
    return feat

async def compute_heavy_features(session, symbol, cache):
    df1m = await fetch_klines(session, symbol, "1m", limit=120)
    df5m = await fetch_klines(session, symbol, "5m", limit=60)
    df1h = await fetch_klines(session, symbol, "1h", limit=240)
    if df1m is None or df1h is None: return None
    p1 = float(df1m['close'].iloc[-1]); p1_prev = float(df1m['close'].iloc[-2])
    p5 = float(df5m['close'].iloc[-1]) if df5m is not None else p1
    p1h = float(df1h['close'].iloc[-1]); p1h_prev = float(df1h['close'].iloc[-2])
    price_ret_1m = safe_div(p1 - p1_prev, p1_prev)
    price_ret_5m = safe_div(p5 - (float(df5m['close'].iloc[-2]) if df5m is not None else p5), p5) if df5m is not None else 0.0
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
            try: vals.append(float(it.get('fundingRate',0)))
            except: pass
        if vals: funding_avg = sum(vals)/len(vals)
    ob_imb = await orderbook_imbalance(session, symbol, limit=100)
    whale = await whale_flow_from_aggtrades(session, symbol, limit=500)
    vwap_val = None
    try:
        v = vwap_from_df(df1h); vwap_val = float(v.iloc[-1])
    except: vwap_val = None
    macd_line = macd_signal = macd_hist = None
    try:
        macd_line, macd_signal, macd_hist = macd(df1h['close'])
        macd_line = float(macd_line.iloc[-1]); macd_signal = float(macd_signal.iloc[-1]); macd_hist = float(macd_hist.iloc[-1])
    except: macd_line = macd_signal = macd_hist = None
    bb_mid=bb_up=bb_low=bb_width=None
    try:
        bb_mid,bb_up,bb_low,bb_width = bollinger(df1h['close'],n=20,k=2)
        bb_mid=float(bb_mid.iloc[-1]); bb_up=float(bb_up.iloc[-1]); bb_low=float(bb_low.iloc[-1]); bb_width=float(bb_width.iloc[-1])
    except: bb_mid=bb_up=bb_low=bb_width=None
    sup_levels = res_levels = []
    try:
        highs = df1h['high']; lows = df1h['low']
        res_levels = sorted(highs[-72:].sort_values(ascending=False).head(3).values.tolist())
        sup_levels = sorted(lows[-72:].sort_values().head(3).values.tolist())
    except:
        sup_levels = res_levels = []
    vwap_diff = None
    if vwap_val is not None:
        try: vwap_diff = float(p1) - float(vwap_val)
        except: vwap_diff = None
    cache[symbol] = {"oi_last": oi_now, "last_price": float(p1)}
    feat = {"symbol": symbol, "time": now_iso(), "price": float(p1), "oi_now": oi_now, "oi_1m": oi_1m,
            "price_ret_1m": price_ret_1m, "price_ret_5m": price_ret_5m, "price_ret_1h": price_ret_1h,
            "rsi": rsi_val, "ema_diff": ema_diff, "atr": atr_val, "vol_ratio": vol_ratio, "funding": funding_avg,
            "orderbook_imb": ob_imb, "whale": whale, "vwap": vwap_val, "vwap_diff": vwap_diff,
            "macd_line": macd_line, "macd_signal": macd_signal, "macd_hist": macd_hist,
            "bb_mid": bb_mid, "bb_up": bb_up, "bb_low": bb_low, "bb_width": bb_width,
            "sup_levels": sup_levels, "res_levels": res_levels}
    FEATURE_HISTORY[symbol].append({"time": time.time(), "feat": feat})
    return feat

# ---------------- Utility: whale stability, trend ----------------
def whale_stability(sym, window_seconds=300):
    arr = [v for (t,v) in LIQ_FLOW.get(sym, []) if time.time() - t <= window_seconds]
    if not arr: return 0.0, 0.0
    a = np.array(arr); return float(a.mean()), float(a.std())

def trend_strength(feat):
    try:
        return abs(feat.get('ema_diff', 0.0))
    except:
        return 0.0

def record_signal_history(sig):
    hist = load_json(SIGNALS_HISTORY_FILE, default=[]) or []
    entry = {"ts": now_iso(), "symbol": sig['symbol'], "side": sig['side'], "entry": sig['entry'], "tp": sig['tp'], "sl": sig['sl'], "score": sig.get('score'), "meta": {"atr": sig.get('feat',{}).get('atr'), "rsi": sig.get('feat',{}).get('rsi'), "ob_imb": sig.get('feat',{}).get('orderbook_imb')}}
    hist.append(entry)
    save_json(SIGNALS_HISTORY_FILE, hist)

# ---------------- ML init/save ----------------
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
    except Exception:
        logging.exception("model load")

def save_model():
    try:
        if MODEL is not None: dump(MODEL, MODEL_FILE)
        if SCALER is not None: dump(SCALER, SCALER_FILE)
    except Exception:
        logging.exception("model save")

def features_vec(feat):
    keys = ["oi_1m","price_ret_1m","price_ret_5m","price_ret_1h","rsi","ema_diff","atr","vol_ratio","orderbook_imb"]
    vec = [float(feat.get(k,0.0) if feat.get(k) is not None else 0.0) for k in keys]
    vec.append(float(feat.get("whale",{}).get("net_whale",0.0) if isinstance(feat.get("whale"),dict) else 0.0))
    vec.append(float(feat.get("vwap_diff",0.0) if feat.get("vwap_diff") is not None else 0.0))
    vec.append(float(feat.get("macd_hist",0.0) if feat.get("macd_hist") is not None else 0.0))
    return np.array(vec, dtype=float)

# ---------------- Signal generation heavy (core) ----------------
def _conf_from_score(score):
    return 1.0/(1.0+math.exp(-abs(score)/2.0))

def _vol_adj_from_atr_ratio(atr_ratio):
    if atr_ratio <= 0: return 1.0
    base = atr_ratio/0.01
    return max(0.7,min(1.5,base))

async def generate_signal_heavy(session, feat):
    # weight map
    w = {"ema":1.0,"rsi":0.9,"macd":1.0,"oi":1.5,"vol":0.8,"ob":1.2,"whale":1.6,"adx":0.6,"bb":0.6,"vwap":0.5,"liq":1.2}
    score = 0.0; reasons=[]
    if feat.get("ema_diff") is not None:
        score += w["ema"]*np.sign(feat["ema_diff"]); reasons.append(f"ema{feat['ema_diff']:.4f}")
    if feat.get("rsi") is not None:
        if feat["rsi"]>60: score += w["rsi"]; reasons.append("rsi_hi")
        elif feat["rsi"]<40: score -= w["rsi"]; reasons.append("rsi_lo")
    # macd hist if present
    if feat.get("macd_line") is not None and feat.get("macd_signal") is not None:
        macd_delta = feat["macd_line"] - feat["macd_signal"]
        score += w["macd"]*np.sign(macd_delta); reasons.append(f"macd{macd_delta:.4f}")
    # vwap
    if feat.get("vwap") is not None:
        if feat["price"] > feat["vwap"]:
            score += w["vwap"]; reasons.append("above_vwap")
        else:
            score -= w["vwap"]; reasons.append("below_vwap")
    # orderbook
    ob = feat.get("orderbook_imb") or 0.0
    if ob > 0.25:
        score += w["ob"]; reasons.append(f"ob+{ob:.2f}")
    elif ob < -0.25:
        score -= w["ob"]; reasons.append(f"ob{ob:.2f}")
    # whales
    wh = feat.get("whale",{}) or {}
    if wh and wh.get("net_whale"):
        if wh["net_whale"] > 0:
            score += w["whale"]; reasons.append(f"wh_buy{wh['net_whale']:.1f}")
        else:
            score -= w["whale"]; reasons.append(f"wh_sell{wh['net_whale']:.1f}")
    # OI minute
    if feat.get("oi_1m") is not None:
        if feat["oi_1m"] >= MINUTE_OI_THRESHOLD_PCT:
            score += w["oi"]; reasons.append(f"oi1m+{feat['oi_1m']:.1f}%")
        elif feat["oi_1m"] <= -MINUTE_OI_THRESHOLD_PCT:
            score -= w["oi"]; reasons.append(f"oi1m{feat['oi_1m']:.1f}%")
    # liquidation flow
    liq_sum = 0.0; nowt=time.time()
    for (t,v) in list(LIQ_FLOW.get(feat["symbol"],[])):
        if nowt - t <= 600: liq_sum += v
    if abs(liq_sum) > 1_000_000:
        score += w["liq"] * np.sign(liq_sum); reasons.append(f"liq{liq_sum:.0f}")
    # whale stability - reduce weight if unstable
    wh_mean, wh_std = whale_stability(feat["symbol"], window_seconds=300)
    if abs(wh_mean) > 0 and wh_std/(abs(wh_mean)+1e-12) > 1.5:
        reasons.append("wh_unstable")
        # downweight whale influence by subtracting half
        score *= 0.9
    # ml boost
    init_model()
    ml_prob = None
    try:
        vec = features_vec(feat).reshape(1,-1)
        if SCALER is not None:
            try: vec_s = SCALER.transform(vec)
            except: vec_s = vec
        else: vec_s = vec
        if MODEL_FITTED and hasattr(MODEL,"predict_proba"):
            ml_prob = float(MODEL.predict_proba(vec_s)[0][1])
            score += 1.5*((ml_prob-0.5)*2.0)
            reasons.append(f"MLp{ml_prob:.2f}")
    except Exception:
        pass
    # trend filter: require either trend or OI confirmation
    ts = trend_strength(feat)
    trend_ok = ts > 0.0005 or abs(feat.get("price_ret_1h",0.0)) > 0.002
    oi_confirm = False
    try:
        if feat.get("oi_1m") is not None and abs(feat.get("oi_1m")) >= MINUTE_OI_THRESHOLD_PCT: oi_confirm = True
        if feat.get("oi_1h") is not None and abs(feat.get("oi_1h")) >= 2.0: oi_confirm = True
    except: oi_confirm = False
    side = None
    if score >= SCORE_THRESHOLD and (trend_ok or oi_confirm or abs(wh_mean)>0):
        side = "LONG"
    elif score <= -SCORE_THRESHOLD and (trend_ok or oi_confirm or abs(wh_mean)>0):
        side = "SHORT"
    else:
        return None
    # compute adaptive TP/SL bounded
    entry = float(feat["price"])
    atrv = float(feat.get("atr") or 0.0)
    atr_ratio = (atrv/entry) if entry>0 else 0.0
    conf = _conf_from_score(score)
    vol_adj = _vol_adj_from_atr_ratio(atr_ratio)
    tp_base = TP_MIN + (TP_MAX-TP_MIN)*conf
    sl_base = SL_MAX - (SL_MAX-SL_MIN)*conf
    tp_pct = tp_base*(1.0 + 0.25*(vol_adj-1.0))
    sl_pct = sl_base*(1.0 + 0.4*(vol_adj-1.0))
    tp_pct = max(TP_MIN, min(TP_MAX, tp_pct))
    sl_pct = max(SL_MIN, min(SL_MAX, sl_pct))
    # sr aware
    sup = feat.get("sup_levels") or []; res = feat.get("res_levels") or []
    cand_sup = [s for s in sup if s < entry]; cand_res = [r for r in res if r > entry]
    nearest_sup = max(cand_sup) if cand_sup else None; nearest_res = min(cand_res) if cand_res else None
    if side == "LONG":
        tp_candidate = entry*(1.0+tp_pct); sl_candidate = entry*(1.0-sl_pct)
        if nearest_res is not None and nearest_res < tp_candidate*1.05: tp = nearest_res*(1.0+0.002)
        else: tp = tp_candidate
        if nearest_sup is not None and nearest_sup > sl_candidate*0.95: sl = nearest_sup*(1.0-0.002)
        else: sl = sl_candidate
    else:
        tp_candidate = entry*(1.0-tp_pct); sl_candidate = entry*(1.0+sl_pct)
        if nearest_sup is not None and nearest_sup > tp_candidate*0.95: tp = nearest_sup*(1.0-0.002)
        else: tp = tp_candidate
        if nearest_res is not None and nearest_res < sl_candidate*1.05: sl = nearest_res*(1.0+0.002)
        else: sl = sl_candidate
    if sl is None or tp is None or tp == sl:
        if side=="LONG":
            sl = entry*(1.0-SL_MIN); tp = entry*(1.0+TP_MIN*RR)
        else:
            sl = entry*(1.0+SL_MIN); tp = entry*(1.0-TP_MIN*RR)
    # enforce bounds
    final_tp_pct = abs(tp/entry - 1.0); final_sl_pct = abs(sl/entry - 1.0)
    if final_tp_pct < TP_MIN:
        tp = entry*(1.0+TP_MIN) if side=="LONG" else entry*(1.0-TP_MIN)
    if final_tp_pct > TP_MAX:
        tp = entry*(1.0+TP_MAX) if side=="LONG" else entry*(1.0-TP_MAX)
    if final_sl_pct < SL_MIN:
        sl = entry*(1.0-SL_MIN) if side=="LONG" else entry*(1.0+SL_MIN)
    if final_sl_pct > SL_MAX:
        sl = entry*(1.0-SL_MAX) if side=="LONG" else entry*(1.0+SL_MAX)
    # anti-duplicate check
    key = f"{feat['symbol']}:{side}"
    now_ts = int(time.time())
    last = SENT.get(key)
    if last and (now_ts - last) < COOLDOWN_SECONDS:
        logging.info("skip due cooldown %s", key); return None
    # compose message
    emoji = "🚀" if side=="LONG" else "🔥"
    txt = (f"{emoji} <b>{side}</b> <code>{feat['symbol']}</code>\n"
           f"💵 <b>Entry:</b> <code>{entry:.8f}</code>\n"
           f"🎯 <b>TP:</b> <code>{tp:.8f}</code>    ⚠️ <b>SL:</b> <code>{sl:.8f}</code>\n"
           f"⭐ <b>Score</b>: {score:.2f} | {' • '.join(reasons[:6])}\n"
           f"📊 ob:{feat.get('orderbook_imb'):.3f}  whales:{feat.get('whale',{}).get('net_whale',0):.1f}  ATR:{atrv:.6f}\n"
           f"🔎 conf:{conf:.3f} vol_adj:{vol_adj:.3f}  ⏱️ {feat['time']}")
    sig = {"symbol": feat["symbol"], "side": side, "entry": entry, "sl": sl, "tp": tp, "score": score, "txt": txt, "feat": feat, "ml_prob": ml_prob}
    record_signal_history(sig)
    return sig

# ---------------- Emulation and trade monitor ----------------
async def open_trade_emulation(sig):
    async with TRADE_LOCK:
        entry = float(sig["entry"])
        size_usd = EMU_USD_SIZE
        # Kelly position sizing (simplified): if model has p and q estimate use p - q / odds
        k_fraction = KELLY_FRACTION
        size = size_usd / entry if entry>0 else 1.0
        trade = {"symbol": sig["symbol"], "side": sig["side"], "entry": entry, "sl": sig["sl"], "tp": sig["tp"],
                 "open_time": now_iso(), "status": "open", "size": float(size), "size_usd": size_usd, "feat_at_open": sig.get("feat",{})}
        TRADE_HISTORY.append(trade); save_json(TRADE_LOG_FILE, TRADE_HISTORY)
        logging.info("EMU OPEN %s %s @ %.8f size=%.6f", trade["symbol"], trade["side"], trade["entry"], trade["size"])
        return trade

async def _train_on_closed_trade(trade):
    global MODEL_FITTED, MODEL, SCALER
    try:
        feat = trade.get("feat_at_open") or {}
        if not feat:
            logging.info("no feat_at_open skip training"); return
        X = features_vec(feat).reshape(1,-1)
        pnl_usd = trade.get("pnl_usd")
        if pnl_usd is None:
            entry = trade.get("entry"); exitp = trade.get("exit_price"); size = trade.get("size",1.0)
            if entry is None or exitp is None: return
            pnl_usd = (exitp - entry)*size if trade.get("side")=="LONG" else (entry - exitp)*size
        label = 1 if pnl_usd>0 else 0
        try:
            if SCALER is not None:
                try: SCALER.partial_fit(X)
                except: SCALER.fit(X)
        except Exception: logging.exception("scaler update")
        try:
            Xs = SCALER.transform(X) if SCALER is not None else X
        except Exception:
            Xs = X
        try:
            if not MODEL_FITTED:
                MODEL.partial_fit(Xs, [label], classes=np.array([0,1])); MODEL_FITTED=True
                logging.info("MODEL initial partial_fit label=%s symbol=%s", label, trade.get("symbol"))
            else:
                MODEL.partial_fit(Xs, [label])
                logging.info("MODEL partial_fit +1 label=%s symbol=%s", label, trade.get("symbol"))
            save_model()
        except Exception:
            logging.exception("model partial_fit")
    except Exception:
        logging.exception("train_on_closed_trade")

async def trade_monitor(session, price_cache):
    interval = 30 if HEAVY else 60
    while True:
        try:
            async with TRADE_LOCK:
                for trade in list(TRADE_HISTORY):
                    if trade.get("status")!="open": continue
                    sym = trade["symbol"]
                    last_price = price_cache.get(sym,{}).get("last_price")
                    if last_price is None:
                        df1m = await fetch_klines(session, sym, "1m", limit=2)
                        if df1m is not None:
                            last_price = float(df1m['close'].iloc[-1])
                    if last_price is None: continue
                    if trade["side"]=="LONG":
                        if last_price >= trade["tp"]:
                            trade.update({"status":"closed","exit_price":last_price,"close_time":now_iso()})
                            trade["pnl"]=(last_price-trade["entry"])*trade["size"]; trade["pnl_usd"]=trade["pnl"]; trade["pnl_pct"]=safe_div(trade["pnl"], trade["size_usd"])
                            logging.info("EMU CLOSE TP %s pnl_usd=%.6f", sym, trade["pnl_usd"])
                            asyncio.create_task(_train_on_closed_trade(trade))
                        elif last_price <= trade["sl"]:
                            trade.update({"status":"closed","exit_price":last_price,"close_time":now_iso()})
                            trade["pnl"]=(last_price-trade["entry"])*trade["size"]; trade["pnl_usd"]=trade["pnl"]; trade["pnl_pct"]=safe_div(trade["pnl"], trade["size_usd"])
                            logging.info("EMU CLOSE SL %s pnl_usd=%.6f", sym, trade["pnl_usd"])
                            asyncio.create_task(_train_on_closed_trade(trade))
                    else:
                        if last_price <= trade["tp"]:
                            trade.update({"status":"closed","exit_price":last_price,"close_time":now_iso()})
                            trade["pnl"]=(trade["entry"]-last_price)*trade["size"]; trade["pnl_usd"]=trade["pnl"]; trade["pnl_pct"]=safe_div(trade["pnl"], trade["size_usd"])
                            logging.info("EMU CLOSE TP %s pnl_usd=%.6f", sym, trade["pnl_usd"])
                            asyncio.create_task(_train_on_closed_trade(trade))
                        elif last_price >= trade["sl"]:
                            trade.update({"status":"closed","exit_price":last_price,"close_time":now_iso()})
                            trade["pnl"]=(trade["entry"]-last_price)*trade["size"]; trade["pnl_usd"]=trade["pnl"]; trade["pnl_pct"]=safe_div(trade["pnl"], trade["size_usd"])
                            logging.info("EMU CLOSE SL %s pnl_usd=%.6f", sym, trade["pnl_usd"])
                            asyncio.create_task(_train_on_closed_trade(trade))
                save_json(TRADE_LOG_FILE, TRADE_HISTORY)
            await asyncio.sleep(interval)
        except Exception:
            logging.exception("trade_monitor")
            await asyncio.sleep(5)

# ---------------- Minute OI worker ----------------
async def minute_oi_worker(session):
    global LAST_OI, SENT
    while True:
        try:
            syms = await fetch_top_symbols(session, limit=TOP_N)
            if not syms: await asyncio.sleep(5); continue
            syms = [s for s in syms if s.endswith("USDT")]
            for s in syms:
                oi_now = await fetch_current_oi(session, s)
                if oi_now is None: continue
                prev = LAST_OI.get(s); LAST_OI[s]=oi_now
                if prev is None or prev==0: continue
                change_pct = (oi_now - prev)/prev*100.0
                if abs(change_pct) >= MINUTE_OI_THRESHOLD_PCT:
                    side = "LONG" if change_pct>0 else "SHORT"
                    key = f"{s}:MIN_OI:{side}"
                    async with SENT_LOCK:
                        last = SENT.get(key); now_ts = int(time.time())
                        if last and (now_ts-last) < COOLDOWN_SECONDS: continue
                        emoji = "⚡"
                        txt = (f"{emoji} <b>Minute OI</b> <code>{s}</code> — <b>{side}</b>\n"
                               f"📊 <b>Δ1m OI:</b> {change_pct:.2f}%\n"
                               f"⏱ <b>Time:</b> {now_iso()}")
                        await send_telegram(session, txt)
                        SENT[key]=now_ts; save_json(SENT_FILE, SENT)
            await asyncio.sleep(60)
        except Exception:
            logging.exception("minute_oi_worker")
            await asyncio.sleep(5)

# ---------------- Telegram send ----------------
async def send_telegram(session, text):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logging.info("Telegram not configured")
        return None
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        async with session.post(url, data=payload, timeout=10) as resp:
            txt = await resp.text(); status = resp.status
            logging.info("tg status %s", status)
            return {"status": status, "text": txt}
    except Exception:
        logging.exception("tg send")
        return None

# ---------------- ML trainer ----------------
async def ml_trainer(session):
    global MODEL, SCALER, MODEL_FITTED
    init_model()
    while True:
        try:
            X=[]; Y=[]
            for sym, dq in list(FEATURE_HISTORY.items()):
                for item in list(dq):
                    sample_time = item["time"]
                    if time.time() - sample_time < LABEL_HORIZON_MINUTES*60: continue
                    feat = item["feat"]
                    cur_price = None
                    try: cur_price = FEATURE_HISTORY[sym][-1]["feat"]["price"]
                    except: cur_price = None
                    if cur_price is None:
                        df = await fetch_klines(session, sym, "1m", limit=2)
                        if df is None: continue
                        cur_price = float(df['close'].iloc[-1])
                    base = feat.get("price")
                    if base is None or base==0:
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
                    except: pass
                    Xs = SCALER.transform(X) if SCALER is not None else X
                    MODEL.partial_fit(Xs, Y, classes=np.array([0,1]))
                    MODEL_FITTED=True
                    logging.info("ML init partial_fit %d samples", len(Y))
                else:
                    try: Xs = SCALER.transform(X)
                    except:
                        try: SCALER.fit(X); Xs = SCALER.transform(X)
                        except: Xs = X
                    MODEL.partial_fit(Xs, Y)
                    logging.info("ML partial_fit +%d", len(Y))
                save_model()
            await asyncio.sleep(30 if HEAVY else 90)
        except Exception:
            logging.exception("ml_trainer")
            await asyncio.sleep(10)

# ---------------- Main loop ----------------
async def main():
    global SENT
    SENT = load_json(SENT_FILE, default={}) or {}
    init_model()
    async with aiohttp.ClientSession() as session:
        loop = asyncio.get_event_loop()
        if TELETHON_AVAILABLE and TG_API_ID and TG_API_HASH:
            loop.create_task(telethon_listener(loop))
        else:
            loop.create_task(html_liq_poller(session))
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
                logging.info("%s scanning %d symbols", now_iso(), len(symbols))
                # Light features
                light_tasks = [compute_light_features(session, s, price_cache) for s in symbols]
                light_results = await asyncio.gather(*light_tasks)
                scored=[]
                for feat in light_results:
                    if not feat: continue
                    # simple light scoring
                    sc = 0.0
                    if feat.get("ema_diff") is not None:
                        sc += np.sign(feat["ema_diff"])
                    if feat.get("rsi") is not None:
                        if feat["rsi"]>60: sc += 1
                        elif feat["rsi"]<40: sc -= 1
                    if feat.get("vol_ratio") and feat["vol_ratio"]>1.6: sc += 0.5
                    scored.append({"symbol":feat["symbol"], "score": sc})
                scored_sorted = sorted([c for c in scored if c], key=lambda x: x["score"], reverse=True)
                candidates = [c["symbol"] for c in scored_sorted if c["score"]>0][:CANDIDATE_LIMIT]
                neg_cands = [c["symbol"] for c in scored_sorted if c["score"]< -0.8][:CANDIDATE_LIMIT]
                for nc in neg_cands:
                    if nc not in candidates: candidates.append(nc)
                logging.info("candidates heavy: %s", candidates)
                heavy_tasks = [compute_heavy_features(session, s, price_cache) for s in candidates]
                heavy_results = await asyncio.gather(*heavy_tasks)
                for feat in heavy_results:
                    if not feat: continue
                    sig = await generate_signal_heavy(session, feat)
                    if not sig: continue
                    key = f"{sig['symbol']}:{sig['side']}"
                    now_ts = int(time.time())
                    async with SENT_LOCK:
                        last = SENT.get(key)
                        if last and (now_ts-last) < COOLDOWN_SECONDS:
                            logging.info("skip cooldown %s", key); continue
                    await send_telegram(session, sig["txt"])
                    async with SENT_LOCK:
                        SENT[key] = now_ts; save_json(SENT_FILE, SENT)
                    await open_trade_emulation(sig)
                await asyncio.sleep(POLL_INTERVAL if not HEAVY else max(10, POLL_INTERVAL//2))
            except Exception:
                logging.exception("main loop")
                await asyncio.sleep(5)

if __name__ == "__main__":
    logging.info("Starting Smart Alpha Bot v2 HEAVY=%s WS=%s BACKTEST=%s", HEAVY, USE_WS, BACKTEST)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("stopped by user")
    except Exception:
        logging.exception("fatal")
