
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kamil_Krypto_BOT v4 — Profesjonalny bot analityczny dla XRP/USDT
- Binance SPOT
- Telegram Alerts
- Sentiment z newsów
- Tryb obserwacji i zarządzanie pozycją
"""

import argparse, csv, os, time, traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta

# ========= KONFIG TELEGRAM =========
TELEGRAM_TOKEN = "8210702612:AAGoN83LknOSTviNRkOKYVug-vpZd9PegV0"      # <- wklej swój token np. 8210...:AA...
TELEGRAM_CHAT_ID = "6372346191"  # <- np. 6372346191
SYMBOL = "XRP/USDT"

# ========= IMPORTS =========
import numpy as np
import pandas as pd
import requests
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import ccxt
from colorama import init as colorama_init, Fore, Style
colorama_init()

# ========= UTILS =========
def tz_now():
    return datetime.now(timezone.utc).astimezone()

def telegram_ready() -> bool:
    return bool(TELEGRAM_TOKEN) and "WSTAW" not in TELEGRAM_TOKEN and bool(TELEGRAM_CHAT_ID)

def send_telegram(msg: str, debug: bool=False):
    if not telegram_ready():
        if debug: print(Fore.YELLOW + "[Telegram OFF] " + msg + Style.RESET_ALL)
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
        if debug: print(Fore.GREEN + "[Telegram OK] Message sent" + Style.RESET_ALL)
    except Exception as e:
        print(Fore.YELLOW + f"[Telegram error] {e}" + Style.RESET_ALL)

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(s: pd.Series, period: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    ru = up.ewm(alpha=1/period, adjust=False).mean()
    rd = dn.ewm(alpha=1/period, adjust=False).mean()
    rs = ru / (rd.replace(0, np.nan))
    return (100 - (100/(1+rs))).fillna(50)

def macd(s: pd.Series, f: int=12, sl: int=26, sg: int=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ef = ema(s, f); es = ema(s, sl)
    line = ef - es
    sig = ema(line, sg)
    hist = line - sig
    return line, sig, hist

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    pc = df['close'].shift(1)
    tr = pd.concat([(df['high']-df['low']),
                    (df['high']-pc).abs(),
                    (df['low']-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up_move = df['high'] - df['high'].shift(1)
    down_move = df['low'].shift(1) - df['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    trur = pd.concat([
        df['high']-df['low'],
        (df['high']-df['close'].shift(1)).abs(),
        (df['low']-df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_ = trur.rolling(window=period).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / atr_)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / atr_)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(window=period).mean().fillna(20)

def bollinger(s: pd.Series, n: int = 20, k: float = 2.0):
    ma = s.rolling(n).mean()
    sd = s.rolling(n).std()
    upper = ma + k*sd
    lower = ma - k*sd
    return ma, upper, lower

# ========= BINANCE =========
BINANCE_ALT_ENDPOINTS = ["https://api1.binance.com","https://api2.binance.com","https://api3.binance.com"]
BINANCE = None

def make_binance_spot(debug: bool=False):
    if debug: print(Fore.CYAN + "[DEBUG] Init Binance SPOT..." + Style.RESET_ALL)
    ex = ccxt.binance({
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": "spot", "adjustForTimeDifference": True}
    })
    ex.load_markets()
    if debug: print(Fore.GREEN + "[DEBUG] load_markets OK" + Style.RESET_ALL)
    return ex

def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 300, debug: bool=False) -> pd.DataFrame:
    """Świece z retry i fallback na hostname"""
    global BINANCE
    last_err = None
    for attempt in range(4):
        try:
            data = BINANCE.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('Europe/Warsaw')
            if debug: print(Fore.GREEN + f"[DEBUG] fetch_ohlcv {timeframe} OK (len={len(df)})" + Style.RESET_ALL)
            return df.set_index('timestamp')
        except Exception as e:
            last_err = e
            if attempt < len(BINANCE_ALT_ENDPOINTS):
                host = BINANCE_ALT_ENDPOINTS[attempt]
                BINANCE.hostname = host.replace("https://","").replace("http://","")
                if debug: print(Fore.YELLOW + f"[DEBUG] switch hostname -> {BINANCE.hostname}" + Style.RESET_ALL)
                time.sleep(1)
            else:
                break
    raise last_err if last_err else Exception("fetch_ohlcv failed")

# ========= SENTIMENT =========
DEFAULT_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
    "https://cryptoslate.com/feed/",
    "https://u.today/rss",
    "https://www.ripple.com/insights/feed/",
]
SENTI_KEYS = ["xrp", "ripple", "xrpl", "sec", "garlinghouse", "xrp ledger"]
_analyzer = SentimentIntensityAnalyzer()

def sentiment_snapshot(window: int = 25, debug: bool=False):
    items: List[Dict] = []
    for url in DEFAULT_FEEDS:
        try:
            parsed = feedparser.parse(url)
            for e in parsed.entries[:60]:
                title = getattr(e, "title", "")
                summary = getattr(e, "summary", "")
                link = getattr(e, "link", "")
                published = getattr(e, "published", "") or getattr(e, "updated", "")
                low = (title+" "+summary).lower()
                if any(k in low for k in SENTI_KEYS):
                    sc = _analyzer.polarity_scores(title + " :: " + summary)["compound"]
                    items.append({"title": title, "score": round(sc,3), "link": link, "published": published})
        except Exception:
            continue
    items = sorted(items, key=lambda x: x.get("published", ""), reverse=True)[:window]
    avg = round(sum(i["score"] for i in items)/len(items), 3) if items else 0.0
    if debug: print(Fore.CYAN + f"[DEBUG] sentiment avg={avg} (items={len(items)})" + Style.RESET_ALL)
    return avg, items

def _truncate(txt: str, n: int = 70) -> str:
    txt = txt.replace('\n', ' ').strip()
    return (txt[:n-1] + '…') if len(txt) > n else txt

def news_block(articles: list, limit: int = 3) -> str:
    if not articles:
        return "Brak świeżych newsów."
    lines = []
    for a in articles[:limit]:
        title = _truncate(a.get("title", ""))
        score = a.get("score", 0.0)
        lines.append(f"• [{score:+.2f}] {title}")
    return "\n".join(lines)

# ========= STRUKTURY =========
@dataclass
class Pos:
    qty: float = 0.0
    avg: float = 0.0
    rem: float = 0.0
    be: float = 0.0
    trail: bool = False
    hold: int = 0

@dataclass
class Risk:
    equity: float = 10000.0
    risk: float = 0.01
    atr_sl: float = 2.0
    tp1: float = 0.02
    trail_atr: float = 1.3
    min_hold: int = 2

# ========= LOGIKA GŁÓWNA =========
def main():
    pa = argparse.ArgumentParser(description="Kamil_Krypto_BOT v4 — SPOT, Telegram, Sentiment, DEBUG")
    pa.add_argument('--poll', type=int, default=60)
    pa.add_argument('--etf', default='1h')
    pa.add_argument('--equity', type=float, default=10000.0)
    pa.add_argument('--sentiment-every', type=int, default=600)
    pa.add_argument('--heartbeat-hours', type=int, default=4)
    pa.add_argument('--debug', action='store_true')
    args = pa.parse_args()

    if args.debug:
        print(Fore.CYAN + "=== DEBUG START ===" + Style.RESET_ALL)
        print("[DEBUG] Telegram configured:", telegram_ready())

    print("\nUstaw pozycję dla XRP/USDT (ENTER = 0):")
    qty_in = input("Ile masz XRP? ").strip().replace(",", ".")
    avg_in = input("Jaka średnia cena zakupu (USDT)? ").strip().replace(",", ".")
    try:
        qty = float(qty_in) if qty_in else 0.0
        avg = float(avg_in) if avg_in else 0.0
    except:
        qty, avg = 0.0, 0.0

    if qty <= 0 or avg <= 0:
        print(Fore.YELLOW + "Tryb OBSERWACJI: bot nie ma pozycji – będzie szukał okazji BUY." + Style.RESET_ALL)
    else:
        print(Fore.GREEN + f"Ustawiono: {qty} XRP @ {avg} USDT" + Style.RESET_ALL)

    rp = Risk(equity=args.equity)
    position = Pos(qty=qty, avg=avg, rem=qty, be=avg, trail=False, hold=0)
    last_status = "INIT"
    last_heartbeat = tz_now()

    global BINANCE
    BINANCE = make_binance_spot(debug=args.debug)
    if args.debug:
        test = fetch_ohlcv(SYMBOL, "1h", 2, debug=args.debug)
        print(Fore.GREEN + f"[DEBUG] Test OHLCV OK: closes={list(test['close'][-2:])}" + Style.RESET_ALL)

    send_telegram("🤖 Bot wystartował! Tryb: OBSERWACJA" if qty==0 else "🤖 Bot wystartował z pozycją", debug=args.debug)

    senti_score, senti_articles = 0.0, []
    last_senti_ts = datetime(1970,1,1, tzinfo=timezone.utc)

    while True:
        try:
            if (tz_now() - last_senti_ts).total_seconds() >= args.sentiment_every:
                senti_score, senti_articles = sentiment_snapshot(window=20, debug=args.debug)
                last_senti_ts = tz_now()

            df = fetch_ohlcv(SYMBOL, args.etf, 300, debug=args.debug)
            df['ema20'] = ema(df['close'],20)
            df['ema50'] = ema(df['close'],50)
            df['ema200'] = ema(df['close'],200)
            df['rsi'] = rsi(df['close'])
            _,_,df['macd'] = macd(df['close'])
            last = df.iloc[-1]; prev = df.iloc[-2]
            price = float(last['close'])
            now = tz_now().strftime("%Y-%m-%d %H:%M")

            status = "HOLD"
            reason = "—"

            if position.qty <= 0:
                if last['ema20'] > last['ema50'] and last['macd'] > 0 and last['rsi'] > 55 and senti_score >= 0:
                    status, reason = "BUY", "Trend wzrostowy + pozytywny sentyment"
            else:
                if last['rsi'] > 65 and last['macd'] < prev['macd']:
                    status, reason = "SELL", "Sygnał wychodzenia z momentum"

            pnl = ((price / position.avg)-1)*100 if position.avg>0 else 0

            color = Fore.YELLOW
            if status=="BUY": color=Fore.GREEN
            elif status=="SELL": color=Fore.RED
            print("="*110)
            print(f"{now} | Cena {price:.4f} | RSI {last['rsi']:.1f} | MACD {last['macd']:+.4f} | EMA20/50/200 {last['ema20']:.4f}/{last['ema50']:.4f}/{last['ema200']:.4f}")
            print(color + f"STATUS: {status} | Powód: {reason} | Sentiment {senti_score:+.2f}" + Style.RESET_ALL)

            if status != last_status:
                news_txt = news_block(senti_articles, limit=3)
                send_telegram(
                    f"🚨 {status} {SYMBOL}\nCena: {price:.4f}\nPowód: {reason}\nSentiment {senti_score:+.2f}\n📰 News:\n{news_txt}",
                    debug=args.debug
                )
                last_heartbeat = tz_now()

            elif (tz_now() - last_heartbeat) >= timedelta(hours=args.heartbeat_hours):
                news_txt = news_block(senti_articles, limit=3)
                send_telegram(
                    f"⏱️ Status {status} | Cena {price:.4f} | RSI {last['rsi']:.1f} | MACD {last['macd']:+.4f}\nSentiment {senti_score:+.2f}\n📰 News:\n{news_txt}",
                    debug=args.debug
                )
                last_heartbeat = tz_now()

            last_status = status
        except KeyboardInterrupt:
            send_telegram("🛑 Bot zatrzymany przez użytkownika.", debug=args.debug)
            break
        except Exception as e:
            print(Fore.RED + f"[ERROR] {e}\n{traceback.format_exc()}" + Style.RESET_ALL)
            send_telegram(f"⚠️ Błąd: {e}")
        time.sleep(args.poll)

if __name__ == "__main__":
    main()
