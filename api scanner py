# ============================================================================
# קובץ 1: requirements.txt (שים בתיקייה הראשית)
# ============================================================================
pandas
numpy
gspread
google-auth
google-auth-oauthlib
xgboost
scikit-learn
joblib
lightgbm
fastapi
uvicorn
yfinance
ta
python-multipart

# ============================================================================
# קובץ 2: Procfile (שים בתיקייה הראשית - ללא סיומת!)
# ============================================================================
web: uvicorn api.server:app --host 0.0.0.0 --port $PORT

# ============================================================================
# קובץ 3: .gitignore (שים בתיקייה הראשית)
# ============================================================================
sa.json
*.joblib
__pycache__/
*.pyc
.env
.DS_Store
node_modules/
venv/
*.log

# ============================================================================
# קובץ 4: api/__init__.py (צור תיקייה api/ ושים בתוכה)
# ============================================================================
from .server import app

__all__ = ['app']

# ============================================================================
# קובץ 5: api/scanner.py (שים בתיקייה api/)
# ============================================================================
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List
import sys
import os

# Import your training functions
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from Train import engineer_features
    TRAIN_AVAILABLE = True
except:
    TRAIN_AVAILABLE = False
    print("⚠️ Train.py not available")

from joblib import load
import pandas as pd

class StockScanner:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        """Load trained models"""
        try:
            if os.path.exists('model.joblib'):
                self.model = load('model.joblib')
                print("✅ Model loaded")
            if os.path.exists('scaler.joblib'):
                self.scaler = load('scaler.joblib')
                print("✅ Scaler loaded")
        except Exception as e:
            print(f"⚠️ Model not loaded: {e}")
    
    def fetch_stock(self, symbol: str) -> List[Dict]:
        """Fetch stock data from yfinance"""
        try:
            stock = yf.Ticker(symbol)
            end = datetime.now()
            start = end - timedelta(days=150)
            
            df = stock.history(start=start, end=end)
            
            if df.empty:
                raise ValueError(f"No data for {symbol}")
            
            data = []
            for date, row in df.iterrows():
                data.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"])
                })
            
            return data[-100:]
        
        except Exception as e:
            raise Exception(f"Error fetching {symbol}: {e}")
    
    def calculate_indicators(self, data: List[Dict]) -> Dict:
        """Calculate technical indicators"""
        import ta
        
        df = pd.DataFrame(data)
        df['close'] = pd.to_numeric(df['close'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        # RSI
        rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi().iloc[-1]
        rsi = float(rsi) if not pd.isna(rsi) else 50.0
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        macd_val = macd.macd_diff().iloc[-1]
        macd_val = float(macd_val) if not pd.isna(macd_val) else 0.0
        
        # Volume
        avg_vol = df['volume'].rolling(20).mean().iloc[-1]
        vol_spike = df['volume'].iloc[-1] / avg_vol if avg_vol > 0 else 1.0
        
        # Breakout
        high_20 = df['high'].rolling(20).max().iloc[-1]
        current = df['close'].iloc[-1]
        breakout = ((current - high_20) / high_20 * 100) if high_20 > 0 else 0.0
        
        # Gap
        gap = 0.0
        if len(df) > 1:
            gap = ((df['open'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100)
        
        # Score
        score = (
            (rsi / 100 * 30) +
            (min(vol_spike / 3, 1) * 30) +
            (1 if macd_val > 0 else 0) * 20 +
            (1 if breakout > 0 else 0) * 20
        )
        
        return {
            'RSI': float(rsi),
            'MACD': float(macd_val),
            'VolSpike': float(vol_spike),
            'Breakout%': float(breakout),
            'Gap%': float(gap),
            'A_Score': float(score)
        }
    
    def predict(self, symbol: str) -> Dict:
        """Predict if stock will breakout"""
        try:
            # Fetch data
            data = self.fetch_stock(symbol)
            
            # Calculate indicators
            indicators = self.calculate_indicators(data)
            
            # If model exists, use it
            if self.model and TRAIN_AVAILABLE:
                features_df = pd.DataFrame([indicators])
                X = engineer_features(features_df)
                
                if self.scaler:
                    X = self.scaler.transform(X)
                
                if hasattr(self.model, 'predict_proba'):
                    prob = self.model.predict_proba(X)[0][1]
                else:
                    prob = float(self.model.predict(X)[0])
            else:
                # Fallback to score-based prediction
                prob = indicators['A_Score'] / 100.0
            
            return {
                "symbol": symbol,
                "probability": float(prob),
                "confidence": float(prob * 100),
                "signal": "🚀 BUY" if prob >= 0.6 else "⚠️ HOLD" if prob >= 0.4 else "❌ AVOID",
                "indicators": indicators,
                "price": data[-1]["close"],
                "date": data[-1]["date"]
            }
        
        except Exception as e:
            raise Exception(f"Prediction error: {e}")

scanner = StockScanner()

# ============================================================================
# קובץ 6: api/server.py (שים בתיקייה api/)
# ============================================================================
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from .scanner import scanner

app = FastAPI(
    title="Stock Breakout Scanner + TradingView",
    description="AI-powered stock scanner with TradingView integration",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ScanRequest(BaseModel):
    symbols: List[str]
    threshold: Optional[float] = 0.5

class TradingViewAlert(BaseModel):
    ticker: Optional[str] = None
    symbol: Optional[str] = None
    price: Optional[float] = None
    indicator: Optional[str] = None

@app.get("/")
async def root():
    return {
        "message": "🚀 Stock Breakout Scanner API",
        "version": "2.0",
        "model_loaded": scanner.model is not None,
        "status": "running",
        "endpoints": {
            "health": "GET /health",
            "analyze_stock": "GET /stock/{symbol}",
            "scan_multiple": "POST /scan",
            "watchlist": "GET /watchlist",
            "tradingview_webhook": "POST /tradingview/webhook"
        },
        "tradingview_setup": {
            "webhook_url": "https://your-app.railway.app/tradingview/webhook",
            "method": "POST",
            "body_format": '{"ticker": "{{ticker}}", "price": {{close}}}'
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": scanner.model is not None,
        "scaler_loaded": scanner.scaler is not None
    }

@app.get("/stock/{symbol}")
async def get_stock(symbol: str):
    """
    Analyze a single stock
    
    Example: GET /stock/AAPL
    """
    try:
        result = scanner.predict(symbol.upper())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scan")
async def scan_stocks(request: ScanRequest):
    """
    Scan multiple stocks
    
    Example POST body:
    {
        "symbols": ["AAPL", "MSFT", "GOOGL"],
        "threshold": 0.5
    }
    """
    results = []
    for symbol in request.symbols:
        try:
            result = scanner.predict(symbol.upper())
            if result["probability"] >= request.threshold:
                results.append(result)
        except Exception as e:
            print(f"Error scanning {symbol}: {e}")
            continue
    
    return {
        "total_scanned": len(request.symbols),
        "signals_found": len(results),
        "results": sorted(results, key=lambda x: x["probability"], reverse=True)
    }

@app.get("/watchlist")
async def scan_watchlist():
    """
    Scan popular tech stocks
    
    Example: GET /watchlist
    """
    symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", 
        "NVDA", "META", "AMD", "NFLX", "DIS"
    ]
    
    results = []
    for symbol in symbols:
        try:
            result = scanner.predict(symbol)
            results.append(result)
        except Exception as e:
            print(f"Error: {symbol} - {e}")
            continue
    
    return {
        "timestamp": scanner.fetch_stock("AAPL")[0]["date"],
        "total_stocks": len(results),
        "top_picks": sorted(results, key=lambda x: x["probability"], reverse=True)[:5],
        "all_results": sorted(results, key=lambda x: x["probability"], reverse=True)
    }

@app.post("/tradingview/webhook")
async def tradingview_webhook(request: Request):
    """
    TradingView Webhook Endpoint
    
    This endpoint receives alerts from TradingView and returns AI predictions.
    
    Setup in TradingView:
    1. Create an alert
    2. Set Webhook URL: https://your-app.railway.app/tradingview/webhook
    3. Message format:
       {
         "ticker": "{{ticker}}",
         "price": {{close}},
         "time": "{{time}}"
       }
    
    Example POST body:
    {
        "ticker": "AAPL",
        "price": 150.25,
        "indicator": "RSI_CROSS"
    }
    """
    try:
        # Parse the request
        try:
            data = await request.json()
        except:
            # If not JSON, try to parse as text
            text = await request.body()
            data = {"raw": text.decode()}
        
        # Extract symbol
        symbol = data.get("ticker") or data.get("symbol")
        if not symbol:
            raise HTTPException(400, "Missing ticker/symbol in request")
        
        # Analyze the stock with our AI
        result = scanner.predict(symbol.upper())
        
        # Determine signal strength
        prob = result["probability"]
        if prob >= 0.7:
            signal_emoji = "🚀"
            signal_text = "STRONG BUY"
            action = "ENTER POSITION"
        elif prob >= 0.55:
            signal_emoji = "✅"
            signal_text = "BUY"
            action = "CONSIDER ENTRY"
        elif prob >= 0.45:
            signal_emoji = "⚠️"
            signal_text = "NEUTRAL"
            action = "WAIT"
        else:
            signal_emoji = "❌"
            signal_text = "AVOID"
            action = "DO NOT ENTER"
        
        # Format response
        response = {
            "success": True,
            "symbol": symbol.upper(),
            "signal": result["signal"],
            "action": action,
            "probability": result["probability"],
            "confidence": f"{result['confidence']:.1f}%",
            "message": f"{signal_emoji} {symbol.upper()} | {signal_text} | Confidence: {result['confidence']:.1f}%",
            "price": result["price"],
            "indicators": result["indicators"],
            "timestamp": result["date"],
            "tradingview_data": data
        }
        
        return response
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(500, f"Error processing webhook: {str(e)}")

@app.post("/tradingview/batch")
async def tradingview_batch(request: Request):
    """
    Batch endpoint for multiple TradingView alerts
    
    Example POST body:
    {
        "alerts": [
            {"ticker": "AAPL"},
            {"ticker": "MSFT"},
            {"ticker": "GOOGL"}
        ]
    }
    """
    try:
        data = await request.json()
        alerts = data.get("alerts", [])
        
        results = []
        for alert in alerts:
            symbol = alert.get("ticker") or alert.get("symbol")
            if symbol:
                try:
                    result = scanner.predict(symbol.upper())
                    results.append(result)
                except:
                    continue
        
        return {
            "success": True,
            "total": len(alerts),
            "analyzed": len(results),
            "results": sorted(results, key=lambda x: x["probability"], reverse=True)
        }
    
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/tradingview/setup")
async def tradingview_setup_guide():
    """
    Instructions for setting up TradingView webhooks
    """
    return {
        "title": "TradingView Webhook Setup Guide",
        "steps": [
            "1. Open TradingView and create a chart",
            "2. Click the Alert button (clock icon)",
            "3. Set your alert conditions (e.g., RSI > 60)",
            "4. In the 'Notifications' tab, check 'Webhook URL'",
            "5. Enter your webhook URL: https://your-app.railway.app/tradingview/webhook",
            "6. In the 'Message' field, enter: {\"ticker\": \"{{ticker}}\", \"price\": {{close}}}",
            "7. Click 'Create'"
        ],
        "webhook_url": "https://your-app.railway.app/tradingview/webhook",
        "message_template": {
            "ticker": "{{ticker}}",
            "price": "{{close}}",
            "volume": "{{volume}}",
            "time": "{{time}}"
        },
        "available_variables": [
            "{{ticker}}", "{{close}}", "{{open}}", "{{high}}", "{{low}}",
            "{{volume}}", "{{time}}", "{{exchange}}"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
