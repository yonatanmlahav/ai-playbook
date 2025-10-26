# api/server.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from .scanner import scanner

app = FastAPI(title="Stock Breakout Scanner + TradingView Integration")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ScanRequest(BaseModel):
    symbols: List[str]

@app.get("/")
async def root():
    return {
        "message": "🚀 Stock Breakout Scanner API",
        "version": "2.0",
        "model_loaded": scanner.model is not None,
        "endpoints": {
            "scan": "POST /scan - Scan multiple stocks",
            "stock": "GET /stock/{symbol} - Get single stock analysis",
            "tradingview": "POST /tradingview/webhook - TradingView webhook"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/stock/{symbol}")
async def get_stock(symbol: str):
    """Analyze single stock"""
    try:
        result = scanner.predict(symbol.upper())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scan")
async def scan_stocks(request: ScanRequest):
    """Scan multiple stocks"""
    results = []
    for symbol in request.symbols:
        try:
            result = scanner.predict(symbol.upper())
            if result["probability"] >= 0.5:
                results.append(result)
        except:
            continue
    
    return sorted(results, key=lambda x: x["probability"], reverse=True)

@app.post("/tradingview/webhook")
async def tradingview_webhook(request: Request):
    """
    TradingView Webhook Endpoint
    
    TradingView sends alerts like:
    {
      "ticker": "AAPL",
      "price": 150.25,
      "indicator": "RSI_CROSS"
    }
    """
    try:
        data = await request.json()
        
        symbol = data.get("ticker") or data.get("symbol")
        if not symbol:
            raise HTTPException(400, "Missing ticker/symbol")
        
        # Analyze the stock
        result = scanner.predict(symbol.upper())
        
        # Format response for TradingView
        signal_emoji = "🚀" if result["probability"] >= 0.7 else "✅" if result["probability"] >= 0.55 else "⚠️"
        
        return {
            "success": True,
            "symbol": symbol.upper(),
            "signal": result["signal"],
            "probability": result["probability"],
            "confidence": f"{result['confidence']:.1f}%",
            "message": f"{signal_emoji} {symbol.upper()} | {result['signal']} | {result['confidence']:.1f}% confidence",
            "price": result["price"],
            "indicators": result["indicators"]
        }
    
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/watchlist")
async def scan_watchlist():
    """Scan popular stocks"""
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "AMD"]
    
    results = []
    for symbol in symbols:
        try:
            result = scanner.predict(symbol)
            results.append(result)
        except:
            continue
    
    return sorted(results, key=lambda x: x["probability"], reverse=True)
```

---

## 🚀 שלב 4: Deploy ל-Railway

1. **Commit הכל ל-GitHub**
2. **ב-Railway:**
   - New Project → Deploy from GitHub
   - בחר `ai-playbook`
   - **Variables:**
```
     SPREADSHEET_ID=1jtPpiHlgl5Fv32A1l23AeULnHYoqyPtO4owddva3bY4
     GCP_SA_JSON_PATH=/app/sa.json
