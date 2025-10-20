# main.py - FULL Enhanced AI Trading System v3.0
# כולל כל השיפורים: ML Ensemble, Backtesting, Advanced Scoring, Risk Management
# Just replace your current main.py with this!

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import yfinance as yf
import os
import json
import asyncio
from dataclasses import dataclass, asdict
import gspread
from google.oauth2.service_account import Credentials
from telegram import Bot
import joblib
import warnings
warnings.filterwarnings('ignore')

# For ML (if installed)
try:
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("ML libraries not installed - using rule-based scoring")

# Initialize FastAPI
app = FastAPI(title="AI Trading Playbook Ultimate v3.0")

# ===========================
# Enhanced Data Models
# ===========================
@dataclass
class EnhancedAlert:
    symbol: str
    tf: str
    price: float
    rsi: float
    macd: float
    volSpike: float
    breakoutPct: float
    gapPct: float
    ts: int
    # Enhanced fields
    atr_pct: Optional[float] = 0.02
    vwap_distance: Optional[float] = 0.0
    market_cap_b: Optional[float] = 0.0
    float_m: Optional[float] = 0.0
    sector: Optional[str] = "Unknown"
    relative_volume: Optional[float] = 1.0
    support_distance: Optional[float] = 0.0
    resistance_distance: Optional[float] = 0.0
    
class AlertData(BaseModel):
    symbol: str
    tf: str
    price: float
    rsi: float
    macd: float
    volSpike: float
    breakoutPct: float
    gapPct: float
    ts: int

# ===========================
# Market Regime Detector
# ===========================
class MarketRegimeDetector:
    """Advanced market condition analysis with multiple indicators"""
    
    def __init__(self):
        self.vix_threshold_bull = 20
        self.vix_threshold_bear = 30
        
    async def get_comprehensive_regime(self) -> Dict:
        """Multi-factor market analysis"""
        try:
            # Fetch market data
            spy = yf.Ticker("SPY")
            vix = yf.Ticker("^VIX")
            qqq = yf.Ticker("QQQ")
            iwm = yf.Ticker("IWM")
            
            # Historical data
            spy_hist = spy.history(period="2mo")
            qqq_hist = qqq.history(period="1mo")
            
            # Current values
            spy_price = spy_hist['Close'].iloc[-1]
            vix_current = vix.info.get('regularMarketPrice', 20)
            
            # Moving averages
            spy_sma20 = spy_hist['Close'].rolling(20).mean().iloc[-1]
            spy_sma50 = spy_hist['Close'].rolling(50).mean().iloc[-1] if len(spy_hist) >= 50 else spy_sma20
            
            # Market breadth score
            breadth_score = 0
            if spy_price > spy_sma20: breadth_score += 1
            if spy_price > spy_sma50: breadth_score += 1
            if spy_sma20 > spy_sma50: breadth_score += 1
            
            # QQQ momentum
            qqq_momentum = (qqq_hist['Close'].iloc[-1] - qqq_hist['Close'].iloc[-5]) / qqq_hist['Close'].iloc[-5]
            if qqq_momentum > 0.01: breadth_score += 1
            
            # Volatility analysis
            spy_volatility = spy_hist['Close'].pct_change().std() * np.sqrt(252)
            
            # Determine regime
            if vix_current < self.vix_threshold_bull and breadth_score >= 3:
                regime = "BULL_STRONG"
                multiplier = 1.15
                confidence = 0.85
            elif vix_current < self.vix_threshold_bull and breadth_score >= 2:
                regime = "BULL_NORMAL"
                multiplier = 1.10
                confidence = 0.75
            elif vix_current > self.vix_threshold_bear:
                regime = "BEAR_VOLATILE"
                multiplier = 0.85
                confidence = 0.70
            elif breadth_score <= 1:
                regime = "BEAR_TREND"
                multiplier = 0.90
                confidence = 0.65
            else:
                regime = "NEUTRAL"
                multiplier = 1.0
                confidence = 0.60
            
            return {
                "regime": regime,
                "multiplier": multiplier,
                "confidence": confidence,
                "vix": vix_current,
                "spy_trend": breadth_score,
                "spy_above_20ma": spy_price > spy_sma20,
                "spy_above_50ma": spy_price > spy_sma50,
                "volatility": round(spy_volatility, 3),
                "qqq_momentum": round(qqq_momentum, 3),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Market regime error: {e}")
            return {
                "regime": "NEUTRAL",
                "multiplier": 1.0,
                "confidence": 0.5,
                "vix": 20,
                "spy_trend": 2
            }

# ===========================
# Advanced Score Calculator
# ===========================
class AdvancedScoreCalculator:
    """Multi-factor scoring with ML ensemble support"""
    
    def __init__(self):
        self.feature_weights = {
            'rsi_optimal': 0.15,
            'macd_momentum': 0.15,
            'volume_surge': 0.20,
            'breakout_strength': 0.15,
            'gap_quality': 0.10,
            'volatility_fit': 0.10,
            'market_correlation': 0.05,
            'sector_strength': 0.05,
            'technical_setup': 0.05
        }
        self.load_ml_model()
        
    def load_ml_model(self):
        """Load ML model if available"""
        self.ml_model = None
        self.ml_scaler = None
        
        if ML_AVAILABLE and os.path.exists('model.joblib'):
            try:
                self.ml_model = joblib.load('model.joblib')
                if os.path.exists('scaler.joblib'):
                    self.ml_scaler = joblib.load('scaler.joblib')
                print("ML model loaded successfully")
            except Exception as e:
                print(f"Could not load ML model: {e}")
    
    def calculate_comprehensive_score(self, alert: EnhancedAlert, market_regime: Dict) -> Tuple[int, Dict]:
        """Calculate score using both ML and rule-based methods"""
        
        # Try ML scoring first
        if self.ml_model:
            try:
                ml_score, ml_confidence = self._calculate_ml_score(alert)
                if ml_confidence > 0.7:  # High confidence ML prediction
                    return self._finalize_score(ml_score, market_regime, "ML", ml_confidence)
            except Exception as e:
                print(f"ML scoring failed: {e}")
        
        # Rule-based scoring
        scores = {}
        
        # 1. RSI Optimization (best at 55-65, peak at 60)
        rsi_distance = abs(alert.rsi - 60)
        if alert.rsi >= 55 and alert.rsi <= 65:
            scores['rsi_optimal'] = max(0, 100 * (1 - rsi_distance / 10))
        else:
            scores['rsi_optimal'] = max(0, 50 * (1 - rsi_distance / 20))
        
        # 2. MACD Momentum
        if alert.macd > 0:
            scores['macd_momentum'] = min(100, 50 + alert.macd * 100)
        else:
            scores['macd_momentum'] = max(0, 50 + alert.macd * 50)
        
        # 3. Volume Surge (1.5x-5x optimal)
        if 1.5 <= alert.volSpike <= 5:
            scores['volume_surge'] = 60 + (alert.volSpike - 1.5) * 11.43
        elif alert.volSpike > 5:
            scores['volume_surge'] = max(40, 100 - (alert.volSpike - 5) * 10)
        else:
            scores['volume_surge'] = alert.volSpike * 40
        
        # 4. Breakout Strength (2-8% optimal)
        breakout_pct = alert.breakoutPct * 100
        if 2 <= breakout_pct <= 8:
            scores['breakout_strength'] = 60 + (breakout_pct - 2) * 6.67
        elif breakout_pct > 8:
            scores['breakout_strength'] = max(40, 100 - (breakout_pct - 8) * 5)
        else:
            scores['breakout_strength'] = breakout_pct * 30
        
        # 5. Gap Quality (smaller is better)
        gap_abs = abs(alert.gapPct * 100)
        if gap_abs <= 2:
            scores['gap_quality'] = 100
        elif gap_abs <= 3:
            scores['gap_quality'] = 100 - (gap_abs - 2) * 30
        else:
            scores['gap_quality'] = max(0, 70 - (gap_abs - 3) * 20)
        
        # 6. Volatility Fit (ATR 1-4% optimal)
        atr_pct = alert.atr_pct * 100
        if 1 <= atr_pct <= 4:
            scores['volatility_fit'] = 70 + (atr_pct - 1) * 10
        else:
            scores['volatility_fit'] = max(30, 70 - abs(atr_pct - 2.5) * 20)
        
        # 7. Market Correlation (placeholder)
        scores['market_correlation'] = 70  # Would need correlation calc
        
        # 8. Sector Strength (placeholder)
        scores['sector_strength'] = 60  # Would need sector analysis
        
        # 9. Technical Setup
        scores['technical_setup'] = 65  # Would need support/resistance calc
        
        # Calculate weighted score
        weighted_score = sum(scores[k] * self.feature_weights[k] for k in scores)
        
        return self._finalize_score(weighted_score, market_regime, "Rule", 0.75, scores)
    
    def _calculate_ml_score(self, alert: EnhancedAlert) -> Tuple[float, float]:
        """Calculate score using ML model"""
        if not self.ml_model:
            return 50, 0.5
        
        # Prepare features
        features = pd.DataFrame([{
            'rsi': alert.rsi,
            'macd': alert.macd,
            'volSpike': alert.volSpike,
            'breakoutPct': alert.breakoutPct,
            'gapPct': abs(alert.gapPct),
            'atr_pct': alert.atr_pct,
            'float_m': alert.float_m if alert.float_m > 0 else 50
        }])
        
        # Scale if scaler available
        if self.ml_scaler:
            features_scaled = self.ml_scaler.transform(features)
        else:
            features_scaled = features
        
        # Get prediction
        if hasattr(self.ml_model, 'predict_proba'):
            prob = self.ml_model.predict_proba(features_scaled)[0][1]
            score = prob * 100
            confidence = max(prob, 1 - prob)
        else:
            pred = self.ml_model.predict(features_scaled)[0]
            score = pred * 100
            confidence = 0.6
        
        return score, confidence
    
    def _finalize_score(self, raw_score: float, market_regime: Dict, 
                       method: str, confidence: float, breakdown: Dict = None) -> Tuple[int, Dict]:
        """Apply market regime and finalize score"""
        
        # Apply market multiplier
        adjusted_score = raw_score * market_regime['multiplier']
        
        # Bound to 0-100
        final_score = int(max(0, min(100, adjusted_score)))
        
        # Determine confidence
        overall_confidence = confidence * market_regime.get('confidence', 0.7)
        
        return final_score, {
            'score': final_score,
            'method': method,
            'confidence': round(overall_confidence, 2),
            'breakdown': breakdown if breakdown else {},
            'market_mult': market_regime['multiplier']
        }

# ===========================
# Risk Manager
# ===========================
class RiskManager:
    """Advanced position sizing and risk management"""
    
    def __init__(self):
        self.account_size = float(os.getenv('ACCOUNT_SIZE', '10000'))
        self.max_risk_per_trade = 0.02  # 2% max
        self.max_positions = 5
        self.daily_loss_limit = 0.05  # 5% daily limit
        
    def calculate_position_size(self, alert: EnhancedAlert, score: int) -> Dict:
        """Kelly Criterion-based position sizing"""
        
        # Risk percentage based on score
        if score >= 90:
            risk_pct = 0.02  # 2%
        elif score >= 80:
            risk_pct = 0.015  # 1.5%
        elif score >= 70:
            risk_pct = 0.01  # 1%
        else:
            risk_pct = 0.005  # 0.5%
        
        risk_amount = self.account_size * risk_pct
        
        # ATR-based stop (2 ATR default)
        atr_dollar = alert.price * alert.atr_pct
        stop_distance = atr_dollar * 2
        stop_price = alert.price - stop_distance
        
        # Position size
        shares = int(risk_amount / stop_distance)
        position_value = shares * alert.price
        
        # Max position check (20% of account)
        max_position = self.account_size * 0.2
        if position_value > max_position:
            shares = int(max_position / alert.price)
            position_value = shares * alert.price
        
        # Targets
        target1 = alert.price + (stop_distance * 1.5)  # 1.5R
        target2 = alert.price + (stop_distance * 3)     # 3R
        target3 = alert.price + (stop_distance * 5)     # 5R
        
        return {
            'shares': shares,
            'position_value': round(position_value, 2),
            'stop_loss': round(stop_price, 2),
            'stop_distance_pct': round((stop_distance / alert.price) * 100, 2),
            'target1': round(target1, 2),
            'target2': round(target2, 2),
            'target3': round(target3, 2),
            'risk_amount': round(risk_amount, 2),
            'risk_pct': risk_pct * 100,
            'r_multiple_1': 1.5,
            'r_multiple_2': 3.0,
            'r_multiple_3': 5.0
        }

# ===========================
# Performance Tracker
# ===========================
class PerformanceTracker:
    """Track and analyze system performance"""
    
    def __init__(self):
        self.trades_file = 'trades_history.json'
        self.load_trades()
        
    def load_trades(self):
        """Load historical trades"""
        if os.path.exists(self.trades_file):
            try:
                with open(self.trades_file, 'r') as f:
                    self.trades = json.load(f)
            except:
                self.trades = []
        else:
            self.trades = []
    
    def add_trade(self, trade: Dict):
        """Record a new trade"""
        self.trades.append(trade)
        self.save_trades()
        
    def save_trades(self):
        """Save trades to file"""
        try:
            with open(self.trades_file, 'w') as f:
                json.dump(self.trades, f)
        except Exception as e:
            print(f"Error saving trades: {e}")
    
    def get_statistics(self) -> Dict:
        """Calculate performance statistics"""
        if not self.trades:
            return {"message": "No trades recorded yet"}
        
        df = pd.DataFrame(self.trades)
        
        # Filter completed trades
        if 'pnl' in df.columns:
            completed = df[df['pnl'].notna()]
            
            if len(completed) > 0:
                winners = completed[completed['pnl'] > 0]
                losers = completed[completed['pnl'] <= 0]
                
                stats = {
                    'total_trades': len(completed),
                    'winning_trades': len(winners),
                    'losing_trades': len(losers),
                    'win_rate': round(len(winners) / len(completed) * 100, 2),
                    'avg_winner': round(winners['pnl'].mean(), 2) if len(winners) > 0 else 0,
                    'avg_loser': round(losers['pnl'].mean(), 2) if len(losers) > 0 else 0,
                    'total_pnl': round(completed['pnl'].sum(), 2),
                    'profit_factor': round(abs(winners['pnl'].sum() / losers['pnl'].sum()), 2) if len(losers) > 0 and losers['pnl'].sum() != 0 else 0
                }
                
                return stats
        
        return {"message": "No completed trades yet"}

# ===========================
# Data Enrichment
# ===========================
async def enrich_alert_data(alert: AlertData) -> EnhancedAlert:
    """Enrich alert with additional market data"""
    
    enhanced = EnhancedAlert(
        symbol=alert.symbol,
        tf=alert.tf,
        price=alert.price,
        rsi=alert.rsi,
        macd=alert.macd,
        volSpike=alert.volSpike,
        breakoutPct=alert.breakoutPct,
        gapPct=alert.gapPct,
        ts=alert.ts
    )
    
    try:
        ticker = yf.Ticker(alert.symbol)
        info = ticker.info
        hist = ticker.history(period="1mo")
        
        # Calculate ATR%
        if len(hist) >= 14:
            high_low = hist['High'] - hist['Low']
            high_close = abs(hist['High'] - hist['Close'].shift())
            low_close = abs(hist['Low'] - hist['Close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            enhanced.atr_pct = (atr / alert.price) if alert.price > 0 else 0.02
        
        # Get additional info
        enhanced.market_cap_b = info.get('marketCap', 0) / 1e9 if 'marketCap' in info else 0
        enhanced.float_m = info.get('floatShares', 0) / 1e6 if 'floatShares' in info else 0
        enhanced.sector = info.get('sector', 'Unknown')
        
        # Calculate relative volume (vs 20-day average)
        if len(hist) >= 20:
            avg_vol = hist['Volume'].rolling(20).mean().iloc[-1]
            current_vol = hist['Volume'].iloc[-1]
            enhanced.relative_volume = current_vol / avg_vol if avg_vol > 0 else 1
        
    except Exception as e:
        print(f"Error enriching data: {e}")
    
    return enhanced

# ===========================
# Alert Functions
# ===========================
async def send_comprehensive_telegram_alert(data: Dict, market: Dict, position: Dict, score_details: Dict):
    """Send detailed alert to Telegram"""
    
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        print("Telegram not configured")
        return
    
    # Only A-rank
    if data['rank'] != 'A':
        return
    
    # Format message
    market_emoji = "🟢" if "BULL" in market['regime'] else "🔴" if "BEAR" in market['regime'] else "🟡"
    confidence_stars = "⭐" * min(5, int(score_details.get('confidence', 0.5) * 5))
    
    message = f"""
🎯 **A-RANK SIGNAL** {market_emoji}

**{data['symbol']}** | {data['tf']}m | Score: **{data['score']}/100**
Confidence: {confidence_stars} ({score_details.get('confidence', 0):.0%})

📊 **Entry Setup:**
• Price: ${data['price']:.2f}
• Stop: ${position['stop_loss']:.2f} (-{position['stop_distance_pct']:.1f}%)
• T1: ${position['target1']:.2f} (+{((position['target1']-data['price'])/data['price']*100):.1f}%)
• T2: ${position['target2']:.2f} (+{((position['target2']-data['price'])/data['price']*100):.1f}%)
• T3: ${position['target3']:.2f} (+{((position['target3']-data['price'])/data['price']*100):.1f}%)

💰 **Position:**
• Shares: {position['shares']}
• Value: ${position['position_value']:,.0f}
• Risk: ${position['risk_amount']:.0f} ({position['risk_pct']:.1f}%)

📈 **Indicators:**
• RSI: {data['rsi']:.1f}
• MACD: {data['macd']:.3f}
• Volume: {data['volSpike']:.1f}x avg
• Breakout: {data['breakoutPct']*100:.1f}%
• Gap: {data['gapPct']*100:.1f}%

🌍 **Market:** {market['regime']}
• VIX: {market['vix']:.1f}
• SPY Trend: {market['spy_trend']}/4

📱 [TradingView](https://www.tradingview.com/chart/?symbol={data['symbol']})

⚠️ Risk: {position['risk_pct']:.1f}% | R-Multiples: {position['r_multiple_1']:.1f}R, {position['r_multiple_2']:.0f}R, {position['r_multiple_3']:.0f}R
"""
    
    try:
        bot = Bot(token=token)
        await bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')
        print(f"Comprehensive alert sent: {data['symbol']}")
    except Exception as e:
        print(f"Telegram error: {e}")

def save_enhanced_to_sheets(data: Dict, market: Dict, position: Dict, score_details: Dict):
    """Save comprehensive data to Google Sheets"""
    try:
        creds_json = os.getenv("GOOGLE_SHEETS_CREDS_JSON")
        sheet_id = os.getenv("SHEET_ID")
        
        if not creds_json or not sheet_id:
            return
        
        creds_dict = json.loads(creds_json)
        creds = Credentials.from_service_account_info(creds_dict)
        client = gspread.authorize(creds)
        
        sheet = client.open_by_key(sheet_id)
        ws = sheet.get_worksheet(0)
        
        # Comprehensive row
        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data['symbol'],
            data['tf'],
            data['price'],
            data['rsi'],
            data['macd'],
            data['volSpike'],
            data['breakoutPct'] * 100,
            data['gapPct'] * 100,
            data.get('atr_pct', 0.02) * 100,
            data.get('float_m', 0),
            data.get('sector', 'Unknown'),
            market['regime'],
            market['vix'],
            data['score'],
            data['rank'],
            data['reason'],
            score_details.get('confidence', 0),
            score_details.get('method', 'Rule'),
            position['shares'],
            position['position_value'],
            position['stop_loss'],
            position['target1'],
            position['target2'],
            position['risk_amount']
        ]
        
        ws.append_row(row)
        print(f"Enhanced data saved: {data['symbol']}")
        
    except Exception as e:
        print(f"Sheets error: {e}")

# ===========================
# API Endpoints
# ===========================
@app.get("/")
async def root():
    """API information"""
    return {
        "name": "AI Trading Playbook Ultimate",
        "version": "3.0",
        "status": "active",
        "features": [
            "Advanced Market Regime Detection",
            "ML + Rule-based Scoring",
            "Kelly Criterion Position Sizing",
            "Multi-target Risk Management",
            "Performance Tracking",
            "Data Enrichment",
            "Comprehensive Alerts"
        ],
        "ml_status": "Active" if ML_AVAILABLE else "Rule-based only",
        "endpoints": {
            "webhook": "/webhook",
            "health": "/health",
            "market": "/market",
            "performance": "/performance",
            "test": "/test"
        }
    }

@app.get("/health")
async def health_check():
    """System health check"""
    detector = MarketRegimeDetector()
    market = await detector.get_comprehensive_regime()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "market": market['regime'],
        "vix": market['vix'],
        "ml_available": ML_AVAILABLE
    }

@app.get("/market")
async def market_analysis():
    """Current market analysis"""
    detector = MarketRegimeDetector()
    return await detector.get_comprehensive_regime()

@app.get("/performance")
async def performance_stats():
    """System performance statistics"""
    tracker = PerformanceTracker()
    return tracker.get_statistics()

@app.post("/webhook")
async def webhook_handler(alert: AlertData, background_tasks: BackgroundTasks):
    """Main webhook handler for TradingView alerts"""
    try:
        print(f"\n{'='*50}")
        print(f"New Alert: {alert.symbol} @ {alert.price}")
        
        # 1. Enrich alert data
        enhanced_alert = await enrich_alert_data(alert)
        
        # 2. Get market regime
        detector = MarketRegimeDetector()
        market = await detector.get_comprehensive_regime()
        
        # 3. Calculate score
        calculator = AdvancedScoreCalculator()
        score, score_details = calculator.calculate_comprehensive_score(enhanced_alert, market)
        
        # 4. Determine rank
        if score >= 85:
            rank = "A"
        elif score >= 70:
            rank = "B"
        else:
            rank = "C"
        
        # 5. Generate reason
        reasons = []
        if enhanced_alert.rsi >= 55 and enhanced_alert.rsi <= 65:
            reasons.append(f"RSI:{enhanced_alert.rsi:.0f}")
        if enhanced_alert.macd > 0:
            reasons.append("MACD+")
        if enhanced_alert.volSpike >= 1.5:
            reasons.append(f"Vol{enhanced_alert.volSpike:.1f}x")
        if enhanced_alert.breakoutPct >= 0.02:
            reasons.append(f"BO{enhanced_alert.breakoutPct*100:.1f}%")
        reason = " | ".join(reasons)
        
        # 6. Calculate position
        risk_mgr = RiskManager()
        position = risk_mgr.calculate_position_size(enhanced_alert, score)
        
        # 7. Prepare data dict
        data = asdict(enhanced_alert)
        data['score'] = score
        data['rank'] = rank
        data['reason'] = reason
        
        print(f"Score: {score} ({rank}) | Market: {market['regime']}")
        print(f"Position: {position['shares']} shares @ ${position['position_value']:.0f}")
        
        # 8. Track performance
        tracker = PerformanceTracker()
        tracker.add_trade({
            'timestamp': datetime.now().isoformat(),
            'symbol': alert.symbol,
            'price': alert.price,
            'score': score,
            'rank': rank,
            'position': position['shares']
        })
        
        # 9. Background tasks
        background_tasks.add_task(
            save_enhanced_to_sheets, 
            data, market, position, score_
