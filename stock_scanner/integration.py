# stock_scanner/integration.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from train import engineer_features
from joblib import load
import pandas as pd
import numpy as np

class IntegratedModel:
    """
    משתמש במודלים שאתה מאמן ב-train.py
    """
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_models()
    
    def load_models(self):
        """טוען את המודלים שנשמרו"""
        try:
            # נסה לטעון את המודל הטוב ביותר
            if os.path.exists('model.joblib'):
                self.model = load('model.joblib')
                print("✅ Loaded main model")
            elif os.path.exists('model_ensemble.joblib'):
                self.model = load('model_ensemble.joblib')
                print("✅ Loaded ensemble model")
            else:
                print("⚠️ No trained model found. Please run train.py first.")
                return False
            
            # טען scaler
            if os.path.exists('scaler.joblib'):
                self.scaler = load('scaler.joblib')
                print("✅ Loaded scaler")
            
            return True
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def is_loaded(self):
        return self.model is not None
    
    def prepare_features(self, stock_data):
        """
        ממיר נתוני מניה לפורמט שהמודל שלך מצפה לו
        """
        # המר את נתוני yfinance לפורמט שלך
        df = pd.DataFrame(stock_data)
        
        # חשב אינדיקטורים טכניים
        features_dict = self._calculate_technical_indicators(df)
        
        # השתמש באותה פונקציית feature engineering מ-train.py
        features_df = pd.DataFrame([features_dict])
        engineered = engineer_features(features_df)
        
        return engineered
    
    def _calculate_technical_indicators(self, df):
        """מחשב RSI, MACD, וכו' מנתוני מניה"""
        import ta
        
        df['close'] = pd.to_numeric(df['close'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        # RSI
        rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi().iloc[-1]
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        macd_value = macd.macd_diff().iloc[-1]
        
        # Volume Spike
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        vol_spike = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Breakout %
        high_20 = df['high'].rolling(20).max().iloc[-1]
        current_price = df['close'].iloc[-1]
        breakout_pct = ((current_price - high_20) / high_20 * 100) if high_20 > 0 else 0
        
        # Gap %
        if len(df) > 1:
            gap_pct = ((df['open'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100)
        else:
            gap_pct = 0
        
        # A_Score (ציון משוקלל)
        score = (
            (rsi / 100 * 30) +
            (min(vol_spike / 3, 1) * 30) +
            (1 if macd_value > 0 else 0) * 20 +
            (1 if breakout_pct > 0 else 0) * 20
        )
        
        return {
            'RSI': rsi,
            'MACD': macd_value,
            'VolSpike': vol_spike,
            'Breakout%': breakout_pct,
            'Gap%': gap_pct,
            'A_Score': score
        }
    
    def predict(self, stock_data):
        """
        חיזוי אם תהיה פריצה
        מחזיר: (probability, prediction_label)
        """
        if not self.is_loaded():
            raise ValueError("Model not loaded. Run train.py first.")
        
        # הכן features
        X = self.prepare_features(stock_data)
        
        # Scale אם יש scaler
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # חזה
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X_scaled)[0]
            return float(proba[1]), 'Win' if proba[1] > 0.5 else 'Loss'
        else:
            pred = self.model.predict(X_scaled)[0]
            return float(pred), 'Win' if pred == 1 else 'Loss'
