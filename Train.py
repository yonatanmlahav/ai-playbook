# train.py - Enhanced ML Training Pipeline v3.0
# ============================================
# כולל: Ensemble models, Feature engineering, Backtesting, Hyperparameter tuning

import os
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif
from joblib import dump, load
import json

# Try importing advanced ML libraries
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("⚠️ LightGBM not installed - using XGBoost and RandomForest only")
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    print("⚠️ CatBoost not installed - using other models")
    CATBOOST_AVAILABLE = False

# ========================================
# CONFIGURATION
# ========================================
SHEET_ID = os.getenv("SPREADSHEET_ID")
SA_PATH = os.getenv("GCP_SA_JSON_PATH", "/app/sa.json")
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.file"
]

# Training parameters
MIN_SAMPLES = 30  # Minimum samples needed for training
TEST_SIZE = 0.2   # 80/20 train/test split
CV_FOLDS = 5      # Cross-validation folds
RANDOM_STATE = 42

# ========================================
# CONNECT TO SHEETS
# ========================================
def connect_to_sheets():
    """Connect to Google Sheets"""
    try:
        creds = Credentials.from_service_account_file(SA_PATH, scopes=SCOPES)
        gc = gspread.authorize(creds)
        if SHEET_ID:
            sh = gc.open_by_key(SHEET_ID)
        else:
            sh = gc.open(os.getenv("SHEET_NAME", "AI_Playbook"))
        return sh
    except Exception as e:
        print(f"❌ Failed to connect to sheets: {e}")
        raise

# ========================================
# FEATURE ENGINEERING
# ========================================
class FeatureEngineer:
    """Advanced feature engineering pipeline"""
    
    def __init__(self):
        self.feature_names = []
        self.scaler = StandardScaler()
        
    def create_features(self, df):
        """Create advanced features from raw data"""
        print("🔧 Engineering features...")
        
        # Original features
        original_features = ["RSI", "MACD", "VolSpike", "Breakout%", "Gap%", "A_Score"]
        
        # Clean original features
        for col in original_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        
        # New engineered features
        features_df = pd.DataFrame()
        
        # 1. Basic features
        features_df['rsi'] = df['RSI']
        features_df['macd'] = df['MACD']
        features_df['vol_spike'] = df['VolSpike']
        features_df['breakout_pct'] = df['Breakout%']
        features_df['gap_pct'] = df['Gap%']
        features_df['a_score'] = df['A_Score']
        
        # 2. RSI features
        features_df['rsi_oversold'] = (df['RSI'] < 30).astype(int)
        features_df['rsi_overbought'] = (df['RSI'] > 70).astype(int)
        features_df['rsi_optimal'] = ((df['RSI'] >= 55) & (df['RSI'] <= 65)).astype(int)
        features_df['rsi_distance_60'] = abs(df['RSI'] - 60)
        
        # 3. MACD features
        features_df['macd_positive'] = (df['MACD'] > 0).astype(int)
        features_df['macd_strength'] = abs(df['MACD'])
        
        # 4. Volume features
        features_df['vol_extreme'] = (df['VolSpike'] > 3).astype(int)
        features_df['vol_moderate'] = ((df['VolSpike'] >= 1.5) & (df['VolSpike'] <= 3)).astype(int)
        features_df['vol_log'] = np.log1p(df['VolSpike'])
        
        # 5. Breakout features
        features_df['breakout_strong'] = (df['Breakout%'] > 5).astype(int)
        features_df['breakout_moderate'] = ((df['Breakout%'] >= 2) & (df['Breakout%'] <= 5)).astype(int)
        
        # 6. Gap features
        features_df['gap_small'] = (abs(df['Gap%']) < 2).astype(int)
        features_df['gap_abs'] = abs(df['Gap%'])
        
        # 7. Interaction features
        features_df['rsi_macd_interaction'] = df['RSI'] * df['MACD']
        features_df['vol_breakout_interaction'] = df['VolSpike'] * df['Breakout%']
        features_df['momentum_score'] = (df['RSI'] - 50) * df['MACD']
        
        # 8. Score features
        features_df['high_score'] = (df['A_Score'] >= 85).astype(int)
        features_df['medium_score'] = ((df['A_Score'] >= 70) & (df['A_Score'] < 85)).astype(int)
        
        # 9. Market features (if available)
        if 'MarketCond' in df.columns:
            features_df['market_bull'] = (df['MarketCond'] == 'Bull').astype(int)
            features_df['market_bear'] =
