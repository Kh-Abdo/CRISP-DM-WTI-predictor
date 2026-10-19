
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib, json, math
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="WTI Price Predictor", page_icon="🛢️", layout="wide")

MODELS_DIR = "models"
SELECTABLE = ["LSTM", "XGBoost"]
COLORS     = {"XGBoost": "steelblue", "LSTM": "darkorange", "SVR": "green", "ARIMAX": "firebrick"}

# ── Artefacts ────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    art = {}
    with open(f"{MODELS_DIR}/feature_names.json") as f:
        art["features"] = json.load(f)
    with open(f"{MODELS_DIR}/eval_metrics.json") as f:
        art["eval"] = json.load(f)
    art["xgb"]        = joblib.load(f"{MODELS_DIR}/xgboost_model.pkl")
    art["xgb_scaler"] = joblib.load(f"{MODELS_DIR}/feature_scaler.pkl")
    art["lstm"], art["lstm_error"] = None, None
    try:
        from tensorflow.keras.models import load_model
        art["lstm"]          = load_model(f"{MODELS_DIR}/lstm_model.keras", compile=False)
        art["lstm_scaler_X"] = joblib.load(f"{MODELS_DIR}/lstm_scaler_X.pkl")
        art["lstm_scaler_y"] = joblib.load(f"{MODELS_DIR}/lstm_scaler_y.pkl")
    except Exception as e:
        art["lstm"], art["lstm_error"] = None, str(e)
    return art

art        = load_artifacts()
FEATURES   = art["features"]
EVAL       = art["eval"]
LF         = EVAL.get("last_fold", {})
TIME_STEPS = int(EVAL.get("time_steps", 30))

def is_num(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
available = [m for m in SELECTABLE if m == "XGBoost" or art["lstm"] is not None]
if art["lstm"] is None:
    st.sidebar.warning("LSTM could not be loaded - XGBoost only. " + str(art["lstm_error"])[:120])
default_winner = EVAL.get("selected_winner", "LSTM")
default_idx    = available.index(default_winner) if default_winner in available else 0

MODEL_NAME = st.sidebar.radio("Primary model", available, index=default_idx,
                              help="Drives the KPIs, metrics and scenario tool. Default = winner chosen in Phase 5.")
OTHER_NAME = next((m for m in available if m != MODEL_NAME), None)

lookback_days = st.sidebar.slider("Lookback period (days for charts)", 30, 365, 90)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model**: {MODEL_NAME}" + (" (Phase 5 winner)" if MODEL_NAME == default_winner else ""))
st.sidebar.markdown("**Target**: Next-day WTI log return")
st.sidebar.markdown(f"**Features**: {len(FEATURES)} technical + macro")
if MODEL_NAME == "LSTM":
    st.sidebar.markdown(f"**Sequence length**: {TIME_STEPS} days")
tw = EVAL.get("test_window")
if tw:
    st.sidebar.caption(f"Backtest window: {tw['start']} to {tw['end']} ({tw['n']} days)")

st.title("🛢️ WTI Crude Oil — Next-Day Price Predictor")
st.caption(f"CRISP-DM Project | Primary model: {MODEL_NAME} | Macro drivers + technicals")
st.markdown("---")

# ── Live data ────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_data():
    tickers = ["CL=F", "DX-Y.NYB", "^GSPC", "GC=F", "^VIX"]
    raw = yf.download(tickers, period="2y", interval="1d", auto_adjust=True)
    df = pd.DataFrame()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[f"WTI_{col}"] = raw[col]["CL=F"]
    df["DXY_Close"]   = raw["Close"]["DX-Y.NYB"]
    df["SP500_Close"] = raw["Close"]["^GSPC"]
    df["Gold_Close"]  = raw["Close"]["GC=F"]
    df["VIX_Close"]   = raw["Close"]["^VIX"]
    df.replace([float("inf"), float("-inf")], float("nan"), inplace=True)
    return df.ffill().dropna()

with st.spinner("Fetching live market data..."):
    df = fetch_data()

# ── Feature engineering ──────────────────────────────────────
def engineer_features(df):
    d = df.copy()
    exp1 = d["WTI_Close"].ewm(span=12, adjust=False).mean()
    exp2 = d["WTI_Close"].ewm(span=26, adjust=False).mean()
    d["MACD_Hist"] = (exp1 - exp2) - (exp1 - exp2).ewm(span=9, adjust=False).mean()
    hl = d["WTI_High"] - d["WTI_Low"]
    hc = abs(d["WTI_High"] - d["WTI_Close"].shift())
    lc = abs(d["WTI_Low"]  - d["WTI_Close"].shift())
    d["ATR_14"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    rh = d["WTI_Close"].rolling(20).max()
    rl = d["WTI_Close"].rolling(20).min()
    d["Dist_Fib_50"] = (d["WTI_Close"] - (rh - (rh - rl) * 0.5)) / d["WTI_Close"]
    d["SMA_10"] = d["WTI_Close"].rolling(10).mean()
    d["SMA_50"] = d["WTI_Close"].rolling(50).mean()
    d["Volatility_20"] = d["WTI_Close"].rolling(20).std()
    delta = d["WTI_Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    d["RSI"] = 100 - (100 / (1 + gain / loss))
    sma20 = d["WTI_Close"].rolling(20).mean()
    std20 = d["WTI_Close"].rolling(20).std()
    d["BB_Width"] = (2 * std20) / sma20
    d["WTI_Gold_Ratio"] = d["WTI_Close"] / d["Gold_Close"].replace(0, float("nan"))
    d["WTI_Ret_Lag1"] = d["WTI_Close"].pct_change().shift(1)
    d["WTI_Ret_Lag2"] = d["WTI_Close"].pct_change().shift(2)
    for col in ["WTI_Close", "DXY_Close", "SP500_Close", "Gold_Close", "VIX_Close"]:
        d[f"{col}_Ret"] = (d[col] / d[col].shift(1)).apply(
            lambda x: float("nan") if x <= 0 else math.log(x))
    return d.dropna()

df_fe     = engineer_features(df)
feat_hist = df_fe[FEATURES]          # raw (unscaled) feature history, most recent row last

# ── Model-aware prediction ───────────────────────────────────
def predict_next_return(model_name, hist, override=None):
    hist = hist.copy()
    if override is not None:                       # scenario: overwrite the latest row
        hist.iloc[-1, :] = [override[f] for f in FEATURES]
    if model_name == "XGBoost":
        x = pd.DataFrame(art["xgb_scaler"].transform(hist.tail(1)), columns=FEATURES)
        return float(art["xgb"].predict(x)[0])
    seq    = hist.tail(TIME_STEPS).values
    seq_sc = art["lstm_scaler_X"].transform(seq).reshape(1, TIME_STEPS, len(FEATURES))
    p      = art["lstm"].predict(seq_sc, verbose=0).flatten()
    return float(art["lstm_scaler_y"].inverse_transform(p.reshape(-1, 1))[0, 0])

def direction_label(r):
    return "📈 UP" if r > 0 else "📉 DOWN"

pred_return  = predict_next_return(MODEL_NAME, feat_hist)
last_price   = float(df["WTI_Close"].iloc[-1])
target_price = last_price * math.exp(pred_return)

pred_other = None
if OTHER_NAME is not None:
    try:
        pred_other = predict_next_return(OTHER_NAME, feat_hist)
    except Exception:
        pred_other = None

# ── KPIs (selected model) ────────────────────────────────────
st.subheader(f"🎯 {MODEL_NAME} — Next-Day Forecast")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current WTI Price",     f"${last_price:.2f}")
c2.metric("Next-Day Direction",    direction_label(pred_return))
c3.metric("Predicted Log Return",  f"{pred_return * 100:.4f}%")
c4.metric("Estimated Target Price", f"${target_price:.2f}")

# ── Backtest metrics (selected model, delta vs the other) ────
st.subheader(f"🧪 {MODEL_NAME} — Backtest Metrics (last walk-forward fold)")
SPECS = [  # key, label, value formatter, delta formatter, lower-is-better
    ("mae",    "MAE",         lambda v: f"{v:.6f}",   lambda d: f"{d:+.6f}",        True),
    ("hit",    "Hit Ratio",   lambda v: f"{v:.2%}",   lambda d: f"{d * 100:+.2f} pts", False),
    ("sharpe", "Sharpe Ratio", lambda v: f"{v:.2f}",  lambda d: f"{d:+.2f}",        False),
    ("r2",     "R²",          lambda v: f"{v:.4f}",   lambda d: f"{d:+.4f}",        False),
]
mcols = st.columns(4)
for col, (key, label, fv, fd, lower_better) in zip(mcols, SPECS):
    v = LF.get(MODEL_NAME, {}).get(key)
    o = LF.get(OTHER_NAME, {}).get(key) if OTHER_NAME else None
    if not is_num(v):
        col.metric(label, "n/a")
        continue
    if is_num(o):
        col.metric(label, fv(v), delta=f"{fd(v - o)} vs {OTHER_NAME}",
                   delta_color="inverse" if lower_better else "normal")
    else:
        col.metric(label, fv(v))

# ── Other model + agreement ──────────────────────────────────
if pred_other is not None:
    agree = (pred_return > 0) == (pred_other > 0)
    msg = (f"{OTHER_NAME} says {direction_label(pred_other)} "
           f"({pred_other * 100:.4f}%) - " + ("models **agree** on direction." if agree
                                              else "models **disagree** on direction."))
    (st.info if agree else st.warning)(msg)

with st.expander("🆚 Model comparison table (all models, last fold)", expanded=False):
    rows = {n: LF[n] for n in ["XGBoost", "LSTM", "SVR", "ARIMAX"] if n in LF}
    tbl = pd.DataFrame(rows).T.rename(columns={"mae": "MAE", "rmse": "RMSE", "r2": "R²",
                                                "hit": "Hit Ratio", "sharpe": "Sharpe",
                                                "max_dd": "Max DD"})
    tbl.index.name = "Model"
    def hl(row):
        on = "background-color: #fff3cd; color: #000000" if row.name == MODEL_NAME else ""
        return [on] * len(row)
    st.dataframe(tbl.style.apply(hl, axis=1).format("{:.4f}", na_rep="n/a"))
    fig, ax = plt.subplots(figsize=(6, 2.6))
    bars = ax.bar(tbl.index, tbl["MAE"].astype(float), color=[COLORS[m] for m in tbl.index])
    for b, m in zip(bars, tbl.index):
        b.set_alpha(1.0 if m == MODEL_NAME else 0.35)
    ax.set_ylabel("MAE (last fold)"); ax.grid(True, alpha=0.3, axis="y")
    st.pyplot(fig); plt.close(fig)
st.markdown("---")

# ── Charts ────────────────────────────────────────────────────
cl, cr = st.columns(2)
with cl:
    st.subheader("WTI Closing Price")
    fig, ax = plt.subplots(figsize=(7, 3))
    df["WTI_Close"].tail(lookback_days).plot(ax=ax, color="black", linewidth=1.2)
    ax.set_ylabel("USD / Barrel"); ax.grid(True, alpha=0.3)
    st.pyplot(fig); plt.close(fig)
with cr:
    st.subheader("RSI-14")
    fig, ax = plt.subplots(figsize=(7, 3))
    df_fe["RSI"].tail(lookback_days).plot(ax=ax, color="purple", linewidth=1.2)
    ax.axhline(70, color="red",   linestyle="--", alpha=0.5, label="Overbought")
    ax.axhline(30, color="green", linestyle="--", alpha=0.5, label="Oversold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    st.pyplot(fig); plt.close(fig)

# ── Manual Scenario (selected model) ─────────────────────────
st.markdown("---")
st.subheader(f"🔧 Manual Scenario Override — {MODEL_NAME}")
with st.expander("Expand to tweak the latest feature values"):
    if MODEL_NAME == "LSTM":
        st.caption(f"Overrides the most recent day of the {TIME_STEPS}-day input sequence.")
    cols = st.columns(4)
    defaults = feat_hist.iloc[-1].to_dict()
    manual = {}
    for i, feat in enumerate(FEATURES):
        manual[feat] = cols[i % 4].number_input(feat, value=float(defaults[feat]),
                                                format="%.6f", key=f"scn_{feat}")
    if st.button("Run Scenario Prediction"):
        m_ret = predict_next_return(MODEL_NAME, feat_hist, override=manual)
        st.success(f"Scenario ({MODEL_NAME}): **{direction_label(m_ret)}** | "
                   f"Log Return: `{m_ret * 100:.4f}%`")

# ── Macro Table ───────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Live Macro Snapshot")
names = ["WTI Crude", "Gold", "S&P 500", "DXY", "VIX"]
cols_ = ["WTI_Close", "Gold_Close", "SP500_Close", "DXY_Close", "VIX_Close"]
snap = pd.DataFrame({
    "Asset":       names,
    "Last Price":  [df[c].iloc[-1] for c in cols_],
    "1-Day Chg %": [df[c].pct_change().iloc[-1] * 100 for c in cols_],
})
snap["Last Price"]  = snap["Last Price"].map("${:.2f}".format)
snap["1-Day Chg %"] = snap["1-Day Chg %"].map("{:.2f}%".format)
st.dataframe(snap, hide_index=True)

st.markdown("---")
st.caption("⚠️ For research purposes only. Not financial advice.")
