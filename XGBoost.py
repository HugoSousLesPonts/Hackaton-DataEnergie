#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════╗
║  Prévision de consommation électrique — XGBoost (30 min)           ║
║  Équipe 4 — Horizon 14 jours (672 pas de 30 min)                  ║
╚══════════════════════════════════════════════════════════════════════╝

Pipeline complet :
  1. Chargement & nettoyage (UTC, dédoublonnage)
  2. Récupération météo exogène via Open-Meteo (Paris/Lyon/Marseille/Lille)
  3. Feature engineering avancé (cyclique, calendaire, lags sécurisés, rolling)
  4. Entraînement XGBRegressor avec early stopping sur validation
  5. Évaluation (MAE, RMSE, MAPE, MASE)
  6. Export CSV de soumission
"""

# ──────────────────────────────────────────────────────────────────────
# 0. IMPORTS
# ──────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import openmeteo_requests
import requests_cache
from retry_requests import retry

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ──────────────────────────────────────────────────────────────────────
# 1. CONSTANTES
# ──────────────────────────────────────────────────────────────────────
HORIZON        = 672          # 14 jours × 48 demi-heures
FREQ           = "30min"      # Pas temporel
INPUT_CSV      = "datas/RES1-3-6.csv"
OUTPUT_CSV     = "equipe_Equipe4_XGBoost_predictions.csv"
SEED           = 42

# Villes de référence pour la météo (lat, lon, poids de pondération)
CITIES = {
    "Paris":     (48.8566,  2.3522,  0.35),
    "Lyon":      (45.7640,  4.8357,  0.25),
    "Marseille": (43.2965,  5.3698,  0.20),
    "Lille":     (50.6292,  3.0573,  0.20),
}

# Jours fériés français (à compléter selon la plage du dataset)
JOURS_FERIES = [
    # ── 2023 ──
    "2023-01-01", "2023-04-10", "2023-05-01", "2023-05-08",
    "2023-05-18", "2023-05-29", "2023-07-14", "2023-08-15",
    "2023-11-01", "2023-11-11", "2023-12-25",
    # ── 2024 ──
    "2024-01-01", "2024-04-01", "2024-05-01", "2024-05-08",
    "2024-05-09", "2024-05-20", "2024-07-14", "2024-08-15",
    "2024-11-01", "2024-11-11", "2024-12-25",
    # ── 2025 ──
    "2025-01-01", "2025-04-21", "2025-05-01", "2025-05-08",
    "2025-05-29", "2025-06-09", "2025-07-14", "2025-08-15",
    "2025-11-01", "2025-11-11", "2025-12-25",
    # ── 2026 ──
    "2026-01-01", "2026-04-06", "2026-05-01", "2026-05-08",
    "2026-05-14", "2026-05-25", "2026-07-14", "2026-08-15",
    "2026-11-01", "2026-11-11", "2026-12-25",
]
JOURS_FERIES = set(pd.to_datetime(JOURS_FERIES).date)

print("=" * 70)
print("  PIPELINE DE PRÉVISION — XGBoost (30 min, H=14j)")
print("=" * 70)

# ──────────────────────────────────────────────────────────────────────
# 2. CHARGEMENT & NETTOYAGE
# ──────────────────────────────────────────────────────────────────────
print("\n[1/6] Chargement des données…")

df = pd.read_csv(INPUT_CSV, sep=";", names=["id", "horodate", "valeur"], skiprows=1)
df["horodate"] = pd.to_datetime(df["horodate"], utc=True)       # Conversion UTC
df["horodate"] = df["horodate"].dt.tz_localize(None)             # Supprime tz pour XGBoost
df = df.rename(columns={"horodate": "datetime", "valeur": "load"})
df = df.drop(columns=["id"])

# Dédoublonnage par moyenne
df = df.groupby("datetime", as_index=False)["load"].mean()
df = df.sort_values("datetime").reset_index(drop=True)
df = df.set_index("datetime")

# Réindexation complète au pas de 30 min pour combler les trous
full_idx = pd.date_range(df.index.min(), df.index.max(), freq=FREQ)
df = df.reindex(full_idx)
df.index.name = "datetime"

# Interpolation linéaire des valeurs manquantes résiduelles
n_miss = df["load"].isna().sum()
if n_miss > 0:
    print(f"   ⚠ {n_miss} valeurs manquantes → interpolation linéaire")
    df["load"] = df["load"].interpolate(method="linear")

print(f"   ✔ Plage : {df.index.min()} → {df.index.max()}")
print(f"   ✔ {len(df)} pas de temps, {df['load'].isna().sum()} NaN restants")

# ──────────────────────────────────────────────────────────────────────
# 3. DONNÉES MÉTÉO EXOGÈNES (Open-Meteo — historique)
# ──────────────────────────────────────────────────────────────────────
print("\n[2/6] Récupération météo Open-Meteo…")

cache_session = requests_cache.CachedSession(".cache_openmeteo", expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
om = openmeteo_requests.Client(session=retry_session)

date_start = df.index.min().strftime("%Y-%m-%d")
date_end   = df.index.max().strftime("%Y-%m-%d")

meteo_frames = []
for city, (lat, lon, weight) in CITIES.items():
    print(f"   → {city} (poids={weight})")
    params = {
        "latitude":  lat,
        "longitude": lon,
        "start_date": date_start,
        "end_date":   date_end,
        "hourly": [
            "temperature_2m",
            "direct_radiation",
            "windspeed_10m",
        ],
        "timezone": "UTC",
    }
    responses = om.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)
    r = responses[0]
    hourly = r.Hourly()

    meteo_city = pd.DataFrame({
        "datetime":     pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True).tz_localize(None),
            periods=hourly.VariablesCount()  # fallback
                     if False else len(hourly.Variables(0).ValuesAsNumpy()),
            freq="h",
        ),
        "temperature":  hourly.Variables(0).ValuesAsNumpy() * weight,
        "radiation":    hourly.Variables(1).ValuesAsNumpy() * weight,
        "wind":         hourly.Variables(2).ValuesAsNumpy() * weight,
    })
    meteo_frames.append(meteo_city)

# Agrégation pondérée
meteo = meteo_frames[0].copy()
for mf in meteo_frames[1:]:
    meteo["temperature"] += mf["temperature"].values
    meteo["radiation"]   += mf["radiation"].values
    meteo["wind"]        += mf["wind"].values

meteo = meteo.set_index("datetime")

# Ré-échantillonnage de 1 h → 30 min (interpolation linéaire)
meteo = meteo.resample(FREQ).interpolate(method="linear")

# Jointure
df = df.join(meteo, how="left")
for col in ["temperature", "radiation", "wind"]:
    df[col] = df[col].interpolate(method="linear").bfill().ffill()

print(f"   ✔ Météo intégrée ({len(meteo)} lignes source)")

# ──────────────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────
print("\n[3/6] Feature engineering…")

dt = df.index

# ── 4a. Encodage cyclique ──
df["hour_sin"]  = np.sin(2 * np.pi * dt.hour / 24)
df["hour_cos"]  = np.cos(2 * np.pi * dt.hour / 24)
df["minute_sin"] = np.sin(2 * np.pi * dt.minute / 60)
df["minute_cos"] = np.cos(2 * np.pi * dt.minute / 60)
df["dow_sin"]   = np.sin(2 * np.pi * dt.dayofweek / 7)
df["dow_cos"]   = np.cos(2 * np.pi * dt.dayofweek / 7)
df["month_sin"] = np.sin(2 * np.pi * dt.month / 12)
df["month_cos"] = np.cos(2 * np.pi * dt.month / 12)

# ── 4b. Features calendaires ──
df["is_weekend"] = (dt.dayofweek >= 5).astype(int)
df["is_ferie"]   = pd.Series(dt.date, index=dt).isin(JOURS_FERIES).astype(int)

# Indicateur veille / lendemain de férié
ferie_dates = sorted(JOURS_FERIES)
veille_ferie = {d - pd.Timedelta(days=1) for d in pd.to_datetime(list(JOURS_FERIES))}
lendemain_ferie = {d + pd.Timedelta(days=1) for d in pd.to_datetime(list(JOURS_FERIES))}
df["is_veille_ferie"]     = pd.Series(dt.date, index=dt).apply(
    lambda d: 1 if pd.Timestamp(d) in veille_ferie else 0
)
df["is_lendemain_ferie"]  = pd.Series(dt.date, index=dt).apply(
    lambda d: 1 if pd.Timestamp(d) in lendemain_ferie else 0
)

# ── 4c. Lags SÉCURISÉS (>= HORIZON = 672) ──
# 672 = 14j, 720 = 15j, 1008 = 21j, 1344 = 28j, 2016 = 42j
SAFE_LAGS = [672, 720, 1008, 1344, 2016]
for lag in SAFE_LAGS:
    df[f"lag_{lag}"] = df["load"].shift(lag)

# ── 4d. Moyennes glissantes sur lags sécurisés ──
# Rolling 7j (336 pas) et 14j (672 pas) calculées sur le signal décalé
for window_days, window_name in [(7, "7d"), (14, "14d")]:
    window = window_days * 48  # en pas de 30 min
    base_shift = HORIZON       # on décale d'abord de HORIZON pour rester safe
    shifted = df["load"].shift(base_shift)
    df[f"rolling_mean_{window_name}"] = shifted.rolling(window, min_periods=1).mean()
    df[f"rolling_std_{window_name}"]  = shifted.rolling(window, min_periods=1).std()

# ── 4e. Features météo enrichies ──
df["temp_squared"] = df["temperature"] ** 2        # non-linéarité chauffage/clim
df["radiation_x_temp"] = df["radiation"] * df["temperature"]

print(f"   ✔ {len(df.columns)} features créées")

# ──────────────────────────────────────────────────────────────────────
# 5. PRÉPARATION TRAIN / VALIDATION / TEST
# ──────────────────────────────────────────────────────────────────────
print("\n[4/6] Préparation des jeux de données…")

# Suppression des lignes avec NaN (dues aux lags)
feature_cols = [c for c in df.columns if c != "load"]
df_model = df.dropna(subset=feature_cols + ["load"]).copy()

# Découpage temporel
# ┌──────────────────┬──────────┬──────────┐
# │      TRAIN       │   VAL    │   TEST   │
# │                  │  672 pts │  672 pts │
# └──────────────────┴──────────┴──────────┘
n = len(df_model)
test_size = HORIZON
val_size  = HORIZON

df_test  = df_model.iloc[-(test_size):]
df_val   = df_model.iloc[-(test_size + val_size):-(test_size)]
df_train = df_model.iloc[:-(test_size + val_size)]

X_train, y_train = df_train[feature_cols], df_train["load"]
X_val,   y_val   = df_val[feature_cols],   df_val["load"]
X_test,  y_test  = df_test[feature_cols],  df_test["load"]

print(f"   ✔ Train : {len(df_train)} ({df_train.index.min()} → {df_train.index.max()})")
print(f"   ✔ Val   : {len(df_val)}  ({df_val.index.min()} → {df_val.index.max()})")
print(f"   ✔ Test  : {len(df_test)}  ({df_test.index.min()} → {df_test.index.max()})")

# ──────────────────────────────────────────────────────────────────────
# 6. ENTRAÎNEMENT XGBOOST
# ──────────────────────────────────────────────────────────────────────
print("\n[5/6] Entraînement XGBoost…")

model = XGBRegressor(
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=7,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=-1,
    tree_method="hist",
    early_stopping_rounds=50,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100,
)

best_iter = model.best_iteration
print(f"   ✔ Meilleur n_estimators = {best_iter}")

# ──────────────────────────────────────────────────────────────────────
# 7. ÉVALUATION
# ──────────────────────────────────────────────────────────────────────
print("\n[6/6] Évaluation sur le jeu de test…")

y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# MAPE (%) — avec protection division par zéro
mask_nonzero = y_test != 0
mape = np.mean(np.abs((y_test[mask_nonzero] - y_pred[mask_nonzero]) / y_test[mask_nonzero])) * 100

# MASE — naïve saisonnier (même demi-heure, 7 jours avant = 336 pas)
SEASONAL_PERIOD = 336  # 7 jours en pas de 30 min
naive_errors = np.abs(np.diff(df_model["load"].values[::1], n=1))
# Calcul plus robuste : erreur naïve = |y(t) - y(t - season)| sur le train
y_full = df_model["load"].values
naive_mae = np.mean(np.abs(y_full[SEASONAL_PERIOD:] - y_full[:-SEASONAL_PERIOD]))
mase = mae / naive_mae if naive_mae > 0 else np.inf

print(f"\n{'─' * 50}")
print(f"  MAE  : {mae:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  MAPE : {mape:.2f} %")
print(f"  MASE : {mase:.4f}")
print(f"{'─' * 50}")

# ──────────────────────────────────────────────────────────────────────
# 8. IMPORTANCE DES FEATURES (Top 15)
# ──────────────────────────────────────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=feature_cols)
importances = importances.sort_values(ascending=False).head(15)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# ── Plot 1 : importance des features ──
importances.plot.barh(ax=axes[0], color="#2563eb")
axes[0].set_title("Top 15 — Importance des features", fontsize=13)
axes[0].invert_yaxis()
axes[0].set_xlabel("Importance (gain)")

# ── Plot 2 : prédiction vs réalité sur la période test ──
axes[1].plot(df_test.index, y_test.values, label="Réel", linewidth=1.0, alpha=0.8)
axes[1].plot(df_test.index, y_pred, label="Prédit", linewidth=1.0, alpha=0.8, linestyle="--")
axes[1].set_title(f"Test — 14 derniers jours  |  MAPE={mape:.2f}%  MASE={mase:.4f}", fontsize=13)
axes[1].legend()
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Charge (MW)")
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("evaluation_xgboost.png", dpi=150, bbox_inches="tight")
plt.show()
print("   ✔ Graphique sauvegardé → evaluation_xgboost.png")

# ──────────────────────────────────────────────────────────────────────
# 9. EXPORT CSV DE SOUMISSION
# ──────────────────────────────────────────────────────────────────────
submission = pd.DataFrame({
    "datetime":     df_test.index,
    "load_mw_pred": np.round(y_pred, 4),
})
submission.to_csv(OUTPUT_CSV, index=False)
print(f"   ✔ Soumission exportée → {OUTPUT_CSV} ({len(submission)} lignes)")

print("\n" + "=" * 70)
print("  PIPELINE TERMINÉ AVEC SUCCÈS")
print("=" * 70)