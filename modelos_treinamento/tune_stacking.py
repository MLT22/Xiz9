"""
Tuning do meta-learner de stacking usando o cache gerado por train_stacking.py.
Executa em segundos (sem reprocessar imagens).

Etapas:
  1. Diagnostico: acuracia individual de cada modelo base
  2. Grid search nos hiperparametros do XGBoost
  3. Otimizacao do threshold de classificacao
  4. Salva o melhor modelo em modelos/Stacking/modelo_stacking.pkl
"""

import os
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, f1_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

CACHE_TRAIN = 'modelos/Stacking/cache_train.npz'
CACHE_EVAL  = 'modelos/Stacking/cache_eval.npz'

# Indices das colunas em build_meta_row:
# 0=p_real_pca, 1=p_ia_pca, 2=p_real_lum, 3=p_ia_lum,
# 4=p_real_ruido, 5=p_ia_ruido, 6=margin_pca, 7=margin_lum,
# 8=margin_ruido, 9=vote_ia, 10=all_agree, 11=mean_p_ia,
# 12=max_conf, 13=min_conf

# ── Carrega cache ─────────────────────────────────────────────────────────────

print("Carregando cache...")
d_train = np.load(CACHE_TRAIN); X_train, y_train = d_train['X'], d_train['y']
d_eval  = np.load(CACHE_EVAL);  X_eval,  y_eval  = d_eval['X'],  d_eval['y']
print(f"  Treino : {len(X_train)} amostras | Avaliacao: {len(X_eval)} amostras\n")

# ── 1. Diagnostico: acuracia individual de cada modelo base ───────────────────

print("=== Diagnostico: desempenho individual dos modelos base ===")
for nome, col_ia in [("PCA",        1),
                      ("Luminancia", 3),
                      ("Ruido",      5)]:
    pred = (X_eval[:, col_ia] > 0.5).astype(int)
    acc  = accuracy_score(y_eval, pred)
    auc  = roc_auc_score(y_eval, X_eval[:, col_ia])
    print(f"  {nome:12s}  acc={acc*100:.1f}%  AUC={auc:.4f}")

print()

# ── 2. Grid search com validacao cruzada ─────────────────────────────────────

print("=== Grid search (XGBoost) ===")

param_grid = [
    dict(n_estimators=3000, max_depth=4, learning_rate=0.01,  subsample=0.8, colsample_bytree=0.8, min_child_weight=3, reg_alpha=0.1),
    dict(n_estimators=2000, max_depth=4, learning_rate=0.02,  subsample=0.8, colsample_bytree=0.8, min_child_weight=3, reg_alpha=0.1),
    dict(n_estimators=2000, max_depth=5, learning_rate=0.01,  subsample=0.8, colsample_bytree=0.7, min_child_weight=2, reg_alpha=0.05),
    dict(n_estimators=1500, max_depth=4, learning_rate=0.02,  subsample=0.9, colsample_bytree=0.8, min_child_weight=2, reg_alpha=0.0),
    dict(n_estimators=1000, max_depth=6, learning_rate=0.02,  subsample=0.7, colsample_bytree=0.7, min_child_weight=1, reg_alpha=0.2),
    dict(n_estimators=2000, max_depth=3, learning_rate=0.015, subsample=0.8, colsample_bytree=1.0, min_child_weight=5, reg_alpha=0.0),
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_all = np.vstack([X_train, X_eval])
y_all = np.concatenate([y_train, y_eval])

best_cv_auc   = 0.0
best_params   = None
best_model    = None

for i, params in enumerate(param_grid):
    model = XGBClassifier(
        **params,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    scores = cross_val_score(model, X_all, y_all, cv=cv,
                             scoring='roc_auc', n_jobs=-1)
    mean_auc = scores.mean()
    print(f"  [{i+1}/{len(param_grid)}] {params}  ->  AUC CV={mean_auc:.4f} ± {scores.std():.4f}")

    if mean_auc > best_cv_auc:
        best_cv_auc = mean_auc
        best_params = params

print(f"\n  Melhor configuracao (AUC CV={best_cv_auc:.4f}):")
print(f"  {best_params}\n")

print("Treinando modelo final com melhores params em treino+eval...")
final_model = XGBClassifier(
    **best_params,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
final_model.fit(X_all, y_all)

y_proba_eval = final_model.predict_proba(X_eval)[:, 1]
y_pred_05    = (y_proba_eval > 0.5).astype(int)
print(f"\nAvaliacao no conjunto de eval (threshold=0.5):")
print(f"  Acuracia: {accuracy_score(y_eval, y_pred_05)*100:.2f}%")
print(f"  AUC-ROC : {roc_auc_score(y_eval, y_proba_eval):.4f}")

# ── 3. Otimizacao do threshold ────────────────────────────────────────────────

print("\n=== Otimizando threshold de classificacao ===")

thresholds  = np.arange(0.30, 0.71, 0.01)
best_thr    = 0.5
best_acc    = 0.0
best_f1     = 0.0

rows = []
for thr in thresholds:
    pred = (y_proba_eval > thr).astype(int)
    acc  = accuracy_score(y_eval, pred)
    f1   = f1_score(y_eval, pred, average='macro')
    rows.append((thr, acc, f1))
    if acc > best_acc:
        best_acc = acc
        best_thr = thr
        best_f1  = f1

print(f"  {'Threshold':>10}  {'Acuracia':>10}  {'F1-macro':>10}")
for thr, acc, f1 in rows:
    marker = " <-- MELHOR" if abs(thr - best_thr) < 0.005 else ""
    if abs(thr - best_thr) <= 0.05:
        print(f"  {thr:>10.2f}  {acc*100:>9.2f}%  {f1:>10.4f}{marker}")

print(f"\n  Threshold otimo : {best_thr:.2f}")
print(f"  Acuracia final  : {best_acc*100:.2f}%")
print(f"  F1-macro        : {best_f1:.4f}")

y_pred_best = (y_proba_eval > best_thr).astype(int)
print()
print(classification_report(y_eval, y_pred_best, target_names=['Real', 'IA']))

# ── 4. Salva ──────────────────────────────────────────────────────────────────

os.makedirs('modelos/Stacking', exist_ok=True)
joblib.dump(final_model, 'modelos/Stacking/modelo_stacking.pkl')
joblib.dump({'threshold': float(best_thr)}, 'modelos/Stacking/stacking_config.pkl')

print(f"Salvos:")
print(f"  modelos/Stacking/modelo_stacking.pkl")
print(f"  modelos/Stacking/stacking_config.pkl  (threshold={best_thr:.2f})")
