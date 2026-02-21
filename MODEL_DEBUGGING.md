# Random Forest Flood Depth Model: Why it seems to "copy" values and how to fix it

Your model can look like it is copying the input targets when one or more of these issues happen:

1. **Data leakage**: a feature is directly or indirectly derived from flood depth.
2. **Too many zero depths**: the model predicts near-zero everywhere because zeros dominate training.
3. **Random split on spatial data**: nearby cells in train/test are very similar, so test metrics look good even if generalization is weak.
4. **No baseline comparison**: you cannot tell if the model is better than a simple constant predictor.

## Practical fixes (in order)

### 1) Check for leakage first
- Confirm none of these are computed using the flood simulation output itself.
- Especially review `FlowAcc`, `TWI`, and any preprocessed layers created from flood maps.
- Remove suspicious features and compare performance.

### 2) Handle zero depths explicitly
- If your objective is **where water exists**, do a two-stage model:
  - Stage A: classifier (`flood/no-flood`)
  - Stage B: regressor for only flooded cells (`Depth > 0`)
- If keeping one regressor, at least stratify and inspect depth distribution.

### 3) Use spatial validation (not only random split)
- Split by spatial blocks (or administrative zones), not random rows.
- Random split in geospatial problems often overestimates performance.

### 4) Add baseline and overfit checks
- Compare against baseline RMSE from predicting `y_train.mean()`.
- Report train RMSE vs test RMSE; a very low train error with much worse test error indicates overfitting.

### 5) Constrain model complexity
Start with:
- `max_depth=12`
- `min_samples_leaf=5`
- `max_features='sqrt'`
- keep `n_estimators=300`

This usually prevents memorization-like behavior.

## Safer training template (drop-in structure)

```python
# Keep only complete rows
df = fishnet_gdf[feature_cols_reg + ['FLOODDEPTH']].replace(-999, np.nan).dropna().copy()

# Optional: remove tiny noise values if physically meaningless
# df = df[df['FLOODDEPTH'] >= 0.01]

X = df[feature_cols_reg]
y = df['FLOODDEPTH'].clip(lower=0)

# Random split (replace with spatial block split when possible)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    max_depth=12,
    min_samples_leaf=5,
    max_features='sqrt'
)
reg.fit(X_train, y_train)

# Compare train/test + baseline
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

pred_train = reg.predict(X_train)
pred_test = reg.predict(X_test)

rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
r2_test = r2_score(y_test, pred_test)

baseline_pred = np.full_like(y_test, y_train.mean(), dtype=float)
rmse_baseline = np.sqrt(mean_squared_error(y_test, baseline_pred))

print(f"Train RMSE: {rmse_train:.3f}")
print(f"Test RMSE : {rmse_test:.3f}")
print(f"Test R2   : {r2_test:.3f}")
print(f"Baseline RMSE (mean predictor): {rmse_baseline:.3f}")
```

## Quick interpretation guide
- If `Test RMSE` is only slightly better than baseline: model has weak signal.
- If `Train RMSE << Test RMSE`: overfitting.
- If performance collapses after removing a feature: that feature may contain leakage or be the only strong predictor.

## Most likely immediate next step for your case
Given your current pipeline, the **first thing** to do is:
1. Remove any potentially leakage-prone variables.
2. Add baseline + train/test diagnostics.
3. Then move from random split to spatial split.

That will tell you whether the model is truly learning flood-depth relationships or exploiting spatial/feature leakage.
