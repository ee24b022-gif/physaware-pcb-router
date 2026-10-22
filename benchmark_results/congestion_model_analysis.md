# Congestion MLP diagnosis and improvement plan

## Scope and limitation

This analysis uses the reproducible fallback harness from the prior validation because the original `pcbroute` checkout was not available in the current sandbox. It evaluates the documented `CongestionPredictionMLP(17→32→1)` contract on the same fixed disjoint seeds: 50 training boards (`1000–1049`) and 50 held-out boards (`2000–2049`). The results characterize the current synthetic target and feature generator; they should be rerun against the actual repository before changing production code.

## Why the MLP lost to the mean predictor

The target is extremely sparse. Only 4% of training examples and 2% of held-out examples have nonzero congestion. The train target mean is `0.007998`, while the held-out mean is only `0.002776`; the test distribution therefore has a strong downward shift. A constant predictor is difficult to beat under these conditions, especially on RMSE when most examples are exactly zero.

The target also includes multiplicative log-normal interaction noise that is not represented in the 17 inputs. The model is therefore asked to predict an irreducible random component. With only 50 boards and a 32-unit hidden layer, the MLP has enough capacity to fit incidental relationships but not enough data to learn stable tail behavior. Its outputs overpredict the many zero cases, which explains its poor held-out error.

There is also a feature/target mismatch. `estimated_congestion` is computed as `competing_nets × (1 − local_obstacle_density) / num_layers`, while the label is driven by total board demand, board capacity, obstacle multiplier, route interaction, width and clearance. In this generator, the feature has near-zero Pearson correlation with the target (`-0.021`) and is not useful in permutation testing. It is not a good proxy for the label.

Finally, the train/test split is by seed range, but the current generator permits a distribution shift in board-level capacity and congestion prevalence. A stronger evaluation should stratify by congestion prevalence or use many independent board families while preserving board-level disjointness.

## Feature analysis

The most useful univariate signals were:

| Feature | Pearson correlation with target |
|---|---:|
| `clearance_mm` | +0.229 |
| `width_mm` | +0.174 |
| `num_layers` | −0.173 |
| `hpwl_mm` | +0.149 |
| `local_obstacle_density` | +0.141 |
| `bbox_width_mm` | +0.127 |
| `free_cell_ratio` | −0.122 |

The permutation test was run against the best non-baseline model found in this experiment, a strongly regularized ridge model. Its largest RMSE increases after permutation were:

| Feature | RMSE increase after permutation |
|---|---:|
| `num_layers` | +0.000369 |
| `width_mm` | +0.000260 |
| `estimated_via_count` | +0.000189 |
| `num_pins` | +0.000101 |

Most other permutation changes were approximately zero or negative, which means their contribution was not stable at this sample size. A negative permutation delta is not evidence that the feature is harmful; it is sampling noise or a sign of correlated/redundant features.

The `bias` dimension should be removed from the input schema. The network already has learned biases, so an always-one input adds no information and complicates feature attribution. `num_pins` and `estimated_via_count` are deterministic duplicates in this generator (`estimated_via_count = max(num_pins − 2, 0)`), so one should be removed. `free_cell_ratio` is also deterministically `1 − obstacle` before noise, and should be retained only if it is measured from actual routeable cells rather than copied from obstacle density.

## Model comparisons

| Model | MAE | RMSE |
|---|---:|---:|
| Mean baseline | 0.010454 | 0.020124 |
| Ridge, λ=100 | **0.009217** | 0.021219 |
| 8-unit ReLU, Huber | 0.013550 | 0.024193 |
| 8-unit ReLU, log1p target | 0.012819 | 0.024203 |
| 8-unit ReLU, MSE | 0.016186 | 0.025793 |
| 32-unit ReLU, MSE | 0.030591 | 0.039274 |

The heavily regularized linear model improves MAE but not RMSE. Consequently, it does not pass the prior integration gate, which requires improvement on both metrics. Larger MLPs are clearly overfitting this small, sparse dataset.

## Recommended feature engineering

1. Replace the hand-built proxy with physically meaningful board-level ratios: total estimated routing demand divided by available track capacity, per-layer demand/capacity, obstacle-expanded area, channel width percentiles, and the number of legal escape edges from each pin.
2. Add interaction features that match the label mechanism: `hpwl × num_nets`, `hpwl × width`, `clearance × width`, `demand / (area × layers × free_ratio)`, and per-layer competing-net density.
3. Compute congestion labels from the actual routed result: peak cell occupancy, mean overflow, total overflow, and fraction of over-capacity cells. Do not use a single noisy scalar if the router already exposes an occupancy map.
4. Aggregate at board level before splitting. Generate multiple routing realizations per board, but keep every realization of one board in exactly one split.
5. Remove constant and duplicate inputs: `bias`, one of `num_pins`/`estimated_via_count`, and any feature that is algebraically derived from another without adding measured information.
6. Normalize positive continuous features with `log1p` or robust scaling. Clip or transform heavy-tailed density and overflow values.

## Proposed alternative architecture and loss

Use a two-stage hurdle model rather than direct unconstrained regression:

```text
shared features
  ├── congestion-probability head: MLP( d → 32 → 16 → 1 ), sigmoid
  └── positive-magnitude head:      MLP( d → 32 → 16 → 1 ), Softplus
```

The first head predicts whether the board/net has nonzero congestion. The second predicts `log1p(congestion)` conditional on a positive label. At inference:

```text
prediction = P(congestion > 0) × expm1(predicted_log_magnitude)
```

Train with a class-weighted binary cross-entropy for the probability head and a Huber loss on the positive examples for the magnitude head:

```text
L = BCE_weighted(y > 0, p) + α × Huber(log1p(y), magnitude) × 1[y > 0]
```

Use moderate positive-class weighting, selected only inside training folds, and tune `α` against both MAE and RMSE. This architecture is appropriate because the current label distribution has a point mass at zero and a continuous positive tail. It avoids forcing one MSE head to compromise between “predict zero accurately” and “fit rare high-congestion cases.”

For an even simpler first revision, use a regularized linear or shallow two-layer model with `log1p` target, weight decay, early stopping, and 5–10 repeated board-level splits. Do not increase hidden width until that baseline is stable.

## Hyperparameter recommendations

- Increase board count substantially before increasing model size; target at least 500–1,000 disjoint boards for a 17-feature MLP.
- Use 5-fold grouped cross-validation by board family or seed range, not one 50/50 split.
- Compare hidden widths 8, 16, and 32 only after adding early stopping and L2 weight decay.
- Sweep learning rates around `1e-3` to `1e-2`, weight decay from `1e-5` to `1e-2`, and batch sizes 16–64.
- Select checkpoints by held-out RMSE and report MAE, RMSE, zero/nonzero classification metrics, and calibration separately.
- Add a naive zero predictor and a training-mean predictor to the baseline suite. Given the current 98% test zero rate, the zero predictor is an important reference.

## Integration recommendation

Do not integrate the congestion model yet. The current best alternative improves MAE only and worsens RMSE; the original MLP loses on both metrics. Keep congestion prediction experimental, remove it from routing decisions, and rerun the integration gate only after the target definition, board-level split, and feature set are corrected.
