### Headline
Deploy the incumbent `logreg / raw / C=1`, which scored 0.662 cross-validated PR-AUC with a fold standard deviation of 0.019, 0.634 PR-AUC on the holdout and a lift of 2.81 in the top 10 percent.

### What the experiments showed
- Nine experiments ran across three rounds and zero gains were confirmed; the loop stopped after two rounds without a confirmed verdict.
- The two nominally best runs, `histgb_raw_slow_shallow` at 0.666 (std 0.021, delta +0.004) and `lgbm_raw_tiny_leaves` at 0.666 (std 0.023, delta +0.004), stayed inconclusive because their gains are well inside a fold spread of about 0.021.
- Engineered features were refuted: `logreg_engineered_C1` moved the mean by +0.000 (0.662, inconclusive), `histgb_eng_slow_deep_iters` reached 0.660 (no_gain) and `lgbm_engineered_reg` only 0.645 (no_gain).
- Stronger shrinkage and dimensionality reduction were also refuted: `logreg_raw_C005` with balanced weights fell to 0.659 (no_gain) and `logreg_minimal_C3` to 0.646 (no_gain).
- Balanced class weights damaged calibration: ECE 0.151 for `logreg_raw_C005` and 0.137 for `lgbm_raw_tiny_leaves`, against 0.029 for the incumbent.

### Recommendation and caveats
Ship the raw logistic regression: no alternative beat it beyond fold noise, and it is the cheapest of the four families tested. CV PR-AUC 0.662 versus holdout 0.634 is a drop of 0.028, larger than the 0.019 fold standard deviation, so treat 0.634 as the working estimate. Holdout Brier is 0.138 and ECE 0.025, so predicted probabilities can be used directly for ranking and for expected-churner counts. Limits: 7043 customers, one holdout split at seed 42, and no calibration, interaction terms or monotonic constraints were tested.

### Next three experiments
1. Fit isotonic calibration (3-fold) on the winner, expected to lower Brier 0.135 and ECE 0.029 with PR-AUC unchanged.
2. Add contract x tenure and internet service x monthly charges interactions, since the linear model already ties boosting at 0.662 to 0.666.
3. Rerun the loop on two more holdout seeds to bound the 0.028 CV-holdout gap.

### Retention actions
- Month-to-month / Fiber optic / tenure 0-6 months: 84 customers, 61 expected churners, mean churn probability 0.732. Offer a 12-month contract with a price lock and an onboarding check-in call in the first month.
- Month-to-month / Fiber optic / tenure 7-12 months: 29 customers, 21 expected churners, electronic check share 0.76. Incentivise a switch to automatic payment.
- Month-to-month / Fiber optic / tenure 13-24 months: 18 customers, 13 expected churners, no-support share 1.0. Bundle tech support or online security at a reduced price.
