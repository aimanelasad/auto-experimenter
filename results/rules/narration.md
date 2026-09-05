### Headline
Deploy logreg_raw (logreg / raw / C=1): cross-validated PR-AUC 0.661 (fold std 0.019), holdout PR-AUC 0.634, lift 2.81 in the top 10 %.
logreg_engineered has the highest CV mean (0.662, 0.000 above the incumbent) but the gain is inside fold noise, so the established model stays.

### What the experiments showed
- 10 experiments ran over 3 rounds; 0 beat the incumbent beyond fold noise. Stop reason: no confirmed gain in 2 consecutive rounds.
- The first baseline logreg_raw reached PR-AUC 0.661; the winner improved on it by 0.000.
- logreg_engineered gained 0.000 PR-AUC over its incumbent (inconclusive): Explicit tenure buckets, add-on count and charge ratio encode known churn patterns that logreg otherwise has to learn from one-hot columns.
- Refuted: logreg_raw_regularised scored 0.661 (-0.001 versus its incumbent). Hypothesis was: Stronger L2 regularisation should reduce variance of the one-hot coefficients.
- Refuted: logreg_raw_balanced scored 0.660 (-0.002 versus its incumbent). Hypothesis was: Balanced class weights push the model to rank rare churners higher; may raise recall@10% at some cost in calibration.

### Recommendation and caveats
Use logreg_raw. The holdout PR-AUC differs from the CV mean by 0.028, within two fold standard deviations. Holdout Brier score 0.138 and ECE 0.025: probabilities are usable as-is. Precision in the top 10 % is 0.745 against a base rate of 0.265. Limits: 7043 customers, one holdout split, and only the options inside the search space were tested.

### Next three experiments
- Calibrate logreg_raw (isotonic, 3-fold); expected to lower Brier and ECE with unchanged PR-AUC.
- Add pairwise interaction terms (contract x tenure, internet service x monthly charges) to the linear model; a small PR-AUC gain is plausible.
- Repeat the whole loop with two more holdout seeds to put an interval on the CV-holdout gap.

### Retention actions
- Month-to-month / Fiber optic / tenure 0-6 months: 84 customers in the top decile, 61 expected churners (observed rate 0.833). Offer a 12-month contract with a price lock (fiber month-to-month is the highest-churn segment); Onboarding check-in call within the first month (most of this segment is six months old or younger); Incentivise a switch to automatic payment (card or bank transfer).
- Month-to-month / Fiber optic / tenure 7-12 months: 29 customers in the top decile, 21 expected churners (observed rate 0.483). Offer a 12-month contract with a price lock (fiber month-to-month is the highest-churn segment); Incentivise a switch to automatic payment (card or bank transfer); Bundle tech support or online security at a reduced price.
- Month-to-month / Fiber optic / tenure 13-24 months: 18 customers in the top decile, 13 expected churners (observed rate 0.611). Offer a 12-month contract with a price lock (fiber month-to-month is the highest-churn segment); Incentivise a switch to automatic payment (card or bank transfer); Bundle tech support or online security at a reduced price.
- all other segments (2 small groups): 10 customers in the top decile, 7 expected churners (observed rate 1.000). Offer a 12-month contract with a first-year discount; Onboarding check-in call within the first month (most of this segment is six months old or younger); Incentivise a switch to automatic payment (card or bank transfer).
