"""Turn churn probabilities into a retention table: who to contact and what to offer.

The top decile of the holdout (by predicted probability) is grouped by contract
and internet service; each group gets its size, expected churners, the observed
churn rate and one or two actions chosen by simple rules on the group's traits.
"""

from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd

from .features import add_engineered

TOP_FRACTION = 0.10
SEGMENT_COLUMNS = ["Contract", "InternetService", "PaymentMethod", "tenure_bucket"]


def churn_by_segment(X: pd.DataFrame, y: pd.Series) -> dict[str, pd.DataFrame]:
    """Descriptive churn rate per value of a few segment columns."""
    df = add_engineered(X)
    df["churn"] = np.asarray(y)
    out = {}
    for col in SEGMENT_COLUMNS:
        g = (
            df.groupby(col, observed=True)["churn"]
            .agg(customers="size", churn_rate="mean")
            .reset_index()
            .rename(columns={col: "segment"})
            .sort_values("churn_rate", ascending=False)
            .reset_index(drop=True)
        )
        g["churn_rate"] = g["churn_rate"].round(4)
        g.insert(0, "dimension", col)
        out[col] = g
    return out


def recommend_actions(
    contract: str,
    internet: str,
    electronic_check_share: float,
    no_support_share: float,
    new_customer_share: float,
    high_charges_share: float,
) -> list[str]:
    actions: list[str] = []
    if contract == "Month-to-month":
        if internet == "Fiber optic":
            actions.append("Offer a 12-month contract with a price lock (fiber month-to-month is the highest-churn segment)")
        else:
            actions.append("Offer a 12-month contract with a first-year discount")
    if new_customer_share >= 0.5:
        actions.append("Onboarding check-in call within the first month (most of this segment is six months old or younger)")
    if electronic_check_share >= 0.5:
        actions.append("Incentivise a switch to automatic payment (card or bank transfer)")
    if no_support_share >= 0.5:
        actions.append("Bundle tech support or online security at a reduced price")
    if high_charges_share >= 0.5 and len(actions) < 3:
        actions.append("Plan review to right-size the monthly charge before the next bill")
    if not actions:
        actions.append("Personal retention call with a loyalty offer")
    return actions[:3]


def retention_table(
    prob: np.ndarray,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    top_fraction: float = TOP_FRACTION,
    min_size: int = 15,
) -> pd.DataFrame:
    """Group the top decile of predicted churn risk into actionable segments."""
    df = add_engineered(X_holdout)
    df["prob"] = np.asarray(prob, dtype=float)
    df["churn"] = np.asarray(y_holdout, dtype=int)
    k = max(1, ceil(top_fraction * len(df)))
    top = df.nlargest(k, "prob")
    high_charge_cut = df["MonthlyCharges"].quantile(0.75)

    def describe(g: pd.DataFrame, label: str, contract: str, internet: str) -> dict:
        echeck = float((g["PaymentMethod"] == "Electronic check").mean())
        nosupport = float(g["has_no_support"].mean())
        new = float(g["new_customer"].mean())
        high = float((g["MonthlyCharges"] > high_charge_cut).mean())
        return {
            "segment": label,
            "customers": int(len(g)),
            "share_of_top_decile": round(100.0 * len(g) / k, 1),
            "mean_churn_prob": round(float(g["prob"].mean()), 3),
            "expected_churners": int(round(float(g["prob"].sum()))),
            "observed_churn_rate": round(float(g["churn"].mean()), 3),
            "electronic_check_share": round(echeck, 2),
            "no_support_share": round(nosupport, 2),
            "new_customer_share": round(new, 2),
            "actions": "; ".join(recommend_actions(contract, internet, echeck, nosupport, new, high)),
        }

    rows = []
    kept_index = []
    for (contract, internet, bucket), g in top.groupby(["Contract", "InternetService", "tenure_bucket"], observed=True):
        if len(g) < min_size:
            continue
        kept_index.extend(g.index.tolist())
        rows.append(describe(g, f"{contract} / {internet} / tenure {bucket} months", contract, internet))
    rows.sort(key=lambda r: -r["expected_churners"])
    rest = top.drop(index=kept_index)
    if len(rest):
        contract = str(rest["Contract"].mode().iat[0])
        internet = str(rest["InternetService"].mode().iat[0])
        rows.append(describe(rest, f"all other segments ({rest.groupby(['Contract', 'InternetService', 'tenure_bucket'], observed=True).ngroups} small groups)", contract, internet))
    table = pd.DataFrame(rows)
    table.attrs["top_decile_size"] = k
    table.attrs["covered"] = len(kept_index)
    table.attrs["top_decile_observed_churn_rate"] = round(float(top["churn"].mean()), 3)
    table.attrs["base_churn_rate"] = round(float(df["churn"].mean()), 3)
    return table
