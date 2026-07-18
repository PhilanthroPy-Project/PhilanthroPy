"""Quick start — train a donor-propensity model and score a prospect pool.

Trains :class:`~philanthropy.models.DonorPropensityModel` on a reproducible
synthetic donor dataset, then ranks a held-out pool by affinity score.

Run it:

    python examples/quickstart.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from philanthropy.datasets import generate_synthetic_donor_data
from philanthropy.models import DonorPropensityModel

FEATURES = ["total_gift_amount", "years_active", "event_attendance_count"]


def main() -> None:
    df = generate_synthetic_donor_data(n_samples=1000, random_state=42)
    X, y = df[FEATURES].to_numpy(), df["is_major_donor"].to_numpy()

    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = DonorPropensityModel(n_estimators=200, random_state=0)
    model.fit(X_train, y_train)

    pool = pd.DataFrame(X_test, columns=FEATURES)
    pool["affinity_score"] = model.predict_affinity_score(X_test)  # 0–100 scale
    pool = pool.sort_values("affinity_score", ascending=False)

    print(f"Trained on {len(X_train)} donors; scoring a {len(X_test)}-prospect pool.\n")
    print("Top 5 prospects by affinity score:")
    print(pool.head(5)[FEATURES + ["affinity_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
