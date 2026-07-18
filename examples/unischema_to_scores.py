"""UniSchema -> PhilanthroPy — score donors straight from an event stream.

`UniSchema <https://github.com/PhilanthroPy-Project/UniSchema>`_ normalises
fragmented advancement webhooks (GiveCampus, Slate, NPSP, Cvent, ...) into one
``ConstituentEvent`` stream. ``philanthropy.ingest`` turns that stream into the
one-row-per-donor feature table the estimators consume — the same columns the
Quick Start trains on, so a UniSchema feed drops in with no glue code.

The realistic flow, end to end:

    1. Aggregate a ConstituentEvent batch into donor features (leakage-safe,
       deduplicated by ``eventId``).
    2. Train a model on your labelled giving history (synthetic stand-in here).
    3. Score the freshly-ingested donors.

Run it:

    python examples/unischema_to_scores.py
"""

from philanthropy.datasets import generate_synthetic_donor_data
from philanthropy.ingest import constituent_events_to_features
from philanthropy.models import DonorPropensityModel

FEATURES = ["total_gift_amount", "years_active", "event_attendance_count"]

# A tiny in-memory ConstituentEvent batch, exactly as UniSchema egresses it.
# In practice you'd load a .ndjson batch or a directory with
# ``read_constituent_events("data/egress/")`` instead of inlining dicts.
EVENTS = [
    {"eventId": "1", "constituentEmail": "ada@uni.edu", "eventType": "DONATION",
     "sourceSystem": "GIVECAMPUS", "amount": 250.0, "createdAt": "2022-03-01T12:00:00Z"},
    {"eventId": "2", "constituentEmail": "ada@uni.edu", "eventType": "EVENT_REGISTRATION",
     "sourceSystem": "CVENT", "createdAt": "2025-06-01T09:00:00Z"},
    {"eventId": "3", "constituentEmail": "grace@uni.edu", "eventType": "DONATION",
     "sourceSystem": "SLATE", "amount": 50000.0, "createdAt": "2018-09-20T15:30:00Z"},
    {"eventId": "4", "constituentEmail": "grace@uni.edu", "eventType": "EVENT_REGISTRATION",
     "sourceSystem": "CVENT", "createdAt": "2025-10-02T18:00:00Z"},
    {"eventId": "5", "constituentEmail": "grace@uni.edu", "eventType": "DONATION",
     "sourceSystem": "GIVECAMPUS", "amount": 75000.0, "createdAt": "2025-11-15T11:00:00Z"},
    # A redelivered webhook (same eventId as #1) — dropped, not double-counted.
    {"eventId": "1", "constituentEmail": "ada@uni.edu", "eventType": "DONATION",
     "sourceSystem": "GIVECAMPUS", "amount": 250.0, "createdAt": "2022-03-01T12:00:00Z"},
]


def main() -> None:
    features = constituent_events_to_features(EVENTS)

    history = generate_synthetic_donor_data(n_samples=1000, random_state=42)
    model = DonorPropensityModel(n_estimators=200, random_state=0)
    model.fit(history[FEATURES].to_numpy(), history["is_major_donor"].to_numpy())

    features["affinity_score"] = model.predict_affinity_score(features[FEATURES].to_numpy())

    print("Donor features aggregated from the UniSchema stream:\n")
    print(features[["constituent_email", *FEATURES, "affinity_score"]].to_string())


if __name__ == "__main__":
    main()
