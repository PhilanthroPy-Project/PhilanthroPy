# Try it in your browser

This page runs PhilanthroPy on [Pyodide](https://pyodide.org), Python compiled to WebAssembly: your browser reads your file and does the modelling itself. **Nothing is uploaded anywhere.**

=== "Donor health"

    A few descriptive metrics that need no training and no history longer than two periods: retention, gift concentration and how much of your revenue the top donors carry. Useful even with a small or thin-data file.

    Upload one CSV of individual gifts, one row per gift:

    | Column | Example |
    |---|---|
    | `donor_id` | `10442` |
    | `gift_date` | `2024-03-15` |
    | `gift_amount` | `250.00` |

    <div id="try-health-app">
      <p>
        <input type="file" id="health-file" accept=".csv,text/csv">
        <button type="button" id="health-sample" class="md-button">Use sample data</button>
      </p>
      <p id="health-status" role="status"></p>
      <div id="health-report"></div>
    </div>

=== "Leadership-upgrade scoring"

    Rank your mid-level donors ($100-$999 a fiscal year) by how likely they are to reach a $1,000 leadership gift next year. This calls [`score_upgrade_prospects`](reference/models.md) with its defaults (July fiscal-year start); it trains on every fully finished fiscal year in your file and checks itself on the most recent one. You need at least three fiscal years of gifts.

    **Gifts (required)**, one row per gift:

    | Column | Example |
    |---|---|
    | `donor_id` | `10442` |
    | `gift_date` | `2024-03-15` |
    | `gift_amount` | `250.00` |

    **Activity (optional)**, one row per event, volunteer shift, email click or any other dated touch. Adding this usually improves the ranking, since it gives the model a second, independent signal on top of giving history.

    | Column | Example |
    |---|---|
    | `donor_id` | `10442` |
    | `activity_date` | `2024-06-01` |
    | `activity_type` | `event` |
    | `hours` (optional) | `2` |
    | `amount` (optional) | `50.00` |

    <div id="try-app">
      <p>
        <label>Gifts: <input type="file" id="try-file" accept=".csv,text/csv"></label><br>
        <label>Activity (optional): <input type="file" id="try-activity-file" accept=".csv,text/csv"></label>
      </p>
      <p>
        <button type="button" id="try-sample" class="md-button">Use sample data (gifts + activity)</button>
      </p>
      <p id="try-status" role="status"></p>
      <div id="try-report"></div>
      <p><a id="try-download" class="md-button md-button--primary" download="upgrade_scores.csv" hidden>Download all scores (CSV)</a></p>
      <div id="try-results"></div>
    </div>

The first run downloads about 30 MB (Python, pandas and scikit-learn). Your browser caches it after that.

## The same thing in Python

```python
import pandas as pd
from philanthropy.datasets import make_donor_panel
from philanthropy.models import score_upgrade_prospects

# In practice: gifts = pd.read_csv("gifts.csv"), activities = pd.read_csv("activity.csv")
gifts = make_donor_panel(n_donors=2000, n_years=6, random_state=0)["gifts"]
scores, report = score_upgrade_prospects(gifts, random_state=0)  # activities= is optional

assert "affinity_score" in scores.columns
```

The CLI equivalent, and how to add event and volunteer data, are in [Use the CLI](how-to/use_the_cli.md).
