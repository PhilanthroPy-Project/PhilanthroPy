// In-browser demo for docs/try.md. Pyodide is fetched only on the first click,
// so no other page pays for it.
const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.29.5/full/";

const PY_SOURCE = `
import io, json
import pandas as pd
from philanthropy.models import score_upgrade_prospects
from philanthropy.metrics import (
    donor_retention_rate,
    gift_concentration_gini,
    top_donor_share,
)

def run(gifts_csv, activity_csv):
    gifts = pd.read_csv(io.StringIO(gifts_csv))
    activities = None
    if activity_csv:
        activities = pd.read_csv(io.StringIO(activity_csv))
        # The page's file is "donor_id" to match the gifts file; the library
        # calls the same column "contact_id".
        activities = activities.rename(columns={"donor_id": "contact_id"})
    scores, report = score_upgrade_prospects(gifts, activities=activities, random_state=0)
    scores = scores.drop(columns="suggested_ask")
    scores["top_reasons"] = [
        ", ".join(f"{name} = {value:,.0f}" for name, value in reasons)
        for reasons in scores["top_reasons"]
    ]
    table = scores.head(25).reset_index().to_html(
        border=0, index=False, float_format="{:.1f}".format
    )
    return table, scores.to_csv(), json.dumps(report)

def sample():
    from philanthropy.datasets import make_donor_panel
    panel = make_donor_panel(n_donors=2000, n_years=6, random_state=0)
    gifts = panel["gifts"][["donor_id", "gift_date", "gift_amount"]]
    return gifts.to_csv(index=False), _sample_activity(gifts)

def _sample_activity(gifts):
    # A light synthetic activity log tied to each donor's own giving, just
    # for a "does the optional file work" demo; not a claim about real data.
    import numpy as np
    rng = np.random.default_rng(0)
    donor_dates = gifts[["donor_id", "gift_date"]].drop_duplicates("donor_id")
    n = len(donor_dates)
    rows = pd.DataFrame({
        "donor_id": donor_dates["donor_id"].to_numpy(),
        "activity_date": pd.to_datetime(donor_dates["gift_date"]) + pd.to_timedelta(
            rng.integers(-60, 60, n), unit="D"
        ),
        "activity_type": rng.choice(["event", "volunteer"], n),
        "hours": np.where(rng.random(n) < 0.3, rng.integers(1, 6, n), np.nan),
    })
    return rows.to_csv(index=False)

def health_run(csv_text):
    gifts = pd.read_csv(io.StringIO(csv_text))
    gifts["gift_date"] = pd.to_datetime(gifts["gift_date"])
    gifts["_year"] = gifts["gift_date"].dt.year
    years = sorted(gifts["_year"].unique())
    if len(years) < 2:
        raise ValueError(
            "Need gifts spanning at least two calendar years to measure retention."
        )
    current_year, prior_year = years[-1], years[-2]
    current_donors = gifts.loc[gifts["_year"] == current_year, "donor_id"]
    prior_donors = gifts.loc[gifts["_year"] == prior_year, "donor_id"]
    retention = donor_retention_rate(current_donors, prior_donors)

    lifetime = gifts.groupby("donor_id")["gift_amount"].sum()
    gini = gift_concentration_gini(lifetime)
    top10 = top_donor_share(lifetime, top_fraction=0.1)

    return json.dumps({
        "current_year": int(current_year),
        "prior_year": int(prior_year),
        "n_current_donors": int(current_donors.nunique()),
        "n_prior_donors": int(prior_donors.nunique()),
        "retention_rate": retention,
        "gini": gini,
        "top10_share": top10,
        "n_donors_total": int(lifetime.shape[0]),
    })

def health_sample():
    from philanthropy.datasets import make_donor_panel
    gifts = make_donor_panel(n_donors=2000, n_years=6, random_state=0)["gifts"]
    return gifts[["donor_id", "gift_date", "gift_amount"]].to_csv(index=False)
`;

let pyodideReady = null;

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src;
    s.onload = resolve;
    s.onerror = () => reject(new Error("Could not load " + src));
    document.head.appendChild(s);
  });
}

async function bootPyodide(setStatus) {
  setStatus("Loading Python, pandas and scikit-learn...");
  await loadScript(PYODIDE_URL + "pyodide.js");
  const py = await loadPyodide({ indexURL: PYODIDE_URL });
  await py.loadPackage(["micropip", "numpy", "pandas", "scikit-learn", "joblib"]);
  setStatus("Installing PhilanthroPy...");
  // The docs build ships a wheel of the same commit as the page; a local
  // `mkdocs serve` has none, so fall back to the latest release on PyPI.
  const base = new URL("../wheels/", location.href);
  const name = await fetch(new URL("latest.txt", base))
    .then((r) => (r.ok ? r.text() : ""))
    .catch(() => "");
  const micropip = py.pyimport("micropip");
  await micropip.install(name.trim() ? new URL(name.trim(), base).href : "philanthropy");
  py.runPython(PY_SOURCE);
  return py;
}

function getPyodide(setStatus) {
  if (!pyodideReady) {
    pyodideReady = bootPyodide(setStatus).catch((err) => {
      pyodideReady = null;
      throw err;
    });
  }
  return pyodideReady;
}

function pct(x) {
  return (100 * x).toFixed(0) + "%";
}

function reportLines(r) {
  const lines = [
    `Trained on ${r.n_training_rows.toLocaleString()} donor-years across ` +
      `${r.n_training_fiscal_years} fiscal years; scored ${r.n_scored.toLocaleString()} ` +
      `donors in the band in FY${r.current_fiscal_year}.`,
  ];
  if (r.validated) {
    const against = [
      `${pct(r.baseline_topn_fy_total_upgrade_rate)} for the top ${r.top_n} by largest FY total`,
    ];
    if (r.baseline_gave_threshold_upgrade_rate != null) {
      against.push(
        `${pct(r.baseline_gave_threshold_upgrade_rate)} of donors who gave ` +
          `$${r.baseline_giving_threshold.toLocaleString()}+ that year`
      );
    }
    against.push(`and ${pct(r.overall_upgrade_rate)} across all ${r.n_validation_rows} candidates`);
    lines.push(
      `Held-out check on FY${r.validation_fiscal_year}: of the top ${r.top_n} donors ` +
        `the model picked, ${pct(r.model_upgrade_rate_top_n)} upgraded, against ` +
        against.join(", ") + "."
    );
  } else {
    lines.push("Only one finished fiscal year, so there was no held-out check.");
  }
  if (r.low_data_message) lines.push(r.low_data_message);
  return lines;
}

function healthLines(h) {
  return [
    `${h.n_donors_total.toLocaleString()} distinct donors, ${h.n_prior_donors.toLocaleString()} ` +
      `gave in ${h.prior_year} and ${h.n_current_donors.toLocaleString()} in ${h.current_year}.`,
    `Retention: ${pct(h.retention_rate)} of ${h.prior_year}'s donors gave again in ${h.current_year}.`,
    `Concentration: the top 10% of donors by lifetime giving account for ${pct(h.top10_share)} of revenue ` +
      `(Gini ${h.gini.toFixed(2)}, where 0 is perfectly even and 1 is a single donor).`,
  ];
}

function initTryPage() {
  const app = document.getElementById("try-app");
  if (app && !app.dataset.ready) {
    app.dataset.ready = "1";

    const file = document.getElementById("try-file");
    const activityFile = document.getElementById("try-activity-file");
    const sampleBtn = document.getElementById("try-sample");
    const status = document.getElementById("try-status");
    const reportEl = document.getElementById("try-report");
    const results = document.getElementById("try-results");
    const download = document.getElementById("try-download");
    const setStatus = (msg) => (status.textContent = msg);
    const inputs = [file, activityFile, sampleBtn];

    async function score(getCsvs) {
      inputs.forEach((el) => (el.disabled = true));
      reportEl.replaceChildren();
      results.replaceChildren();
      download.hidden = true;
      try {
        const py = await getPyodide(setStatus);
        const [giftsCsv, activityCsv] = await getCsvs(py);
        setStatus("Training and scoring (this can take up to a minute)...");
        // Let the status paint before runPython blocks the main thread.
        // ponytail: runs on the main thread, move to a Web Worker if big files freeze the tab.
        await new Promise((r) => setTimeout(r, 50));
        const [table, fullCsv, reportJson] = py.globals
          .get("run")(giftsCsv, activityCsv || "")
          .toJs();
        for (const line of reportLines(JSON.parse(reportJson))) {
          const p = document.createElement("p");
          p.textContent = line;
          reportEl.appendChild(p);
        }
        // pandas escapes every cell in to_html, so the user's own values are inert here.
        results.innerHTML = table;
        results.querySelector("table")?.removeAttribute("class");
        if (download.href) URL.revokeObjectURL(download.href);
        download.href = URL.createObjectURL(new Blob([fullCsv], { type: "text/csv" }));
        download.hidden = false;
        setStatus("Done. The table shows the top 25; the download has everyone.");
      } catch (err) {
        const msg = String(err.message || err).trim().split("\n").pop();
        setStatus("Could not score this file: " + msg);
      } finally {
        inputs.forEach((el) => (el.disabled = false));
      }
    }

    file.addEventListener("change", () => {
      const chosen = file.files[0];
      file.value = "";
      if (chosen) {
        score(async () => [
          await chosen.text(),
          activityFile.files[0] ? await activityFile.files[0].text() : "",
        ]);
      }
    });
    sampleBtn.addEventListener("click", () =>
      score(async (py) => py.globals.get("sample")().toJs())
    );
  }

  const healthApp = document.getElementById("try-health-app");
  if (healthApp && !healthApp.dataset.ready) {
    healthApp.dataset.ready = "1";

    const file = document.getElementById("health-file");
    const sampleBtn = document.getElementById("health-sample");
    const status = document.getElementById("health-status");
    const reportEl = document.getElementById("health-report");
    const setStatus = (msg) => (status.textContent = msg);

    async function run(getCsv) {
      file.disabled = sampleBtn.disabled = true;
      reportEl.replaceChildren();
      try {
        const py = await getPyodide(setStatus);
        const csv = await getCsv();
        setStatus("Computing...");
        const health = JSON.parse(py.globals.get("health_run")(csv));
        for (const line of healthLines(health)) {
          const p = document.createElement("p");
          p.textContent = line;
          reportEl.appendChild(p);
        }
        setStatus("Done.");
      } catch (err) {
        const msg = String(err.message || err).trim().split("\n").pop();
        setStatus("Could not compute this file: " + msg);
      } finally {
        file.disabled = sampleBtn.disabled = false;
      }
    }

    file.addEventListener("change", () => {
      const chosen = file.files[0];
      file.value = "";
      if (chosen) run(() => chosen.text());
    });
    sampleBtn.addEventListener("click", () =>
      run(async () => (await getPyodide(setStatus)).globals.get("health_sample")())
    );
  }
}

// Material's instant navigation swaps pages without a reload; document$ fires on each.
if (typeof document$ !== "undefined") {
  document$.subscribe(initTryPage);
} else {
  document.addEventListener("DOMContentLoaded", initTryPage);
}
