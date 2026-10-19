"""Results pages must not drift from ``docs/assets/results/results.json``.

The index table is generated from results.json, so the committed snippet must
equal a fresh render. The prose numbers that are still hand-written are
checked two ways: every "N out of 100" phrase on a results page must match
some number in results.json, and the worked example on the how-to-read page
is pinned to the result it is scaled from.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "make_results_pages.py"
RESULTS = json.loads((REPO_ROOT / "docs" / "assets" / "results" / "results.json").read_text())
PAGES = REPO_ROOT / "docs" / "results"


@pytest.fixture(scope="module")
def mrp():
    spec = importlib.util.spec_from_file_location("make_results_pages", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _numbers(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _numbers(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield float(obj)


def test_index_table_matches_results_json(mrp, tmp_path):
    out = tmp_path / "index_table.md"
    mrp.render_index_table(mrp.build_index(RESULTS), out)
    committed = PAGES / "_verdicts" / "index_table.md"
    assert committed.read_text() == out.read_text(), (
        "docs/results/_verdicts/index_table.md is stale; rerun scripts/make_results_pages.py"
    )


@pytest.mark.parametrize("key", ["who_to_mail_kdd98", "who_to_mail_cup98val"])
def test_cost_sweep_tables_match_results_json(mrp, tmp_path, key):
    out = tmp_path / "sweep.md"
    mrp.render_cost_sweep_table(RESULTS[key]["cost_sweep"], out)
    committed = PAGES / "_verdicts" / f"{key}_cost_sweep.md"
    assert committed.read_text() == out.read_text(), f"{committed} is stale; rerun scripts/make_results_pages.py"


def test_results_json_index_matches_a_fresh_build(mrp):
    fresh = mrp.build_index(RESULTS)
    for question, row in RESULTS["_index"].items():
        assert row["bottom_line"] == fresh[question]["bottom_line"], question
        assert row["cells"] == {ds: c["text"] for ds, c in fresh[question]["cells"].items()}, question


def test_every_out_of_100_phrase_is_in_results_json():
    known = set()
    for x in _numbers(RESULTS):
        known.add(round(x))
        known.add(round(100 - x))  # "N out of 100 keep giving" is 100 minus a lapse base rate
    stale = []
    for page in sorted(PAGES.glob("*.md")):
        for m in re.finditer(r"(\d+(?:\.\d+)?) out of 100", page.read_text()):
            if round(float(m.group(1))) not in known:
                stale.append(f"{page.name}: {m.group(0)}")
    assert not stale, f"numbers not found in results.json: {stale}"


def test_transfer_sentence_matches_the_intervals():
    # "ahead in every one of the test waves" / "does not carry over": the
    # per-wave minimum gap is above 0 on PSID, the bootstrap upper bound
    # below 0 on KDD98.
    assert RESULTS["upgrade_transfer_psid"]["top10pct"]["diff_lo"] > 0
    assert RESULTS["upgrade_transfer_kdd98"]["top10pct"]["diff_hi"] < 0


def test_how_to_read_worked_example_matches_donorschoose_retention():
    text = (PAGES / "how_to_read.md").read_text()
    ret = RESULTS["lapse_donorschoose_retention"]["top10pct"]
    base = RESULTS["lapse_donorschoose"]["metadata"]["base_rate_pct"]
    # Scaled to 1,000 names and rounded to the nearest 10 ("about 450").
    for label, value in (("Random", 100 - base), ("The simple rule", ret["rule"]), ("The model", ret["model"])):
        m = re.search(rf"\*\*{label}:\*\*.*?about ([\d,]+)", text)
        assert m, label
        assert abs(int(m.group(1).replace(",", "")) - value * 10) <= 10, (label, m.group(1), value * 10)


def _get(path):
    node = RESULTS
    for part in path.split("."):
        node = node[part]
    return node


def _folds(key):
    meta = RESULTS[key]["metadata"]
    return meta.get("fold_years") or meta.get("fold_waves")


def _pieces(key):
    mailed, total = re.search(r"pieces=(\d+)/(\d+)", RESULTS[key]["note"]).groups()
    return int(mailed), int(total)


def _wtm_kdd98():
    mailed, total = _pieces("who_to_mail_kdd98")
    e = RESULTS["who_to_mail_kdd98"]
    return [total - mailed, total, e["net_revenue_model"] - e["net_revenue_mail_everyone"]]


def _per_10k(pct):
    # "N of 100" in the top 10% -> donors among the first 1,000 of 10,000, to the nearest 10.
    return round(pct * 10, -1)


INTERVAL_REAL = ("kdd98", "donorschoose", "psid", "karlan_list")


def _lv(ds, level="90"):
    return RESULTS[f"interval_{ds}"]["levels"][level]


def _n_test(key):
    return int(re.search(r"n_test=(\d+)", RESULTS[key]["metadata"]["note"]).group(1))


INTERVAL_NAMES = {"kdd98": "KDD Cup 1998", "donorschoose": "DonorsChoose", "psid": "PSID household survey",
                  "karlan_list": "Karlan and List"}


def test_interval_analyst_table_matches_results_json():
    text = (PAGES / "intervals.md").read_text()
    for ds, name in INTERVAL_NAMES.items():
        for level, v in RESULTS[f"interval_{ds}"]["levels"].items():
            row = f"| {name} | {level}% | {v['attained']:.0f} of 100 | ${v['median_width']:,.0f} |"
            assert row in text, row


def test_every_real_interval_level_is_on_target():
    # The page and the CHANGELOG say "within 3 points of the level asked
    # for" on every real file. The coverage verdict "wins" is one-sided (it
    # only penalises under-coverage), so the two-sided claim is checked too.
    for ds in INTERVAL_REAL:
        for level, v in RESULTS[f"interval_{ds}"]["levels"].items():
            assert v["verdict"] == "wins", (ds, level, v)
            assert abs(v["attained"] - float(level)) <= 3, (ds, level, v)


# (page, sentence template, the numbers it must show, in order). Every number
# in a bold verdict sentence, in the opening paragraphs of each real-file tab,
# and every hand-written number on these pages traces to a results.json key
# here; a number is compared at the precision the page prints it. The page
# text is whitespace-collapsed first, so templates are single-spaced.
NUMBER_MAP = [
    # Leadership upgrade
    ("leadership.md", r"Of our top 10% of picks, (\d+) out of every 100 crossed \$1,000\. Ranking by this year's "
     r"giving total alone found (\d+) out of every 100",
     lambda: [_get("upgrade_synthetic.top10pct.model"), _get("upgrade_synthetic.top10pct.rule")]),
    ("leadership.md", r"On PSID, the model's top 10% of picks found (\d+) out of every 100 households crossing \$1,000, "
     r"against (\d+) for the best rule, ahead in every one of the (\d+) test waves\. On KDD Cup 1998's \$50 proxy "
     r"question it found (\d+) out of every 100, against (\d+) for the best rule",
     lambda: [_get("upgrade_transfer_psid.top10pct.model"), _get("upgrade_transfer_psid.top10pct.rule"),
              len(_folds("upgrade_psid")), _get("upgrade_transfer_kdd98.top10pct.model"),
              _get("upgrade_transfer_kdd98.top10pct.rule")]),
    ("leadership.md", r"Of the model's top 10% of picks, (\d+) out of every 100 crossed \$50\. The best rule \(the "
     r"largest single gift already in the eligible band\) found (\d+) out of every 100",
     lambda: [_get("upgrade_kdd98.top10pct.model"), _get("upgrade_kdd98.top10pct.rule")]),
    ("leadership.md", r"goes the other way: the model's top 1% found (\d+) in 100 against (\d+) in 100 for the rule",
     lambda: [_get("upgrade_kdd98.top1pct.model"), _get("upgrade_kdd98.top1pct.rule")]),
    ("leadership.md", r"Of the model's top 10% of picks, (\d+) out of every 100 crossed \$1,000 the following fiscal "
     r"year\. The best of the 3 simple rules checked here \(this year's total alone, the winner in all (\d+) test "
     r"years\) found (\d+) out of every 100",
     lambda: [_get("upgrade_donorschoose.top10pct.model"), len(_folds("upgrade_donorschoose")),
              _get("upgrade_donorschoose.top10pct.rule")]),
    ("leadership.md", r"its top 1% found (\d+) in 100 against (\d+) in 100 for the rule\. Upgrading to \$1,000\+ is "
     r"rare in this file: only about (\d+) in 100",
     lambda: [_get("upgrade_donorschoose.top1pct.model"), _get("upgrade_donorschoose.top1pct.rule"),
              _get("upgrade_donorschoose.metadata.base_rate_pct")]),
    ("leadership.md", r"the model's top 10% hit rate is (\d+) out of 100 with it, (\d+) without\. The top 1% shifts "
     r"from (\d+) to (\d+) out of 100",
     lambda: [_get("upgrade_donorschoose_momentum.top10pct.model"), _get("upgrade_donorschoose.top10pct.model"),
              _get("upgrade_donorschoose.top1pct.model"), _get("upgrade_donorschoose_momentum.top1pct.model")]),
    ("leadership.md", r"between ([\d,]+) and ([\d,]+) households per test wave, and about (\d+) in 100 of them "
     r"crossed \$1,000",
     lambda: [min(_get("upgrade_psid.metadata.n_per_fold")), max(_get("upgrade_psid.metadata.n_per_fold")),
              _get("upgrade_psid.metadata.base_rate_pct")]),
    ("leadership.md", r"Of the model's top 10% of picks, (\d+) out of every 100 crossed \$1,000 by the next wave\. "
     r"The best of the 3 simple rules checked here \(this wave's total alone, the winner in all (\d+) test waves\) "
     r"found (\d+) out of every 100",
     lambda: [_get("upgrade_psid.top10pct.model"), len(_folds("upgrade_psid")), _get("upgrade_psid.top10pct.rule")]),
    ("leadership.md", r"At the top 5% it found (\d+) against (\d+)\.",
     lambda: [_get("upgrade_psid.top5pct.model"), _get("upgrade_psid.top5pct.rule")]),
    ("leadership.md", r"\((\d+) in 100 against (\d+) for the rule\), but that group",
     lambda: [_get("upgrade_psid.top1pct.model"), _get("upgrade_psid.top1pct.rule")]),
    ("leadership.md", r"moved the top 10% from (\d+) to (\d+)",
     lambda: [_get("upgrade_psid.top10pct.model"), _get("upgrade_psid_momentum.top10pct.model")]),
    # Response
    ("response.md", r"the model's top 1% found (\d+) in 100 and its top 5% found (\d+) in 100, against (\d+) and "
     r"(\d+) in 100 for the best rule\. At the top 10% it was about the same as the rule \((\d+) in 100 against "
     r"(\d+)\)",
     lambda: [_get("response_kdd98.top1pct.model"), _get("response_kdd98.top5pct.model"),
              _get("response_kdd98.top1pct.rule"), _get("response_kdd98.top5pct.rule"),
              _get("response_kdd98.top10pct.model"), _get("response_kdd98.top10pct.rule")]),
    ("response.md", r"the model's top 1% found (\d+) in 100 and its top 10% found (\d+) in 100, against (\d+) and "
     r"(\d+) in 100 for the best rule",
     lambda: [_get("response_cup98val.top1pct.model"), _get("response_cup98val.top10pct.model"),
              _get("response_cup98val.top1pct.rule"), _get("response_cup98val.top10pct.rule")]),
    ("response.md", r"Of our top 5% of picks, (\d+) out of every 100 gave again\. .*? found (\d+) out of every 100",
     lambda: [_get("response_synthetic.top5pct.model"), _get("response_synthetic.top5pct.rule")]),
    ("response.md", r"the model's top 10% found (\d+\.\d) in 100 who gave, against (\d+\.\d) in 100 for the best rule "
     r"\(most gifts first\)\. At the top 1% it found (\d+) in 100 against (\d+)\. Picking at random finds about "
     r"(\d+) in 100",
     lambda: [_get("response_karlan_list.top10pct.model"), _get("response_karlan_list.top10pct.rule"),
              _get("response_karlan_list.top1pct.model"), _get("response_karlan_list.top1pct.rule"),
              _get("response_karlan_list.metadata.base_rate_pct")]),
    ("response.md", r"the offer raised giving by about (\d+\.\d) in 100\. Among the 30% the uplift model ranked highest, "
     r"it raised giving by about (\d+\.\d) in 100, against (\d+\.\d) for the best of two simple rules \(most recent "
     r"donors first\)\. The range on that gap runs from the model about (\d+) in 100 behind to about (\d+) in 100 ahead",
     lambda: [_get("uplift_karlan_list.everyone"), _get("uplift_karlan_list.top30pct.model"),
              _get("uplift_karlan_list.top30pct.rule"), -_get("uplift_karlan_list.top30pct.diff_lo"),
              _get("uplift_karlan_list.top30pct.diff_hi")]),
    # Lapse
    ("lapse.md", r"about (\d+) out of every 100 donors gave nothing to the next mailing",
     lambda: [_get("lapse_kdd98.base_rate_pct")]),
    ("lapse.md", r"Of our top 10% of picks, (\d+) out of every 100 lapsed\. The best of the simple rules we compare "
     r"against here, years since the donor's last gift, found (\d+) out of every 100\. Picking at random also finds "
     r"about (\d+) out of every 100",
     lambda: [_get("lapse_kdd98.top10pct.model"), _get("lapse_kdd98.top10pct.rule"),
              _get("lapse_kdd98.base_rate_pct")]),
    ("lapse.md", r"at the top 5% of that ranking, (\d+) out of every 100 gave again, against (\d+) out of every 100 "
     r"for the same rule inverted",
     lambda: [_get("lapse_kdd98_retention.top5pct.model"), _get("lapse_kdd98_retention.top5pct.rule")]),
    ("lapse.md", r"At the top 10% the model is also ahead \((\d+\.\d) vs (\d+\.\d)\); at the top 1% the two are too "
     r"close to call \((\d+) vs (\d+)\)\. Picking at random finds about (\d+) out of every 100",
     lambda: [_get("lapse_kdd98_retention.top10pct.model"), _get("lapse_kdd98_retention.top10pct.rule"),
              _get("lapse_kdd98_retention.top1pct.model"), _get("lapse_kdd98_retention.top1pct.rule"),
              100 - _get("lapse_kdd98.base_rate_pct")]),
    ("lapse.md", r"About (\d+) out of every 100 of them gave nothing in the following fiscal year",
     lambda: [_get("lapse_donorschoose.metadata.base_rate_pct")]),
    ("lapse.md", r"Of the model's top 10% of picks, (\d+) out of every 100 lapsed\. The best simple rule found (\d+) "
     r"out of every 100\. About the same as the rule here\. The two are level at the top 1% too \((\d+) "
     r"vs (\d+)\)",
     lambda: [_get("lapse_donorschoose.top10pct.model"), _get("lapse_donorschoose.top10pct.rule"),
              _get("lapse_donorschoose.top1pct.model"), _get("lapse_donorschoose.top1pct.rule")]),
    ("lapse.md", r"At the top 10% of that ranking, (\d+) out of every 100 gave again, against (\d+) out of every 100 "
     r"for the same rule inverted\. At the top 1%, (\d+) out of 100 against (\d+) out of 100 for the rule\. Picking "
     r"at random finds about (\d+) out of every 100",
     lambda: [_get("lapse_donorschoose_retention.top10pct.model"), _get("lapse_donorschoose_retention.top10pct.rule"),
              _get("lapse_donorschoose_retention.top1pct.model"), _get("lapse_donorschoose_retention.top1pct.rule"),
              100 - _get("lapse_donorschoose.metadata.base_rate_pct")]),
    ("lapse.md", r"\(top 10% lapse hit rate: (\d+) out of 100 with it, (\d+) without\)\. === .KDD Cup 1998",
     lambda: [_get("lapse_donorschoose_momentum.top10pct.model"), _get("lapse_donorschoose.top10pct.model")]),
    ("lapse.md", r"between ([\d,]+) and ([\d,]+) households per test wave\. Unlike the donor files in the other tabs, lapsing "
     r"is a minority outcome here: about (\d+) out of every 100 households lapsed",
     lambda: [min(_get("lapse_psid.metadata.n_per_fold")), max(_get("lapse_psid.metadata.n_per_fold")),
              _get("lapse_psid.metadata.base_rate_pct")]),
    ("lapse.md", r"Of the model's top 10% of picks, (\d+) out of every 100 lapsed\. The best simple rule \(the smallest "
     r"giving this wave first, the best of the 3 rules checked here in all (\d+) test waves\) found (\d+) out of "
     r"every 100\. The model was ahead in every test wave\. At the top 1% it found (\d+) against (\d+)\.",
     lambda: [_get("lapse_psid.top10pct.model"), len(_folds("lapse_psid")), _get("lapse_psid.top10pct.rule"),
              _get("lapse_psid.top1pct.model"), _get("lapse_psid.top1pct.rule")]),
    ("lapse.md", r"about (\d+) out of every 100 households gave again\. Of the 10% the model ranks least likely to "
     r"lapse, (\d+) out of every 100 gave again, against (\d+) for the best rule",
     lambda: [100 - _get("lapse_psid.metadata.base_rate_pct"), _get("lapse_psid_retention.top10pct.model"),
              _get("lapse_psid_retention.top10pct.rule")]),
    ("lapse.md", r"At the top 5% it is (\d+) against (\d+), and at the top 1% (\d+) against (\d+)\.",
     lambda: [_get("lapse_psid_retention.top5pct.model"), _get("lapse_psid_retention.top5pct.rule"),
              _get("lapse_psid_retention.top1pct.model"), _get("lapse_psid_retention.top1pct.rule")]),
    ("lapse.md", r"\(top 10% lapse hit rate: (\d+) out of 100 with it, (\d+) without\)\. Only aggregates",
     lambda: [_get("lapse_psid_momentum.top10pct.model"), _get("lapse_psid.top10pct.model")]),
    ("lapse.md", r"\((\d+) out of 100\), DonorsChoose Open Data, where most multi-year donors still lapse \((\d+) out of 100\), "
     r"and the PSID household survey, where it is a minority outcome \(about (\d+) out of 100\)",
     lambda: [_get("lapse_kdd98.base_rate_pct"), _get("lapse_donorschoose.metadata.base_rate_pct"),
              _get("lapse_psid.metadata.base_rate_pct")]),
    # Suggested ask
    ("ask.md", r"Of every 100 suggested amounts, (\d+) landed within 25% of what the donor actually gave\. The best "
     r"of the simple rules .*? for (\d+) out of every 100",
     lambda: [_get("ask_kdd98.within25pct_model"), _get("ask_kdd98.within25pct_last_gift")]),
    ("ask.md", r"Of every 100 suggested amounts, (\d+) landed within 25% of what the donor actually gave the "
     r"following year\. The best of the 3 simple rules checked here \(last period's total alone, the winner in all "
     r"(\d+) test years\) landed within 25% for (\d+) out of every 100",
     lambda: [_get("ask_donorschoose.within25pct_model"), len(_folds("ask_donorschoose")),
              _get("ask_donorschoose.within25pct_last_gift")]),
    ("ask.md", r"\((\d+) out of 100 within 25% with it, versus (\d+) without\)\. Aggregates only",
     lambda: [_get("ask_donorschoose_momentum.within25pct_model"), _get("ask_donorschoose.within25pct_model")]),
    ("ask.md", r"between ([\d,]+) and ([\d,]+) households per test wave\. Of every 100 suggested amounts, (\d+) "
     r"landed .*? for (\d+) out of every 100",
     lambda: [min(_get("ask_psid.metadata.n_per_fold")), max(_get("ask_psid.metadata.n_per_fold")),
              _get("ask_psid.within25pct_model"), _get("ask_psid.within25pct_last_gift")]),
    ("ask.md", r"On average it was off by \$([\d,]+), against \$([\d,]+) for the rule",
     lambda: [_get("ask_psid.mae_model"), _get("ask_psid.mae_rule")]),
    ("ask.md", r"\((\d+) out of 100 within 25% with it, versus (\d+) without\)\. Only aggregates",
     lambda: [_get("ask_psid_momentum.within25pct_model"), _get("ask_psid.within25pct_model")]),
    ("ask.md", r"about (\d+) in 100 within 25%, versus about (\d+) in 100",
     lambda: [_get("ask_synthetic.within25pct_model"), _get("ask_synthetic.within25pct_last_gift")]),
    ("ask.md", r"held-out 30% of the donors: (\d+) who gave\. Of every 100 predicted amounts, (\d+) landed within 25% "
     r"of the actual gift\. The best rule, the donor's largest past gift, landed within 25% for (\d+) out of every "
     r"100\. On average the model was off by \$(\d+), against \$(\d+) for the rule",
     lambda: [_get("ask_karlan_list.metadata.n_test"), _get("ask_karlan_list.within25pct_model"),
              _get("ask_karlan_list.within25pct_last_gift"), _get("ask_karlan_list.mae_model"),
              _get("ask_karlan_list.mae_rule")]),
    ("ask.md", r"\| KDD Cup 1998 \| \$(\d+) of every \$100 \| \$(\d+) \|",
     lambda: [_get("ask_kdd98.revenue_top10pct_model"), _get("ask_kdd98.revenue_top10pct_rule")]),
    ("ask.md", r"\| DonorsChoose \| \$(\d+) of every \$100 \| \$(\d+) \|",
     lambda: [_get("ask_donorschoose.revenue_top10pct_model"), _get("ask_donorschoose.revenue_top10pct_rule")]),
    ("ask.md", r"\| PSID household survey \| \$(\d+) of every \$100 \| \$(\d+) \|",
     lambda: [_get("ask_psid.revenue_top10pct_model"), _get("ask_psid.revenue_top10pct_rule")]),
    ("ask.md", r"\| Karlan and List \| \$(\d+) of every \$100 \| \$(\d+) \|",
     lambda: [_get("ask_karlan_list.revenue_top10pct_model"), _get("ask_karlan_list.revenue_top10pct_rule")]),
    # "What this means for your file": counts for a 10,000-donor file, to the nearest 10
    ("leadership.md", r"first 1,000 names include about ([\d,]+) who reach \$1,000 the next wave; ranking by this "
     r"wave's giving finds about ([\d,]+)",
     lambda: [_per_10k(_get("upgrade_psid.top10pct.model")), _per_10k(_get("upgrade_psid.top10pct.rule"))]),
    ("leadership.md", r"first 100 names out of 10,000 include about (\d+) upgraders, against about (\d+) for the rule",
     lambda: [_get("upgrade_donorschoose.top1pct.model"), _get("upgrade_donorschoose.top1pct.rule")]),
    ("response.md", r"first 1,000 names include about (\d+) who give; the best simple rule's first 1,000 include about "
     r"(\d+)",
     lambda: [_get("response_karlan_list.top10pct.model") * 10, _get("response_karlan_list.top10pct.rule") * 10]),
    ("lapse.md", r"first 1,000 names include about ([\d,]+) who stop giving; ranking by the smallest giving first "
     r"finds about ([\d,]+)",
     lambda: [_per_10k(_get("lapse_psid.top10pct.model")), _per_10k(_get("lapse_psid.top10pct.rule"))]),
    ("lapse.md", r"of the 1,000 donors the model ranks least likely to lapse, about ([\d,]+) give again, against about "
     r"([\d,]+) for the rule",
     lambda: [_per_10k(_get("lapse_donorschoose_retention.top10pct.model")),
              _per_10k(_get("lapse_donorschoose_retention.top10pct.rule"))]),
    ("ask.md", r"the simple rule lands within 25% of the next gift for about ([\d,]+) donors and the model for about "
     r"([\d,]+)",
     lambda: [round(_get("ask_kdd98.within25pct_last_gift") * 100, -1), round(_get("ask_kdd98.within25pct_model") * 100, -1)]),
    ("who_to_mail.md", r"the model skips about ([\d,]+) letters and still brings in about \$([\d,]+) more",
     lambda: [round((_pieces("who_to_mail_kdd98")[1] - _pieces("who_to_mail_kdd98")[0]) / _pieces("who_to_mail_kdd98")[1]
                    * 10_000, -1),
              round((_get("who_to_mail_kdd98.net_revenue_model") - _get("who_to_mail_kdd98.net_revenue_mail_everyone"))
                    / _pieces("who_to_mail_kdd98")[1] * 10_000, -1)]),
    # How sure are we?
    ("intervals.md", r"checked them on the held-out 30%: ([\d,]+) donors who gave\. Asked to hold 90 out of 100 gifts, "
     r"the range held (\d+) out of 100\. The typical 90% range was (\d+) dollars wide",
     lambda: [_n_test("interval_kdd98"), _lv("kdd98")["attained"], _lv("kdd98")["median_width"]]),
    ("intervals.md", r"the range held (\d+) out of 100 on average, and (\d+) out of 100 in the weakest year\. "
     r"The typical 90% range was ([\d,]+) dollars wide",
     lambda: [_lv("donorschoose")["attained"], _lv("donorschoose")["attained_lo"], _lv("donorschoose")["median_width"]]),
    ("intervals.md", r"the range held (\d+) out of 100 on average, and (\d+) out of 100 in the weakest wave\. "
     r"The typical 90% range was ([\d,]+) dollars wide",
     lambda: [_lv("psid")["attained"], _lv("psid")["attained_lo"], _lv("psid")["median_width"]]),
    ("intervals.md", r"(\d+) donors who gave in the held-out 30%\. Asked to hold 90 out of 100 gifts, "
     r"the range held (\d+) out of 100\. The typical 90% range was (\d+) dollars wide",
     lambda: [_n_test("interval_karlan_list"), _lv("karlan_list")["attained"], _lv("karlan_list")["median_width"]]),
    ("intervals.md", r"On our generated donors, asked to hold 90 out of 100 gifts, the range held (\d+) out of 100",
     lambda: [_lv("synthetic")["attained"]]),
    ("intervals.md", r"between about ([\d,]+) and ([\d,]+) of their next gifts land inside it",
     lambda: [round(min(_lv(f)["attained"] for f in INTERVAL_REAL) * 100, -1),
              round(max(_lv(f)["attained"] for f in INTERVAL_REAL) * 100, -1)]),
    # Who to mail
    ("who_to_mail.md", r"We skipped ([\d,]+) of ([\d,]+) letters and still raised \$([\d,]+) more", _wtm_kdd98),
    ("who_to_mail.md", r"mailing ([\d,]+) of them, is marked.*?brings in \$([\d,]+) after costs, against \$([\d,]+) "
     r"for mailing everyone",
     lambda: [_pieces("who_to_mail_kdd98")[0], _get("who_to_mail_kdd98.net_revenue_model"),
              _get("who_to_mail_kdd98.net_revenue_mail_everyone")]),
    ("who_to_mail.md", r"mailing the ([\d,]+) of ([\d,]+) donors the model expected to be worth it brought in "
     r"\$([\d,]+) after costs, against \$([\d,]+) for mailing everyone",
     lambda: [*_pieces("who_to_mail_cup98val"), _get("who_to_mail_cup98val.net_revenue_model"),
              _get("who_to_mail_cup98val.net_revenue_mail_everyone")]),
]


@pytest.mark.parametrize("page,template,values", NUMBER_MAP, ids=[f"{p}:{i}" for i, (p, _, _) in enumerate(NUMBER_MAP)])
def test_page_number_matches_its_results_key(page, template, values):
    text = " ".join((PAGES / page).read_text().split())
    m = re.search(template, text)
    assert m, f"{page}: sentence not found: {template}"
    expected = values()
    assert len(m.groups()) == len(expected)
    for shown, value in zip(m.groups(), expected):
        digits = shown.replace(",", "")
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        assert digits == format(value, f".{decimals}f"), f"{page}: shows {shown}, results.json has {value}"


REVENUE_WORDS = {"wins": "Beats the rule", "modest": "About the same", "loses": "Loses to the rule"}


@pytest.mark.parametrize("label,key", [
    ("KDD Cup 1998", "ask_kdd98"),
    ("DonorsChoose", "ask_donorschoose"),
    ("PSID household survey", "ask_psid"),
    ("Karlan and List", "ask_karlan_list"),
])
def test_revenue_table_verdict_matches_results_json(label, key):
    text = (PAGES / "ask.md").read_text()
    m = re.search(rf"^\| {label} \| [^|]+ \| [^|]+ \| ([^|]+) \|$", text, re.M)
    assert m, f"ask.md: revenue row for {label} not found"
    assert m.group(1).startswith(REVENUE_WORDS[_get(f"{key}.revenue_top10pct_verdict")])


def test_bottom_line_rule(mrp):
    def cell(verdict, org="A", folds=1, top=False, retention=False, counted=True):
        return {"verdict": verdict, "org": org, "folds": folds, "top_slice_win": top,
                "retention": retention, "counted": counted}

    assert mrp.bottom_line([cell("wins", "A"), cell("wins", "B")]) == "Use the model"
    assert mrp.bottom_line([cell("wins", folds=4)]) == "Use the model"
    one_org = "Use the model (tested on one organisation so far)"
    assert mrp.bottom_line([cell("wins", "A"), cell("wins", "A")]) == one_org
    assert mrp.bottom_line([cell("wins")]) == one_org
    assert mrp.bottom_line([cell("wins", "A"), cell("modest", "B", top=True)]) == one_org
    assert mrp.bottom_line([cell("modest", top=True), cell("modest")]) == "Use the model for your top slice only"
    assert mrp.bottom_line([cell("wins", "A"), cell("wins", "B"), cell("loses", "C")]) == "Can't tell yet"
    assert mrp.bottom_line([cell("modest", top=True), cell("loses")]) == "Use the simple rule"
    assert mrp.bottom_line([cell("modest"), cell("loses")]) == "Use the simple rule"
    assert mrp.bottom_line([cell("wins", "A", retention=True), cell("wins", "B", retention=True)]) == (
        "Use the retention list"
    )
    assert mrp.bottom_line([cell("wins", "A", retention=True), cell("wins", "B", folds=4)]) == "Use the model"
    # A proxy-question cell is shown but never counted, so its loss does not block a win.
    assert mrp.bottom_line([cell("loses", counted=False), cell("wins", "B", folds=4)]) == "Use the model"
    assert mrp.bottom_line([cell("loses", counted=False)]) == "Can't tell yet"
    assert mrp.bottom_line([{"verdict": None, "counted": False}]) == "Can't tell yet"
