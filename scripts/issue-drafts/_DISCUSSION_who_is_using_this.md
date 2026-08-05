# Draft: pinned Discussion — "Are you using PhilanthroPy?"

Not posted. Create it under the **General** category at
<https://github.com/PhilanthroPy-Project/PhilanthroPy/discussions/new?category=general>,
then pin it and link it from the README.

Rationale: the package carries no telemetry and never will (`tests/test_no_network.py`
enforces that), so this thread is the only adoption signal that exists. GitHub
dependents currently read 0 repositories / 0 packages.

---

**Title:** Are you using PhilanthroPy? Tell us your org + use case

**Body:**

If you are using PhilanthroPy — or evaluated it and decided against it — please say so here. One reply is enough.

There is no telemetry in this package and there never will be ([it makes zero network calls](https://philanthropy-project.github.io/PhilanthroPy/explanation/security_review_answers/), enforced in CI), which means this thread is genuinely the only way I can know anyone is running it.

**If you are using it,** whatever you are willing to share:

- Organisation type — university, hospital foundation, independent nonprofit, consultancy, or research
- Which estimators or metrics you actually use
- Which CRM your data comes out of, and whether `philanthropy.ingest` read it or you wrote your own loader
- Anything you had to work around

**If you evaluated it and passed,** that is more useful to me than a star. Especially: was it missing a capability, was Python itself the barrier, was it a procurement or privacy review, or was your team working in R?

You are welcome to be vague about your employer. "A mid-size hospital foundation in the US" tells me what I need. Please do not post donor data, record counts tied to a named institution, or anything from a live extract.

Why I'm asking: this library has real usage signals of exactly zero right now — 0 GitHub dependents, no documented deployment. A single named user unlocks things a hundred stars cannot, starting with a JOSS submission. If you would rather not say it in public, email works too.
