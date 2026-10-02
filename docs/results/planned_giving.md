# Planned giving

**Not yet tested on real bequest data.** No public dataset we ship, or know of,
records which donors actually left a bequest, so there is nothing honest to
report here yet: no accuracy number, no chart. Anything we showed here would
be scoring the model against a stand-in for bequest intent, not bequest
intent itself, and would look like proof it isn't.

## What the model would need

`PlannedGivingIntentScorer` needs a training file where donors are labelled
with whether they made a planned gift (a bequest, a charitable trust, a
beneficiary designation), not just whether they kept giving. That label
usually only exists inside a gift officer's own CRM, built up over years of
identifying bequest intentions and cross-checking against the DMS/wealth
data field for legacy giving. There is also no established simple rule for
planned-giving prospects to compare against, the way "rank by giving so far"
exists for major gifts, so a real test needs both a labelled file and someone
to define what "beating no model" looks like for this question.

## How to test it on your own file

If your shop tracks planned-giving intent (even informally, as a flag or a
moves-management stage), you can fit and check the model yourself:

```python
import numpy as np
from philanthropy.models import PlannedGivingIntentScorer

# Replace with your own donor feature matrix and bequest-intent flags
# (1 = made a planned gift, 0 = did not); this uses random arrays only to
# show the API runs end to end.
rng = np.random.default_rng(0)
X_train, y_bequest_train = rng.random((200, 3)), rng.integers(0, 2, 200)
X_test, y_bequest_test = rng.random((50, 3)), rng.integers(0, 2, 50)

model = PlannedGivingIntentScorer(random_state=0).fit(X_train, y_bequest_train)
score = model.predict_intent_score(X_test)
```

Compare `score`'s ranking against `y_bequest_test` using whatever top-N hit
rate matters to your program, the same way the other Results pages do. If
you're willing to share an anonymized version of that comparison, it would
let this page report a real number instead of "untested."

## What the model looks at

The drivers below come from the same giving-response stand-in label used
above, not real bequest intent; read them as "what the estimator weighs when
fit on this kind of label," not as a finding about who actually leaves a
bequest.

--8<-- "results/_features/planned_giving__planned_giving_synthetic.md"
