## Summary
`philanthropy check` and `philanthropy train --task leadership` only accepted the four sources hardcoded in the CLI. This adds Bloomerang, DonorPerfect and Blackbaud CRM, so both commands accept every `read_gifts` preset and read each one through the same row filter `features` uses.

## Changes
- `_bloomerang.py`, `_donorperfect.py`, `_blackbaud_crm.py`: added private `_<crm>_received_rows` helpers (same pattern as the #337 helpers), and their `*_to_features` functions now call them.
- `cli.py`: added a `_raw_gift_readers()` registry covering every `ingest.GIFT_SOURCES` name; `_read_raw_gifts` looks the source up there instead of an if/elif chain (an unknown source now fails loudly instead of falling through to CiviCRM). `check` and `train --source` choices are now `GIFT_SOURCES`.
- `tests/test_cli.py`: `test_raw_gift_log_drops_the_rows_features_drops` now covers all seven sources; added `test_raw_gift_readers_cover_every_gift_source` and an unknown-source rejection test.
- `CHANGELOG.md`/`CONTRIBUTORS.md`: entry and credit.

## Testing
```sh
python -m pytest tests/test_cli.py tests/test_ingest.py -q
python -m mypy philanthropy
python -m flake8 philanthropy/cli.py philanthropy/ingest/_bloomerang.py philanthropy/ingest/_donorperfect.py philanthropy/ingest/_blackbaud_crm.py tests/test_cli.py
```
- 94 tests passed
- mypy: no issues in 70 source files
- flake8: clean

## Related
- Closes #341

## Checklist
- [x] Duplicate check passed (no competing PR; issue unclaimed)
- [x] Tests added/updated for the change
- [x] Lint/format passes
- [x] No unrelated changes
- [x] Docs updated if behavior changed
