# Get your CRM export in

One call, `read_gifts(path_or_rows, source=...)`, turns a gift export from
any of the five CRMs below into the donor-level feature table every model
in this library expects. Pick your CRM's tab for the exact column mapping
and a runnable example; export steps vary by CRM configuration and version,
so treat them as a starting point, not exact click-by-click instructions.

=== "Raiser's Edge / RE NXT"

    Run a gift export or query (Raiser's Edge NXT: **Reports > Gift
    reports**, or **Query** for a custom pull) and include at minimum the
    constituent ID, gift date, gift amount, and gift type columns, saved as
    CSV.

    | Your column | Maps to | Notes |
    |---|---|---|
    | `Constituent ID` (or `constit_id`, `donor_id`) | `contact_id` | required |
    | `Gift Date` (or `Pledged on` for a pledge row) | `receive_date` | required |
    | `Gift Amount` | `total_amount` | required |
    | `Gift Type` | excluded via `exclude_gift_types` | pledges are excluded by default so a promise isn't double-counted with its payments |

    ```python
    from philanthropy.ingest import read_gifts

    rows = [
        {"Constituent ID": "88", "Gift Date": "2025-01-10",
         "Gift Amount": "1200.00", "Gift Type": "Pledge"},
        {"Constituent ID": "88", "Gift Date": "2025-02-10",
         "Gift Amount": "100.00", "Gift Type": "Pay-Cash"},
    ]
    features = read_gifts(rows, source="raisers_edge")
    print(features.loc["88", "total_gift_amount"])  # 100.0: the pledge is excluded
    ```

    A real export is a CSV file with these same column headers; pass its
    path instead of a list of rows: `read_gifts("gifts.csv",
    source="raisers_edge")`.

=== "Salesforce NPSP"

    Export your Opportunity report (**Reports > New Report**, an Opportunity
    report type) with the account/contact key, close date, amount, and
    stage columns, saved as CSV.

    | Your column | Maps to | Notes |
    |---|---|---|
    | `Account ID` (or `AccountId`, `Primary Contact`) | `contact_id` | required |
    | `Close Date` | `receive_date` | required |
    | `Amount` | `total_amount` | required |
    | `Stage` | filtered via `include_stages` | only `Closed Won` counts as money received by default; `Pledged` is excluded |

    ```python
    from philanthropy.ingest import read_gifts

    rows = [
        {"Account ID": "88", "Close Date": "2025-01-10",
         "Amount": "100.00", "Stage": "Pledged"},
        {"Account ID": "88", "Close Date": "2025-02-10",
         "Amount": "100.00", "Stage": "Closed Won"},
    ]
    features = read_gifts(rows, source="npsp")
    print(features.loc["88", "total_gift_amount"])  # 100.0: the pledge is excluded
    ```

    A real export is a CSV file with these same column headers; pass its
    path instead of a list of rows: `read_gifts("gifts.csv", source="npsp")`.

=== "Bloomerang"

    Export your transaction list (**Transactions > Export**) with the
    account number, date, amount, and transaction type columns, saved as
    CSV.

    | Your column | Maps to | Notes |
    |---|---|---|
    | `Account Number` (or `AccountId`) | `contact_id` | required |
    | `Date` | `receive_date` | required |
    | `Amount` | `total_amount` | required |
    | `Transaction Type` | filtered via `exclude_entry_types` | `Pledge` rows are excluded by default so the promise isn't double-counted with its `PledgePayment` rows |

    ```python
    from philanthropy.ingest import read_gifts

    rows = [
        {"Account Number": "88", "Date": "2025-01-10",
         "Amount": "1200.00", "Transaction Type": "Pledge"},
        {"Account Number": "88", "Date": "2025-02-10",
         "Amount": "100.00", "Transaction Type": "PledgePayment"},
    ]
    features = read_gifts(rows, source="bloomerang")
    print(features.loc["88", "total_gift_amount"])  # 100.0: the pledge is excluded
    ```

    A real export is a CSV file with these same column headers; pass its
    path instead of a list of rows: `read_gifts("gifts.csv",
    source="bloomerang")`.

=== "DonorPerfect"

    Export a gift report (**Reports > Report Writer**, or the "Gift/Pledge
    detail" canned report) with the donor ID, gift date, gift amount, and
    record type columns, saved as CSV.

    | Your column | Maps to | Notes |
    |---|---|---|
    | `Donor ID` (or `DonorID`) | `contact_id` | required |
    | `Gift Date` | `receive_date` | required |
    | `Gift Amount` | `total_amount` | required |
    | `Record Type` | filtered via `exclude_record_types` | `P` (Pledge) and `M` (a split gift's Main total) are excluded by default; `G` (a regular gift or pledge payment) counts |

    ```python
    from philanthropy.ingest import read_gifts

    rows = [
        {"Donor ID": "88", "Gift Date": "2025-01-10",
         "Gift Amount": "1200.00", "Record Type": "P"},
        {"Donor ID": "88", "Gift Date": "2025-02-10",
         "Gift Amount": "100.00", "Record Type": "G"},
    ]
    features = read_gifts(rows, source="donorperfect")
    print(features.loc["88", "total_gift_amount"])  # 100.0: the pledge is excluded
    ```

    A real export is a CSV file with these same column headers; pass its
    path instead of a list of rows: `read_gifts("gifts.csv",
    source="donorperfect")`.

=== "CiviCRM"

    Export a contribution search result (**Search > Find Contributions**,
    or the equivalent APIv4 `Contribution.get` call) with the contact ID,
    contribution date, total amount, and contribution status columns, saved
    as CSV.

    | Your column | Maps to | Notes |
    |---|---|---|
    | `Contact ID` | `contact_id` | required |
    | `Contribution Date` (or `receive_date` from an APIv4 pull) | `receive_date` | required |
    | `Total Amount` | `total_amount` | required |
    | `Contribution Status` | filtered via `statuses` | only `Completed` counts by default; `Failed`, `Cancelled`, and pending statuses are excluded |

    ```python
    from philanthropy.ingest import read_gifts

    rows = [
        {"Contact ID": "101", "Contribution Date": "2025-01-15",
         "Total Amount": "250.00", "Contribution Status": "Completed"},
        {"Contact ID": "101", "Contribution Date": "2025-06-02",
         "Total Amount": "99.00", "Contribution Status": "Failed"},
    ]
    features = read_gifts(rows, source="civicrm")
    print(features.loc["101", "total_gift_amount"])  # 250.0: the failed contribution is excluded
    ```

    A real export is a CSV file with these same column headers; pass its
    path instead of a list of rows: `read_gifts("gifts.csv",
    source="civicrm")`.

## Next step

`read_gifts` returns one row per donor, indexed by `contact_id`, with
columns like `total_gift_amount`, `gift_count`, and `recency_days`. Feed
that straight into a model, e.g. `DonorPropensityModel` or
`LapsePredictor`, the same way the [Results](../results/index.md) pages do,
or see [Use the CLI](use_the_cli.md) to run this from the command line
without writing any code.
