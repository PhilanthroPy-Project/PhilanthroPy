# Renders docs/assets/demo.gif. Run from the repo root: vhs docs/assets/demo.tape
# Needs `pip install philanthropy` (or an editable install) on PATH. Target: under 2 MB, 20 to 30 seconds.
Output docs/assets/demo.gif
Set Shell "bash"
Set FontSize 18
Set Width 1100
Set Height 560
Set Padding 20
Set TypingSpeed 40ms

# Hidden setup: build a synthetic raw gift export (the library ships a generator, not a real file).
Hide
Type `python -c "from philanthropy.datasets import make_donor_panel as m; g=m(n_donors=400, n_years=5, random_state=0)['gifts']; g.rename(columns={'donor_id':'Constituent ID','gift_date':'Gift Date','gift_amount':'Gift Amount'})[['Constituent ID','Gift Date','Gift Amount']].assign(**{'Gift Type':'Cash'}).to_csv('/tmp/demo_gifts.csv', index=False)"`
Enter
Sleep 3s
Type "clear"
Enter
Show

Type "# A raw CRM gift export: one row per gift"
Enter
Sleep 1s
Type "head -4 /tmp/demo_gifts.csv"
Enter
Sleep 4s
Type "# Does a model beat the simple rule we already use?"
Enter
Sleep 1s
Type "philanthropy check --task lapse --source raisers_edge --data /tmp/demo_gifts.csv"
Enter
Sleep 12s
