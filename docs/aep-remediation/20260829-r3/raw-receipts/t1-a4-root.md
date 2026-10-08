# T1 A4 root backend/reliability receipt

Target: root dirty worktree at `65c9085cd048e8a7351a53e87666fdd5639e612b`. No product file was edited.

## QSCCI dual-failure probe

The exact executable probe is preserved in `findings/AEP-20260829-r3-WT001-001.md`. Controlling output:

```text
{'raised_type': 'QSCCIError', 'raised_text': 'service restoration failure: QSCCIError: SECONDARY_RESTORE_FAILURE', 'cause': 'None', 'context': 'None', 'quarantine_members': [[]], 'contains_primary': False}
Ran 2 tests in 0.020s
OK
```

Disposition: net-new Medium boundary finding `WT001-001`. Round 2 `WT-001` remains the separate controlling Critical for restoration before child terminal/reaped proof.

## Paired-intervention bounded check

```powershell
python -B -m unittest tests.test_paired_intervention_experiments -v
```

```text
Ran 25 tests in 17.478s
OK
```

No new finding was derived from that module. This focused pass does not accept the root cohort or the repository aggregate.
