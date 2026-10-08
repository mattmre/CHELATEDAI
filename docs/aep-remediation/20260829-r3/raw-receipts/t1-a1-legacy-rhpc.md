# T1 A1 legacy and RHPC worktree receipt

Scope: the `h2-rerun`, EGV training, and RHPC1 dirty worktrees. No file was written and Tier 2 PR material was not opened.

## Stable worktree results

```text
h2-rerun HEAD=6c3e1847830ec231157a421735a5a34d7d71a658 paths=24
EGV-TRAINING HEAD=97a8237e3f1b92425b6802895c1fcbdeafeb2681 visible_environment_paths=3898
RHPC1 HEAD=305986427850dcb393a6078900201a7aa84c5b33 dirty_paths=13
```

- The 22 deleted h2 results and two deleted manifests reproduce Round 1 `WT003-001` without a new cause.
- `.venv-egv-prod/` remains visible because the tracked ignore rule covers `venv/`, reproducing `WT019-001`.
- RHPC tracked-to-untracked wiring, source identity, predecessor reread, and check/replace publication reproduce `WT020-001` through `WT020-004`.

## RHPC matched-control probe

```powershell
$wt = 'D:\GITHUB\CHELATEDAI-RHPC1'
Push-Location $wt
try {
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $probe = @'
from rhpc.stage_a import StageAConfig, _codebook, _permutation_parameters, _encode_path, _corrupt_code, _route_from_trial, _initial_hint, _resolve_route, _method_metrics
for run_id in (7, 11):
    c = StageAConfig(run_id=run_id)
    codes = _codebook(c)
    ordered = _permutation_parameters(c, ordered=True)
    orderless = _permutation_parameters(c, ordered=False)
    current, matched = [], []
    for i in range(c.trials):
        truth = _route_from_trial(c, i)
        initial, margins = _initial_hint(c, i, truth)
        current.append((truth, _resolve_route(_corrupt_code(_encode_path(truth, codes, ordered), c, i), codes, orderless, initial, margins, c)))
        matched.append((truth, _resolve_route(_corrupt_code(_encode_path(truth, codes, orderless), c, i), codes, orderless, initial, margins, c)))
    a = _method_metrics(current)["path_accuracy"]
    b = _method_metrics(matched)["path_accuracy"]
    print({"run_id": run_id, "current_cross_encoding_accuracy": a, "matched_no_permutation_accuracy": b, "current_gate_advantage": 1.0-a, "matched_gate_advantage": 1.0-b})
'@
    python -B -c $probe
} finally { Pop-Location }
```

```text
{'run_id': 7, 'current_cross_encoding_accuracy': 0.0625, 'matched_no_permutation_accuracy': 1.0, 'current_gate_advantage': 0.9375, 'matched_gate_advantage': 0.0}
{'run_id': 11, 'current_cross_encoding_accuracy': 0.0546875, 'matched_no_permutation_accuracy': 1.0, 'current_gate_advantage': 0.9453125, 'matched_gate_advantage': 0.0}
```

Source inspection also found `_decode_canonical(result_path, ...)` followed by a second `result_path.read_bytes()` in `rhpc/artifacts.py:244-245`. Dispositions: net-new `WT025-001` and `WT025-002`; prior packets remain duplicates.
