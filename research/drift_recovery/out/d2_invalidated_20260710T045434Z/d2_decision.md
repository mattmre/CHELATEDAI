# D2 decision

Overall: **NO_GLOBAL_G3_VERDICT_UNDERPOWERED_OR_UNAVAILABLE**.

G2 is applied before G3. `NO_G3_VERDICT` is not a corrector kill claim.

| Cell | G2 median half-width | Power | Harm AUPRC | G3 verdict |
|---|---:|---|---:|---|
| SciFact / 8 clusters | 0.0190 | FAIL | 0.043 | NO_G3_VERDICT |
| SciFact / 16 clusters | 0.0148 | PASS | 0.082 | KILL_CORRECTOR |
| NFCorpus / 8 clusters | 0.0276 | FAIL | 0.166 | NO_G3_VERDICT |
| NFCorpus / 16 clusters | 0.0255 | FAIL | 0.365 | NO_G3_VERDICT |

## Cell evidence

- SciFact / 8: minimum oracle gap 0.0000; decision `NO_G3_VERDICT`.
  G3 was not interpreted; rough query budget at the observed variance is 97.
- SciFact / 16: minimum oracle gap -0.0046; decision `KILL_CORRECTOR`.
- NFCorpus / 8: minimum oracle gap -0.0078; decision `NO_G3_VERDICT`.
  G3 was not interpreted; rough query budget at the observed variance is 203.
- NFCorpus / 16: minimum oracle gap -0.0091; decision `NO_G3_VERDICT`.
  G3 was not interpreted; rough query budget at the observed variance is 174.
