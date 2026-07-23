# Adversarially review the new test file tests/test_sparse_local_preflight.py

A fresh implementer just wrote it. Try to DISPROVE that it provides real coverage:
- Are the non-affine / non-radial assertions real (fit-an-affine-and-check-residual) or tautological?
- Would the clean-ID-leakage test actually CATCH a leak (behavioral check), or does it only read a flag?
- Does the sanity ladder genuinely order global < oracle-ID local, with meaningful margins?
- RUN the suite yourself (python -m unittest research.drift_recovery.tests.test_sparse_local_preflight -v)
  and confirm it passes; also run the two pre-existing D2 suites (test_d2_decisions, test_synthetic_collapse)
  to confirm no regression.
Return PASS / PASS-WITH-FIXES / FAIL with file:line specifics and a one-line bottom line.
