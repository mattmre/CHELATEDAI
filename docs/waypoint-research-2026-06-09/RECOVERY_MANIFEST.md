# Waypoint research recovery manifest

Status: recovered ignored research; not merged acceptance evidence

Recovery date: 2026-07-22

Recovery branch: `codex/recover-waypoint-research-20260722`

Base commit: `34ce4b5632e0d9cd2a16c29e0e1acc42e645b9c2`

Original path: `docs/waypoint-research-2026-06-09/`

## Preservation record

The original directory was excluded by `.git/info/exclude`; 198 of its 201 files were not represented by reachable Git blobs. It was copied byte-for-byte into an isolated worktree before staging. The original 201-file collection contains 62,042,044 bytes.

The source and recovery copy were compared by relative path and per-file SHA-256. All files matched. The aggregate copy digest is:

`sha256(sorted(relative_path + " " + file_sha256), UTF-8 with LF separators) = 5459e2116acb02dfce8f4dc16d2f6032ac8f5802bc1ab4a770180c10865d6999`

This manifest is an added 202nd file and is not part of that digest.

## Material findings preserved here

- `review-notes-cleanup-pass.md` records a full-corpus SciFact direction test: MiniLM-to-MPNet new-query-to-old-index NDCG@10 was 0.543, old-documents-to-new-space was 0.557, and the old/old baseline was 0.648. This falsified the proposed directional asymmetry; it did not test query-conditional route selection.
- `panel/rung15-cut-decision.md` records that the actual Evidence DAG gives query nodes zero in-degree and no query-to-query edges, so the proposed GNN cannot express the claimed multi-hop advantage on that graph. It identifies route assignment under encoder-swap drift as the meaningful successor.
- The wedge design and red-team records close the known-dictionary nonlinear-decoder comparison as theorem/harness validation, while leaving controlled dictionary uncertainty and sketched overcomplete-ICA identification as speculative open lanes.
- `panel/prereg-harmonic-invariance-HI1-draft.md` is explicitly `CUT - DO NOT RUN`; it is preserved as a negative design result, not a positive finding.
- The Phase-II audit separates scientific conclusions from open-PR and merge-state claims, which are time-sensitive and must be refreshed before reuse.

The panel directory also contains large raw agent/reviewer transcripts. They are retained for provenance, not treated as independent evidence.

## Safety boundary

A scan found no common AWS access-key, GitHub token, private-key header, or inline API-key/client-secret/access-token pattern in this collection. That scan is not a guarantee that the raw transcripts contain no private contextual material. Remote publication still requires explicit authorization.
