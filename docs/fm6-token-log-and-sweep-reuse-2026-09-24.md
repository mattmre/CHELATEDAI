# FM-6 token logs and sweep engine reuse (2026-09-24)

Checked on `4bb3685`.

## Query tokens

Authorization reads the `Authorization: Bearer` header only. `do_GET` does not accept `?token=`. The stdlib request log still printed `self.requestline`, so `GET /api/summary?token=secret HTTP/1.1` would have landed in stdout. `DashboardHandler.log_message` now runs those strings through `redact_url_credentials`, which replaces `token`, `access_token`, and `auth` query values with `[redacted]`.

`test_query_token_is_not_accepted_as_authorization` and `test_log_message_redacts_query_token` cover that. A browser session was not opened.

## V-02 sweep engine reuse

`run_sedimentation_cycle` does write vectors back. After training it calls `sync_vectors_to_qdrant` (`antigravity_engine.py` just before `run_inference`), and `sedimentation_trainer.sync_vectors_to_qdrant` calls `qdrant.upsert`. The log line "Updating corpus vectors in Qdrant" belongs to `run_offline_distillation`. That is a second writer, not the only one. An earlier draft of this note said sedimentation does not upsert. That sentence was wrong.

The sweep still keeps one `AntigravityEngine` so it does not open a second Qdrant client on the same path. Before each configuration replaces the adapter, `restore_collection` upserts the snapshot taken after the baseline evaluation. The configuration's own post-score still sees the vectors that sedimentation just wrote. The next configuration starts from the snapshot again.

## Still open

Phase I step 1 has no projection training loss on the teacher-target path. Step 2 still uses other in-batch targets as negatives. A browser 401 fire and a Spark BEIR run were not executed. Rung 15 was not started.
