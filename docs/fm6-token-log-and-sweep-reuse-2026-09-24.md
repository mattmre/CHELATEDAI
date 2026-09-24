# FM-6 token logs and sweep engine reuse (2026-09-24)

Checked on `4bb3685`.

## Query tokens

Authorization reads the `Authorization: Bearer` header only. `do_GET` does not accept `?token=`. The stdlib request log still printed `self.requestline`, so `GET /api/summary?token=secret HTTP/1.1` would have landed in stdout. `DashboardHandler.log_message` now runs those strings through `redact_url_credentials`, which replaces `token`, `access_token`, and `auth` query values with `[redacted]`.

`test_query_token_is_not_accepted_as_authorization` and `test_log_message_redacts_query_token` cover that. A browser session was not opened.

## V-02 sweep engine reuse

`run_large_sweep.py` and `run_sweep.py` still set `engine = base_engine` inside the config loop and say that reuse avoids a Qdrant file lock. Each iteration replaces `engine.adapter`, clears `chelation_log`, and restores the noise and push-magnitude config.

`run_sedimentation_cycle` does not upsert into Qdrant. The upsert under "Updating corpus vectors in Qdrant" is `run_offline_distillation` (`antigravity_engine.py` around the method that starts near line 2092). The sweep calls `run_sedimentation_cycle`, not that offline updater. Reuse therefore does not carry sedimentation-written vectors from one config to the next. This note does not change that reuse. Opening a second Qdrant client on the same path is the lock the comment is avoiding.

## Still open

Phase I step 1 has no projection training loss on the teacher-target path. Step 2 still uses other in-batch targets as negatives. A browser 401 fire and a Spark BEIR run were not executed. Rung 15 was not started.
