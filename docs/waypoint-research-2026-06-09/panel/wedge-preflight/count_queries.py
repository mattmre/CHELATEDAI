import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
from datasets import load_dataset

for name in ["mteb/scifact", "mteb/nfcorpus", "mteb/fiqa", "mteb/arguana"]:
    try:
        ds = load_dataset(name, "queries")["queries"]
        print(f"{name}: total queries in 'queries' config = {len(ds)}")
    except Exception as e:
        print(f"{name}: queries config failed -> {type(e).__name__}: {str(e)[:120]}")
    for cfg in ["default"]:
        try:
            d = load_dataset(name, cfg)
            for split in d:
                qs = set(d[split]["query-id"]) if "query-id" in d[split].column_names else None
                print(f"   {name} [{cfg}/{split}] rows={len(d[split])} "
                      f"unique_query_ids={len(qs) if qs is not None else 'n/a'}")
        except Exception as e:
            print(f"   {name} [{cfg}] failed -> {type(e).__name__}: {str(e)[:120]}")
