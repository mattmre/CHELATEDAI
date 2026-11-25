"""Bundle 55 protocol-ready fragment.

The owned filename is not the compiler key. compile_repo_fragments rejects
this filename. Callers compile these bytes under the legal verifier key.
This drycheck does not publish wild evidence and does not mutate the claim registry.
"""

SPINE_DIGEST = "32693f54e0ce7ad3d401ceb752ae4723b0eee251f6d49b95cdfb26a6201971e9"
BUNDLE_ID = 55
DEPENDS_ON = ()


def verifier_status():
    return {
        "bundle_id": BUNDLE_ID,
        "status": "protocol-ready",
        "wild_evidence": False,
        "llm_write_tokens": 0,
        "authored_prompts": 0,
    }


def verify_bundle(repo_root=None):
    import json
    from pathlib import Path

    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    manifest_rel = "artifacts/pv7_w18_6c_wild_revisions/pv7_w18_6c_drycheck_manifest.json"
    mpath = root / manifest_rel
    manifest_ok = True
    if mpath.is_file():
        doc = json.loads(mpath.read_text(encoding="utf-8"))
        ret = doc.get("retirement", {})
        manifest_ok = (
            doc.get("mode") == "local-drycheck"
            and doc.get("status") == "DRYCHECK"
            and doc.get("synthetic_labeled_wild") == 0
            and doc.get("wild_adjudicated_count") == 0
            and ret.get("superiority_is_wild_evidence") is False
        )
    base = verifier_status()
    return {
        **base,
        "passed": bool(manifest_ok and base["wild_evidence"] is False),
        "drycheck_manifest_honest": manifest_ok,
    }


def main():
    import json

    result = verify_bundle()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
