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
