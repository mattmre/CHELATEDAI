#!/usr/bin/env bash
# ChelatedAI related-works research harness.
# Deterministically validates the research-plan artifacts and reports
# coverage of the repo's core claims by catalogued related works.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLAN_DIR="$HERE/research_plan"

# --- 1. Validate inputs exist ------------------------------------------------
for f in RESEARCH_PLAN.md claims.tsv related_works.tsv; do
  if [[ ! -s "$PLAN_DIR/$f" ]]; then
    echo "FATAL: $PLAN_DIR/$f missing or empty" >&2
    exit 1
  fi
done

# --- 2. Load claims (id, track, text) ----------------------------------------
declare -A CLAIM_TRACK CLAIM_TEXT
while IFS=$'\t' read -r cid track text; do
  [[ "$cid" =~ ^#.*$ || -z "$cid" || "$cid" == "cid" ]] && continue
  if [[ -z "$track" || -z "$text" ]]; then
    echo "FATAL: claims.tsv row '$cid' malformed" >&2
    exit 1
  fi
  CLAIM_TRACK["$cid"]="$track"
  CLAIM_TEXT["$cid"]="$text"
done < "$PLAN_DIR/claims.tsv"

# --- 3. Parse related-works catalog; validate + score ------------------------
# Columns: id | category | strength | title | authors | ref | mapped_claim_ids | notes
declare -A WORK_STRENGTH CAT_STRONG
declare -a ALL_WORKS
n_total=0 n_strong=0
declare -i n_total=0
declare -i n_strong=0
while IFS=$'\t' read -r wid category strength title authors ref claims notes; do
  [[ "$wid" =~ ^#.*$ || -z "$wid" || "$wid" == "id" ]] && continue
  n_total=$((n_total+1))
  case "$category" in
    relevant|cousin|structural_duplicate|additive|math_foundation) : ;;
    *) echo "FATAL: $wid invalid category '$category'" >&2; exit 1 ;;
  esac
  case "$strength" in
    strong|supporting|weak) : ;;
    *) echo "FATAL: $wid invalid strength '$strength'" >&2; exit 1 ;;
  esac
  if [[ -z "$title" || -z "$ref" ]]; then
    echo "FATAL: $wid missing title/ref" >&2
    exit 1
  fi
  if [[ "$strength" == "strong" ]]; then
    n_strong=$((n_strong+1))
    # every mapped claim must exist
    IFS=';' read -r -a mapped <<< "${claims//[[:space:]]/}"
    for m in "${mapped[@]}"; do
      [[ -z "$m" ]] && continue
      if [[ -z "${CLAIM_TRACK[$m]+x}" ]]; then
        echo "FATAL: $wid maps to unknown claim '$m'" >&2
        exit 1
      fi
    done
  fi
  ALL_WORKS+=("$wid")
done < "$PLAN_DIR/related_works.tsv"

# --- 4. Coverage = claims hit by at least one strong work --------------------
declare -A COVERED
while IFS=$'\t' read -r wid category strength title authors ref claims notes; do
  [[ "$wid" =~ ^#.*$ || -z "$wid" || "$wid" == "id" ]] && continue
  [[ "$strength" == "strong" ]] || continue
  IFS=';' read -r -a mapped <<< "${claims//[[:space:]]/}"
  for m in "${mapped[@]}"; do
    [[ -n "$m" ]] && COVERED["$m"]=1
  done
done < "$PLAN_DIR/related_works.tsv"

n_claims=${#CLAIM_TRACK[@]}
n_covered=${#COVERED[@]}
if (( n_claims > 0 )); then
  coverage=$(awk -v c="$n_covered" -v t="$n_claims" 'BEGIN { printf "%.1f", 100.0*c/t }')
else
  coverage=0.0
fi

# --- 5. Rich-mix check: at least one structural_duplicate considered? --------
n_dup=0
while IFS=$'\t' read -r wid category strength title authors ref claims notes; do
  [[ "$wid" =~ ^#.*$ || -z "$wid" || "$wid" == "id" ]] && continue
  [[ "$category" == "structural_duplicate" ]] && n_dup=$((n_dup+1))
done < "$PLAN_DIR/related_works.tsv"

echo "METRIC claim_coverage=$coverage"
echo "METRIC related_works_total=$n_total"
echo "METRIC related_works_strong=$n_strong"
echo "METRIC structural_duplicates_considered=$n_dup"

exit 0
