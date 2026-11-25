import ChelatedFormal.Theorem3_EK5_CyclicWFS
import Lean.Data.Json

/-!
# `ek5_ref_cli` — Lean reference evaluator for `ek5_v3_stratified.py` (Strata 0–3)

Reads newline-delimited JSON snapshots (stdin, or `--input <path>`), one materialized `EK5` graph per
line, and prints one JSON result line per snapshot.

Snapshot schema (one JSON object per line):

```
{ "snapshot_id": "<optional echo>",
  "strict_dag_supersede": true,                       -- optional, default true
  "nodes": [ {"id": "a", "tier": 3, "domain": "general"}, ... ],
  "missing": ["x", ...],                              -- optional: nodes with an invisible mandatory parent
  "edges": [ {"src": "a", "dst": "b", "type": "REFUTE"}, ... ] }   -- distinct (src,dst,type) triples
```

Result line: `{"ok":true,"nodes":[{id,eff_auth,val,cur,eligibility,reason,pos_cyc}...],
"blocked_authority_edges":[[u,w,type]...],"quarantined_supersede_cycles":[[id...]...]}`
or `{"ok":false,"error":"..."}`.

**Provenance of each stratum** (the `Theorem3_EK5_CyclicWFS` module is the only kernel-checked part):

* Stratum 2 (`val`): computed by `ChelatedFormal.evalStratum2Wfs` — kernel-bound to `lfp(Γ²)` by
  `ChelatedFormal.evalStratum2Wfs_sound`.  Forced-`F` nodes (positive `D_V` cycles / missing mandatory
  parents) are modelled by a positive self-dependency (the "phantom parent" trick), which makes them
  unsupportable in `posClosure`.
* Strata 0, 1, 3 here are a direct functional port (transitive-closure SCCs instead of Tarjan); they are
  **conformance-tested, not proved**.
-/

open Lean ChelatedFormal

namespace Ek5Ref

/-! ## Data model -/

structure Edge where
  src : String
  dst : String
  etype : String

structure Snapshot where
  snapId : Option String
  strict : Bool
  ids : Array String
  tiers : Array Nat
  domains : Array String
  missingIn : Array String
  edges : Array Edge

abbrev BMat := Array (Array Bool)

def validTypes : List String :=
  ["REFUTE", "SUPERSEDE", "VERSION_LINEAGE_SUPERSEDE", "COPY_OF", "DERIVES_CURRENT", "DERIVES_FROM",
   "AMENDS", "DERIVES_SNAPSHOT", "CITES_AUDIT", "SUPPORT"]

def depCurTypes : List String := ["COPY_OF", "DERIVES_CURRENT", "DERIVES_FROM"]

def depEpOnlyTypes : List String := ["AMENDS", "DERIVES_SNAPSHOT"]

def isDepType (t : String) : Bool := depCurTypes.contains t || depEpOnlyTypes.contains t

/-! ## Parsing -/

def parseNode (j : Json) : Except String (String × Nat × String) := do
  let id ← j.getObjValAs? String "id"
  let tier ← j.getObjValAs? Nat "tier"
  let dom ← match j.getObjVal? "domain" with
    | .ok v => v.getStr?
    | .error _ => pure "general"
  pure (id, tier, dom)

def parseEdge (j : Json) : Except String Edge := do
  let s ← j.getObjValAs? String "src"
  let d ← j.getObjValAs? String "dst"
  let t ← j.getObjValAs? String "type"
  pure ⟨s, d, t⟩

def parseSnapshot (j : Json) : Except String Snapshot := do
  let strict ← match j.getObjVal? "strict_dag_supersede" with
    | .ok v => v.getBool?
    | .error _ => pure true
  let snapId : Option String := match j.getObjVal? "snapshot_id" with
    | .ok (Json.str s) => some s
    | _ => none
  let nodesJ ← (← j.getObjVal? "nodes").getArr?
  let nodes ← nodesJ.mapM parseNode
  let missingIn ← match j.getObjVal? "missing" with
    | .ok v => do
      let a ← v.getArr?
      a.mapM Json.getStr?
    | .error _ => pure #[]
  let edgesJ ← match j.getObjVal? "edges" with
    | .ok v => v.getArr?
    | .error _ => pure #[]
  let edges ← edgesJ.mapM parseEdge
  pure { snapId := snapId, strict := strict,
         ids := nodes.map (·.1), tiers := nodes.map (·.2.1), domains := nodes.map (·.2.2),
         missingIn := missingIn, edges := edges }

/-! ## Boolean-matrix helpers (Warshall transitive closure = SCC / cycle detection) -/

def newMat (n : Nat) : BMat := Array.replicate n (Array.replicate n false)

def setMat (m : BMat) (i j : Nat) : BMat := m.set! i (m[i]!.set! j true)

/-- Paths of length ≥ 1: `(tclosure adj)[i][j]` iff `j` is reachable from `i` by a non-empty path. -/
def tclosure (n : Nat) (adj : BMat) : BMat := Id.run do
  let mut r := adj
  for k in [0:n] do
    for i in [0:n] do
      if (r[i]!)[k]! then
        for j in [0:n] do
          if (r[k]!)[j]! then
            r := setMat r i j
  return r

/-- Smallest member of the strongly connected component of `u` (singleton SCC: `u` itself). -/
def sccRep (n : Nat) (r : BMat) (u : Nat) : Nat := Id.run do
  let mut best := u
  for v in [0:n] do
    if v < best && (r[u]!)[v]! && (r[v]!)[u]! then best := v
  return best

/-! ## Stratum 1: anti-laundering authority ceiling over `D_V` -/

def stratum1 (n : Nat) (tier : Array Nat) (missing : Array Bool) (depV : BMat) : Array Nat × BMat :=
  let r := tclosure n depV
  let base : Array Nat := Array.ofFn (n := n) fun i =>
    if missing[i.val]! || (r[i.val]!)[i.val]! then 0 else tier[i.val]!
  let eff : Array Nat := Array.ofFn (n := n) fun i => Id.run do
    let mut m := base[i.val]!
    for v in [0:n] do
      if (r[i.val]!)[v]! && base[v]! < m then m := base[v]!
    return m
  (eff, r)

/-- Scoped capability (`_effective_authority_for_domain`), executable twin of `scopedAuth`. -/
def effFor (domIdx : Array Nat) (gen : Nat) (eff : Array Nat) (u : Nat) (targetDom : Nat) : Nat :=
  if domIdx[u]! != gen && targetDom != gen && domIdx[u]! != targetDom then 0 else eff[u]!

/-! ## Stratum 2: well-founded semantics through the proved evaluator -/

def evalS2 (n nd gen : Nat) (domIdx : Array Nat) (eff : Array Nat) (refs deps : BMat)
    (forced : Array Bool) : Array Tri :=
  let doms := nd + 1
  let domF : Fin n → Fin doms := fun i => ⟨domIdx[i.val]! % doms, Nat.mod_lt _ (Nat.succ_pos _)⟩
  let genF : Fin doms := ⟨gen % doms, Nat.mod_lt _ (Nat.succ_pos _)⟩
  let effF : Fin n → ℕ := fun i => eff[i.val]!
  let G : CyclicEpistemicGraph n doms :=
    { domainOf := domF
      effAuthForDomain := scopedAuth genF domF effF
      effAuthSelf := effF
      rawRefutes := fun u w => (refs[u.val]!)[w.val]!
      rawDeps := fun u w => (deps[u.val]!)[w.val]! || (forced[u.val]! && decide (u = w)) }
  let res := evalStratum2Wfs G
  Array.ofFn (n := n) fun i => triOf res.1 res.2 i

def triStr : Tri → String
  | Tri.t => "T"
  | Tri.f => "F"
  | Tri.u => "U"

/-! ## Stratum 3: temporal currency -/

structure Stratum3Out where
  cur : Array String
  reason : Array String
  qSupNodes : Array Bool

def anyIdx (n : Nat) (p : Nat → Bool) : Bool := Id.run do
  let mut b := false
  for i in [0:n] do
    if p i then b := true
  return b

def stratum3 (n : Nat) (strict : Bool) (val : Array String) (posCyc missing : Array Bool)
    (sup lsup depC : BMat) : Stratum3Out := Id.run do
  -- (a) quarantined SUPERSEDE cycles (only in non-strict / imported-cycle mode)
  let mut qSup : Array Bool := Array.replicate n false
  if !strict then
    let mut act := newMat n
    for u in [0:n] do
      if val[u]! != "F" then
        for w in [0:n] do
          if (sup[u]!)[w]! then act := setMat act u w
    let r := tclosure n act
    for u in [0:n] do
      qSup := qSup.set! u (r[u]!)[u]!
  -- (b) declared version-lineage SCC condensation
  let rl := tclosure n lsup
  let inRegion : Array Bool := Array.ofFn (n := n) fun i =>
    anyIdx n (fun j => (lsup[i.val]!)[j]! || (lsup[j]!)[i.val]!)
  let sid : Array Nat := Array.ofFn (n := n) fun i => sccRep n rl i.val
  let isQ : Array Bool := Array.ofFn (n := n) fun i =>
    inRegion[i.val]! && (rl[sid[i.val]!]!)[sid[i.val]!]! &&
      anyIdx n (fun v => sid[v]! == sid[i.val]! && inRegion[v]! && val[v]! != "F")
  let mut hd : Array Bool := Array.replicate n false
  let mut hp : Array Bool := Array.replicate n false
  let mut changed := true
  let mut guard := 0
  while changed && guard ≤ n + 1 do
    changed := false
    guard := guard + 1
    for u in [0:n] do
      for w in [0:n] do
        if (lsup[u]!)[w]! && sid[u]! != sid[w]! then
          let c := sid[u]!
          let d := sid[w]!
          if isQ[c]! then
            if !hp[d]! then
              hp := hp.set! d true
              changed := true
          else
            let compT := hd[c]! || anyIdx n (fun v => sid[v]! == c && inRegion[v]! && val[v]! == "T")
            let compU := hp[c]! || anyIdx n (fun v => sid[v]! == c && inRegion[v]! && val[v]! == "U")
            if compT && !hd[d]! then
              hd := hd.set! d true
              changed := true
            if compU && !hp[d]! then
              hp := hp.set! d true
              changed := true
  -- (c) local blocker evaluation
  let mut localCur : Array String := Array.replicate n "T"
  let mut localReason : Array String := Array.replicate n "CURRENT"
  for v in [0:n] do
    if val[v]! == "F" then
      localCur := localCur.set! v "F"
      localReason := localReason.set! v
        (if missing[v]! then "MISSING_MANDATORY_PARENT"
         else if posCyc[v]! then "UNFOUNDED_CYCLE" else "REFUTED")
    else
      let mut bExt := "F"
      let mut bCyc := "F"
      for j in [0:n] do
        if (sup[j]!)[v]! then
          let vj := val[j]!
          if vj != "F" then
            if qSup[j]! then bCyc := "U"
            else if vj == "T" then bExt := "T"
            else if vj == "U" && bExt != "T" then bExt := "U"
      if inRegion[v]! then
        let s := sid[v]!
        if hd[s]! then bExt := "T"
        else if hp[s]! || isQ[v]! then bCyc := "U"
      if bExt == "T" then
        localCur := localCur.set! v "F"
        localReason := localReason.set! v "SUPERSEDED"
      else if val[v]! == "U" then
        localCur := localCur.set! v "U"
        localReason := localReason.set! v "EPISTEMIC_CONTESTED"
      else if bExt == "U" || bCyc == "U" then
        localCur := localCur.set! v "U"
        localReason := localReason.set! v "TEMPORAL_CONTESTED"
      else
        localCur := localCur.set! v "T"
        localReason := localReason.set! v "CURRENT"
  -- (d) monotone currency propagation along D_C (cycle nodes forced F)
  let rc := tclosure n depC
  let mut cur := localCur
  let mut reason := localReason
  for u in [0:n] do
    if (rc[u]!)[u]! then
      cur := cur.set! u "F"
      reason := reason.set! u "UNFOUNDED_CYCLE"
  for _ in [0:n + 1] do
    let prevCur := cur
    for u in [0:n] do
      if !(rc[u]!)[u]! && localCur[u]! != "F" then
        let mut anyF := false
        let mut anyU := false
        for d in [0:n] do
          if (depC[u]!)[d]! then
            if prevCur[d]! == "F" then anyF := true
            else if prevCur[d]! == "U" then anyU := true
        if anyF then
          cur := cur.set! u "F"
          reason := reason.set! u "SUPERSEDED_ANCESTOR"
        else if anyU && localCur[u]! == "T" then
          cur := cur.set! u "U"
          reason := reason.set! u "TEMPORAL_ANCESTOR_CONTESTED"
        else
          cur := cur.set! u localCur[u]!
          reason := reason.set! u localReason[u]!
  return ⟨cur, reason, qSup⟩

/-! ## Whole-snapshot evaluation (Strata 0–3) -/

def strLt (a b : String) : Bool := decide (a < b)

def listLt : List String → List String → Bool
  | [], [] => false
  | [], _ :: _ => true
  | _ :: _, [] => false
  | a :: as, b :: bs => if a == b then listLt as bs else strLt a b

def evaluate (s : Snapshot) : Except String Json := do
  let n := s.ids.size
  -- Stratum 0: validate nodes
  for i in [0:n] do
    for j in [i + 1:n] do
      if s.ids[i]! == s.ids[j]! then throw s!"duplicate node id {s.ids[i]!}"
  let idxOf (x : String) : Option Nat := s.ids.findIdx? (· == x)
  let domList : Array String := s.domains.foldl (fun acc d => if acc.contains d then acc else acc.push d) #[]
  let nd := domList.size
  let genIdx := match domList.findIdx? (· == "general") with | some g => g | none => nd
  let domIdx : Array Nat := s.domains.map fun d => (domList.findIdx? (· == d)).getD 0
  let mut missing : Array Bool := Array.replicate n false
  for m in s.missingIn do
    match idxOf m with
    | some i => missing := missing.set! i true
    | none => throw s!"missing references unknown node {m}"
  let mut refs := newMat n
  let mut sup := newMat n
  let mut lsup := newMat n
  let mut depV := newMat n
  let mut depC := newMat n
  let mut seen : Array (String × String × String) := #[]
  for e in s.edges do
    unless validTypes.contains e.etype do throw s!"Unknown edge_type={e.etype}"
    if (e.etype == "SUPERSEDE" || e.etype == "VERSION_LINEAGE_SUPERSEDE") && e.src == e.dst then
      throw s!"Self-{e.etype} rejected on {e.src}"
    if seen.contains (e.src, e.dst, e.etype) then throw s!"duplicate edge {e.src}-{e.etype}->{e.dst}"
    seen := seen.push (e.src, e.dst, e.etype)
    match idxOf e.src, idxOf e.dst with
    | some a, some b =>
      if e.etype == "REFUTE" then refs := setMat refs a b
      else if e.etype == "SUPERSEDE" then sup := setMat sup a b
      else if e.etype == "VERSION_LINEAGE_SUPERSEDE" then lsup := setMat lsup a b
      else if isDepType e.etype then
        depV := setMat depV a b
        if depCurTypes.contains e.etype then depC := setMat depC a b
    | some a, none =>
      if isDepType e.etype then missing := missing.set! a true
      else throw s!"Dangling edge endpoint in {e.src} -[{e.etype}]-> {e.dst}"
    | _, _ => throw s!"Dangling edge endpoint in {e.src} -[{e.etype}]-> {e.dst}"
  -- Stratum 1
  let (eff, rV) := stratum1 n s.tiers missing depV
  let posCyc : Array Bool := Array.ofFn (n := n) fun i => (rV[i.val]!)[i.val]!
  let forced : Array Bool := Array.ofFn (n := n) fun i => posCyc[i.val]! || missing[i.val]!
  -- authority-qualified Stratum 2/3 edges
  let allowed (a b : Nat) : Bool := effFor domIdx genIdx eff a domIdx[b]! ≥ eff[b]!
  let mut refsAct := newMat n
  let mut supAct := newMat n
  let mut lsupAct := newMat n
  let mut blocked : Array (String × String × String) := #[]
  for a in [0:n] do
    for b in [0:n] do
      if (refs[a]!)[b]! then
        if allowed a b then refsAct := setMat refsAct a b
        else blocked := blocked.push (s.ids[a]!, s.ids[b]!, "REFUTE")
      if (sup[a]!)[b]! then
        if allowed a b then supAct := setMat supAct a b
        else blocked := blocked.push (s.ids[a]!, s.ids[b]!, "SUPERSEDE")
      if (lsup[a]!)[b]! then
        if allowed a b then lsupAct := setMat lsupAct a b
        else blocked := blocked.push (s.ids[a]!, s.ids[b]!, "VERSION_LINEAGE_SUPERSEDE")
  -- Stratum 2 (kernel-bound evaluator)
  let tri := evalS2 n nd genIdx domIdx eff refsAct depV forced
  let val : Array String := tri.map triStr
  -- strict mode refuses cyclic raw SUPERSEDE / lineage edges (native write rejection)
  if s.strict then
    let rs := tclosure n sup
    let rl := tclosure n lsup
    for u in [0:n] do
      if (rs[u]!)[u]! then throw s!"SUPERSEDE cycle rejected through {s.ids[u]!}"
      if (rl[u]!)[u]! then throw s!"Lineage SUPERSEDE cycle rejected through {s.ids[u]!}"
  -- Stratum 3
  let o3 := stratum3 n s.strict val posCyc missing supAct lsupAct depC
  -- quarantined SUPERSEDE cycles (sorted tuples of sorted ids)
  let mut qCycles : Array (List String) := #[]
  if !s.strict then
    let mut act := newMat n
    for u in [0:n] do
      if val[u]! != "F" then
        for w in [0:n] do
          if (supAct[u]!)[w]! then act := setMat act u w
    let r := tclosure n act
    for u in [0:n] do
      if (r[u]!)[u]! && sccRep n r u == u then
        let comp := (Array.range n).filter (fun v => v == u || ((r[u]!)[v]! && (r[v]!)[u]!))
        let ids := (comp.map (fun v => s.ids[v]!)).qsort strLt
        qCycles := qCycles.push ids.toList
  let qSorted := qCycles.qsort listLt
  let blockedSorted := blocked.qsort fun a b => listLt [a.1, a.2.1, a.2.2] [b.1, b.2.1, b.2.2]
  -- eligibility + output (nodes sorted by id)
  let order := (Array.range n).qsort fun a b => strLt s.ids[a]! s.ids[b]!
  let nodesOut := order.map fun i =>
    let elig :=
      if val[i]! == "T" && o3.cur[i]! == "T" then "ADMIT"
      else if val[i]! == "F" || o3.cur[i]! == "F" then "AUDIT_ONLY"
      else "CONTESTED"
    Json.mkObj
      [ ("id", Json.str s.ids[i]!), ("eff_auth", toJson eff[i]!), ("val", Json.str val[i]!),
        ("cur", Json.str o3.cur[i]!), ("eligibility", Json.str elig),
        ("reason", Json.str o3.reason[i]!), ("pos_cyc", Json.bool (posCyc[i]!)) ]
  let base : List (String × Json) :=
    [ ("ok", Json.bool true), ("nodes", Json.arr nodesOut),
      ("blocked_authority_edges",
        Json.arr (blockedSorted.map fun b => Json.arr #[Json.str b.1, Json.str b.2.1, Json.str b.2.2])),
      ("quarantined_supersede_cycles",
        Json.arr (qSorted.map fun c => Json.arr ((c.map Json.str).toArray))) ]
  let withId := match s.snapId with
    | some sid => ("snapshot_id", Json.str sid) :: base
    | none => base
  pure (Json.mkObj withId)

def errorLine (msg : String) : String :=
  (Json.mkObj [("ok", Json.bool false), ("error", Json.str msg)]).compress

def processLine (line : String) : String :=
  match Json.parse line with
  | .error e => errorLine s!"json parse error: {e}"
  | .ok j =>
    match parseSnapshot j with
    | .error e => errorLine s!"schema error: {e}"
    | .ok s =>
      match evaluate s with
      | .error e => errorLine e
      | .ok out => out.compress

partial def readAllLines (h : IO.FS.Stream) (acc : Array String) : IO (Array String) := do
  let l ← h.getLine
  if l.isEmpty then pure acc else readAllLines h (acc.push l)

end Ek5Ref

def main (args : List String) : IO UInt32 := do
  let lines ← match args with
    | [] => do
      let stdin ← IO.getStdin
      Ek5Ref.readAllLines stdin #[]
    | ["--input", p] => do
      let c ← IO.FS.readFile p
      pure (c.splitOn "\n").toArray
    | _ => do
      IO.eprintln "usage: ek5_ref_cli [--input <path>]   (newline-delimited JSON snapshots on stdin)"
      pure #[]
  if lines.isEmpty && !args.isEmpty && args != ["--input", args.getD 1 ""] then
    return 2
  let stdout ← IO.getStdout
  for line in lines do
    let l := line.trimAscii.toString
    if l.isEmpty then continue
    stdout.putStrLn (Ek5Ref.processLine l)
  stdout.flush
  return 0
