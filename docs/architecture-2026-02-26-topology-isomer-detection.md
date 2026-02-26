# Architecture: Topology-Aware Retrieval + Isomer Detection
Date: 2026-02-26 | Session: 21 | Status: Approved

## Summary
TopologyAnalyzer models embedding space as bond-typed graph. IsomerDetector finds queries with structurally different results across chelation states.

## New Files
- topology_analyzer.py: TopologyAnalyzer class
- isomer_detector.py: IsomerDetector class

## TopologyAnalyzer
- Bond classification: covalent (>=0.90), hydrogen (>=0.70), vdw (>=0.40), none (<0.40)
- build_bond_matrix(): Cosine similarity -> typed adjacency matrix
- compute_cluster_connectivity(): Inter/intra-cluster bond statistics
- compute_topology_change(): Pre/post sedimentation comparison
  - collapse_pressure: change in covalent ratio (positive = bad)
  - topology_distance: normalized Frobenius norm of bond type changes
- Snapshot recording for historical tracking

## IsomerDetector
- Detects "retrieval isomers": same query, different results due to adapter state
- isomer_strength = 1.0 - jaccard (higher = more different)
- Convenience methods: detect_sedimentation_isomers(), detect_chelation_isomers()
- Query similarity grouping: find_similar_query_isomers()
- Frequency and distribution statistics

## Practical Value
- TopologyAnalyzer: Sedimentation impact assessment, collapse early warning, cluster health
- IsomerDetector: Chelation reliability scoring, query difficulty classification, regression detection

## Engine Integration
- enable_topology_analysis(), enable_isomer_detection()
- get_structural_health_report(): Unified health from stability + topology + isomers
- Health classification: healthy/degrading/critical

## Config Presets
- topology: tight, balanced, loose
- isomer: sensitive, balanced, strict

## Test Plan: ~60 tests
- test_topology_analyzer.py (27), test_isomer_detector.py (25), config additions (8)
