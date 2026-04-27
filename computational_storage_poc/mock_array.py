from __future__ import annotations

from collections import Counter

import numpy as np

from device_profiles import DeviceClass, StorageDeviceProfile, get_profile

# Theoretical constants
INTERNAL_FLASH_LATENCY = 100e-6
CPU_BRANCH_SELECTION_LATENCY = 1e-6


class MockSSD_Node:
    """A minimal SSD model used for deterministic latency accounting."""

    def __init__(self, drive_id):
        self.drive_id = drive_id

    def fetch_node(self, node_id: int):
        return f"Node_{node_id}_Data"

    def local_ann_search(self, query_vector, vectors: dict, top_k: int = 10):
        """Return local top-k vector IDs by cosine similarity."""

        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        query = np.array(query_vector, dtype=float)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            raise ValueError("query_vector must be non-zero")

        scored = []
        for vector_id, vector in vectors.items():
            candidate = np.array(vector, dtype=float)
            candidate_norm = np.linalg.norm(candidate)
            if candidate_norm == 0:
                similarity = 0.0
            else:
                similarity = float(np.dot(query, candidate) / (query_norm * candidate_norm))
            scored.append((str(vector_id), similarity))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]


class ArraySimulation:
    def __init__(self, num_drives=4, device_profile: StorageDeviceProfile | None = None):
        self.drives = [MockSSD_Node(i) for i in range(num_drives)]
        self.device_profile = device_profile or get_profile(DeviceClass.CONSUMER_NVME)

    def single_thread_execution(self, path_of_nodes: list):
        """Deterministic sequential latency with one blocking fetch per node."""
        return self.device_profile.latency_seconds(len(path_of_nodes))

    def speculative_multipath_racing(self, probability_branches: list):
        """
        Deterministic speculative latency.

        Each branch is assigned to a drive by dispatch order. The slowest drive queue
        determines total flash wait time, plus a small CPU branch-selection latency.
        """
        if not probability_branches:
            return 0.0

        drive_depths = Counter(index % len(self.drives) for index, _ in enumerate(probability_branches))
        slowest_drive_depth = max(drive_depths.values())
        return self.device_profile.latency_seconds(slowest_drive_depth)

    def compare_execution_modes(self, path_of_nodes: list, probability_branches: list):
        sequential_time = self.single_thread_execution(path_of_nodes)
        race_time = self.speculative_multipath_racing(probability_branches)
        latency_hidden_pct = 0.0 if sequential_time == 0 else 100 - (race_time / sequential_time) * 100
        return {
            "sequential_time_ms": sequential_time * 1000,
            "speculative_time_ms": race_time * 1000,
            "latency_hidden_pct": latency_hidden_pct,
        }

    def sharded_population_evaluation(self, candidate_scores: list):
        """Simulate near-data evaluation of ES population members on drive shards.

        Args:
            candidate_scores: List of dictionaries with `candidate_id` and
                scalar `fitness`. Candidates are assigned to drives by dispatch
                order, representing storage-sharded population scoring.

        Returns:
            Summary with best candidate, per-drive queue depths, and deterministic
            latency based on the slowest shard queue.
        """
        if not candidate_scores:
            return {
                "best_candidate_id": None,
                "best_fitness": None,
                "drive_depths": {},
                "storage_latency_ms": 0.0,
            }

        drive_depths = Counter()
        best_candidate = None
        best_fitness = None
        for index, candidate in enumerate(candidate_scores):
            if "candidate_id" not in candidate or "fitness" not in candidate:
                raise ValueError("Each candidate must include candidate_id and fitness")
            drive_id = index % len(self.drives)
            drive_depths[drive_id] += 1
            fitness = float(candidate["fitness"])
            if best_fitness is None or fitness > best_fitness:
                best_fitness = fitness
                best_candidate = candidate["candidate_id"]

        slowest_drive_depth = max(drive_depths.values())
        latency = self.device_profile.latency_seconds(slowest_drive_depth)
        return {
            "best_candidate_id": best_candidate,
            "best_fitness": best_fitness,
            "drive_depths": dict(drive_depths),
            "storage_latency_ms": latency * 1000,
            "device_profile": self.device_profile.to_dict(),
        }

    def storage_resident_ann_query(self, query_vector, shard_vectors: dict, top_k: int = 10):
        """Simulate per-drive ANN filtering before host-side merge.

        Args:
            query_vector: Query embedding.
            shard_vectors: Mapping of drive id to `{doc_id: vector}` dictionaries.
            top_k: Number of merged candidates to return.
        """
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        merged = []
        drive_depths = Counter()
        for drive_id, vectors in shard_vectors.items():
            drive = self.drives[int(drive_id) % len(self.drives)]
            local_hits = drive.local_ann_search(query_vector, vectors, top_k=top_k)
            drive_depths[drive.drive_id] += len(vectors)
            merged.extend({
                "drive_id": drive.drive_id,
                "doc_id": doc_id,
                "score": score,
            } for doc_id, score in local_hits)

        merged.sort(key=lambda item: item["score"], reverse=True)
        slowest_drive_depth = max(drive_depths.values(), default=0)
        latency = self.device_profile.latency_seconds(slowest_drive_depth)
        return {
            "top_hits": merged[:top_k],
            "drive_depths": dict(drive_depths),
            "storage_latency_ms": latency * 1000,
            "device_profile": self.device_profile.to_dict(),
            "scope": "simulation_only",
        }


if __name__ == "__main__":
    array = ArraySimulation(num_drives=4)

    print("--- Phase 4: Speculative SSD Array Simulation ---")
    path_nodes = [10, 22, 5, 99]

    print("\nSimulating 4 sequential synchronous jumps (e.g. 1 inference step finding the next)...")
    seq_time = array.single_thread_execution(path_nodes)

    print("\nSimulating Speculative Multipath Racing (Fetching the 4 most likely paths concurrently)...")
    race_time = array.speculative_multipath_racing(path_nodes)

    print(f"\nSequential Time: {seq_time * 1000:.3f} ms")
    print(f"Speculative Racing Time: {race_time * 1000:.3f} ms")
    print(f"\nLatency Hidden via Speculation: {100 - (race_time / seq_time) * 100:.1f}%")
    print("Because the SSDs act independently, we hide the continuous latency penalty of sequential logic.")

    if race_time >= seq_time:
        raise SystemExit("Speculative execution should beat sequential execution for the unique-drive demo")
