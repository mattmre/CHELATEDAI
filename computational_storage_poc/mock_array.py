from collections import Counter

# Theoretical constants
INTERNAL_FLASH_LATENCY = 100e-6
CPU_BRANCH_SELECTION_LATENCY = 1e-6


class MockSSD_Node:
    """A minimal SSD model used for deterministic latency accounting."""

    def __init__(self, drive_id):
        self.drive_id = drive_id

    def fetch_node(self, node_id: int):
        return f"Node_{node_id}_Data"


class ArraySimulation:
    def __init__(self, num_drives=4):
        self.drives = [MockSSD_Node(i) for i in range(num_drives)]

    def single_thread_execution(self, path_of_nodes: list):
        """Deterministic sequential latency with one blocking fetch per node."""
        return len(path_of_nodes) * INTERNAL_FLASH_LATENCY

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
        return CPU_BRANCH_SELECTION_LATENCY + (slowest_drive_depth * INTERNAL_FLASH_LATENCY)

    def compare_execution_modes(self, path_of_nodes: list, probability_branches: list):
        sequential_time = self.single_thread_execution(path_of_nodes)
        race_time = self.speculative_multipath_racing(probability_branches)
        latency_hidden_pct = 0.0 if sequential_time == 0 else 100 - (race_time / sequential_time) * 100
        return {
            "sequential_time_ms": sequential_time * 1000,
            "speculative_time_ms": race_time * 1000,
            "latency_hidden_pct": latency_hidden_pct,
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
