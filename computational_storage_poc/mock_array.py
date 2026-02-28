import time
import concurrent.futures
import threading

# Theoretical constants
PCIE_ROUND_TRIP_LATENCY = 10e-6
INTERNAL_FLASH_LATENCY = 100e-6
I_O_CONTENTION_PENALTY = 50e-6 # Time penalty if two cores hit the same physical SSD 

class MockSSD_Node:
    """A dumb mock of an SSD that tracks if it's currently busy (I/O Contention)."""
    def __init__(self, drive_id):
        self.drive_id = drive_id
        self.lock = threading.Lock()
        
    def fetch_node(self, node_id: int):
        with self.lock:
            # We assume reading the flash natively without PCIe overhead
            time.sleep(INTERNAL_FLASH_LATENCY)
            return f"Node_{node_id}_Data"


class ArraySimulation:
    def __init__(self, num_drives=4):
        self.drives = [MockSSD_Node(i) for i in range(num_drives)]
        
    def single_thread_execution(self, path_of_nodes: list):
        """Simulates standard synchronous inference (one step at a time)"""
        start = time.perf_counter()
        results = []
        for node in path_of_nodes:
            # Pick a random drive (assuming data is stripped)
            drive = self.drives[node % len(self.drives)]
            res = drive.fetch_node(node)
            results.append(res)
        end = time.perf_counter()
        return end - start
        
    def speculative_multipath_racing(self, probability_branches: list):
        """
        Simulates Speculative Multithreading.
        Instead of waiting for the math to guess the branch, we race 4 cores 
        to fetch 4 possible branch nodes simultaneously from 4 different SSDs.
        Because they are on different physical NVMes, there is zero I/O Contention.
        """
        start = time.perf_counter()
        
        # We assume the "Correct" path was branch index 0 (which the CPU figures out slightly later)
        correct_branch_index = 0
        
        def run_speculation(branch_idx, node_id):
            drive = self.drives[branch_idx % len(self.drives)]
            return drive.fetch_node(node_id)
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(probability_branches)) as executor:
            # Submit all branches at exactly the same time (racing)
            futures = []
            for i, node_id in enumerate(probability_branches):
                futures.append(executor.submit(run_speculation, i, node_id))
            
            # CPU figures out the math while the SSDs are fetching...
            time.sleep(1e-6) # 1 microsecond CPU math time
            
            # We only CARE about the correct branch's result.
            # The other 3 futures finish their fetches and are instantly discarded (L1 cache flush).
            final_result = futures[correct_branch_index].result()
            
        end = time.perf_counter()
        return end - start

if __name__ == "__main__":
    array = ArraySimulation(num_drives=4)
    
    print("--- Phase 4: Speculative SSD Array Simulation ---")
    
    # 1. Single-Threaded Path (Sequential: Path -> A -> B -> C -> D)
    print("\nSimulating 4 sequential synchronous jumps (e.g. 1 inference step finding the next)...")
    seq_time = array.single_thread_execution([10, 22, 5, 99])
    
    # 2. Speculative Racing (Parallel: We race A, B, C, and D based on 4 probability vectors instantly)
    # The CPU throws all 4 requests down the PCIe lane at once to 4 different SSDs.
    print("\nSimulating Speculative Multipath Racing (Fetching the 4 most likely paths concurrently)...")
    race_time = array.speculative_multipath_racing([10, 22, 5, 99])
    
    print(f"\nSequential Time: {seq_time * 1000:.3f} ms")
    print(f"Speculative Racing Time: {race_time * 1000:.3f} ms")
    
    print(f"\nLatency Hidden via Speculation: {100 - (race_time / seq_time)*100:.1f}%")
    print("Because the SSDs act independently, we hide the continuous latency penalty of sequential logic.")
