import numpy as np
import struct
import time

BLOCK_SIZE = 512
PARAM_TYPE = np.float16
BYTES_PER_PARAM = 2
MATRIX_BYTES = BLOCK_SIZE * BLOCK_SIZE * BYTES_PER_PARAM
POINTER_BYTES = 8
TOTAL_BLOCK_BYTES = MATRIX_BYTES + POINTER_BYTES

# Theoretical latencies in seconds
PCIE_ROUND_TRIP_LATENCY = 10e-6   # 10 microseconds
PCIE_BANDWIDTH_BPS = 7.88e9       # PCIe Gen4 x4 (~7.88 GB/s)
INTERNAL_FLASH_LATENCY = 100e-6   # 100 microseconds per internal access

class MockNVMeDrive:
    def __init__(self, binary_path: str):
        with open(binary_path, "rb") as f:
            self.flash_memory = f.read()
            
    def host_read_block(self, offset: int, size: int):
        transfer_time = size / PCIE_BANDWIDTH_BPS
        latency_cost = PCIE_ROUND_TRIP_LATENCY + INTERNAL_FLASH_LATENCY + transfer_time
        return self.flash_memory[offset : offset + size], latency_cost
        
    def computational_inference(self, trigger_offset: int, input_activations: np.ndarray):
        total_latency = 0.0
        
        # CPU to SSD transfer
        input_bytes = input_activations.tobytes()
        transfer_time_in = len(input_bytes) / PCIE_BANDWIDTH_BPS
        total_latency += PCIE_ROUND_TRIP_LATENCY + transfer_time_in
        
        current_offset = trigger_offset
        current_activations = input_activations.astype(np.float32)
        
        while current_offset != 0:
            # NVMe internally reading flash (no PCIe cost)
            total_latency += INTERNAL_FLASH_LATENCY
            
            block_data = self.flash_memory[current_offset : current_offset + TOTAL_BLOCK_BYTES]
            matrix_bytes = block_data[:MATRIX_BYTES]
            pointer_bytes = block_data[MATRIX_BYTES:]
            next_offset = struct.unpack('<Q', pointer_bytes)[0]
            w_block = np.frombuffer(matrix_bytes, dtype=PARAM_TYPE).reshape((BLOCK_SIZE, BLOCK_SIZE)).astype(np.float32)
            
            # Simulated internal hardware matrix compute delay
            # We assume a dedicated ASIC doing 512x512 MACs
            # For a 512x512 matrix, that's 262,144 MACs.
            # At 1 TOPS, this takes ~0.26 microseconds.
            compute_latency = 0.26e-6 
            current_activations = current_activations @ w_block
            total_latency += compute_latency
            
            current_offset = next_offset
            
        # SSD to CPU transfer
        out_bytes = current_activations.tobytes()
        transfer_time_out = len(out_bytes) / PCIE_BANDWIDTH_BPS
        total_latency += PCIE_ROUND_TRIP_LATENCY + transfer_time_out
        
        return current_activations, total_latency


def traditional_host_inference(drive: MockNVMeDrive, trigger_offset: int, input_activations: np.ndarray):
    total_latency = 0.0
    current_offset = trigger_offset
    current_activations = input_activations.astype(np.float32)
    
    while current_offset != 0:
        # PCIe read
        block_data, read_latency = drive.host_read_block(current_offset, TOTAL_BLOCK_BYTES)
        total_latency += read_latency
        
        matrix_bytes = block_data[:MATRIX_BYTES]
        pointer_bytes = block_data[MATRIX_BYTES:]
        next_offset = struct.unpack('<Q', pointer_bytes)[0]
        w_block = np.frombuffer(matrix_bytes, dtype=PARAM_TYPE).reshape((BLOCK_SIZE, BLOCK_SIZE)).astype(np.float32)
        
        # CPU compute (measured via perf_counter)
        t0 = time.perf_counter()
        current_activations = current_activations @ w_block
        t1 = time.perf_counter()
        total_latency += (t1 - t0)
        
        current_offset = next_offset
        
    return current_activations, total_latency


if __name__ == "__main__":
    drive = MockNVMeDrive("model.bin")
    inputs = np.zeros((1, 512), dtype=PARAM_TYPE)
    inputs[0, :256] = np.random.randn(256) * 0.1 
    
    out_trad, trad_time = traditional_host_inference(drive, 0x0, inputs)
    out_comp, comp_time = drive.computational_inference(0x0, inputs)
    
    print("\n--- Theoretical Latency Profiling (2 Layers) ---")
    print(f"Traditional Inference: {trad_time * 1000000:.1f} microseconds")
    print(f"Computational Storage: {comp_time * 1000000:.1f} microseconds")
    
    diff = np.max(np.abs(out_trad - out_comp))
    print(f"\nVerification: Outputs match? {'YES' if diff < 1e-4 else 'NO'} (Max diff: {diff})")
    print(f"Speedup Factor: {trad_time / comp_time:.2f}x\n")
    
    # Note: On a model with 10,000 layers, this scales linearly:
    layers = 10000
    print(f"--- Extrapolated to 10,000 layers ({min(layers*BLOCK_SIZE*BLOCK_SIZE*BYTES_PER_PARAM/1e9, 9999):.2f} GB model) ---")
    print(f"Traditional Inference: {(trad_time/2) * layers * 1000:.1f} ms")
    print(f"Computational Storage: {(comp_time/2) * layers * 1000:.1f} ms")
