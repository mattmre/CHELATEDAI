import numpy as np
import struct
import os

BLOCK_SIZE = 512
PARAM_TYPE = np.float16 # Using 16-bit precision for the POC
BYTES_PER_PARAM = 2
MATRIX_BYTES = BLOCK_SIZE * BLOCK_SIZE * BYTES_PER_PARAM
POINTER_BYTES = 8 # 64-bit guide node
TOTAL_BLOCK_BYTES = MATRIX_BYTES + POINTER_BYTES

# Format of each block:
# [512x512 matrix data (524,288 bytes for FP16)] + [64-bit unsigned int offset pointer (8 bytes)]
# Total Block Size = 524,296 bytes.

def create_block(matrix: np.ndarray, next_block_offset: int) -> bytes:
    """
    Takes a matrix (up to 512x512) and a 64-bit integer representing the byte offset 
    of the *next* block to read in the file (0 if it's the last node).
    Returns the raw bytes for the block.
    """
    assert len(matrix.shape) == 2, "Matrix must be 2D"
    rows, cols = matrix.shape
    assert rows <= BLOCK_SIZE and cols <= BLOCK_SIZE, f"Matrix exceeds {BLOCK_SIZE}x{BLOCK_SIZE}"
    
    # Pad to 512x512
    padded = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=PARAM_TYPE)
    padded[:rows, :cols] = matrix.astype(PARAM_TYPE)
    
    matrix_bytes = padded.tobytes()
    # Pack the 64-bit unsigned long long (little-endian)
    pointer_bytes = struct.pack('<Q', next_block_offset)
    
    assert len(matrix_bytes) == MATRIX_BYTES, f"Matrix bytes mismatch: {len(matrix_bytes)} vs {MATRIX_BYTES}"
    assert len(pointer_bytes) == POINTER_BYTES, f"Pointer bytes mismatch: {len(pointer_bytes)} vs {POINTER_BYTES}"
    
    return matrix_bytes + pointer_bytes

def generate_mock_mnist():
    """
    Generates a tiny mockup of a 3-layer MLP meant to represent small test inference.
    To avoid tiling for the simplest POC, we'd assume a 16x16=256 input size.
    Input: 256
    Hidden: 128
    Output: 10 (classes)
    
    Matrices:
    W1: 256 x 128
    W2: 128 x 10
    """
    print("Generating mock random weights for a tiny 256->128->10 MLP...")
    np.random.seed(42)
    # Transposing logic for typical dense layers: Y = X @ W
    # where X is (Batch, 256), W is (256, 128), Y is (Batch, 128)
    w1 = np.random.randn(256, 128) * 0.1
    w2 = np.random.randn(128, 10) * 0.1
    
    # We will lay them out in a file sequentially.
    # Block 0: W1, points to Block 1
    # Block 1: W2, points to offset 0 (end of graph)
    
    block1_offset = TOTAL_BLOCK_BYTES
    end_offset = 0 # 0 means stop / end of graph
    
    b0 = create_block(w1, block1_offset)
    b1 = create_block(w2, end_offset)
    
    return b0 + b1

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model.bin")
    
    payload = generate_mock_mnist()
    with open(out_path, "wb") as f:
        f.write(payload)
    
    print(f"Compiled binary graph to {out_path}")
    print(f"Total size: {len(payload)} bytes")
    print("Graph Structure:")
    print("  Block 0 (Offset 0x0) -> W1 (256x128) -> Points to Block 1 (Offset 0x80008)")
    print("  Block 1 (Offset 0x80008) -> W2 (128x10) -> Points to 0x0 (End)")
