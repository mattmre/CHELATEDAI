import os
import numpy as np

from block_graph import build_graph_payload

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
    rng = np.random.default_rng(42)
    # Transposing logic for typical dense layers: Y = X @ W
    # where X is (Batch, 256), W is (256, 128), Y is (Batch, 128)
    w1 = rng.normal(size=(256, 128)).astype(np.float32) * 0.1
    w2 = rng.normal(size=(128, 10)).astype(np.float32) * 0.1
    return build_graph_payload([w1, w2])

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
