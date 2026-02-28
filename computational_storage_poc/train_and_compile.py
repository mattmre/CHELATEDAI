import numpy as np
import struct
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BLOCK_SIZE = 512
PARAM_TYPE = np.float16
BYTES_PER_PARAM = 2
MATRIX_BYTES = BLOCK_SIZE * BLOCK_SIZE * BYTES_PER_PARAM
POINTER_BYTES = 8
TOTAL_BLOCK_BYTES = MATRIX_BYTES + POINTER_BYTES

# Set up a simple PyTorch model: 64 -> 128 -> 10
class TinyDigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128, bias=False)
        self.fc2 = nn.Linear(128, 10, bias=False)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def create_block(matrix: np.ndarray, next_block_offset: int) -> bytes:
    assert len(matrix.shape) == 2, "Matrix must be 2D"
    rows, cols = matrix.shape
    assert rows <= BLOCK_SIZE and cols <= BLOCK_SIZE, f"Matrix exceeds {BLOCK_SIZE}x{BLOCK_SIZE}"
    
    padded = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=PARAM_TYPE)
    padded[:rows, :cols] = matrix.astype(PARAM_TYPE)
    
    matrix_bytes = padded.tobytes()
    pointer_bytes = struct.pack('<Q', next_block_offset)
    
    return matrix_bytes + pointer_bytes

def train_and_compile():
    print("Loading Digits dataset (8x8 images = 64 inputs)...")
    digits = load_digits()
    X, y = digits.data, digits.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)
    
    model = TinyDigitClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("Training model...")
    for epoch in range(50):
        optimizer.zero_grad()
        out = model(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        test_out = model(X_test_t)
        preds = torch.argmax(test_out, dim=1)
        acc = (preds == y_test_t).float().mean()
        print(f"Test Accuracy: {acc.item() * 100:.2f}%")
        
    print("Extracting weights for Computational Storage serialization...")
    # NOTE: PyTorch linear layer weights are (out_features, in_features). 
    # For inference Y = X @ W, we need to transpose them to (in_features, out_features).
    w1 = model.fc1.weight.detach().numpy().T  # Shape: (64, 128)
    w2 = model.fc2.weight.detach().numpy().T  # Shape: (128, 10)
    
    block1_offset = TOTAL_BLOCK_BYTES
    end_offset = 0 
    
    b0 = create_block(w1, block1_offset)
    b1 = create_block(w2, end_offset)
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "real_model.bin")
    
    payload = b0 + b1
    with open(out_path, "wb") as f:
        f.write(payload)
        
    print(f"\nCompiled REAL binary graph to {out_path}")
    print(f"Total size: {len(payload)} bytes")
    print("Graph Structure:")
    print(f"  Block 0 (Offset 0x0) -> W1 {w1.shape} -> Points to Block 1 (Offset {hex(block1_offset)})")
    print(f"  Block 1 (Offset {hex(block1_offset)}) -> W2 {w2.shape} -> Points to {hex(end_offset)} (End)")

if __name__ == "__main__":
    train_and_compile()
