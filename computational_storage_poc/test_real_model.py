import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mock_nvme import MockNVMeDrive

# We'll test our newly compiled "Real" model running through our Mock NVMe SSD

def test_real_model():
    print("Loading test dataset...")
    digits = load_digits()
    X, y = digits.data, digits.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Needs to match train set split so we test on unseen data
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Initializing Mock NVMe Array with real_model.bin...")
    try:
        drive = MockNVMeDrive("real_model.bin")
    except FileNotFoundError:
        print("Please run train_and_compile.py first.")
        return
        
    correct = 0
    total = len(X_test)
    total_comp_latency = 0.0
    
    print("\nRunning Inference on Test Set natively via Computational Storage...")
    for i in range(total):
        # Format input (pad to 512)
        input_act = np.zeros((1, 512), dtype=np.float16)
        input_act[0, :64] = X_test[i]
        
        out_comp, latency = drive.computational_inference(0x0, input_act)
        total_comp_latency += latency
        
        # The output is 1x512. The first 10 columns are the class logits.
        logits = out_comp[0, :10]
        pred = np.argmax(logits)
        
        if pred == y_test[i]:
            correct += 1

    accuracy = correct / total * 100
    print(f"\nResults over {total} test samples:")
    print(f"Accuracy via Computational SSD: {accuracy:.2f}%")
    print(f"Total Theoretical Latency: {total_comp_latency * 1000:.2f} ms")
    
if __name__ == "__main__":
    test_real_model()
