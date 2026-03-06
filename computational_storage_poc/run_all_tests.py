import os
import subprocess
import sys
import time

def run_script(script_name, description):
    print("\n" + "="*60)
    print(f"> RUNNING PHASE: {description}")
    print(f"> SCRIPT: {script_name}")
    print("="*60 + "\n")
    
    start_time = time.time()
    result = subprocess.run([sys.executable, script_name], check=False)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\nFAILED: {script_name} (Exit code: {result.returncode})")
        sys.exit(result.returncode)
    else:
        print(f"\nSUCCESS: {script_name} completed in {elapsed:.2f}s")

if __name__ == "__main__":
    print("============================================================")
    print("  CHELATEDAI Computational Storage POC - Full Test Suite")
    print("============================================================")
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    run_script("compiler.py", "1. Conceptual Block Memory Layout & Mock Compiler")
    run_script("mock_nvme.py", "2. Software-Only Simulation & Latency Profiling")
    run_script("mock_array.py", "3. Array Simulation & Speculative Multithreading Analysis")
    run_script("train_and_compile.py", "4. Train Real PyTorch Model & Burn to Block Model Format")
    run_script("test_real_model.py", "5. Test Block Model Accuracy on Mock NVMe")
    run_script("CHELATEDAI_integration_demo.py", "6. Run CHELATEDAI Engine Integration Demo")
    
    print("\n" + "="*60)
    print("ALL PHASES COMPLETED SUCCESSFULLY")
    print("="*60)
    print("\nThe software array simulation is fully verified.")
    print("For physical hardware evaluation, please check firmware/BUILD_GUIDE.md")
    print("or use the auto-generated .uf2 file from the GitHub Actions release.")
