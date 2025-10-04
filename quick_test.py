#!/usr/bin/env python3
"""
快速测试脚本 - 只测试一个最小配置
"""

import subprocess
import sys

def quick_test():
    print("🚀 Quick Test")
    print("=============")
    
    # 只测试一个最小配置
    cmd = [sys.executable, "gpu/myconv.py", "--input_size", "16", "--kernel_size", "3"]
    
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ PyTorch Baseline: SUCCESS")
            
            # 检查输出
            if "GPU Wall Time" in result.stdout:
                print("✅ GPU timing data available")
            if "Correctness check: True" in result.stdout:
                print("✅ Correctness verified")
            
            return True
        else:
            print(f"❌ PyTorch Baseline: FAILED")
            print(f"Error: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n🎉 Quick test passed! Ready for full experiments.")
    else:
        print("\n❌ Quick test failed! Check the issues above.")
    sys.exit(0 if success else 1)
