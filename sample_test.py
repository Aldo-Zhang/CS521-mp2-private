#!/usr/bin/env python3
"""
样板测试脚本 - 在虚拟机正式运行前验证代码正确性
使用小模型快速验证所有三个版本都能正常工作
"""

import subprocess
import sys
import time
import os

def run_sample_test(script, input_size, kernel_size, timeout=30):
    """运行单个样板测试"""
    cmd = [sys.executable, script, '--input_size', str(input_size), '--kernel_size', str(kernel_size)]
    
    print(f"🧪 Testing {script} with {input_size}x{input_size} input, {kernel_size}x{kernel_size} kernel...")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✅ SUCCESS ({end_time - start_time:.1f}s)")
            
            # 解析关键信息
            lines = result.stdout.split('\n')
            gpu_wall_time = None
            gpu_kernel_time = None
            correctness = None
            
            for line in lines:
                if "GPU Wall Time" in line and "μs" in line:
                    import re
                    match = re.search(r'(\d+\.\d+)\s+μs', line)
                    if match:
                        gpu_wall_time = float(match.group(1))
                
                if "GPU Kernel Time" in line and "μs" in line:
                    import re
                    match = re.search(r'(\d+\.\d+)\s+μs', line)
                    if match:
                        gpu_kernel_time = float(match.group(1))
                
                if "Correctness check:" in line:
                    correctness = "True" in line
            
            # 显示关键结果
            if gpu_wall_time and gpu_kernel_time:
                efficiency = (gpu_kernel_time / gpu_wall_time) * 100
                print(f"   📊 GPU Wall Time: {gpu_wall_time:.1f} μs")
                print(f"   📊 GPU Kernel Time: {gpu_kernel_time:.1f} μs")
                print(f"   📊 Kernel Efficiency: {efficiency:.1f}%")
            
            if correctness is not None:
                print(f"   ✅ Correctness: {correctness}")
            
            return True
        else:
            print(f"❌ FAILED (exit code: {result.returncode})")
            print(f"   Error: {result.stderr[:200]}...")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT ({timeout}s)")
        return False
    except Exception as e:
        print(f"💥 ERROR: {e}")
        return False

def check_environment():
    """检查运行环境"""
    print("🔍 Checking environment...")
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"   Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查必要的包
    try:
        import torch
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
    except ImportError:
        print("   ❌ PyTorch not available")
        return False
    
    try:
        import jax
        print(f"   JAX: {jax.__version__}")
        print(f"   JAX devices: {jax.devices()}")
    except ImportError:
        print("   ❌ JAX not available")
        return False
    
    # 检查文件存在
    scripts = ["gpu/myconv.py", "gpu/myconv_inductor.py", "gpu/myconv_jax.py"]
    for script in scripts:
        if os.path.exists(script):
            print(f"   ✅ {script}")
        else:
            print(f"   ❌ {script} not found")
            return False
    
    return True

def main():
    print("🚀 Sample Test for VM Deployment")
    print("=" * 50)
    print()
    
    # 检查环境
    if not check_environment():
        print("❌ Environment check failed!")
        return 1
    
    print()
    
    # 样板测试配置 - 使用很小的模型
    sample_configs = [
        (16, 3),   # 很小的输入，小卷积核
        (32, 5),   # 稍大一点
    ]
    
    scripts = [
        "gpu/myconv.py",
        "gpu/myconv_inductor.py", 
        "gpu/myconv_jax.py"
    ]
    
    print("🧪 Running sample tests...")
    print()
    
    total_tests = 0
    passed_tests = 0
    
    for script in scripts:
        script_name = script.split('/')[-1].replace('.py', '')
        print(f"📁 {script_name.upper()}")
        print("-" * 30)
        
        for input_size, kernel_size in sample_configs:
            total_tests += 1
            if run_sample_test(script, input_size, kernel_size, timeout=60):
                passed_tests += 1
            print()
    
    # 总结
    print("=" * 50)
    print("📊 SAMPLE TEST SUMMARY")
    print("=" * 50)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print()
        print("🎉 ALL TESTS PASSED!")
        print("✅ Ready for full experiment deployment")
        print()
        print("Next steps:")
        print("1. Upload all files to your VM")
        print("2. Run: python3 run_experiments.sh")
        print("3. Wait for results and automatic shutdown")
        return 0
    else:
        print()
        print("❌ SOME TESTS FAILED!")
        print("🔧 Please fix issues before deployment")
        return 1

if __name__ == "__main__":
    exit(main())
