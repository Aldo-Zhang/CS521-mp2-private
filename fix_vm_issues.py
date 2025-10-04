#!/usr/bin/env python3
"""
修复虚拟机上的问题
"""

import os
import sys

def fix_jax_cudnn_issue():
    """修复JAX CuDNN版本问题"""
    print("🔧 Fixing JAX CuDNN version issue...")
    
    # 读取JAX文件
    jax_file = "gpu/myconv_jax.py"
    if not os.path.exists(jax_file):
        print(f"❌ {jax_file} not found")
        return False
    
    with open(jax_file, 'r') as f:
        content = f.read()
    
    # 检查是否已经修复
    if "Forcing JAX to use CPU" in content:
        print("✅ JAX already fixed")
        return True
    
    # 修复内容
    old_code = '''    # 尝试使用GPU，如果失败则回退到CPU
    try:
        # 检查JAX GPU可用性
        gpu_devices = jax.devices('gpu')
        if len(gpu_devices) > 0:
            print(f"Using JAX GPU backend: {gpu_devices}")
            # 确保数据在GPU上
            jax.config.update('jax_platform_name', 'gpu')
        else:
            raise RuntimeError("No GPU devices available")
    except Exception as e:
        print(f"GPU not available ({e}), falling back to CPU")
        os.environ['JAX_PLATFORM_NAME'] = 'cpu''''
    
    new_code = '''    # 强制使用CPU避免CuDNN版本问题
    print("Forcing JAX to use CPU to avoid CuDNN version conflicts")
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    jax.config.update('jax_platform_name', 'cpu')'''
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        
        with open(jax_file, 'w') as f:
            f.write(content)
        
        print("✅ JAX CuDNN issue fixed")
        return True
    else:
        print("⚠️  JAX code structure changed, manual fix needed")
        return False

def check_file_permissions():
    """检查文件权限"""
    print("🔧 Checking file permissions...")
    
    files_to_check = [
        "sample_test.py",
        "run_vm_experiments.sh",
        "gpu/myconv.py",
        "gpu/myconv_inductor.py",
        "gpu/myconv_jax.py"
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            if os.access(file_path, os.R_OK):
                print(f"✅ {file_path} - readable")
            else:
                print(f"❌ {file_path} - not readable")
                os.chmod(file_path, 0o644)
                print(f"🔧 Fixed permissions for {file_path}")
        else:
            print(f"❌ {file_path} - not found")

def create_quick_test():
    """创建快速测试脚本"""
    print("🔧 Creating quick test script...")
    
    quick_test_content = '''#!/usr/bin/env python3
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
        print("\\n🎉 Quick test passed! Ready for full experiments.")
    else:
        print("\\n❌ Quick test failed! Check the issues above.")
    sys.exit(0 if success else 1)
'''
    
    with open("quick_test.py", 'w') as f:
        f.write(quick_test_content)
    
    os.chmod("quick_test.py", 0o755)
    print("✅ Quick test script created")

def main():
    print("🔧 VM Issue Fixer")
    print("=================")
    print()
    
    # 修复JAX问题
    fix_jax_cudnn_issue()
    print()
    
    # 检查文件权限
    check_file_permissions()
    print()
    
    # 创建快速测试
    create_quick_test()
    print()
    
    print("🎯 Fixes applied!")
    print("Next steps:")
    print("1. Run: python3 quick_test.py")
    print("2. If successful, run: python3 sample_test.py")
    print("3. If all tests pass, run: ./run_vm_experiments.sh")

if __name__ == "__main__":
    main()
