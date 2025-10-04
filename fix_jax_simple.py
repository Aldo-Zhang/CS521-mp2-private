#!/usr/bin/env python3
"""
简单的JAX修复脚本
"""

import os

def fix_jax():
    """修复JAX CuDNN问题"""
    print("🔧 Fixing JAX CuDNN issue...")
    
    jax_file = "gpu/myconv_jax.py"
    if not os.path.exists(jax_file):
        print(f"❌ {jax_file} not found")
        return False
    
    # 读取文件
    with open(jax_file, 'r') as f:
        content = f.read()
    
    # 检查是否已经修复
    if "Forcing JAX to use CPU" in content:
        print("✅ JAX already fixed")
        return True
    
    # 查找需要替换的代码段
    old_pattern = '''    # 尝试使用GPU，如果失败则回退到CPU
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
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
        jax.config.update('jax_platform_name', 'cpu')'''
    
    new_pattern = '''    # 强制使用CPU避免CuDNN版本问题
    print("Forcing JAX to use CPU to avoid CuDNN version conflicts")
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    jax.config.update('jax_platform_name', 'cpu')'''
    
    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)
        
        with open(jax_file, 'w') as f:
            f.write(content)
        
        print("✅ JAX CuDNN issue fixed")
        return True
    else:
        print("⚠️  JAX code structure changed, trying alternative fix...")
        
        # 尝试更简单的替换
        if "jax.devices('gpu')" in content:
            # 找到相关行并替换
            lines = content.split('\n')
            new_lines = []
            skip_next = False
            
            for i, line in enumerate(lines):
                if skip_next:
                    skip_next = False
                    continue
                    
                if "尝试使用GPU" in line or "检查JAX GPU可用性" in line:
                    # 替换整个GPU检测块
                    new_lines.append("    # 强制使用CPU避免CuDNN版本问题")
                    new_lines.append("    print(\"Forcing JAX to use CPU to avoid CuDNN version conflicts\")")
                    new_lines.append("    os.environ['JAX_PLATFORM_NAME'] = 'cpu'")
                    new_lines.append("    jax.config.update('jax_platform_name', 'cpu')")
                    skip_next = True
                else:
                    new_lines.append(line)
            
            new_content = '\n'.join(new_lines)
            
            with open(jax_file, 'w') as f:
                f.write(new_content)
            
            print("✅ JAX CuDNN issue fixed (alternative method)")
            return True
        else:
            print("❌ Could not find JAX GPU code to fix")
            return False

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
    print("🔧 Simple VM Fixer")
    print("==================")
    print()
    
    # 修复JAX问题
    fix_jax()
    print()
    
    # 创建快速测试
    create_quick_test()
    print()
    
    print("🎯 Fixes applied!")
    print("Next steps:")
    print("1. Run: python3 quick_test.py")
    print("2. If successful, run: python3 sample_test.py")
    print("3. If all tests pass, run: ./run_vm_experiments_safe.sh")

if __name__ == "__main__":
    main()
