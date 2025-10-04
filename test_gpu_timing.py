#!/usr/bin/env python3
"""
测试GPU Wall Time vs GPU Kernel Time的详细分析
"""

import subprocess
import sys
import json
import re

def run_test(script, input_size, kernel_size):
    """运行单个测试并解析结果"""
    cmd = [sys.executable, script, '--input_size', str(input_size), '--kernel_size', str(kernel_size)]
    
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            return parse_output(result.stdout, script, input_size, kernel_size)
        else:
            print(f"✗ FAILED: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        print("✗ TIMEOUT")
        return None
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return None

def parse_output(output, script, input_size, kernel_size):
    """解析输出，提取GPU时间信息"""
    lines = output.split('\n')
    
    result = {
        'script': script,
        'input_size': input_size,
        'kernel_size': kernel_size,
        'gpu_wall_time_us': None,
        'gpu_kernel_time_us': None,
        'gpu_overhead_us': None,
        'memory_time_us': None,
        'other_time_us': None,
        'kernel_efficiency': None,
        'memory_overhead': None,
        'other_overhead': None
    }
    
    for line in lines:
        # 解析GPU Wall Time
        if "GPU Wall Time" in line and "μs" in line:
            match = re.search(r'(\d+\.\d+)\s+μs', line)
            if match:
                result['gpu_wall_time_us'] = float(match.group(1))
        
        # 解析GPU Kernel Time
        if "GPU Kernel Time" in line and "μs" in line:
            match = re.search(r'(\d+\.\d+)\s+μs', line)
            if match:
                result['gpu_kernel_time_us'] = float(match.group(1))
        
        # 解析GPU Overhead
        if "GPU Overhead:" in line and "μs" in line:
            match = re.search(r'(\d+\.\d+)\s+μs', line)
            if match:
                result['gpu_overhead_us'] = float(match.group(1))
        
        # 解析Memory Operations
        if "Memory Operations:" in line and "μs" in line:
            match = re.search(r'(\d+\.\d+)\s+μs', line)
            if match:
                result['memory_time_us'] = float(match.group(1))
        
        # 解析Other Operations
        if "Other Operations:" in line and "μs" in line:
            match = re.search(r'(\d+\.\d+)\s+μs', line)
            if match:
                result['other_time_us'] = float(match.group(1))
        
        # 解析效率百分比
        if "Kernel efficiency:" in line:
            match = re.search(r'(\d+\.\d+)%', line)
            if match:
                result['kernel_efficiency'] = float(match.group(1))
        
        if "Memory overhead:" in line:
            match = re.search(r'(\d+\.\d+)%', line)
            if match:
                result['memory_overhead'] = float(match.group(1))
        
        if "Other overhead:" in line:
            match = re.search(r'(\d+\.\d+)%', line)
            if match:
                result['other_overhead'] = float(match.group(1))
    
    return result

def analyze_results(results):
    """分析结果，生成报告"""
    print("\n" + "="*80)
    print("GPU WALL TIME vs GPU KERNEL TIME ANALYSIS")
    print("="*80)
    
    for result in results:
        if result is None:
            continue
            
        script_name = result['script'].split('/')[-1].replace('.py', '')
        print(f"\n{script_name.upper()} - Input: {result['input_size']}x{result['input_size']}, Kernel: {result['kernel_size']}x{result['kernel_size']}")
        print("-" * 60)
        
        if result['gpu_wall_time_us'] and result['gpu_kernel_time_us']:
            wall_time = result['gpu_wall_time_us']
            kernel_time = result['gpu_kernel_time_us']
            overhead = wall_time - kernel_time
            overhead_percent = (overhead / wall_time) * 100
            
            print(f"GPU Wall Time:    {wall_time:8.3f} μs")
            print(f"GPU Kernel Time:  {kernel_time:8.3f} μs")
            print(f"GPU Overhead:     {overhead:8.3f} μs ({overhead_percent:5.1f}%)")
            
            if result['memory_time_us']:
                memory_percent = (result['memory_time_us'] / wall_time) * 100
                print(f"Memory Transfer:  {result['memory_time_us']:8.3f} μs ({memory_percent:5.1f}%)")
            
            if result['other_time_us']:
                other_percent = (result['other_time_us'] / wall_time) * 100
                print(f"Other Operations: {result['other_time_us']:8.3f} μs ({other_percent:5.1f}%)")
            
            print(f"Kernel Efficiency: {result['kernel_efficiency']:5.1f}%")
            
            # 分析开销来源
            print("\nOverhead Analysis:")
            if result['memory_time_us']:
                print(f"  - Memory transfer overhead: {result['memory_time_us']:8.3f} μs")
            if result['other_time_us']:
                print(f"  - Synchronization overhead: {result['other_time_us']:8.3f} μs")
            
            # 计算理论最大效率
            theoretical_efficiency = (kernel_time / wall_time) * 100
            print(f"  - Theoretical max efficiency: {theoretical_efficiency:5.1f}%")
        else:
            print("❌ Failed to parse GPU timing data")

def main():
    print("=== GPU Timing Analysis Test ===")
    print()
    
    # 测试配置
    test_configs = [
        (32, 3),
        (64, 5),
        (128, 7)
    ]
    
    scripts = [
        "gpu/myconv.py",
        "gpu/myconv_inductor.py"
    ]
    
    results = []
    
    for script in scripts:
        print(f"\nTesting {script}:")
        for input_size, kernel_size in test_configs:
            result = run_test(script, input_size, kernel_size)
            if result:
                results.append(result)
                print("✓ SUCCESS")
            else:
                print("✗ FAILED")
    
    # 分析结果
    analyze_results(results)
    
    # 保存结果到JSON文件
    with open('gpu_timing_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Results saved to: gpu_timing_analysis.json")
    print("🎯 Analysis complete!")
    
    return 0

if __name__ == "__main__":
    exit(main())
