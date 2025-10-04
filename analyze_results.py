#!/usr/bin/env python3
"""
结果分析脚本 - 解析实验输出并生成性能对比报告
"""

import re
import sys
from collections import defaultdict
import argparse

def parse_experiment_file(filename):
    """解析实验结果文件"""
    results = defaultdict(dict)
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # 分割配置
    configs = re.split(r'Configuration: Input (\d+)x\1, Kernel (\d+)x\2', content)
    
    for i in range(1, len(configs), 3):
        if i + 2 < len(configs):
            input_size = int(configs[i])
            kernel_size = int(configs[i + 1])
            config_content = configs[i + 2]
            
            # 解析每个版本的结果
            versions = ['PyTorch Baseline', 'PyTorch Inductor', 'JAX']
            for version in versions:
                version_results = parse_version_results(config_content, version)
                if version_results:
                    results[(input_size, kernel_size)][version] = version_results
    
    return results

def parse_version_results(content, version):
    """解析特定版本的结果"""
    # 查找版本部分
    version_start = content.find(f'--- {version} ---')
    if version_start == -1:
        return None
    
    version_end = content.find('---', version_start + 1)
    if version_end == -1:
        version_content = content[version_start:]
    else:
        version_content = content[version_start:version_end]
    
    result = {}
    
    # 解析性能数据
    if version == 'JAX':
        # JAX结果解析
        total_time_match = re.search(r'Total time: ([\d.]+) μs \(([\d.]+) ms\)', version_content)
        if total_time_match:
            result['total_time_us'] = float(total_time_match.group(1))
            result['total_time_ms'] = float(total_time_match.group(2))
    else:
        # PyTorch结果解析
        cuda_match = re.search(r'CUDA time: ([\d.]+) μs \(([\d.]+) ms\)', version_content)
        cpu_match = re.search(r'CPU time:\s+([\d.]+) μs \(([\d.]+) ms\)', version_content)
        
        if cuda_match:
            result['cuda_time_us'] = float(cuda_match.group(1))
            result['cuda_time_ms'] = float(cuda_match.group(2))
        
        if cpu_match:
            result['cpu_time_us'] = float(cpu_match.group(1))
            result['cpu_time_ms'] = float(cpu_match.group(2))
        
        # 计算总时间
        if 'cuda_time_us' in result and 'cpu_time_us' in result:
            result['total_time_us'] = result['cuda_time_us'] + result['cpu_time_us']
            result['total_time_ms'] = result['total_time_ms'] = result['total_time_us'] / 1000.0
    
    # 解析正确性检查
    shape_match = re.search(r'Shape check: (True|False)', version_content)
    correctness_match = re.search(r'Correctness check: (True|False)', version_content)
    
    if shape_match:
        result['shape_check'] = shape_match.group(1) == 'True'
    if correctness_match:
        result['correctness_check'] = correctness_match.group(1) == 'True'
    
    return result if result else None

def generate_report(results):
    """生成性能对比报告"""
    print("=" * 80)
    print("PERFORMANCE COMPARISON REPORT")
    print("=" * 80)
    print()
    
    # 按配置分组显示结果
    for (input_size, kernel_size) in sorted(results.keys()):
        print(f"Configuration: {input_size}x{input_size}, Kernel {kernel_size}x{kernel_size}")
        print("-" * 60)
        
        config_results = results[(input_size, kernel_size)]
        
        # 显示每个版本的结果
        for version in ['PyTorch Baseline', 'PyTorch Inductor', 'JAX']:
            if version in config_results:
                result = config_results[version]
                print(f"{version:20s}: ", end="")
                
                if 'total_time_ms' in result:
                    print(f"{result['total_time_ms']:8.3f} ms", end="")
                else:
                    print("N/A", end="")
                
                # 显示正确性
                if 'shape_check' in result and 'correctness_check' in result:
                    status = "✓" if (result['shape_check'] and result['correctness_check']) else "✗"
                    print(f" {status}", end="")
                
                print()
            else:
                print(f"{version:20s}: FAILED")
        
        print()
    
    # 性能对比总结
    print("=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    
    # 计算平均性能
    versions = ['PyTorch Baseline', 'PyTorch Inductor', 'JAX']
    avg_times = {}
    
    for version in versions:
        times = []
        for config_results in results.values():
            if version in config_results and 'total_time_ms' in config_results[version]:
                times.append(config_results[version]['total_time_ms'])
        
        if times:
            avg_times[version] = sum(times) / len(times)
    
    print("Average execution time (ms):")
    for version in versions:
        if version in avg_times:
            print(f"  {version:20s}: {avg_times[version]:8.3f} ms")
        else:
            print(f"  {version:20s}: N/A")
    
    # 相对性能比较
    if 'PyTorch Baseline' in avg_times:
        baseline_time = avg_times['PyTorch Baseline']
        print("\nRelative performance (vs PyTorch Baseline):")
        for version in versions:
            if version in avg_times and version != 'PyTorch Baseline':
                speedup = baseline_time / avg_times[version]
                print(f"  {version:20s}: {speedup:6.2f}x")
    
    print()

def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('result_file', help='Path to the result file')
    
    args = parser.parse_args()
    
    try:
        results = parse_experiment_file(args.result_file)
        generate_report(results)
    except Exception as e:
        print(f"Error analyzing results: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
