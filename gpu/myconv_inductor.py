import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from myconv import ConvModel
import argparse

def measure_kernel_and_host(run_fn, tag="run"):
    acts = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    with profile(activities=acts, record_shapes=False, profile_memory=False) as prof:
        torch.cuda.synchronize()
        with record_function(tag):
            run_fn()                      
        torch.cuda.synchronize()

    ka = prof.key_averages()

    def cuda_us(e, self_only=True):
        # PyTorch profiler returns time in microseconds (μs)
        # Try different attribute names
        us = None
        if hasattr(e, "self_cuda_time_total"):
            us = e.self_cuda_time_total
        elif hasattr(e, "cuda_time_total"):
            us = e.cuda_time_total
        elif hasattr(e, "device_time"):  # Use device_time instead of deprecated cuda_time
            us = e.device_time
        return us or 0.0

    def cpu_us(e, self_only=True):
        # PyTorch profiler returns time in microseconds (μs)
        us = None
        if hasattr(e, "self_cpu_time_total"):
            us = e.self_cpu_time_total
        elif hasattr(e, "cpu_time_total"):
            us = e.cpu_time_total
        elif hasattr(e, "cpu_time"):
            us = e.cpu_time
        return us or 0.0

    def is_compilation_event(e):
        key = e.key.lower()
        compilation_keywords = [
            'dynamo', 'compile', 'graph', 'pass', 'joint', 'recursive',
            'inductor', 'torch-compiled', 'compiledfunction'
        ]
        return any(keyword in key for keyword in compilation_keywords)
    
    # 分类所有事件
    compilation_events = [e for e in ka if is_compilation_event(e)]
    inference_events = [e for e in ka if not is_compilation_event(e)]
    
    # 计算编译时间
    compilation_cuda_us = sum(cuda_us(e, self_only=True) for e in compilation_events)
    compilation_cpu_us = sum(cpu_us(e, self_only=True) for e in compilation_events)
    
    # 计算推理时间
    inference_cuda_us = sum(cuda_us(e, self_only=True) for e in inference_events)
    inference_cpu_us = sum(cpu_us(e, self_only=True) for e in inference_events)
    
    # 计算总时间（所有事件）
    total_cuda_us = sum(cuda_us(e, self_only=True) for e in ka)
    total_cpu_us = sum(cpu_us(e, self_only=True) for e in ka)
    
    # 验证时间计算
    print(f"[DEBUG] Time verification:")
    print(f"  Compilation CUDA: {compilation_cuda_us:.3f} μs")
    print(f"  Compilation CPU:  {compilation_cpu_us:.3f} μs")
    print(f"  Inference CUDA:   {inference_cuda_us:.3f} μs")
    print(f"  Inference CPU:    {inference_cpu_us:.3f} μs")
    print(f"  Total CUDA:       {total_cuda_us:.3f} μs")
    print(f"  Total CPU:        {total_cpu_us:.3f} μs")
    print(f"  Sum verification: {compilation_cuda_us + compilation_cpu_us + inference_cuda_us + inference_cpu_us:.3f} μs")
    
    # 转换为毫秒
    inference_kernel_ms = inference_cuda_us / 1000.0
    inference_host_ms = inference_cpu_us / 1000.0
    compilation_kernel_ms = compilation_cuda_us / 1000.0
    compilation_host_ms = compilation_cpu_us / 1000.0
    
    # 添加详细调试信息
    print(f"[DEBUG] Total events: {len(ka)}")
    print(f"[DEBUG] Compilation events: {len(compilation_events)}")
    print(f"[DEBUG] Inference events: {len(inference_events)}")
    
    # 分析推理GPU时间组成
    inference_cuda_events = [(e.key, cuda_us(e), cpu_us(e)) for e in inference_events if cuda_us(e) > 0]
    inference_cuda_events.sort(key=lambda x: x[1], reverse=True)
    
    # 分类推理GPU操作
    inference_kernel_ops = []
    inference_memory_ops = []
    inference_other_ops = []
    
    for key, cuda_time, cpu_time in inference_cuda_events:
        if 'memcpy' in key.lower() or 'memset' in key.lower():
            inference_memory_ops.append((key, cuda_time, cpu_time))
        elif any(kw in key.lower() for kw in ['conv', 'gemm', 'add', 'mul', 'relu']):
            inference_kernel_ops.append((key, cuda_time, cpu_time))
        else:
            inference_other_ops.append((key, cuda_time, cpu_time))
    
    # 计算推理各类操作的时间
    inference_kernel_time = sum(cuda_time for _, cuda_time, _ in inference_kernel_ops)
    inference_memory_time = sum(cuda_time for _, cuda_time, _ in inference_memory_ops)
    inference_other_time = sum(cuda_time for _, cuda_time, _ in inference_other_ops)
    
    print(f"[DEBUG] Inference GPU Time Analysis:")
    print(f"  Total inference GPU time: {inference_cuda_us:.3f} μs")
    print(f"  Pure kernel operations: {inference_kernel_time:.3f} μs ({inference_kernel_time/inference_cuda_us*100:.1f}%)")
    print(f"  Memory operations: {inference_memory_time:.3f} μs ({inference_memory_time/inference_cuda_us*100:.1f}%)")
    print(f"  Other operations: {inference_other_time:.3f} μs ({inference_other_time/inference_cuda_us*100:.1f}%)")
    
    print(f"[DEBUG] Top inference GPU operations:")
    for key, cuda_time, cpu_time in inference_cuda_events[:5]:
        print(f"  {key:40s} GPU: {cuda_time:8.3f} μs  CPU: {cpu_time:8.3f} μs")
    
    # 显示编译操作
    compilation_cuda_events = [(e.key, cuda_us(e), cpu_us(e)) for e in compilation_events if cuda_us(e) > 0]
    compilation_cuda_events.sort(key=lambda x: x[1], reverse=True)
    print(f"[DEBUG] Top compilation operations:")
    for key, cuda_time, cpu_time in compilation_cuda_events[:3]:
        print(f"  {key:40s} GPU: {cuda_time:8.3f} μs  CPU: {cpu_time:8.3f} μs")
    
    # 计算GPU wall time和kernel time
    gpu_wall_time = inference_cuda_us
    gpu_kernel_time = inference_kernel_time
    
    print(f"[DEBUG] GPU Time Breakdown:")
    print(f"  GPU Wall Time (inference): {gpu_wall_time:.3f} μs")
    print(f"  GPU Kernel Time (pure): {gpu_kernel_time:.3f} μs")
    print(f"  GPU Overhead: {gpu_wall_time - gpu_kernel_time:.3f} μs ({((gpu_wall_time - gpu_kernel_time)/gpu_wall_time*100):.1f}%)")

    return {
        "inference_kernel_ms": inference_kernel_ms,
        "inference_kernel_us": inference_cuda_us,
        "inference_host_ms": inference_host_ms,
        "inference_host_us": inference_cpu_us,
        "compilation_cuda_us": compilation_cuda_us,
        "compilation_cpu_us": compilation_cpu_us,
        "total_cuda_us": total_cuda_us,
        "total_cpu_us": total_cpu_us,
        "gpu_wall_time_us": gpu_wall_time,
        "gpu_kernel_time_us": gpu_kernel_time,
        "gpu_overhead_us": gpu_wall_time - gpu_kernel_time,
        "memory_time_us": inference_memory_time,
        "other_time_us": inference_other_time,
        "prof": prof
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Inductor Performance Test')
    parser.add_argument('--input_size', type=int, required=True, 
                       help='Input size (H=W, e.g., 32 for 32x32)')
    parser.add_argument('--kernel_size', type=int, required=True,
                       help='Kernel size (e.g., 3, 5, 7)')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size (default: 2)')
    parser.add_argument('--channels', type=int, default=3,
                       help='Input channels (default: 4)')
    parser.add_argument('--out_channels', type=int, default=8,
                       help='Output channels (default: 8)')
    
    args = parser.parse_args()
    
    torch.manual_seed(0)
    
    # 从命令行参数获取配置
    H = W = args.input_size
    kernel_size = args.kernel_size
    N = args.batch_size
    C = args.channels
    out_channels = args.out_channels
    
    print(f"=== PyTorch Inductor Performance Test ===")
    print(f"Input size: {H}x{W}")
    print(f"Kernel size: {kernel_size}")
    print(f"Batch size: {N}")
    print(f"Input channels: {C}")
    print(f"Output channels: {out_channels}")
    print()
    
    # 创建输入数据
    x = torch.randn(N, C, H, W).cuda()
    
    # 创建模型
    model = ConvModel(H, W, C, out_channels, kernel_size, stride=1, padding=1).cuda().eval()
    
    # 定义run_fn: 包含compile + 推理
    def run_fn():
        scripted_model = torch.compile(model, backend="inductor")
        out = scripted_model(x)
        return out
    
    # 测量compile + 推理的总时间
    result = measure_kernel_and_host(
        run_fn, 
        tag=f"inductor_{H}x{W}_k{kernel_size}"
    )
    

    print()
    print("=== Results ===")
    print()
    print("=== Performance Summary ===")
    print(f"GPU Wall Time (inference): {result['gpu_wall_time_us']:.3f} μs ({result['gpu_wall_time_us']/1000:.3f} ms)")
    print(f"GPU Kernel Time (pure):   {result['gpu_kernel_time_us']:.3f} μs ({result['gpu_kernel_time_us']/1000:.3f} ms)")
    print(f"GPU Overhead:             {result['gpu_overhead_us']:.3f} μs ({result['gpu_overhead_us']/1000:.3f} ms)")
    print(f"Memory Operations:        {result['memory_time_us']:.3f} μs ({result['memory_time_us']/1000:.3f} ms)")
    print(f"Other Operations:         {result['other_time_us']:.3f} μs ({result['other_time_us']/1000:.3f} ms)")
    print(f"CPU Host Time:            {result['inference_host_us']:.3f} μs ({result['inference_host_ms']:.3f} ms)")
    print(f"Compilation CUDA time:    {result['compilation_cuda_us']:.3f} μs ({result['compilation_cuda_us']/1000:.3f} ms)")
    print(f"Compilation CPU time:     {result['compilation_cpu_us']:.3f} μs ({result['compilation_cpu_us']/1000:.3f} ms)")
    print(f"Total time (all):        {result['total_cuda_us'] + result['total_cpu_us']:.3f} μs ({(result['total_cuda_us'] + result['total_cpu_us'])/1000:.3f} ms)")
    print()
    print("=== GPU Time Analysis ===")
    overhead_percent = (result['gpu_overhead_us'] / result['gpu_wall_time_us']) * 100
    kernel_percent = (result['gpu_kernel_time_us'] / result['gpu_wall_time_us']) * 100
    memory_percent = (result['memory_time_us'] / result['gpu_wall_time_us']) * 100
    print(f"Kernel efficiency: {kernel_percent:.1f}% (pure computation)")
    print(f"Memory overhead:   {memory_percent:.1f}% (data transfer)")
    print(f"Other overhead:    {overhead_percent - memory_percent:.1f}% (synchronization, etc.)")
    print()
    print("=== Time Breakdown ===")
    print(f"Inference only:          {result['inference_kernel_us'] + result['inference_host_us']:.3f} μs")
    print(f"Compilation only:        {result['compilation_cuda_us'] + result['compilation_cpu_us']:.3f} μs")
    print(f"Verification:            {result['inference_kernel_us'] + result['inference_host_us'] + result['compilation_cuda_us'] + result['compilation_cpu_us']:.3f} μs")