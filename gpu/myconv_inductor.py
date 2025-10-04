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

    # 过滤掉编译相关的操作
    def is_compilation_event(e):
        key = e.key.lower()
        compilation_keywords = [
            'dynamo', 'compile', 'graph', 'pass', 'joint', 'recursive',
            'inductor', 'torch-compiled', 'compiledfunction'
        ]
        return any(keyword in key for keyword in compilation_keywords)
    
    # 只计算实际的kernel操作时间
    kernel_events = [e for e in ka if not is_compilation_event(e) and cuda_us(e) > 0]
    cpu_events = [e for e in ka if not is_compilation_event(e) and cpu_us(e) > 0]
    
    # 计算总时间（微秒）
    total_kernel_us = sum(cuda_us(e, self_only=True) for e in kernel_events)
    total_host_us   = sum(cpu_us(e,  self_only=True) for e in cpu_events)
    
    # 计算编译时间
    compilation_events = [e for e in ka if is_compilation_event(e)]
    compilation_cuda_us = sum(cuda_us(e, self_only=True) for e in compilation_events)
    compilation_cpu_us = sum(cpu_us(e, self_only=True) for e in compilation_events)
    
    # 转换为毫秒
    total_kernel_ms = total_kernel_us / 1000.0
    total_host_ms = total_host_us / 1000.0
    
    # 添加详细调试信息
    print(f"[DEBUG] Total events: {len(ka)}")
    print(f"[DEBUG] Compilation events: {len(compilation_events)}")
    print(f"[DEBUG] Kernel events (filtered): {len(kernel_events)}")
    print(f"[DEBUG] CPU events (filtered): {len(cpu_events)}")
    
    # 显示编译时间
    print(f"[DEBUG] Compilation time:")
    print(f"  CUDA: {compilation_cuda_us:.3f} μs ({compilation_cuda_us/1000:.3f} ms)")
    print(f"  CPU:  {compilation_cpu_us:.3f} μs ({compilation_cpu_us/1000:.3f} ms)")
    
    # 显示前几个最耗时的kernel操作
    kernel_events_sorted = [(e.key, cuda_us(e)) for e in kernel_events]
    kernel_events_sorted.sort(key=lambda x: x[1], reverse=True)
    print(f"[DEBUG] Top kernel operations (μs):")
    for key, time_us in kernel_events_sorted[:5]:
        print(f"  {key}: {time_us:.3f} μs")

    return {"kernel_ms": total_kernel_ms,
            "kernel_us": total_kernel_us,
            "host_ms": total_host_ms,
            "host_us": total_host_us,
            "compilation_cuda_us": compilation_cuda_us,
            "compilation_cpu_us": compilation_cpu_us,
            "prof": prof}

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
    print(f"Kernel time (inference): {result['kernel_us']:.3f} μs ({result['kernel_ms']:.3f} ms) - CUDA operations")
    print(f"Host time (inference):   {result['host_us']:.3f} μs ({result['host_ms']:.3f} ms) - CPU operations")
    print(f"Compilation CUDA time:   {result['compilation_cuda_us']:.3f} μs ({result['compilation_cuda_us']/1000:.3f} ms)")
    print(f"Compilation CPU time:    {result['compilation_cpu_us']:.3f} μs ({result['compilation_cpu_us']/1000:.3f} ms)")
    print(f"Total inference time:    {result['kernel_us'] + result['host_us']:.3f} μs ({(result['kernel_ms'] + result['host_ms']):.3f} ms)")
    print(f"Total time (all):        {result['kernel_us'] + result['host_us'] + result['compilation_cuda_us'] + result['compilation_cpu_us']:.3f} μs")