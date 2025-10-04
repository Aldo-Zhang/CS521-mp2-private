import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from myconv import ConvModel

if __name__ == "__main__":
    torch.manual_seed(0)

    # Instantiate your PyTorch model
    N, C, H, W = 2, 3, 19, 19
    x = torch.randn(N, C, H, W).cuda()
    
    model = ConvModel(H, W, in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1).cuda().eval()

    # Torch-Inductor compilation
    scripted_model = torch.compile(model, backend="inductor")
    out = scripted_model(x) # Warmup

    # Profiling tests
    kernel_sizes = [3, 5, 7]
    input_sizes = [32, 64, 128]

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        device = "cuda"
        activities += [ProfilerActivity.CUDA]

    test_count = 0
    
    for H in input_sizes:
        W = H
        x = torch.randn(N, C, H, W).cuda()
        
        for kernel_size in kernel_sizes:
            test_count += 1
            print(f"Running Test {test_count}: Input size {H}x{W}, Kernel size {kernel_size}")
            
            model = ConvModel(H, W, in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1).cuda().eval()
            
            torch.cuda.synchronize()
            with profile(activities=activities, record_shapes=True) as prof:
                with record_function("inductor_model_inference"):
                    scripted_model = torch.compile(model, backend="inductor")
                    out = scripted_model(x)
                torch.cuda.synchronize()
            
            prof.export_chrome_trace(f"inductor_trace_test_{test_count}.json")
            
    print(f"Completed {test_count} tests with PyTorch Inductor")
    
    # # Test your solution
    # conv_ref = F.conv2d(x, model.weight, model.bias, stride=1, padding=1)
    # print("Inductor --- shape check:", out.shape == conv_ref.shape)
    # print("Inductor --- correctness check:", torch.allclose(out, conv_ref, atol=1e-4))