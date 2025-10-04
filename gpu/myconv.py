import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

class ConvModel(nn.Module):
    def __init__(self, H, W, in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding

        self.H = H
        self.W = W

        # TO DO: Define static shapes here. 

        # Precompute output size
        #out_size = (in_size + 2*padding - kernel_size) / stride + 1
        self.out_h = (H + 2 * padding - kernel_size) // stride + 1
        self.out_w = (W + 2 * padding - kernel_size) // stride + 1

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        

    def im2col_manual(self, x):
        N = x.shape[0]        # batch size can remain dynamic
        C = self.in_channels
        KH = KW = self.kernel_size
        S = self.stride
        P = self.padding
        out_h = self.out_h
        out_w = self.out_w

        # Pad input
        x_pad = F.pad(x, (P, P, P, P))

        # TO DO: Convert input (x) into shape (N, out_h*out_w, C*KH*KW). 
        # Refer to Lecture 3 for implementing this operation.
        
        # Create output tensor to store patches
        patches = torch.zeros(N, out_h * out_w, C * KH * KW, device=x.device, dtype=x.dtype)
        
        # Iterate over each output position and decide the corresponding input
        for i in range(out_h):
            for j in range(out_w):
                start_h = i * S
                start_w = j * S
                end_h = start_h + KH
                end_w = start_w + KW
                
                # Extract from the original X[N, C, H, W]
                window = x_pad[:, :, start_h:end_h, start_w:end_w]
                flat_window = window.reshape(N, C * KH * KW) 
                
                # Store the patches
                patches[:, i * out_w + j, :] = flat_window
        
        return patches

    def conv2d_manual(self, x):
        N = x.shape[0]
        C_out = self.out_channels
        KH = KW = self.kernel_size
        C = self.in_channels

        # TO DO: 1) convert input (x) into shape (N, out_h*out_w, C*KH*KW).
        cols = self.im2col_manual(x)

        # TO DO: 2) flatten self.weight into shape (C_out, C*KH*KW).
        flat_weight = self.weight.reshape(C_out, C*KH*KW)

        # TO DO: 3) perform tiled matmul after required reshaping is done.
        cols_reshaped = cols.reshape(-1, C*KH*KW)  # Reshape the cols to be a 2-d matrix
        R = cols_reshaped.shape[0]
        out_flat = torch.empty(R, C_out, device=x.device, dtype=x.dtype)
        RB = 128    # rows tile
        CB = 128    # cols tile
        for r0 in range(0, R, RB):
            r1 = min(r0 + RB, R)
            A = cols_reshaped[r0:r1, :]             # (rb, K)

            for c0 in range(0, C_out, CB):
                c1 = min(c0 + CB, C_out)
                B = flat_weight[c0:c1, :].T         # (K, cb)
                out_flat[r0:r1, c0:c1] = torch.mm(A, B)
        
        # TO DO: 4) Add bias.
        out_flat = out_flat + self.bias  # (N*out_h*out_w, C_out)

        # TO DO: 5) reshape output into shape (N, C_out, out_h, out_w).
        out = out_flat.reshape(N, self.out_h * self.out_w, C_out)  
        out = out.permute(0, 2, 1) 
        out = out.reshape(N, C_out, self.out_h, self.out_w)

        return out

    def forward(self, x):
        return self.conv2d_manual(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    N, C, H, W = 2, 4, 32, 32 # change the size to be 32 * 32
    x = torch.randn(N, C, H, W).cuda()
    out_channels=8
    kernel_size=3 # change kernel size to 3
    model = ConvModel(H, W, C, out_channels, kernel_size, stride=1, padding=1).to(x.device)
    # Instantiate a profiler
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        device = "cuda"
        activities += [ProfilerActivity.CUDA]

    # Warmup
    torch.cuda.synchronize()
    _ = model(x)                 # warmup 1
    _ = F.conv2d(x, model.weight, model.bias, stride=1, padding=1)  # warmup 2
    torch.cuda.synchronize()

    # Test1 input size 32 * 32 Kernel Size: 3
    torch.cuda.synchronize()
    with profile(activities=activities, record_shapes=True) as prof1:
        with record_function("model_inference"):
            out = model(x)
        torch.cuda.synchronize()

    # Test2 input size 32 * 32 Kernel Size: 5
    kernel_size = 5
    model = ConvModel(H, W, C, out_channels, kernel_size, stride=1, padding=1).to(x.device)
    torch.cuda.synchronize()
    with profile(activities=activities, record_shapes=True) as prof2:
        with record_function("model_inference"):
            out = model(x)
        torch.cuda.synchronize()

    # Test3 input size 32 * 32 Kernel Size: 7
    kernel_size = 7
    model = ConvModel(H, W, C, out_channels, kernel_size, stride=1, padding=1).to(x.device)
    torch.cuda.synchronize()
    with profile(activities=activities, record_shapes=True) as prof3:
        with record_function("model_inference"):
            out = model(x)
        torch.cuda.synchronize()
    # Test4 input size 64 * 64 Kernel Size: 3
    kernel_size = 3
    H, W = 64, 64
    x = torch.randn(N, C, H, W).cuda()
    model = ConvModel(H, W, C, out_channels, kernel_size, stride=1, padding=1).to(x.device)
    torch.cuda.synchronize()
    with profile(activities=activities, record_shapes=True) as prof4:
        with record_function("model_inference"):
            out = model(x)
        torch.cuda.synchronize()

    # Test5 input size 64 * 64 Kernel Size: 5
    kernel_size = 5
    model = ConvModel(H, W, C, out_channels, kernel_size, stride=1, padding=1).to(x.device)
    torch.cuda.synchronize()
    with profile(activities=activities, record_shapes=True) as prof5:
        with record_function("model_inference"):
            out = model(x)
        torch.cuda.synchronize()

    # Test6 input size 64 * 64 Kernel Size: 7
    kernel_size = 7
    model = ConvModel(H, W, C, out_channels, kernel_size, stride=1, padding=1).to(x.device)
    torch.cuda.synchronize()
    with profile(activities=activities, record_shapes=True) as prof6:
        with record_function("model_inference"):
            out = model(x)
        torch.cuda.synchronize()
    #Test7 input size 128 * 128 Kernel Size: 3
    kernel_size = 3
    H, W = 128, 128
    x = torch.randn(N, C, H, W).cuda()
    model = ConvModel(H, W, C, out_channels, kernel_size, stride=1, padding=1).to(x.device)
    torch.cuda.synchronize()
    with profile(activities=activities, record_shapes=True) as prof7:
        with record_function("model_inference"):
            out = model(x)
        torch.cuda.synchronize()

    # # Test8 input size 128 * 128 Kernel Size: 5
    # H, W = 128, 128
    # kernel_size = 5
    # x = torch.randn(N, C, H, W).cuda()
    # # Warmup
    # torch.cuda.synchronize()
    # _ = model(x)                 # warmup 1
    # _ = F.conv2d(x, model.weight, model.bias, stride=1, padding=1)  # warmup 2
    # torch.cuda.synchronize()

    # model = ConvModel(H, W, C, out_channels, kernel_size, stride=1, padding=1).to(x.device)
    # torch.cuda.synchronize()
    # with profile(activities=activities, record_shapes=True) as prof8:
    #     with record_function("model_inference"):
    #         out = model(x)
    #     torch.cuda.synchronize()
    # # Test9 input size 128 * 128 Kernel Size: 7
    # kernel_size = 7
    # model = ConvModel(H, W, C, out_channels, kernel_size, stride=1, padding=1).to(x.device)
    # torch.cuda.synchronize()
    # with profile(activities=activities, record_shapes=True) as prof9:
    #     with record_function("model_inference"):
    #         out = model(x)
    #     torch.cuda.synchronize()

    prof1.export_chrome_trace("baseline_trace_test_1.json")
    prof2.export_chrome_trace("baseline_trace_test_2.json")
    prof3.export_chrome_trace("baseline_trace_test_3.json")
    prof4.export_chrome_trace("baseline_trace_test_4.json")
    prof5.export_chrome_trace("baseline_trace_test_5.json")
    prof6.export_chrome_trace("baseline_trace_test_6.json")
    prof7.export_chrome_trace("baseline_trace_test_7.json")
    # prof8.export_chrome_trace("baseline_trace_test_8.json")
    # prof9.export_chrome_trace("baseline_trace_test_9.json")
    # Test your solution (shape and correctness)
    # conv_ref = F.conv2d(x, model.weight, model.bias, stride=1, padding=1)
    # print("PyTorch --- shape check:", out.shape == conv_ref.shape)
    # print("PyTorch --- correctness check:", torch.allclose(out, conv_ref, atol=1e-4))
