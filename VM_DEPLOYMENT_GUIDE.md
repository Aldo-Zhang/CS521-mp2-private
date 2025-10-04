# 虚拟机部署指南

## 🚀 快速开始

### 1. 上传文件到虚拟机
```bash
# 方法1: 使用scp上传整个项目
scp -r /Users/aldo/UIUC/25Fall/CS521/MP2/mp2/* username@vm-ip:/path/to/project/

# 方法2: 使用rsync同步
rsync -avz --exclude='__pycache__' /Users/aldo/UIUC/25Fall/CS521/MP2/mp2/ username@vm-ip:/path/to/project/
```

### 2. 在虚拟机上运行样板测试
```bash
# 进入项目目录
cd /path/to/project

# 首先运行修复脚本（解决JAX CuDNN问题）
python3 fix_vm_issues.py

# 运行快速测试（只测试一个最小配置）
python3 quick_test.py

# 如果快速测试通过，运行完整样板测试
python3 sample_test.py
```

### 3. 如果样板测试通过，运行完整实验
```bash
# 运行完整实验（会自动关机）
./run_vm_experiments.sh
```

## 📁 需要上传的文件

### 核心文件
- `gpu/myconv.py` - PyTorch基线实现
- `gpu/myconv_inductor.py` - PyTorch Inductor版本
- `gpu/myconv_jax.py` - JAX版本
- `gpu/myconv_interface.py` - CUDA接口（未修改）
- `gpu/myconv_kernel.cu` - CUDA内核（未实现）

### 测试和实验脚本
- `sample_test.py` - 样板测试脚本
- `run_vm_experiments.sh` - 完整实验脚本
- `test_gpu_timing.py` - GPU时间分析脚本
- `test_all_versions.py` - 所有版本测试脚本

### 可选文件
- `generate_walltime_plot.py` - 图表生成脚本
- `VM_DEPLOYMENT_GUIDE.md` - 本指南

## 🔧 环境要求

### Python包
```bash
pip install torch torchvision torchaudio
pip install jax jaxlib
pip install matplotlib numpy
```

### GPU支持
- CUDA 11.8+ (推荐)
- cuDNN 8.0+
- 确保PyTorch和JAX都能检测到GPU

## 📊 实验配置

### 样板测试（快速验证）
- 输入尺寸: 16x16, 32x32
- 卷积核: 3x3, 5x5
- 超时: 60秒/测试

### 完整实验
- 输入尺寸: 32x32, 64x64, 128x128
- 卷积核: 3x3, 5x5, 7x7
- 总实验数: 27个 (3×3×3)

## 🎯 预期输出

### 样板测试输出
```
🚀 Sample Test for VM Deployment
==================================================

🔍 Checking environment...
   Python: 3.8.10
   PyTorch: 2.0.1
   CUDA available: True
   CUDA version: 11.8
   GPU count: 1
   JAX: 0.4.13
   JAX devices: [GpuDevice(id=0)]
   ✅ gpu/myconv.py
   ✅ gpu/myconv_inductor.py
   ✅ gpu/myconv_jax.py

🧪 Running sample tests...

📁 MYCONV
------------------------------
🧪 Testing gpu/myconv.py with 16x16 input, 3x3 kernel...
✅ SUCCESS (2.3s)
   📊 GPU Wall Time: 45.2 μs
   📊 GPU Kernel Time: 38.1 μs
   📊 Kernel Efficiency: 84.3%
   ✅ Correctness: True

...

🎉 ALL TESTS PASSED!
✅ Ready for full experiment deployment
```

### 完整实验输出
- 结果文件: `results/results_YYYYMMDD_HHMMSS.txt`
- 图表文件: `plots/walltime_*.png`
- Git提交: 自动提交所有结果

## ⚠️ 故障排除

### 如果样板测试失败
1. 检查Python环境
2. 检查GPU可用性
3. 检查文件权限
4. 查看错误日志

### 如果JAX GPU失败
- 代码会自动回退到CPU
- 这是正常的，不影响实验

### 如果实验中断
- 检查磁盘空间
- 检查内存使用
- 查看系统日志

## 📈 结果分析

实验完成后，你会得到：
1. **详细时间数据**: GPU Wall Time vs GPU Kernel Time
2. **效率分析**: 内核效率、内存开销、同步开销
3. **性能对比**: 三个框架的性能对比
4. **可视化图表**: Wall Time随输入尺寸和卷积核尺寸的变化

## 🔄 自动化流程

1. 上传代码 → 2. 运行样板测试 → 3. 运行完整实验 → 4. 自动关机

整个过程预计需要30-60分钟，取决于GPU性能。
