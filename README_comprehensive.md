# 综合性能测试系统

## 概述

这个系统提供了完整的性能测试框架，可以比较三种不同实现的卷积性能：
1. **PyTorch Baseline** - 手动实现的im2col卷积
2. **PyTorch Inductor** - 使用torch.compile优化的版本
3. **JAX** - JAX JIT编译版本

## 文件结构

```
gpu/
├── myconv.py              # PyTorch Baseline实现
├── myconv_inductor.py     # PyTorch Inductor实现
├── myconv_jax.py          # JAX实现
└── myconv_interface.py    # CUDA扩展接口

scripts/
├── run_all_experiments.sh # 统一测试脚本
├── analyze_results.py     # 结果分析脚本
└── run_experiments*.sh    # 单独测试脚本
```

## 主要改进

### 1. 统一的性能测量
- 所有版本都使用相同的性能测量函数
- 微秒和毫秒双重显示
- 详细的调试信息

### 2. 命令行参数支持
- 支持自定义输入尺寸、卷积核尺寸等参数
- 统一的参数格式

### 3. 正确性验证
- 所有版本都与PyTorch标准实现对比
- 形状和数值正确性检查

## 使用方法

### 运行单个测试

```bash
# PyTorch Baseline
python3 gpu/myconv.py --input_size 32 --kernel_size 3

# PyTorch Inductor
python3 gpu/myconv_inductor.py --input_size 32 --kernel_size 3

# JAX
python3 gpu/myconv_jax.py --input_size 32 --kernel_size 3
```

### 运行完整测试套件

```bash
# 运行所有27个实验 (3版本 × 9配置)
./run_all_experiments.sh

# 后台运行
nohup ./run_all_experiments.sh > experiment_output.log 2>&1 &

# 使用screen
screen -S experiments
./run_all_experiments.sh
# 按 Ctrl+A, D 分离会话
```

### 分析结果

```bash
# 分析实验结果
python3 analyze_results.py results/all_experiments_YYYYMMDD_HHMMSS.txt
```

## 实验配置

### 测试矩阵
- **输入尺寸**: 32×32, 64×64, 128×128
- **卷积核尺寸**: 3×3, 5×5, 7×7
- **版本**: PyTorch Baseline, PyTorch Inductor, JAX
- **总计**: 27个实验

### 性能指标
- **CUDA时间**: GPU操作时间
- **CPU时间**: CPU操作时间
- **总时间**: 总执行时间
- **编译时间**: 仅Inductor和JAX
- **正确性**: 形状和数值验证

## 输出格式

### 控制台输出
```
=== PyTorch Baseline Performance Test ===
Input size: 32x32
Kernel size: 3
Batch size: 2
Input channels: 4
Output channels: 8

[DEBUG] Total events: 15
[DEBUG] CUDA events with time > 0: 8
[DEBUG] CPU events with time > 0: 7
[DEBUG] Top CUDA operations (μs):
  aten::conv2d: 1234.567 μs
  ...

=== Performance Summary ===
CUDA time: 1234.567 μs (1.235 ms) - CUDA operations
CPU time:  12.345 μs (0.012 ms) - CPU operations
Total time: 1246.912 μs (1.247 ms)

=== Correctness Check ===
Shape check: True
Correctness check: True
```

### 结果文件
- 所有输出保存到带时间戳的文本文件
- 包含详细的性能数据和调试信息
- 自动git提交

## 性能分析

### 预期结果
1. **PyTorch Baseline**: 基准性能，包含手动实现的im2col
2. **PyTorch Inductor**: 编译优化后的性能，包含编译开销
3. **JAX**: JIT编译性能，CPU执行

### 关键指标
- **编译时间**: Inductor和JAX的编译开销
- **推理时间**: 纯计算时间
- **内存使用**: 不同实现的内存效率
- **正确性**: 数值精度验证

## 故障排除

### 常见问题
1. **CUDA不可用**: 检查CUDA环境
2. **JAX错误**: 强制使用CPU模式
3. **内存不足**: 减小batch size或输入尺寸

### 调试命令
```bash
# 检查CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# 检查JAX
python3 -c "import jax; print(jax.default_backend())"

# 手动测试
python3 gpu/myconv.py --input_size 32 --kernel_size 3
```

## 自动化部署

### AWS实例使用
```bash
# 1. 上传代码到实例
scp -i key.pem -r . ec2-user@instance-ip:~/project/

# 2. 在实例上运行
ssh -i key.pem ec2-user@instance-ip
cd project
./run_all_experiments.sh

# 3. 下载结果
scp -i key.pem ec2-user@instance-ip:~/project/results_*.txt ./
```

### 监控进度
```bash
# 查看实时输出
tail -f experiment_output.log

# 查看结果文件
ls -la results_*.txt

# 重新连接screen
screen -r experiments
```

## 结果解读

### 性能对比
- 比较不同版本的执行时间
- 分析编译开销vs推理性能
- 识别性能瓶颈

### 正确性验证
- 确保所有实现产生相同结果
- 验证数值精度
- 检查边界情况

### 报告生成
- 自动生成性能对比报告
- 计算相对性能提升
- 提供可视化建议

这个系统提供了完整的性能测试和分析框架，帮助你深入理解不同编译器的优化效果！
