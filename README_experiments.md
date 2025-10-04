# 实验脚本使用说明

## 脚本说明

我为你创建了3个实验脚本，用于自动化运行9个PyTorch Inductor实验：

### 1. `run_experiments_simple.sh` (推荐)
- **最简单版本**，适合快速执行
- 运行9个实验配置
- 自动保存结果到文件
- 自动git提交
- 10秒后自动关闭实例

### 2. `run_experiments.sh` 
- **标准版本**，包含详细输出
- 实时显示进度
- 错误处理
- 结果验证

### 3. `run_experiments_detailed.sh`
- **最详细版本**，包含完整日志
- 环境检查
- 超时处理
- 详细统计信息

## 使用方法

### 在AWS实例上运行：

```bash
# 方法1: 直接运行简单版本
./run_experiments_simple.sh

# 方法2: 后台运行（推荐）
nohup ./run_experiments_simple.sh > experiment_output.log 2>&1 &

# 方法3: 使用screen（推荐）
screen -S experiments
./run_experiments_simple.sh
# 按 Ctrl+A, D 分离会话
```

### 监控进度：

```bash
# 查看实时输出
tail -f experiment_output.log

# 查看结果文件
ls -la results_*.txt

# 重新连接screen会话
screen -r experiments
```

## 实验配置

脚本会运行以下9个配置：

| 测试 | 输入尺寸 | 卷积核尺寸 |
|------|----------|------------|
| 1    | 32x32    | 3x3        |
| 2    | 32x32    | 5x5        |
| 3    | 32x32    | 7x7        |
| 4    | 64x64    | 3x3        |
| 5    | 64x64    | 5x5        |
| 6    | 64x64    | 7x7        |
| 7    | 128x128  | 3x3        |
| 8    | 128x128  | 5x5        |
| 9    | 128x128  | 7x7        |

## 输出文件

- `results_YYYYMMDD_HHMMSS.txt`: 实验结果文件
- `experiment_output.log`: 运行日志（如果使用nohup）
- Git提交记录

## 注意事项

1. **确保在正确的目录**: 脚本需要在包含`gpu/myconv_inductor.py`的目录中运行
2. **CUDA环境**: 确保CUDA可用
3. **Git配置**: 确保git已配置用户信息
4. **权限**: 脚本需要sudo权限来关闭实例

## 故障排除

如果实验失败：
1. 检查CUDA环境: `python3 -c "import torch; print(torch.cuda.is_available())"`
2. 检查Python环境: `python3 --version`
3. 手动运行单个测试: `python3 gpu/myconv_inductor.py --input_size 32 --kernel_size 3`

## 推荐使用方式

```bash
# 1. 进入项目目录
cd /path/to/your/project

# 2. 使用screen运行（推荐）
screen -S experiments
./run_experiments_simple.sh

# 3. 分离会话（实验继续运行）
# 按 Ctrl+A, 然后按 D

# 4. 重新连接查看进度
screen -r experiments
```
