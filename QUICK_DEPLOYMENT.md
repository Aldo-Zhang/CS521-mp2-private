# 快速部署指南

## 🚀 在虚拟机上的操作步骤

### 1. 拉取最新代码
```bash
git pull
```

### 2. 运行快速测试
```bash
python3 quick_test.py
```

### 3. 如果快速测试通过，运行样板测试
```bash
python3 sample_test.py
```

### 4. 如果样板测试通过，运行完整实验
```bash
# 选择其中一个：
./run_vm_experiments_safe.sh    # 推荐：安全模式 (12个实验)
./run_vm_experiments.sh          # 完整模式 (27个实验)
```

## 🔧 如果遇到问题

### 问题1: JAX CuDNN错误
**解决方案**: JAX已经强制使用CPU，应该不会出现这个问题

### 问题2: Inductor超时
**解决方案**: 使用安全模式脚本，有更长的超时时间

### 问题3: 权限问题
```bash
chmod +x *.py *.sh
```

## 📊 预期输出

### 快速测试成功
```
🚀 Quick Test
=============
Running: python3 gpu/myconv.py --input_size 16 --kernel_size 3
✅ PyTorch Baseline: SUCCESS
✅ GPU timing data available
✅ Correctness verified

🎉 Quick test passed! Ready for full experiments.
```

### 样板测试成功
```
📊 SAMPLE TEST SUMMARY
==================================================
Total tests: 6
Passed: 6
Failed: 0
Success rate: 100.0%

🎉 ALL TESTS PASSED!
✅ Ready for full experiment deployment
```

## ⚡ 快速命令序列

```bash
# 一键部署命令
git pull && python3 quick_test.py && python3 sample_test.py && ./run_vm_experiments_safe.sh
```

## 🎯 成功标准

- ✅ 快速测试通过
- ✅ 样板测试成功率 ≥ 80%
- ✅ 至少PyTorch基线正常工作
- ✅ 能生成GPU Wall Time vs Kernel Time数据

满足以上条件即可运行完整实验！
