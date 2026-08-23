## 运行步骤

### 第1步: 数据预处理

```bash
python /path/to/prepare_glpo_data.py \
    --input  processed_data/grpo_train.jsonl \
    --output processed_data/grpo_train_M4.jsonl \
    --M 4
```

检查输出行数是原来的4倍。

### 第2步: 替换 plugin

```bash
cp glpo_swift_plugin_v3.py /root/data/MyLearn/YuYi/
```

### 第3步: 启动训练

```bash
bash run_glpo.sh
```