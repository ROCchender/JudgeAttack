### （一）下载模型
#### 1. 安装Git LFS（大文件支持）
```powershell
# 安装Git LFS
git lfs install
```

#### 2. 从ModelScope下载模型
```powershell
# 进入model文件夹
cd JudgeAttack\model

# 克隆模型仓库(选择自己需要的)
git clone https://www.modelscope.cn/Qwen/Qwen3-0.6B.git
git clone https://www.modelscope.cn/Qwen/Qwen2.5-0.5B-Instruct.git
```

### （二）创建并激活虚拟环境
#### 1. 创建虚拟环境
```powershell
# 使用Python启动器创建虚拟环境
py -m venv venv

#### 2. 激活虚拟环境
```powershell
# 在PowerShell中激活虚拟环境
./venv/Scripts/Activate
```

### （三）安装必要依赖（首次运行必须）

```powershell
# 清华大学源 - GPU版本
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple requests python-dotenv transformers==4.30.0 pandas matplotlib cuda-python accelerate

# CPU版本
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==2.0.0 requests python-dotenv transformers==4.30.0 pandas matplotlib accelerate
```

### （四）使用虚拟环境运行程序

激活虚拟环境后，使用以下命令运行程序：

```powershell
# 在虚拟环境中运行
python judge.py --data data\test_data.json --output results.json --model "model/Qwen3-0.6B"
python judge.py --data data\test_data.json --output results.json --model "model/Qwen2.5-0.5B-Instruct"
```

### （五）参数说明
- `--data`: 数据文件路径
- `--results_report`: 结果输出文件路径
- `--model`: 模型路径，固定为"Qwen/Qwen3-0.6B"
- `--max_attacks`: 单样本最大攻击次数（为10）
- `--num_samples`: 评估样本数

### （六）文件说明
- `judge.py`: 主程序文件，包含攻击逻辑和评估指标计算
- `judge_attack.py`: 攻击策略实现文件，包含自定义攻击策略
- `evaluation_metrics.py`: 评估指标计算文件，包含ASR、平均查询次数等指标计算
- `simple_judge_attack.py`: 简单攻击策略实现文件，包含基于文本替换的攻击策略
- `evaluate_attack.py`: 评估攻击脚本，包含攻击策略评估和结果输出
- `validate_json.py`: 验证JSON文件格式是否正确，包含样本数、标签格式等验证

### （七）注意事项
- 确保在项目根目录下运行命令
- 首次运行前必须先下载模型到本地model文件夹
- 首次运行前必须先安装依赖包
- 如果使用完整路径命令失败，请检查您系统中Python的实际安装位置
- 模型默认路径已配置为本地model文件夹，无需额外指定模型路径参数

### （八）结果验证逻辑
1. 加载模型成功提示：`LLM loaded successfully!`；
2. 样本处理日志：`Processing sample X/10`（显示当前处理样本序号）；
3. 攻击结果日志：成功时显示`Attack successful! Queries used: X`，失败时显示`Attack failed. Queries used: 5`；
4. 最终输出评估指标：包含ASR、平均查询次数、总攻击次数等核心数据。

## 实施步骤
1. 环境搭建：安装Python 3.7+及依赖包（torch、transformers等）；
2. 数据划分：用`train_test_split`将train_data.json划分为训练集（80%）和验证集（20%），用于调试攻击策略；
3. 攻击策略实现：继承`JudgeAttack`类，重写`attack`方法，实现自定义攻击策略（基于文本攻击类型扩展）；
4. 测试与评估：调用`evaluate`方法，用验证集测试攻击效果，优化策略后等待助教用test_data.json（150条）最终评估。

## 改动说明

### 1. 代码改动说明

#### 从简单攻击到完整攻击的代码演进
- **基础架构改进**：优化了`JudgeAttack`抽象基类的设备选择逻辑，实现了GPU/CPU智能切换
- **模型加载优化**：更新了`_load_llm`方法，支持根据设备类型自动选择合适的数据类型（GPU使用float16，CPU使用float32）
- **多模型支持**：添加了对Qwen2.5-0.5B-Instruct模型的完整支持，可以在Qwen3和Qwen2.5模型间灵活切换
- **命令行参数增强**：
  - 在judge.py中添加了`--model-type`参数，支持auto/qwen2.5/qwen3模型类型自动优化
  - 在evaluate_attack.py中添加了完整的命令行参数支持
- **依赖管理更新**：在`requirements.txt`中添加了支持CUDA的PyTorch依赖，确保GPU加速功能
- **安装指南完善**：在readme.md中提供了GPU和CPU版本的详细安装命令

### 2. 测试数据说明

#### test_data.json（150条数据）
- **数据规模**：包含150条比较高质量的判断任务数据，覆盖多领域专业知识
- **数据格式**：每条数据包含`question_id`、`instruction`、`response_a`和`response_b`四个字段
- **领域覆盖**：涵盖人工智能、量子通信、数字金融、国际关系等多个专业领域
- **数据质量**：所有数据经过格式验证，确保JSON结构正确，可以直接用于攻击测试

### 3. 如何利用test_data.json检验代码

#### 验证流程
1. **数据验证**：运行以下命令验证test_data.json的格式是否正确
   ```powershell
   python validate_json.py
   ```
   成功输出：`文件格式正确，包含150条数据记录`

2. **攻击测试**：使用以下命令运行攻击评估
   ```powershell
   # 使用Qwen3-0.6B模型（GPU版本）
   python judge.py --data data\test_data.json --output results.json --model "model/Qwen3-0.6B" --model-type qwen3
   
   # 使用Qwen2.5-0.5B-Instruct模型（GPU版本）
   python judge.py --data data\test_data.json --output results.json --model "model/Qwen2.5-0.5B-Instruct" --model-type qwen2.5
   
   # 使用evaluate_attack.py进行快速评估（支持参数化配置）
   python evaluate_attack.py --model "model/Qwen2.5-0.5B-Instruct" --num-samples 5 --max-attacks 10
   ```

3. **结果分析**：
   - 程序会自动处理全部150条测试数据
   - 输出文件`results.json`包含详细的攻击结果
   - `results_report`文件夹中生成评估指标和可视化图表

4. **关键验证点**：
   - 确认程序输出`使用GPU: CUDA:0`（已正确安装GPU依赖时）
   - 验证模型路径正确加载，如`正在加载模型: D:\github\JudgeAttack\model\Qwen2.5-0.5B-Instruct`
   - 确认处理进度显示`Processing sample X/150`
   - 检查攻击策略尝试和结果，如`尝试攻击策略 1/10: 直接要求选A`
   - 验证最终ASR（攻击成功率）等评估指标
   - 确保成功在Qwen2.5和Qwen3模型间切换，两种模型都能正常工作
   - 确认所有150条样本都能被正确处理，无格式错误或路径错误
