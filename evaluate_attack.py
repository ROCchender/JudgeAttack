## JudgeAttack 评估脚本
# 根据代码示例创建的测试和评估脚本

import argparse
import os
import json
from simple_judge_attack import SimpleJudgeAttack

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='JudgeAttack 评估脚本')
    parser.add_argument(
        '--model', 
        type=str, 
        default='model/Qwen2.5-0.5B-Instruct',
        help='模型路径，支持相对路径和绝对路径 (默认: model/Qwen2.5-0.5B-Instruct)'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/test_data.json',
        help='测试数据文件路径 (默认: data/test_data.json)'
    )
    parser.add_argument(
        '--max-attacks',
        type=int,
        default=10,
        help='单个样本最大攻击次数 (默认: 10)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='评估样本数量 (默认: 100)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results.json',
        help='结果输出文件路径 (默认: results.json)'
    )
    return parser.parse_args()

# 解析命令行参数
args = parse_args()

# 配置参数
MODEL_PATH = args.model
DATA_PATH = args.data
MAX_ATTACKS = args.max_attacks
NUM_SAMPLES = args.num_samples
OUTPUT_PATH = args.output

# 转换为绝对路径（如果需要）
base_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.isabs(DATA_PATH):
    DATA_PATH = os.path.join(base_dir, DATA_PATH)
if not os.path.isabs(OUTPUT_PATH):
    OUTPUT_PATH = os.path.join(base_dir, OUTPUT_PATH)

print("===== JudgeAttack 评估开始 =====")
print(f"使用模型: {MODEL_PATH}")
print(f"测试数据: {DATA_PATH}")
print(f"最大攻击次数: {MAX_ATTACKS}")
print(f"评估样本数量: {NUM_SAMPLES}")

# 创建攻击实例
print("创建 SimpleJudgeAttack 实例...")
judge_attack = SimpleJudgeAttack(
    llm_path=MODEL_PATH,  # 指定训练好的语言模型路径
    max_attacks=MAX_ATTACKS  # 指定最大攻击次数
)

# 运行评估
print(f"开始评估，使用前 {NUM_SAMPLES} 个样本...")
results = judge_attack.evaluate(data_path=DATA_PATH, num_samples=NUM_SAMPLES)  # 调用evaluate方法，对指定数量样本进行评估

# 查看结果
print("\n===== 评估结果 =====")
print(f"ASR: {results['Attack_Success_Rate']:.4f}")  # 打印攻击成功率(ASR)，保留4位小数
print(f"Average Queries: {results['Average_Queries']:.2f}")  # 打印每次攻击的平均查询次数

# 保存结果到文件
print(f"\n保存结果到 {OUTPUT_PATH}...")
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n===== JudgeAttack 评估完成 =====")