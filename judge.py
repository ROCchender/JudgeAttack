#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JudgeAttack主脚本
提供命令行接口和评估逻辑
"""

import argparse
import json
import os
import time
from typing import List, Dict, Any

from simple_judge_attack import SimpleJudgeAttack
from evaluation_metrics import AttackEvaluation

class JudgeAttackCLI:
    """
    JudgeAttack命令行工具类
    处理命令行参数和评估流程
    """
    
    def __init__(self):
        """
        初始化CLI工具
        设置命令行参数解析器
        """
        self.parser = argparse.ArgumentParser(
            description='JudgeAttack: 大模型评判攻击工具',
            formatter_class=argparse.RawTextHelpFormatter
        )
        
        # 必要参数
        self.parser.add_argument(
            '--model', 
            type=str, 
            default='model/Qwen2.5-0.5B-Instruct',
            help='模型路径，支持相对路径和绝对路径\n' +
                 '推荐使用相对路径: model/Qwen2.5-0.5B-Instruct 或 model/Qwen3-0.6B'
        )
        
        self.parser.add_argument(
            '--data',
            type=str,
            required=True,
            help='评估数据文件路径 (JSON格式)'
        )
        
        # 可选参数
        self.parser.add_argument(
            '--output',
            type=str,
            default='attack_results.json',
            help='输出结果文件路径 (默认: attack_results.json)'
        )
        
        self.parser.add_argument(
            '--max-attacks',
            type=int,
            default=10,
            help='单个样本最大攻击次数 (默认: 10)'
        )
        
        self.parser.add_argument(
            '--batch-size',
            type=int,
            default=1,
            help='批处理大小 (默认: 1)'
        )
        
        self.parser.add_argument(
            '--attack-type',
            type=str,
            default='simple',
            choices=['simple'],
            help='攻击类型 (默认: simple)'
        )
        
        self.parser.add_argument(
            '--debug',
            action='store_true',
            help='启用调试模式，输出详细信息'
        )
        
        self.parser.add_argument(
            '--model-type',
            type=str,
            default='auto',
            choices=['auto', 'qwen2.5', 'qwen3'],
            help='模型类型提示，用于优化参数设置 (默认: auto自动检测)'
        )
    
    def load_data(self, data_path: str) -> List[Dict[str, str]]:
        """
        加载评估数据
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            List[Dict]: 评估数据列表
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 验证数据格式
            if not isinstance(data, list):
                raise ValueError("数据文件必须是JSON数组格式")
            
            # 验证每条数据的格式
            required_fields = ['question_id', 'instruction', 'response_a', 'response_b']
            for i, item in enumerate(data):
                missing_fields = [field for field in required_fields if field not in item]
                if missing_fields:
                    raise ValueError(f"数据项{i+1}缺少必要字段: {', '.join(missing_fields)}")
            
            print(f"成功加载 {len(data)} 条评估数据")
            return data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"数据文件格式错误: {str(e)}")
    
    def create_attacker(self, args: argparse.Namespace) -> SimpleJudgeAttack:
        """
        创建攻击实例
        
        Args:
            args: 命令行参数
            
        Returns:
            SimpleJudgeAttack: 攻击实例
        """
        if args.attack_type == 'simple':
            return SimpleJudgeAttack(
                llm_path=args.model,
                max_attacks=args.max_attacks
            )
        else:
            raise ValueError(f"不支持的攻击类型: {args.attack_type}")
    
    def run_attack(self, attacker: SimpleJudgeAttack, data: Dict[str, str]) -> Dict[str, Any]:
        """
        对单个样本执行攻击
        
        Args:
            attacker: 攻击实例
            data: 样本数据
            
        Returns:
            Dict: 攻击结果
        """
        try:
            start_time = time.time()
            
            # 执行攻击
            success, queries, new_judgment = attacker.attack(
                instruction=data['instruction'],
                response_a=data['response_a'],
                response_b=data['response_b']
            )
            
            end_time = time.time()
            
            # 构建结果
            result = {
                'question_id': data['question_id'],
                'attack_success': success,
                'queries': queries,
                'new_judgment': new_judgment,
                'execution_time': end_time - start_time,
                'error': None
            }
            
            return result
            
        except Exception as e:
            print(f"处理问题 {data['question_id']} 时出错: {str(e)}")
            return {
                'question_id': data['question_id'],
                'attack_success': False,
                'queries': 0,
                'new_judgment': '',
                'execution_time': 0,
                'error': str(e)
            }
    
    def evaluate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        评估攻击结果
        使用AttackEvaluation模块计算5个核心指标
        
        Args:
            results: 所有样本的攻击结果
            
        Returns:
            Dict: 评估指标
        """
        # 使用专门的评估模块
        evaluator = AttackEvaluation()
        return evaluator.calculate_metrics(results)
    
    def save_results(self, results: List[Dict[str, Any]], metrics: Dict[str, Any], output_path: str):
        """
        保存结果到文件
        使用AttackEvaluation模块的保存功能
        
        Args:
            results: 所有样本的攻击结果
            metrics: 评估指标
            output_path: 输出文件路径
        """
        evaluator = AttackEvaluation()
        evaluator.metrics = metrics
        evaluator.results = results
        evaluator.save_results(output_path)
        
        # 生成报告目录
        report_dir = output_path.replace('.json', '_report')
        evaluator.generate_report(report_dir)
    
    def display_metrics(self, metrics: Dict[str, Any]):
        """
        显示评估指标
        使用AttackEvaluation模块的打印功能
        
        Args:
            metrics: 评估指标
        """
        evaluator = AttackEvaluation()
        evaluator.metrics = metrics
        evaluator.print_metrics()
    
    def run(self):
        """
        运行主程序
        """
        # 解析命令行参数
        args = self.parser.parse_args()
        
        print("JudgeAttack 评估工具启动")
        print(f"模型: {args.model}")
        print(f"数据文件: {args.data}")
        print(f"攻击类型: {args.attack_type}")
        print(f"最大攻击次数: {args.max_attacks}")
        
        try:
            # 加载数据
            data = self.load_data(args.data)
            
            # 创建攻击实例
            attacker = self.create_attacker(args)
            
            # 执行攻击评估
            results = []
            print(f"\n开始执行攻击评估 ({len(data)} 个样本)")
            
            for i, sample in enumerate(data):
                print(f"\n处理样本 {i+1}/{len(data)} (ID: {sample['question_id']})")
                result = self.run_attack(attacker, sample)
                results.append(result)
                
                if args.debug:
                    status = "成功" if result['attack_success'] else "失败"
                    print(f"  状态: {status}")
                    if result['error']:
                        print(f"  错误: {result['error']}")
            
            # 计算评估指标
            metrics = self.evaluate_results(results)
            
            # 显示结果
            self.display_metrics(metrics)
            
            # 保存结果
            self.save_results(results, metrics, args.output)
            
            print("\nJudgeAttack 评估完成!")
            
        except Exception as e:
            print(f"\n错误: {str(e)}")
            if args.debug:
                import traceback
                traceback.print_exc()
            exit(1)

def main():
    """
    主函数入口
    """
    cli = JudgeAttackCLI()
    cli.run()

if __name__ == "__main__":
    main()