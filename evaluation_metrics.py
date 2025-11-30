#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
攻击评估指标模块
实现JudgeAttack任务的5个核心评估指标
"""

import json
import os
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class AttackEvaluation:
    """
    攻击评估类
    实现攻击指标计算和结果可视化
    """
    
    def __init__(self):
        """
        初始化评估类
        """
        self.metrics = {}
        self.results = []
    
    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算5个核心评估指标
        
        Args:
            results: 攻击结果列表
            
        Returns:
            Dict: 包含5个核心指标的字典
        """
        # 过滤有效结果
        valid_results = [r for r in results if r.get('error') is None]
        
        # 指标1: 总攻击次数
        total_attacks = len(valid_results)
        
        # 指标2: 成功攻击次数
        successful_attacks = sum(1 for r in valid_results if r.get('attack_success', False))
        
        # 指标3: 总查询次数
        total_queries = sum(r.get('queries', 0) for r in valid_results)
        
        # 指标4: 攻击成功率 (ASR)
        attack_success_rate = successful_attacks / total_attacks if total_attacks > 0 else 0.0
        
        # 指标5: 平均查询次数
        avg_queries = total_queries / total_attacks if total_attacks > 0 else 0.0
        
        # 附加指标
        avg_execution_time = sum(r.get('execution_time', 0) for r in valid_results) / total_attacks if total_attacks > 0 else 0.0
        
        self.metrics = {
            'total_samples': len(results),
            'valid_samples': total_attacks,
            'total_attacks': total_attacks,        # 指标1
            'successful_attacks': successful_attacks,  # 指标2
            'total_queries': total_queries,         # 指标3
            'attack_success_rate': attack_success_rate,  # 指标4 (ASR)
            'avg_queries': avg_queries,            # 指标5
            'avg_execution_time': avg_execution_time
        }
        
        self.results = results
        return self.metrics
    
    def print_metrics(self):
        """
        格式化打印评估指标
        """
        if not self.metrics:
            print("没有计算评估指标，请先调用calculate_metrics方法")
            return
        
        print("\n========== JudgeAttack 评估指标 ==========")
        print(f"总样本数: {self.metrics['total_samples']}")
        print(f"有效样本数: {self.metrics['valid_samples']}")
        print("\n[核心指标]")
        print(f"1. 总攻击次数: {self.metrics['total_attacks']}")
        print(f"2. 成功攻击次数: {self.metrics['successful_attacks']}")
        print(f"3. 总查询次数: {self.metrics['total_queries']}")
        print(f"4. 攻击成功率 (ASR): {self.metrics['attack_success_rate']*100:.2f}% ({self.metrics['successful_attacks']}/{self.metrics['total_attacks']})")
        print(f"5. 平均查询次数: {self.metrics['avg_queries']:.2f}")
        print("\n[附加指标]")
        print(f"平均执行时间: {self.metrics['avg_execution_time']:.4f} 秒")
        print("=========================================")
    
    def save_results(self, output_path: str, detailed: bool = True):
        """
        保存评估结果到文件
        
        Args:
            output_path: 输出文件路径
            detailed: 是否保存详细结果
        """
        if not self.metrics:
            print("没有计算评估指标，请先调用calculate_metrics方法")
            return
        
        try:
            # 构建结果对象
            result_obj = {
                'metrics': self.metrics,
                'summary': {
                    '攻击成功率 (ASR)': f"{self.metrics['attack_success_rate']*100:.2f}%",
                    '成功攻击次数': self.metrics['successful_attacks'],
                    '总攻击次数': self.metrics['total_attacks'],
                    '平均查询次数': f"{self.metrics['avg_queries']:.2f}"
                }
            }
            
            # 保存详细结果
            if detailed and self.results:
                result_obj['detailed_results'] = self.results
            
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 保存到JSON文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_obj, f, ensure_ascii=False, indent=2)
            
            print(f"评估结果已保存到: {output_path}")
            
            # 保存简单的文本摘要
            txt_path = output_path.replace('.json', '.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("JudgeAttack 评估结果摘要\n")
                f.write("============================\n\n")
                f.write(f"攻击成功率 (ASR): {self.metrics['attack_success_rate']*100:.2f}%\n")
                f.write(f"成功攻击次数: {self.metrics['successful_attacks']}\n")
                f.write(f"总攻击次数: {self.metrics['total_attacks']}\n")
                f.write(f"总查询次数: {self.metrics['total_queries']}\n")
                f.write(f"平均查询次数: {self.metrics['avg_queries']:.2f}\n")
            
            print(f"文本摘要已保存到: {txt_path}")
            
        except Exception as e:
            print(f"保存结果失败: {str(e)}")
    
    def analyze_attack_patterns(self):
        """
        分析攻击模式
        返回各攻击策略的成功率统计
        
        Returns:
            Dict: 攻击模式统计
        """
        if not self.results:
            return {}
        
        # 这里可以扩展实现更复杂的攻击模式分析
        # 目前返回基础统计信息
        
        # 统计每种判断结果的分布
        judgment_counts = {}
        success_by_question = []
        
        for result in self.results:
            if result.get('error') is None:
                # 统计判断结果
                new_judgment = result.get('new_judgment', '').strip().upper()
                judgment_counts[new_judgment] = judgment_counts.get(new_judgment, 0) + 1
                
                # 记录每个问题的攻击成功情况
                success_by_question.append({
                    'question_id': result.get('question_id', ''),
                    'success': result.get('attack_success', False)
                })
        
        return {
            'judgment_distribution': judgment_counts,
            'success_by_question': success_by_question
        }
    
    def generate_report(self, report_dir: str):
        """
        生成完整的评估报告
        
        Args:
            report_dir: 报告输出目录
        """
        if not self.metrics:
            print("没有计算评估指标，请先调用calculate_metrics方法")
            return
        
        try:
            # 确保报告目录存在
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)
            
            # 保存详细结果
            json_path = os.path.join(report_dir, 'evaluation_results.json')
            self.save_results(json_path, detailed=True)
            
            # 生成简单的可视化图表
            self._generate_charts(report_dir)
            
            print(f"评估报告已生成到目录: {report_dir}")
            
        except Exception as e:
            print(f"生成报告失败: {str(e)}")
    
    def _generate_charts(self, output_dir: str):
        """
        生成可视化图表
        注意：这是可选功能，需要matplotlib等库
        
        Args:
            output_dir: 输出目录
        """
        try:
            # 检查是否安装了必要的库
            import matplotlib
            import matplotlib.pyplot as plt
            import pandas
            
            # 设置中文字体支持
            plt.rcParams['font.sans-serif'] = ['SimHei']  
            plt.rcParams['axes.unicode_minus'] = False   
            
            # 攻击成功率饼图
            plt.figure(figsize=(10, 8))
            
            # 饼图 - 攻击成功/失败分布
            labels = ['攻击成功', '攻击失败']
            sizes = [
                self.metrics['successful_attacks'],
                self.metrics['total_attacks'] - self.metrics['successful_attacks']
            ]
            colors = ['#4CAF50', '#FF5722']
            
            plt.subplot(2, 2, 1)
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title('攻击成功率分布')
            
            # 柱状图 - 核心指标
            plt.subplot(2, 2, 2)
            metrics_names = ['总攻击次数', '成功攻击次数', '总查询次数']
            metrics_values = [
                self.metrics['total_attacks'],
                self.metrics['successful_attacks'],
                self.metrics['total_queries']
            ]
            plt.bar(metrics_names, metrics_values, color=['#2196F3', '#4CAF50', '#FFC107'])
            plt.title('攻击指标统计')
            plt.xticks(rotation=45, ha='right')
            
            # 显示图表
            plt.tight_layout()
            chart_path = os.path.join(output_dir, 'attack_metrics_charts.png')
            plt.savefig(chart_path)
            plt.close()
            
            print(f"可视化图表已保存到: {chart_path}")
            
        except ImportError:
            print("警告: 无法生成可视化图表，需要安装 matplotlib 和 pandas")
        except Exception as e:
            print(f"生成图表失败: {str(e)}")

# 测试函数
def test_evaluation():
    """
    测试评估模块的功能
    """
    # 示例攻击结果
    example_results = [
        {
            'question_id': 'q001',
            'attack_success': True,
            'queries': 3,
            'new_judgment': 'B',
            'execution_time': 1.2,
            'error': None
        },
        {
            'question_id': 'q002',
            'attack_success': False,
            'queries': 5,
            'new_judgment': 'A',
            'execution_time': 1.5,
            'error': None
        },
        {
            'question_id': 'q003',
            'attack_success': True,
            'queries': 2,
            'new_judgment': 'B',
            'execution_time': 1.0,
            'error': None
        }
    ]
    
    # 创建评估实例
    evaluator = AttackEvaluation()
    
    # 计算指标
    metrics = evaluator.calculate_metrics(example_results)
    
    # 打印指标
    evaluator.print_metrics()
    
    # 分析攻击模式
    patterns = evaluator.analyze_attack_patterns()
    print(f"\n攻击模式分析: {patterns}")
    
    print("\n评估模块测试完成")

# 示例用法
if __name__ == "__main__":
    # 运行测试
    test_evaluation()
    print("\n攻击评估指标模块已加载")