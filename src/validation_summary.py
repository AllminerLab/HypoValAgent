#!/usr/bin/env python3
"""
医学假设验证代码分析与总结脚本
功能：
1. 使用Claude API分析医学假设验证代码的功能
2. 使用DeepSeek API结合分析结果和实验数据总结验证结论
"""

import os
import json
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
import requests
from anthropic import Anthropic
import openai
import os
import logging
import sys
from llm_api_config import DEEPSEEK_KEY_API, CLAUDE_KEY_API
from logger import init_logger
from time_util import TimeTracker, TokenTracker

base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
if not logger.handlers:
    logger = init_logger('../data/log')

@dataclass
class ExperimentResults:
    def __init__(self, metrics_path: str, execution_time_path: str, ):

        with open(metrics_path, "r", encoding="utf-8") as f:
            self.metrics_result = json.load(f)

        with open(execution_time_path, "r", encoding="utf-8") as f:
            self.execution_result = json.load(f)

    def get_all_results(self):
        return self.metrics_result, self.execution_result


class MedicalHypothesisAnalyzer:
    """医学假设验证分析器"""

    def __init__(self, intent:dict=None, metrics_path: str =None, execution_time_path: str = None, multicenter_retrieved_feature_dict: Dict = None, reset_token=True):
        """
        初始化分析器
        """
        self.metrics_path = metrics_path
        self.execution_time_path = execution_time_path
        self.task_goal = intent['任务目标']
        self.intent = intent
        self.multicenter_retrieved_feature_dict = multicenter_retrieved_feature_dict

        self.claude_client = Anthropic(api_key="sk-wstU6K18T87m4NziIXC2AEwaiiBN74xFzHQilY6ZCWjDWhcX")
        self.deepseek_api_key = 'sk-f665ba25b1ed407b80085e36599b926b'
        self.deepseek_base_url = "https://api.deepseek.com"

        self.reset_token = reset_token
        self.token_recorder = TokenTracker(file_path=f'../data/interim/任务目标({self.task_goal})',
                                           filename="token_records.json",
                                           module='验证总结',
                                           reset=self.reset_token)

        self.time_tracker = TimeTracker(file_path=f'../data/interim/任务目标({self.task_goal})',
                                        filename="time_records.json")

        self.val_start_time = time.time()



    def analyze_code(self, code: str) -> Dict[str, Any]:
        """
        使用Claude分析Python代码功能

        Args:
            code: 要分析的Python代码

        Returns:
            包含代码分析结果的字典
        """
        prompt = f"""请分析以下医学假设验证相关的Python代码，详细说明：
                    1. 代码的主要功能和目的
                    2. 输入参数（包括数据含义）
                    3. 输出结果（包括返回值类型和含义）
                    4. 使用的主要算法或方法
                    5. 涉及的医学概念或指标
                    6. 代码的关键步骤
                    
                    代码：
                    ```python
                    {code}
                    ```
                    
                    请以JSON格式返回分析结果。"""

        openai.api_key = CLAUDE_KEY_API #"sk-wstU6K18T87m4NziIXC2AEwaiiBN74xFzHQilY6ZCWjDWhcX"
        openai.base_url = "https://chat.cloudapi.vip/v1/"
        openai.default_headers = {"x-foo": "true"}

        try:
            response = openai.chat.completions.create(
                model="gpt-5",
                temperature=0.2,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            # 解析Claude的响应
            analysis_text = response.choices[0].message.content


            # 尝试提取JSON部分
            try:
                # 查找JSON内容
                import re
                json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
                if json_match:
                    analysis_result = json.loads(json_match.group())

                    self.token_recorder.add_record(prompt_tokens=response.usage.prompt_tokens,
                                                   completion_tokens=response.usage.completion_tokens,
                                                   total_tokens=response.usage.total_tokens)
                else:
                    # 如果没有找到JSON格式，返回原始文本
                    analysis_result = {
                        "raw_analysis": analysis_text,
                        "parse_error": "无法解析为JSON格式"
                    }
            except json.JSONDecodeError:
                analysis_result = {
                    "raw_analysis": analysis_text,
                    "parse_error": "JSON解析失败"
                }

            return {
                "success": True,
                "analysis": analysis_result
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }

    def get_multi_center_validation_result_prompt(self, task_goal, center_code_analysis, center_metrics_result):
        validation_result_prompt = f""""验证目标：{task_goal}的验证方案和评估结果如下：\n"""
        for center, code_analysis in center_code_analysis.items():
            metrics_result = center_metrics_result[center]
            validation_result_prompt = validation_result_prompt + f"""{center}数据中心结果如下：\n
            代码验证方案总结：{code_analysis}.\n\n
            评价指标结果：{metrics_result}\n\n"""

        return validation_result_prompt

    def get_execution_time_prompt(self, task_goal, execution_time_stat):
        execution_time_prompt = f""""为了验证医学假设验证({task_goal}),通过以下步骤：语义解析与意图识别环节、特征检索、模型设计与实现、模型代码调整，每个步骤的耗时如下：\n"""

        for module, time_stat in execution_time_stat.items():
            execution_time_prompt = execution_time_prompt + f"""{module}为{time_stat}, """

        return execution_time_prompt


    def call_llm_api(self, prompt, system_prompt="你是一位专业的医学研究分析专家，擅长解读实验结果并结合最新研究进展提供见解。"):
        for deepseek_api_key in DEEPSEEK_KEY_API:

            try:
                # 调用DeepSeek API
                headers = {
                    "Authorization": f"Bearer {deepseek_api_key}",
                    "Content-Type": "application/json"
                }

                data = {
                    "model": "deepseek-chat",
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.7,
                    "max_tokens": 8096
                }

                response = requests.post(
                    f"{self.deepseek_base_url}/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    summary = result['choices'][0]['message']['content']

                    self.token_recorder.add_record(prompt_tokens=result['usage'].get('prompt_tokens'),
                                                   completion_tokens=result['usage'].get('completion_tokens'),
                                                   total_tokens=result['usage'].get('total_tokens'))

                    return summary

                    return {
                        "success": True,
                        "summary": summary,
                        "model": result.get('model', 'deepseek-chat'),
                        "usage": result.get('usage', {}),
                        "timestamp": time.time()
                    }
                else:
                    logger.info(f"DeepSeek API错误: {response.status_code} - {response.text}, 更换api key中")
                    continue

            except Exception as e:
                logger.info(f"出现错误: {e}, 更换api key中")
                continue

        logger.info("api_key已调用完毕，调用失败！")
        return None


    def summarize_thinking_records(self, task_goal):
        thinking_path = f'../data/interim/任务目标({self.task_goal})/think_records.json'
        with open(thinking_path, "r", encoding="utf-8") as f:
            thinking_result = json.load(f)

        thinking_prompt = f""""针对验证目标：{task_goal}，根据每个环节的思考过程进行总结，要求总结要精准、简洁，每个环节的具体思考过程如下：\n"""
        for process, thinking in thinking_result.items():
            process_thinking_result = thinking_result[process]
            thinking_prompt = thinking_prompt + f"""{process}环节：\n{process_thinking_result}。\n"""

        thinking_prompt = thinking_prompt

        thinking_summary = self.call_llm_api(prompt=thinking_prompt,
                                             system_prompt="你是一位专业的人工智能和医学交叉专家，擅长进行总结归纳分析。")

        return thinking_summary





    def summarize_validation(self,
                                center_code_analysis,
                                center_metrics_result,
                                execution_time_stat,
                                task_goal: str) -> Dict[str, Any]:

        """
        使用DeepSeek总结验证结果

        Args:
            code_analysis: Claude的代码分析结果
            experiment_results: 实验结果数据

        Returns:
            包含总结的字典
        """
        validation_result_prompt = self.get_multi_center_validation_result_prompt(task_goal=task_goal,
                                                                                  center_code_analysis=center_code_analysis,
                                                                                  center_metrics_result=center_metrics_result)

        execution_time_prompt = self.get_execution_time_prompt(task_goal=task_goal, execution_time_stat=execution_time_stat)


        thinking_summary = self.summarize_thinking_records(task_goal=task_goal)
        # 构建提示词
        prompt = f"""基于以下医学假设验证({task_goal})的信息，请提供详细的验证结果总结：

## 代码功能分析与实验结果如下：\n
{validation_result_prompt}\n\n

###各个环节执行耗时相关信息\n
- 各个环节执行耗时：{execution_time_prompt}\n

###各个环节的思考过程如下\n
-{thinking_summary}
\n

请提供以下内容的总结：
请根据代码功能分析与实验结果、各个环节执行耗时相关信息，各个环节的思考过程。面向没有计算机背景的医生，以医学的语句判断该医学假设是否合理或者是否有意义？如果合理或者有意义，请结合验证时间的效率方案提供解释和意义。如果不合理或没有意义，请说出原因。\n
\n\n如果医学假设验证结果有意义，则分析下面的内容：\n
 1、验证时间效率，需结合近几年同类型或同领域的医学验证进行对比，凸显其高效性。\n
 2、下一步研究方向建议。\n

\n\n如果医学假设验证结果没有意义，请结合提供的素材说明原因。\n

请结合医学领域的最新进展（2024-2025年）进行分析，并以结构化的方式呈现总结。"""

        return self.call_llm_api(prompt = prompt, system_prompt="你是计算机和医学交叉专家，善于把计算机的结论简单形象地转化成医学结论。"), thinking_summary



    def analyze_and_summarize(self) -> Dict[str, Any]:

        # 判断之前的环节是否有数据
        if self.intent is None or self.execution_time_path is None or self.multicenter_retrieved_feature_dict is None:
            validation = self.intent['医学假设']
            summary = f'当前医学数据库没有该医学假设（{validation}）的相关数据，无法进行验证！！！'
            total_result = {
                "code_analysis": None,
                "metrics_result": None,
                'thinking_summary': None,
                "summary": summary
            }
            logger.info(summary)
            output_file = f"../data/interim/任务目标({self.task_goal})/summary_result.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(total_result, f, ensure_ascii=False, indent=2)
            logger.info(f"\n结果已保存到: {output_file}")
            return total_result

        center_code_analysis = {}
        center_metrics_result = {}

        for center, retrieved_features in self.multicenter_retrieved_feature_dict.items():
            if center == '耗时（min）':
                continue
            task_type = retrieved_features['任务类型']

            validation_code_script = f'task_related_code_{self.task_goal}_{center}.py'
            with open(validation_code_script, 'r') as file:
                validation_code = file.read()

            if task_type != '生成任务':
                metrics_path = f'../data/model/任务目标({self.task_goal})_{center}/evaluation_result.json'
                with open(metrics_path, "r",encoding="utf-8") as f:
                    metrics_result = json.load(f)
                    center_metrics_result[center] = metrics_result
            else:
                metrics_path = f'../data/llm/saves/{self.task_goal}_{center}/predict/all_results.json'
                with open(metrics_path, "r",encoding="utf-8") as f:
                    metrics_result = json.load(f)
                    center_metrics_result[center] = metrics_result

            logger.info(f"正在分析{center}中心代码...")
            code_analysis = self.analyze_code(code=validation_code)
            center_code_analysis[center]=code_analysis["analysis"]

        with open(self.execution_time_path, "r", encoding="utf-8") as f:
            execution_time_stat = json.load(f)

        logger.info("总结验证结果...")
        summary, thinking_summary = self.summarize_validation(center_code_analysis=center_code_analysis,
                                            center_metrics_result=center_metrics_result,
                                            execution_time_stat=execution_time_stat,
                                            task_goal=self.task_goal)
        total_result = {
            "code_analysis": center_code_analysis,
            "metrics_result": center_metrics_result,
            'thinking_summary': thinking_summary,
            "summary": summary
        }

        logger.info("\n### 代码功能分析 ###")
        logger.info(json.dumps(total_result['code_analysis'],
                         ensure_ascii=False, indent=2))

        logger.info("\n### 评估指标分析 ###")
        logger.info(json.dumps(total_result['metrics_result'],ensure_ascii=False, indent=2))

        logger.info("\n### 思考过程总结 ###")
        logger.info(total_result['thinking_summary'])

        logger.info("\n### 验证结果总结 ###")
        logger.info(total_result['summary'])

        # 保存结果到文件
        output_file = f"../data/interim/任务目标({self.task_goal})/summary_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(total_result, f, ensure_ascii=False, indent=2)
        logger.info(f"\n结果已保存到: {output_file}")

        self.time_tracker.add_record(start_time=self.val_start_time,
                                     end_time=time.time(),
                                     record_name=f'验证总结耗时（min）')

        return total_result