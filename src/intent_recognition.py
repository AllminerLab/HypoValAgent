import time

import requests
import json
from typing import Dict, Any, Tuple
import re

import os
import logging
import sys
from logger import init_logger
from time_util import ThinkTracker, TokenTracker
from time_util import TimeTracker
import llm_api_config

base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
if not logger.handlers:
    logger = init_logger('../data/log')

def create_dir_if_not_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

class IntentRecognition:
    """医护人员需求分析系统"""
    
    def __init__(self, api_key: str = None):
        """
        初始化分析器
        
        Args:
            api_key: Deepseek API密钥
        """
        self.api_key = ''
        self.api_url = "https://api.deepseek.com/chat/completions"

        self.intent_start_time = None

        self.token_stat = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}



        if not self.api_key:
            logger.info("错误：API Key不能为空")
            return
        
    def build_prompt(self, user_requirement: str) -> str:
        """
        构建提示词
        
        Args:
            user_requirement: 用户输入的医护人员需求
            
        Returns:
            构建好的提示词
        """
        prompt = f"""请分析以下医护人员的需求，识别其意图，并严格按照指定格式返回JSON结果。
        医护人员需求：{user_requirement}
        请识别需求中的任务目标和各类特征，返回格式如下：
        {{
          "任务目标": "简要描述任务的主要目标",
          "任务类型": "如果是数据分析相关需求则填写'数据分析任务'，否则填写'null'。注意：数据分析任务是指运用数学方法描述数据特征（均值、方差）或进行回归分析等，例如通过均值判断数据集中趋势，用置信区间评估统计显著性等。在医学领域，分诊、预后预测等不属于数据分析任务",\n
          "输入": {{
            "连续特征": "数值型连续变量，如血压值、体温等，如果没有则为'null'",
            "离散特征": "分类变量，如性别、血型、疾病类型等，如果没有则为'null'",
            "文本特征": "文本描述，如病历记录、症状描述等，如果没有则为'null'",
            "图像特征": "医学影像，如X光片、CT、MRI等，如果没有则为'null'",
            "时序特征": "时间序列数据，如心电图、脑电图、连续监测数据等，如果没有则为'null'"
          }},
          "输出": "任务期望的输出结果"，
          "思考结果"："简要说明识别到的意图的理由,要求有理有据"
        }}
        \n
        \n
        只返回JSON格式的结果，不要添加其他说明文字。
        
       """
        return prompt
#  # 只返回JSON格式的结果，不要添加其他说明文字。

    def add_token_stat(self, prompt_tokens, completion_tokens, total_tokens):
        self.token_stat['prompt_tokens'] = self.token_stat['prompt_tokens'] + prompt_tokens
        self.token_stat['completion_tokens'] = self.token_stat['completion_tokens'] + completion_tokens
        self.token_stat['total_tokens'] = self.token_stat['total_tokens'] + total_tokens


    def call_llm_api(self, prompt: str, api_key: str) -> str:
        """
        调用Deepseek API

        Args:
            prompt: 分析提示词

        Returns:
            API响应内容
        """
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "deepseek-reasoner",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "temperature": 0.1,  # 降低随机性，提高一致性
            "max_tokens": 2000
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=600)
            response.raise_for_status()

            result = response.json()
            self.add_token_stat(prompt_tokens=result['usage']['prompt_tokens'], completion_tokens=result['usage']['completion_tokens'], total_tokens=result['usage']['total_tokens'])
            return result["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            # raise Exception(f"API调用失败: {str(e)}")
            logger.info(f"API调用失败: {str(e)}")
            return None
        except KeyError as e:
            # raise Exception(f"API响应内容格式错误: {str(e)}")
            logger.info(f"API响应内容格式错误: {str(e)}")
            return None

    def parse_api_response(self, response_text: str) -> Dict[str, Any]:
        """
        解析API响应，提取JSON结果

        Args:
            response_text: API返回的文本

        Returns:
            解析后的字典结果
        """
        # 清理响应文本
        cleaned_text = response_text.strip()

        try:
            # 尝试直接解析JSON
            result = json.loads(cleaned_text)
            return result
        except json.JSONDecodeError:
            # 使用正则表达式提取JSON
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, cleaned_text, re.DOTALL)

            for match in matches:
                try:
                    result = json.loads(match)
                    # 检查是否包含所需的基本结构
                    if "任务目标" in result and "输入" in result:
                        return result
                except json.JSONDecodeError:
                    logger.info("无法从API响应中提取有效的JSON格式")
                    # raise Exception("无法从API响应中提取有效的JSON格式")
                    return None


    def analyze_requirement(self, requirement: str) -> Dict[str, Any]:
        """
        分析医护人员需求

        Args:
            requirement: 医护人员需求描述

        Returns:
            包含任务目标、输入特征和输出的字典
        """
        if not requirement.strip():
            raise ValueError("需求描述不能为空")

        # 创建提示词
        prompt = self.build_prompt(requirement)

        result = None

        # 调用API
        llm_api_key_list = llm_api_config.DEEPSEEK_KEY_API

        for llm_api_key in llm_api_key_list:

            api_response = self.call_llm_api(prompt=prompt, api_key=llm_api_key)

            # 如果访问失败，则尝试下一个api_key,否则退出
            if api_response is None or api_response == '':
                logger.info('api_response没有内容，尝试换用API KEY')
                continue

            # 解析响应
            result = self.parse_api_response(api_response)

            if result is None:
                logger.info('返回内容解析失败，尝试换用API KEY')
                continue

            # 验证格式
            is_valid, message = self.validate_result_format(result)
            if not is_valid:
                logger.info(f'结果格式验证失败: {message}，尝试换用API KEY')
                continue
            else:
                break


        if api_response is None or api_response == '' or result is None or not is_valid:
            logger.info('意图解析失败！！！')


        return result

    def validate_result_format(self, result: Dict[str, Any]) -> Tuple[bool, str]:
        """
        验证结果格式是否符合要求

        Args:
            result: 解析后的结果字典

        Returns:
            (是否有效, 验证消息)
        """
        required_keys = ["任务目标", "任务类型", "输入", "输出"]
        input_keys = ["连续特征", "离散特征", "文本特征", "图像特征", "时序特征"]

        # 检查主要键
        for key in required_keys:
            if key not in result:
                return False, f"缺少必需的键: {key}"

        # 检查输入字段
        if not isinstance(result["输入"], dict):
            return False, "输入字段必须是字典格式"

        for key in input_keys:
            if key not in result["输入"]:
                return False, f"输入字段缺少必需的键: {key}"

        # 验证任务类型
        task_type = result["任务类型"]
        if task_type not in ["数据分析任务", "null"]:
            return False, f"任务类型必须是'数据分析任务'或'null'，当前值: {task_type}"

        return True, "格式验证通过"


    def chinese_to_english(self, text):
        """
        调用Deepseek API

        Args:
            text: 分析提示词

        Returns:
            API响应内容
        """
        prompt = f"""
        请总结下列的中文语句，并用英文精简地表达出来，只返回表达的英文，不要有其他内容。
        中文语句：{text}
        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "deepseek-reasoner",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "temperature": 0.1,  # 降低随机性，提高一致性
            "max_tokens": 1000
        }

        response = requests.post(self.api_url, headers=headers, json=data, timeout=300)
        result = response.json()

        return result["choices"][0]["message"]["content"]

    def save_result(self, result: Dict[str, Any], intent_save_root_path: str = None) -> str:
        """
        保存分析结果到文件
        
        Args:
            result: 分析结果
            filename: 文件名（可选）
            
        Returns:
            保存的文件路径
        """
        if intent_save_root_path is None:
            intent_save_root_path = "../data/interim"

        task_goal = result['任务目标']

        intent_save_dir = create_dir_if_not_exist("/".join([intent_save_root_path, f'任务目标({task_goal})']))

        intent_save_path = "/".join([intent_save_dir, 'medical_intent_recognition.json'])

        intent_end_time = time.time()
        logger.info(f"意图识别耗时：{(intent_end_time - self.intent_start_time) / 60:.2f} 分钟")
        result['耗时（min）'] = round((intent_end_time - self.intent_start_time) / 60, 2)

        tracker = TimeTracker(file_path=f'../data/interim/任务目标({task_goal})', filename="time_records.json")
        tracker.add_record(start_time=self.intent_start_time, end_time=intent_end_time, record_name='意图识别耗时（min）')

        tracker = ThinkTracker(file_path=f'../data/interim/任务目标({task_goal})', filename="think_records.json")
        tracker.add_record(module='语义解析与意图识别',think=result['思考结果'])

        tracker = TokenTracker(file_path=f'../data/interim/任务目标({task_goal})', filename="token_records.json", module='语义解析与意图识别')
        tracker.add_record(module='语义解析与意图识别', prompt_tokens=self.token_stat['prompt_tokens'],
                           completion_tokens=self.token_stat['completion_tokens'],
                           total_tokens=self.token_stat['total_tokens'])

        with open(intent_save_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"结果已保存到: {intent_save_path}")
        
        return intent_save_path
    
    def format_result(self, result: Dict[str, Any]) -> str:
        """
        格式化输出结果
        
        Args:
            result: 分析结果
            
        Returns:
            格式化的字符串
        """
        output = []
        output.append(f"\n任务目标: {result['任务目标']}")
        output.append(f"\n任务类型: {result['任务类型']}")
        output.append("\n输入特征:")
        
        for feature_type, feature_value in result['输入'].items():
            output.append(f"  - {feature_type}: {feature_value}")
        
        output.append(f"\n输出: {result['输出']}")
        output.append(f"\n\n思考结果: {result['思考结果']}")
        output.append("="*50)
        
        return "\n".join(output)


    def query(self, requirement, intent_save_root_path):
        if not requirement:
            logger.info("需求描述不能为空，请重新输入")

        self.intent_start_time = time.time()
        logger.info("正在分析需求...")

        result = self.analyze_requirement(requirement)

        # 打印格式化结果
        logger.info(self.format_result(result))

        result['医学假设'] = requirement



        self.save_result(result=result, intent_save_root_path=intent_save_root_path)

        return result
