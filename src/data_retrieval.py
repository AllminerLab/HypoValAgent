import os
import re
import time
from pathlib import Path

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import glob

from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import requests
import json

from difflib import SequenceMatcher
import jieba  # 中文分词
import os
import logging
import sys
from logger import init_logger
from time_util import TimeTracker, ThinkTracker, TokenTracker
import llm_api_config
from streaming_llm_handler import deepseek_stream_calling

warnings.filterwarnings('ignore')
base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
if not logger.handlers:
    logger = init_logger('../data/log')
def create_dir_if_not_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

class IntentTableFeatureExtractor:
    """根据意图从表格中提取相关特征的类"""

    def __init__(self, emergency_database_dir="../data/raw", feature_type_dir="../data/feature_type", task_related_data_dir=None, retrieve_features_path=None):
        self.emergency_database_dir = emergency_database_dir
        self.tables = {}
        self.merged_table = None
        self.similarity_list = []
        self.task_related_data_dir = task_related_data_dir
        self.retrieve_features_path = retrieve_features_path
        self.feature_type_dir = feature_type_dir
        self.column_type_dict = {}
        self.retrieve_start_time = None
        self.token_recorder = None


    def scan_single_center_files(self, center, context, center_path):
        # 中文分词
        keywords = jieba.cut(context)
        keywords = [word for word in keywords if len(word) > 1]
        # 支持的文件格式
        file_patterns = ['*.csv', '*.xlsx', '*.xls']

        # 保存中心搜索结果
        single_center_tables = {}

        for pattern in file_patterns:
            files = glob.glob(os.path.join(center_path, pattern))
            for file in files:
                try:
                    if file.endswith('.csv'):
                        df = pd.read_csv(file, encoding='utf-8')
                    else:
                        df = pd.read_excel(file)

                    file_name = os.path.basename(file)

                    # 计算相似度
                    similarity = max(
                        SequenceMatcher(None, keyword, file_name).ratio()
                        for keyword in keywords
                    )
                    if similarity >= 0.2:  # 阈值可调整
                        single_center_tables[file_name] = df
                        logger.info(f"成功寻找表格: {file_name} ({center}), 形状: {df.shape}")

                except Exception as e:
                    logger.info(f"加载表格 {file} 失败: {str(e)}")

        return single_center_tables

    def scan_center_files(self, context):
        """
        扫描emergency_database_dir目录下的各个中心文件夹，返回每个中心的文件列表

        参数:
            data_dir: 数据目录路径，默认为""../data/raw""

        返回:
            字典格式，包含各中心及其文件列表
        """
        scan_result = {}

        # 检查数据库目录是否存在
        if not os.path.exists(self.emergency_database_dir):
            logger.info(f"错误：目录 '{self.emergency_database_dir}' 不存在")
            return scan_result

        # 遍历data目录下的所有项目
        for center in os.listdir(self.emergency_database_dir):
            center_path = os.path.join(self.emergency_database_dir, center)

            single_center_tables = self.scan_single_center_files(center=center, context=context, center_path=center_path)

            if len(single_center_tables) > 0:
                scan_result[center] = single_center_tables

        return scan_result

    def merge_single_center_tables(self, single_center_tables: dict, center: str) -> pd.DataFrame:
        ##
        def get_feature_dict(name, center):
            path = Path(name)
            feature_file_name = f"{path.stem}_feature_type.json"

            with open(f'../data/feature_type/{center}/{feature_file_name}', 'r', encoding='utf-8') as file:
                feature_dict = json.load(file)

            return feature_dict.copy()

        if len(single_center_tables) == 0:
            raise ValueError("没有加载任何表格")

        if len(single_center_tables) == 1 or len(single_center_tables) > 2:
            merged_table = list(single_center_tables.values())[0]
            return merged_table, get_feature_dict(list(single_center_tables.keys())[0], center)

        """
            逐步合并多个DataFrame

            参数:
                dataframes: DataFrame列表
                names: DataFrame名称列表（用于打印信息）

            返回:
                最终合并的DataFrame
            """


        # 暂时只考虑两个dataframe的合并
        dataframes = list(single_center_tables.values())
        names = list(single_center_tables.keys())


        # 从第一个DataFrame开始
        merged_table = dataframes[0]
        logger.info(f"\n开始合并过程...")
        logger.info(f"初始DataFrame: {names[0]}, 形状: {merged_table.shape}")

        merge_feature_dict = get_feature_dict(names[0], center)

        # 合并其他DataFrame
        for i in range(1, len(dataframes)):
            logger.info(f"\n--- 合并第 {i + 1}/{len(dataframes)} 个DataFrame ---")
            logger.info(f"当前要合并: {names[i]}, 形状: {dataframes[i].shape}")

            # 找出共同列
            common_cols = list(set(merged_table.columns) & set(dataframes[i].columns))

            # 合并
            merged_table = pd.merge(merged_table, dataframes[i], on=common_cols, how='outer')
            logger.info(f"合并后形状: {merged_table.shape}")

            merge_feature_dict.update(get_feature_dict(names[i], center))

        return merged_table, merge_feature_dict

    def merge_tables(self, scan_result):
        """为每个中心合并多个表格"""
        if len(scan_result) == 0:
            raise ValueError("没有加载任何表格")

        multicenter_merged_table = {}
        multicenter_feature_type_dict = {}

        for center, single_center_files in scan_result.items():
            multicenter_merged_table[center], multicenter_feature_type_dict[center] = self.merge_single_center_tables(single_center_files, center)

        return multicenter_merged_table, multicenter_feature_type_dict


    def classify_column_type(self, col: str, series: pd.Series, feature_type_dict: Dict) -> str:
        """批量分类所有列的类型"""
        return feature_type_dict.get(col, None)

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
            pattern = r'\{.*\}'
            json_match = re.search(pattern, cleaned_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(), strict=False)
            else:
                logger.info("无法从API响应中提取有效的JSON格式")
                # raise Exception("无法从API响应中提取有效的JSON格式")
                return None



    def call_llm_api(self, prompt: str, temperature: float = 0.2):

        api_key_list = llm_api_config.DEEPSEEK_KEY_API

        result = None

        for api_key in api_key_list:
            result = self.call_single_llm_api(prompt=prompt, api_key=api_key, temperature=temperature)

            if result is not None and result != "":
                parsed_result = self.parse_api_response(result)
                if parsed_result is not None:
                    return parsed_result['选择特征'], parsed_result['思考理由']
                else:
                    logger.info('返回内容解析失败，尝试换用API KEY')
                    continue
            else:
                logger.info('调用失败，尝试换用API KEY')
                continue

        if result is None or result == "":
            logger.info('现有API KEY全部调用失败！！！')
            return None
        return None

    def call_single_llm_api(self, prompt: str, api_key: str, temperature: float=0.2):

        try:
            res = deepseek_stream_calling(api_key=api_key, prompt=prompt, model="deepseek-chat", token_recorder=self.token_recorder, temperature=temperature)
            return res
        except:
            logger.info('response解析失败')
            return None

    def build_same_meaning_prompt(self, feature_name: str, column_names: list,
                                  task_goal: str, column_data: pd.Series = None) -> str:
        # column_names = [f"<{tag}/>" for tag in column_names]

        column_names = "\t".join(column_names)

        prompt = f"""作为医学数据分析专家，请在给定的任务背景下，输出候选特征中哪个特征与输出特征含义是一样的。如果没有,输出nan。\n
            任务背景：{task_goal} \n
            输出特征：{feature_name} \n
            候选特征：{column_names} \n
            
            注意：
            1、候选特征以逗号'\t'进行分割 \n
            2、如果有多个特征，只输出一个含义最相同的特征，记住，最多一个特征。\n

            比如医学领域特别注意：\n
            - 脓毒症的预后与分级：生命体征（血压、心率、体温）、感染指标（白细胞、CRP、PCT）、器官功能指标高度相关\n
            - 心血管疾病：血压、血脂、血糖、BMI等代谢指标相关\n
            - 呼吸系统：氧饱和度、呼吸频率、血气分析相关\n
            
            注意，图像特征和数值特征的区别。\n
            
            请以JSON格式返回结果，格式如下：
            {{
            "思考理由": 在这里说明选择的理由，含义为什么是一样的，或者为什么不一样，要求简洁、明了。一定要简洁！\n
            "选择特征": 返回找到的意义相同的候选特征或者nan \n
            }}    
            """

        return prompt

    def build_similarity_prompt(self, feature_name: str, column_names: list,
                                task_goal: str, column_data: pd.Series = None) -> str:

        column_names = "\t".join(column_names)

        prompt = f"""你是一位资深医学数据分析专家，精通临床医学、生物信息学和统计学，请在给定的任务背景下，寻找相关的医学专业知识，在输出候选特征中哪些候选特征与输出特征或任务背景至少是中度相关的。\n
        
            如果候选特征有多个,则以"\t"分开。如果没有找到候选特征,则输出'null'。\n

            任务背景：{task_goal} \n
            输出特征：{feature_name} \n
            候选特征：{column_names}  \n
            注意：
            1、候选特征以逗号"\t"进行分割； \n
            2、请确保候选特征的完整性，不要忽略候选特征中括号部分，比如"(1：否；2：是）"，"(mmHg)","(°C)"等内容；
            比如特征为 "基础疾病：高血压（1：否；2：是）"，不要只返回"基础疾病：高血压"，而要返回"基础疾病：高血压（1：否；2：是）";\n
     

             评估标准：\n
            1. 直接相关（0.9-1.0）：字段名称相同或是明确的同义词\n
            2. 高度相关（0.7-0.9）：在该任务中有重要的临床或统计意义\n
            3. 中度相关（0.5-0.7）：可能有用但不是核心特征\n
            4. 低度相关（0.2-0.5）：关联较弱\n
            5. 不相关（0-0.2）：基本无关\n

            比如医学领域特别注意：\n
            - 脓毒症的预后与分级：生命体征（血压、心率、体温）、感染指标（白细胞、CRP、PCT）、器官功能指标高度相关\n
            - 心血管疾病：血压、血脂、血糖、BMI等代谢指标相关\n
            - 呼吸系统：氧饱和度、呼吸频率、血气分析相关\n
            
            请以JSON格式返回结果，格式如下：
            {{
            "思考理由": 在这里说明选择的理由，含义为什么是相似或高度相关的，或者为什么找不到，要求简洁、明了。一定要简洁！\n
            "选择特征": 返回找到的高度相关的候选特征或者nan \n
            }}    。\n
            """

        return prompt

    def add_col(self, result, df, col, feature_type_dict):
        col_type = self.classify_column_type(col, df[col], feature_type_dict)

        if col_type is None:
            logger.info(f"{col} is not in feature_type_dict!!!")
            return False

        if col_type in result['特征字段名']:
            result['特征字段名'][col_type].append(f"<{col}>")
        return True

    def task_classify(self, series: pd.Series, feature_type_dict, feature_name) -> Tuple[str, float]:
        """分析pandas Series的数据类型"""

        feature_type = feature_type_dict[feature_name]

        if feature_type == '连续特征':
            return '回归任务', 1

        if feature_type == '离散特征':
            unique_ratio = series.nunique() / len(series)
            if unique_ratio > 0.5:
                return '回归任务', 0.8
            else:
                return '分类任务', 1

        if feature_type == '文本特征':
            if series.nunique() > len(series) * 0.1:
                return '生成任务', 1
            else:
                return '分类任务', 0.8

        logger.info("无法识别任务类型！！")

        return 'unknown', 0.0

    def find_matching_columns(self, intent: Dict[str, Any], df: pd.DataFrame, feature_type_dict) -> Dict[str, Any]:
        task_type = intent.get('任务类型', '')

        if task_type != '数据分析任务':

            column_mapping = self.find_matching_columns_for_class_regress_gene(intent=intent, df=df, feature_type_dict=feature_type_dict)

            return column_mapping

        elif task_type == '数据分析任务':
            column_mapping = self.find_matching_columns_for_data_analysis(intent=intent, df=df, feature_type_dict=feature_type_dict)

            return column_mapping

    def find_matching_columns_for_data_analysis(self, intent: Dict[str, Any], df: pd.DataFrame, feature_type_dict) -> Dict[str, Any]:
        result = {
            '任务目标': None,
            '任务类型': None,
            '特征字段名': {
                '连续特征': [],
                '离散特征': [],
                '文本特征': [],
                '时序特征': [],
                '图像特征': []
            },
            '标签名': None,
            '标签类型': 'null',
            '医学假设': None,
            '思考理由': None
        }

        # 提取意图信息
        task_goal = intent.get('任务目标', '')
        result['任务目标'] = task_goal
        result['医学假设'] = intent.get('医学假设', '')

        input_features = intent.get('输入', {})
        output_target = intent.get('输出', '')
        result['任务类型'] = intent.get('任务类型', '')
        thinking_dict = {}

        result['标签名'] = output_target

        # 记录已使用的列，避免重复
        used_columns = set()

        column_names = [col for col in df.columns]

        # 寻找输入特征中明确指定的特征,用于数据分析
        fea_num = 0
        for feature_type, feature_names in input_features.items():
            if feature_names and feature_names != 'null' and pd.notna(feature_names):
                if isinstance(feature_names, str):
                    feature_list = [f.strip() for f in feature_names.split(',')]
                else:
                    feature_list = feature_names if isinstance(feature_names, list) else [feature_names]

                for feature in feature_list:
                    column_names2 = [col for col in column_names if col not in used_columns]
                    prompt = self.build_same_meaning_prompt(feature_name=feature,
                                                            column_names=column_names2, task_goal=task_goal)
                    selected_cols, selected_thinking = self.call_llm_api(prompt=prompt)

                    thinking_dict[f'与任务指定特征<{feature}>意义相同的<{selected_cols}>的思考理由'] = selected_thinking
                    logger.info(f'selected_cols: {selected_cols}')
                    if selected_cols != 'null':
                        for col in selected_cols.split("\t"):
                            self.add_col(result=result, col=col, df=df, feature_type_dict=feature_type_dict)
                            used_columns.add(col)
                            fea_num += 1
                    else:
                        logger.info(f'寻找不到任务指定特征<{feature}>！！！')

        # 只有两个特征以上才能进行数据分析
        if fea_num >= 2:
            return result
        else:
            logger.info('特征数量小于2个')
            return None

    def find_matching_columns_for_class_regress_gene(self, intent: Dict[str, Any], df: pd.DataFrame, feature_type_dict) -> Dict[str, Any]:
        """根据意图寻找匹配的列"""
        result = {
            '任务目标': None,
            '任务类型': None,
            '特征字段名': {
                '连续特征': [],
                '离散特征': [],
                '文本特征': [],
                '时序特征': [],
                '图像特征': []
            },
            '标签名': None,
            '标签类型': None,
            '医学假设': None,
            '思考理由': None
        }

        # 提取意图信息
        task_goal = intent.get('任务目标', '')
        result['任务目标'] = task_goal
        result['医学假设'] = intent.get('医学假设', '')

        input_features = intent.get('输入', {})
        output_target = intent.get('输出', '')
        task_type = intent.get('任务类型', '')
        thinking_dict = {}

        # 记录已使用的列，避免重复
        used_columns = set()

        selected_output = None

        column_names = [col for col in df.columns]

        fea_num = 0

        # 1. 首先寻找输出标签
        prompt = self.build_same_meaning_prompt(feature_name=output_target, column_names=column_names,
                                                task_goal=task_goal)
        selected_output, selected_thinking = self.call_llm_api(prompt=prompt)
        thinking_dict[f'与输出<{output_target}>意义相同的输出标签为<{selected_output}>的思考理由'] = selected_thinking
        # logger.info(f'已检索到的标签名: {selected_output}')
        if selected_output != 'null':
            result['标签名'] = f"<{selected_output}>"
            result['标签类型'] = self.classify_column_type(col=selected_output,
                                                           series=df[selected_output],
                                                           feature_type_dict=feature_type_dict)
            if result['标签类型'] is None:
                logger.info("标签类型为None")
                return None
            used_columns.add(selected_output)

            # 如果不是分析任务，更新任务类型
            if task_type != '数据分析任务':
                result['任务类型'], _ = self.task_classify(df[selected_output], feature_type_dict=feature_type_dict, feature_name=selected_output)
        else:
            logger.info('无法找到输出目标特征!!!')
            return None

        # 2. 寻找输入特征中明确指定的特征
        for feature_type, feature_names in input_features.items():
            if feature_names and feature_names != 'null' and pd.notna(feature_names):
                if isinstance(feature_names, str):
                    feature_list = [f.strip() for f in feature_names.split(',')]
                else:
                    feature_list = feature_names if isinstance(feature_names, list) else [feature_names]

                for feature in feature_list:
                    column_names2 = [col for col in column_names if col not in used_columns]
                    prompt = self.build_same_meaning_prompt(feature_name=feature,
                                                            column_names=column_names2, task_goal=task_goal)
                    selected_cols, selected_thinking = self.call_llm_api(prompt=prompt)

                    thinking_dict[f'与任务指定特征<{feature}>意义相同的<{selected_cols}>的思考理由'] = selected_thinking
                    logger.info(f'selected_cols: {selected_cols}')
                    if selected_cols != 'null':
                        for col in selected_cols.split("\t"):
                            if self.add_col(result=result, col=col, df=df, feature_type_dict=feature_type_dict):
                                used_columns.add(col)
                                fea_num += 1
                            else:
                                return None
                    else:
                        logger.info(f'未能找到指定特征<{feature}>，原因如下：{selected_thinking}')

        # 3. 寻找与任务目标相关的其他特征
        column_names1 = [col for col in column_names if col not in used_columns]
        if len(column_names1) > 0:
            prompt = self.build_similarity_prompt(feature_name=output_target, column_names=column_names1,
                                                  task_goal=task_goal)
            similarity_feas, similarity_thinking = self.call_llm_api(prompt=prompt, temperature=0.3)
            thinking_dict[f'与目标任务相关特征为<{similarity_feas}>的思考理由'] = similarity_thinking
        # logger.info(f'已检索到的特征字段名:{similarity_feas}')

            if similarity_feas != 'null':
                for col in similarity_feas.split("\t"):
                    if col != selected_output:
                        if self.add_col(result=result, col=col, df=df, feature_type_dict=feature_type_dict):
                            used_columns.add(col)
                            fea_num += 1
                        else:
                            return None

        if fea_num <= 0:
            logger.info('与任务相关的输入特征和任务指定的输入特征都无法找到！！！')
            return None


        # 4. 将空列表替换为'null'
        for key in result['特征字段名']:
            if not result['特征字段名'][key]:
                result['特征字段名'][key] = 'null'

        if result['标签名'] is None:
            result['标签名'] = 'null'
        result['思考理由'] = thinking_dict

        print(f'result: {result}')

        return result

    def extract_relevant_data(self, intent: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
        """主函数：提取相关特征和数据"""
        # 1. 加载表格
        '''
        scan_result = {
           中心1：{文件名 1: Dataframe1},
           中心2：{文件名 2: Dataframe1}
        }
        '''
        scan_result = self.scan_center_files(intent['任务目标'])

        if not scan_result:
            raise ValueError("没有找到任何表格文件")

        # 2. 合并表格（如果有多个）
        multicenter_merged_table, multicenter_feature_type_dict = self.merge_tables(scan_result=scan_result)

        multicenter_retrieved_feature_dict = {}
        multicenter_relevant_data_dict = {}

        retrieve_success_flag = False

        for center, merged_df in multicenter_merged_table.items():
            logger.info(f"{center} 合并后的表格形状: {merged_df.shape}")

            feature_type_dict = multicenter_feature_type_dict[center]

            # 3. 为每个中心寻找匹配的列
            column_mapping = self.find_matching_columns(intent, merged_df, feature_type_dict).copy()

            if column_mapping is None:
                logger.info(f'{center}数据中心没有符合此次检索的数据！')
                continue
            else:
                retrieve_success_flag = True

            multicenter_retrieved_feature_dict[center] = column_mapping

            # 4. 提取相关数据
            relevant_columns = []

            # 收集所有相关列名
            for feature_type, columns in column_mapping['特征字段名'].items():
                if columns != 'null':
                    for col in columns:
                        # 去除<>标记
                        col_name = col.strip('<>')
                        if col_name in merged_df.columns:
                            relevant_columns.append(col_name)

            # 添加标签列
            if column_mapping['标签名'] != 'null':
                label_col = column_mapping['标签名'].strip('<>')
                if label_col in merged_df.columns:
                    relevant_columns.append(label_col)

            # 提取相关数据
            if relevant_columns:
                relevant_data = merged_df[relevant_columns].copy()
                print(f"relevant_data: {relevant_data.shape}")
                relevant_data = self.process_relevant_data(relevant_data=relevant_data,
                                                           column_mapping=column_mapping)
            else:
                relevant_data = pd.DataFrame()

            multicenter_relevant_data_dict[center] = relevant_data

        if retrieve_success_flag:
            return multicenter_retrieved_feature_dict, multicenter_relevant_data_dict
        else:
            logger.info(f'当前医学数据库没有符合此次检索的数据！')
            return None, None

    def process_relevant_data(self, column_mapping, relevant_data):
        # 处理所有列的基础清洗
        for col in relevant_data.columns:
            if relevant_data[col].dtype == 'object':
                # 移除前后空格
                relevant_data[col] = relevant_data[col].astype(str).str.strip()

                # 处理常见的缺失值表示
                relevant_data[col] = relevant_data[col].replace({
                    'nan': np.nan, 'NaN': np.nan, 'NULL': np.nan, 'null': np.nan,
                    'None': np.nan, 'none': np.nan, '': np.nan, ' ': np.nan,
                    'N/A': np.nan, 'n/a': np.nan, '#N/A': np.nan, '#NULL!': np.nan,
                    '#DIV/0!': np.nan, '#REF!': np.nan, '#NAME?': np.nan,
                    '-': np.nan, '--': np.nan, '?': np.nan, '未知': np.nan,
                    '无': np.nan, '暂无': np.nan, '缺失': np.nan
                })

        # 删除标签为空的数据
        if column_mapping['标签名'] != 'null':
            if column_mapping['标签名'].strip('<>') in relevant_data.columns:
                relevant_data.dropna(subset=[column_mapping['标签名'].strip('<>')], inplace=True)

        return relevant_data

    def retrieve(self, intent):
        self.retrieve_start_time = time.time()

        task_goal = intent['任务目标']

        self.token_recorder = TokenTracker(file_path=f'../data/interim/任务目标({task_goal})', filename="token_records.json", module='特征检索')

        multicenter_retrieved_feature_dict = None

        for i in range(3):
            multicenter_retrieved_feature_dict = self.single_retrieve(intent=intent)

            if multicenter_retrieved_feature_dict is None:
                logger.info('出现错误或没有检索数据，重新检索中...')
                continue
            else:
                retrieve_end_time = time.time()
                logger.info(f"特征检索耗时：{(retrieve_end_time - self.retrieve_start_time) / 60:.2f} 分钟")

                think_tracker = ThinkTracker(file_path=f'../data/interim/任务目标({task_goal})', filename="think_records.json")

                for center, _ in multicenter_retrieved_feature_dict.items():
                    think_tracker.add_record(module='特征检索', sub_module=center, think=multicenter_retrieved_feature_dict[center]['思考理由'])

                multicenter_retrieved_feature_dict['耗时（min）'] = round((retrieve_end_time - self.retrieve_start_time) / 60, 2)

                tracker = TimeTracker(file_path=f'../data/interim/任务目标({task_goal})', filename="time_records.json")
                tracker.add_record(start_time=self.retrieve_start_time, end_time=retrieve_end_time,
                                   record_name='特征检索耗时（min）')

                with open(self.retrieve_features_path, 'w', encoding='utf-8') as f:
                    json.dump(multicenter_retrieved_feature_dict, f, ensure_ascii=False, indent=2)

                logger.info(f"\n特征字段已保存到 {self.retrieve_features_path}")
                break

        if multicenter_retrieved_feature_dict is None:
            validation = intent['医学假设']
            logger.info(f'检索失败！！！当前数据库没有可验证此医学假设（{validation}）的数据！！！')

        return multicenter_retrieved_feature_dict


    def single_retrieve(self, intent):
        try:
            # 提取特征和数据
            multicenter_retrieved_feature_dict, multicenter_relevant_data_dict = self.extract_relevant_data(intent)

            if multicenter_retrieved_feature_dict is None or multicenter_relevant_data_dict is None:
                return None

            task_goal = intent['任务目标']

            for center, column_mapping in multicenter_retrieved_feature_dict.items():

                logger.info(f"\n{center}中心识别结果如下:")
                logger.info("特征字段名：")
                for f_type in column_mapping['特征字段名']:
                    logger.info(f"{f_type}:{column_mapping['特征字段名'][f_type]}")

                logger.info(f"标签名:{column_mapping['标签名']}")
                logger.info(f"标签类型:{column_mapping['标签类型']}")

                relevant_data = multicenter_relevant_data_dict[center]


                logger.info(f"\n提取的数据量:{relevant_data.shape}")

                # 保存结果
                task_related_data_path = f'{self.task_related_data_dir}/task_related_data_{center}.csv'
                relevant_data.to_csv(task_related_data_path, index=False)
                logger.info(f"\n数据已保存到 {task_related_data_path}")

                multicenter_retrieved_feature_dict[center]['任务相关数据路径'] = task_related_data_path

            return multicenter_retrieved_feature_dict


        except Exception as e:
            logger.info(f"处理过程中出现错误: {str(e)}")
            return None

