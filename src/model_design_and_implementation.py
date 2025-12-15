import os
import json
import re
import time

import requests
from typing import Optional, Tuple
import anthropic
import openai
import os
import logging
import sys
from llm_api_config import  CLAUDE_KEY_API
from check_result import check_features_openai
import json
from streaming_llm_handler import stream_openai_response
from logger import init_logger
from time_util import TimeTracker, ThinkTracker, TokenTracker
base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
if not logger.handlers:
    logger = init_logger('../data/log')


def create_dir_if_not_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

class CodeGenerator:
    """AI代码生成器类"""

    def __init__(self, api_key: object = None, base_url: object = None, task_goal: object = None, reset_token = True) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.reset_token = reset_token

        self.token_recorder = TokenTracker(file_path=f'../data/interim/任务目标({task_goal})',
                                           filename="token_records.json",
                                           module='模型设计与实现',
                                           reset=self.reset_token)

    def call_llm_api_for_code(self, prompt: str) -> Optional[str]:

        """使用Anthropic Claude API生成代码"""
        if self.api_key is None:
            openai.api_key = CLAUDE_KEY_API
        if self.base_url is None:
            openai.base_url = "https://chat.cloudapi.vip/v1/"
            openai.default_headers = {"x-foo": "true"}

        try:
            res = stream_openai_response(base_url="https://chat.cloudapi.vip/v1/",
                                         api_key=CLAUDE_KEY_API,
                                         prompt=prompt,
                                         model="claude-opus-4-1-20250805",
                                         timeout=6000,
                                         temperature=0.3,
                                         token_recorder=self.token_recorder)


            return res
        except Exception as e:
            logger.info(f"An error occurred: {e}")
            return None

    def parse_response_to_code_reasoning(self, response: str) -> Tuple[str, str]:
        """
        解析API响应，分离思考理由和代码

        Args:
            content: API返回的完整内容

        Returns:
            (思考理由, 代码)的元组
        """
        reasoning = ""
        code = ""

        # 尝试提取思考理由部分
        reasoning_pattern = r"##?\s*思考理由\s*\n(.*?)(?=##?\s*模型代码|```python|$)"
        reasoning_match = re.search(reasoning_pattern, response, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        # 尝试提取代码部分
        code_pattern = r"```python\n(.*?)```"
        code_match = re.search(code_pattern, response, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            # 如果没有找到代码块，尝试其他模式
            code_pattern2 = r"##?\s*模型代码\s*\n(.*?)(?=##|$)"
            code_match2 = re.search(code_pattern2, response, re.DOTALL)
            if code_match2:
                code = code_match2.group(1).strip()

        # 如果还是没有找到，尝试更宽松的匹配
        if not reasoning and not code:
            # 将内容按照某些关键词分割
            if "```" in response:
                parts = response.split("```")
                if len(parts) >= 2:
                    reasoning = parts[0].strip()
                    code = parts[1].replace("python", "").strip()

        return code, reasoning


    def save_code(self, code: str, filename: str):
        """保存生成的代码到文件"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(code)
            logger.info(f"\n代码已保存到: {filename}")
        except Exception as e:
            logger.info(f"保存文件失败: {e}")

    def extract_code_blocks(self, response: str) -> str:
        """从响应中提取代码块"""
        # 尝试提取markdown代码块
        import re
        code_pattern = r'```python\n(.*?)```'
        matches = re.findall(code_pattern, response, re.DOTALL)

        if matches:
            return '\n\n'.join(matches)

        # 如果没有代码块标记，返回整个响应
        return response

    def build_generation_task_prompt(self, retrieved_features, task_related_data_path,database_center,
                                     task_related_train_name='lora_train_data.json',
                                     task_related_test_name='lora_test_data.json',
                                     train_yaml_path='train_config.yaml',
                                     predict_yaml_path='test_config.yaml',
                                     model_name_or_path=f'../data/llm/model/Qwen3-8B'):

        task_goal = retrieved_features["任务目标"]
        input_features = retrieved_features["特征字段名"]
        output_label = retrieved_features["标签名"]
        config_dir = f'../data/llm/config/{task_goal}_{database_center}'
        train_output_dir = f'../data/llm/saves/{task_goal}_{database_center}/train'
        predict_output_dir = f'../data/llm/saves/{task_goal}_{database_center}/predict'
        dataset_dir = f'../data/llm/dataset'

        input_features_prompt = ""
        for feature_type in input_features:
            if input_features[feature_type] != "nan":
                input_features_prompt += f"""{feature_type}为{','.join(input_features[feature_type])} ;\n """

        prompt = f"""请微调一个任务目标为{task_goal}的大语言模型，步骤如下：\n
                     1、请将{task_related_data_path}的数据转换成以下的JSON格式，包含instruction，input 和 output；可根据具体情况修改包含instruction和input，要求符合大模型提示词的规范要求。\n
                     instruction为任务目标{task_goal}，\n
                     input由特征字段{input_features_prompt}的内容组成(每个特征由<>包围)，\n
                     input指的是标签名{output_label}。\n
                     请将生成的数据按照8：2的格式进行划分，训练集放置在../data/llm/dataset/{task_goal}_{database_center}/{task_related_train_name}, 
                     测试集放置在../data/llm/dataset/{task_goal}_{database_center}/{task_related_test_name}  \n          
                    格式如下：
                   [
                      {{
                        "instruction": "识别并解释给定列表中的两个科学理论：细胞理论和日心说。",
                        "input": "",
                        "output": "细胞理论是生物科学的一个理论，它认为所有生命体都是由微小的基本单元——细胞所构成。这是生物学的一个基础理论，认为细胞是所有生物的基本结构和功能单位，所有的生物都是由一个或多个细胞组成，细胞只能通过细胞分裂产生新的细胞。这一理论由薛定谔、施瓦内和雪莱于1839年首次提出。\n\n日心说是指太阳是太阳系的中心，也就是说，行星围绕太阳旋转的理论。这个理论打破了传统的地心说观点，认为地球并不是宇宙的中心。日心说的提出者是尼古拉·哥白尼，他在16世纪初发表了他的著作《天体运行论》，阐述了太阳系行星围绕太阳运行的模型，为天文学的发展做出了巨大贡献。"
                      }},
                      {{
                        "instruction": "输入三支篮球队的名称并生成一个适当的口号。",
                        "input": "输入：俄克拉荷马城雷霆队，芝加哥公牛队，布鲁克林网队。",
                        "output": "输出： "雷霆，公牛和网队：各显神通，角逐群雄!""
                      }}
                   ] \n

                   2、向"../data/llm/dataset/dataset_info.json"文件中新增以下内容：\n
                   {task_related_train_name[:-len('.json')]}: {{
                    "file_name": {task_goal}_{database_center}/{task_related_train_name}
                   }},
                   \n
                   {task_related_test_name[:-len('.json')]}:{{
                   "file_name": {task_goal}_{database_center}/{task_related_test_name}
                   }}
                   \n 
                   3、请import llm_llama_factory_api_v2.py脚本，确保导入正确，并调用llm_train方法进行训练微调，赋值的参数如下：\n
                      dataset={task_related_train_name[:-len('.json')]} , output_dir={train_output_dir},
                      yaml_file_path = {config_dir}/{train_yaml_path}, dataset_dir={dataset_dir},
                      model_name_or_path = {model_name_or_path}，train_max_samples = 具体训练集的大小
                      \n
                   4、请import llm_llama_factory_api_v2.py脚本，确保导入正确，并调用llm_predict方法进行推理测试，赋值的参数如下：\n
                      eval_dataset={task_related_test_name[:-len('.json')]} , output_dir={predict_output_dir},
                      yaml_file_path = {config_dir}/{predict_yaml_path}， adapter_name_or_path={train_output_dir},
                      dataset_dir={dataset_dir}, model_name_or_path={model_name_or_path}， test_max_samples = 具体测试集的大小 \n\n
                      
                      注意： lm_llama_factory_api_v2.py脚本 为 基于LLaMA-Factory框架， 自动生成训练/推理 YAML（含 LoRA 配置、数据集、模板、超参等），并调用命令执行任务。
                   
                   5、本任务模型采用的大模型是Qwen3-8B,，请结合医学领域大模型应用及具体的任务目标进行是否微调的判断，如果不需要使用微调的大模型，
                   请将adapter_name_or_path设置为None,而不用设置为{train_output_dir}。\n
                      注意：本任务模型采用的大模型是Qwen3-8B, 候选的模型为medgemma-4b-it。\n
                      
                   6、要求python脚本是可执行的，存在 if __name__ == "__main__" 启动入口。\n
                   7、请确保生成的代码符合上述要求，此类任务是生成任务，不用对特征进行处理。\n
                   8、请分别返回python脚本，以及设计整个模型方案的理由。生成完代码后，请对代码进行检查，是否符合上述所有要求，如果没有，则重新生成。\n
                     按照以下格式输出：           
                      ## 思考理由
                       [在这里详细说明你的设计思路、技术选型理由、架构考虑等，要求简洁、明了。一定要简洁！]
                    
                     ## 模型代码
                    ```python
                    [在这里输出完整的模型代码]
                    ```
        """

        return prompt

    def build_regression_task_prompt(self, retrieved_features, database_center):
        task_goal = retrieved_features["任务目标"]
        input_features = retrieved_features["特征字段名"]
        output_label = retrieved_features["标签名"]
        output_label_type = retrieved_features["标签类型"]
        task_related_data_path = retrieved_features['任务相关数据路径']
        result_root_path = create_dir_if_not_exist(f'../data/model/任务目标({task_goal})_{database_center}')

        input_features_prompt = ""
        for feature_type in input_features:
            if input_features[feature_type] != "nan":
                input_features_prompt += f"""{feature_type}为{','.join(input_features[feature_type])} ;\n """

        prompt = f"""请开发一个任务目标为{task_goal}的回归模型。\n
                                       输入为{input_features_prompt}。\n
                                       输出的标签为{output_label}。\n
                                       输出的标签类型为{output_label_type}。\n
                                       1、用python代码进行实现，返回一个可行性的python脚本，要求存在 if __name__ == "__main__" 启动入口。\n
                                       2、要求代码能够执行训练与推理，训练集和测试集的比例为8：2。\n
                                       请先划分训练集和测试集，再进行编码处理。先对训练集进行编码，再对测试集采用训练集的编码标准,保存编码标准字典或文件。\n
                                       训练集放置在../data/model/任务目标({task_goal})_{database_center}/train_data.csv, 
                                       测试集放置在../data/model/任务目标({task_goal})_{database_center}/test_data.csv。\n
                                       3、评估指标请使用MAE, RMSE, MAPE等指标， 并输出测试集的评估指标结果，评估结果存放在{result_root_path}下面，用单独的 evaluation_result.json文件保存。\n
                                       4、如果有多个模型，评价指标结果只保存最佳模型的评价指标。同时， 对应的预测结果也需要进行保存。\n
                                       5、训练与推理的数据存放路径为{task_related_data_path}。\n
                                       6、如有训练，则模型参数存放的目录为{result_root_path}。\n
                                       7、保持路径<"{result_root_path}">和<"{task_related_data_path}">中出现<>任务目标<\>这四个字，不要写错了。\n
                                       8、如果涉及文本编码，请调用SentenceTransformer模型，其内置语言模型用paraphrase-multilingual-MiniLM-L12-v2，路径为../data/llm/model/paraphrase-multilingual-MiniLM-L12-v2。\n
                                       注意：
                                       1、请自动分析任务目标和特征具体情况，充分思考并使用合适的模型。\n
                                       2、不需要对输出标签进行编码处理，但需要清除空值。\n
                                       4、离散特征需要对特征进行离散编码处理，使之能够进行映射，符合模型的要求。\n
                                       5、连续特征需要对特征进行z-score标准化编码处理，不同的连续特征分别进行标准化。\n
                                       6、文本特征需要使用语言模型对文本进行嵌入化。\n
                                        7、对于连续特征缺失的情况，在标准化编码之后请进行填充处理。对于连续特征，用平均值填充。\n
                               对于离散特征的缺失值，用离散特征编码后的第一个值进行填充。连续特征和离散特征中，所有缺失值要生成相应的mask值供模型处理。\n
                                       8、要求对数据的处理，处理前和处理后的数据量是一致的。\n
                                       9、文本编码的内置语言模型一定要引用正确，即为../data/llm/model/paraphrase-multilingual-MiniLM-L12-v2。\n
                                       10、请分别返回python脚本，以及设计整个模型方案的理由。\n
                                       11.请按照以下格式输出：\n

                                      ## 思考理由
                                       [在这里详细说明你的设计思路、技术选型理由、架构考虑等，要求简洁、明了。一定要简洁！]

                                     ## 模型代码
                                    ```python
                                    [在这里输出完整的模型代码]
                                    ```"""

        return prompt

    def build_classification_task_prompt(self, retrieved_features, database_center):
        task_goal = retrieved_features["任务目标"]
        input_features = retrieved_features["特征字段名"]
        output_label = retrieved_features["标签名"]
        output_label_type = retrieved_features["标签类型"]
        task_related_data_path = retrieved_features['任务相关数据路径']
        result_root_path = create_dir_if_not_exist(f'../data/model/任务目标({task_goal})_{database_center}')

        input_features_prompt = ""
        for feature_type in input_features:
            if input_features[feature_type] != "nan":
                input_features_prompt += f"""{feature_type}为{','.join(input_features[feature_type])} ;\n """

        prompt = f"""请开发一个任务目标为{task_goal}的分类模型。\n
                               输入为{input_features_prompt}。\n
                               输出的标签为{output_label}。\n
                               输出的标签类型为{output_label_type}。\n
                               1、用python代码进行实现，返回一个可行性的python脚本，要求存在 if __name__ == "__main__" 启动入口。\n
                               2、要求代码能够执行训练与推理，训练集和测试集的比例为8：2。\n
                               请先划分训练集和测试集，再进行编码处理。先对训练集进行编码，再对测试集采用训练集的编码标准,保存编码标准字典或文件。\n
                               训练集放置在../data/model/任务目标({task_goal})_{database_center}/train_data.csv, 
                               测试集放置在../data/model/任务目标({task_goal})_{database_center}/test_data.csv。\n
                               3、评估指标请使用AUC-ROC, Recall, Precision, F1-score, ACC, AUPRC等指标， 并输出测试集的评估指标结果。\n
                               请根据不同的分类数量进行采用不同的指标计算方式计算整体结果：单标签多分类（大于二分类）的话请输出每个分类的结果，以及所有指标的整体平均结果，包含micro，macro, weighted模式（这三个模式中，仅限AUC-ROC, Recall, Precision, F1-score，AUPRC）。
                               单标签二分类的话输出正类、负类的结果，整体结果用binary模式,输出AUC-ROC, Recall, Precision, F1-score, ACC, AUPRC等指标。\n
                               评估结果存放在{result_root_path}下面，用单独的 evaluation_result.json文件保存。\n
                               4、如果有多个模型，评价指标结果只保存最佳模型的评价指标。同时，对应的输出概率、预测结果和ground truth与对应的测试集拼接后进行保存，存放目录为{result_root_path}。\n
                               5、训练与推理的数据存放路径为{task_related_data_path}。\n
                               6、如有训练，则模型参数存放的目录为{result_root_path}。\n
                               7、保持路径<"{result_root_path}">和<"{task_related_data_path}">中出现<>任务目标<\>这四个字，不要写错了。\n
                               8、如果涉及文本编码，请调用SentenceTransformer模型，其内置语言模型用paraphrase-multilingual-MiniLM-L12-v2，路径为../data/llm/model/paraphrase-multilingual-MiniLM-L12-v2。\n
                               注意：
                               1、请自动分析任务目标和特征具体情况，充分思考并使用合适的模型。\n
                               2、需要对输出标签进行编码处理，使之能够更新模型参数，符合模型的要求。\n
                               4、离散特征需要对特征进行编码处理，使之能够进行映射，符合模型的要求。\n
                               5、连续特征需要对特征进行z-score标准化处理，不同的连续特征分别进行标准化。\n
                               6、文本特征需要使用语言模型对文本进行嵌入化。\n
                               7、对于连续特征缺失的情况，在标准化编码之后请进行填充处理。对于连续特征，用平均值填充。\n
                               对于离散特征的缺失值，用离散特征编码后的第一个值进行填充。连续特征和离散特征中，所有缺失值要生成相应的mask值供模型处理。\n
                               8、要求对数据的处理，处理前和处理后的数据量是一致的。\n
                               9、针对测试集的未知标签，要对数据进行删除。\n
                               10、请确保每个类别都能算到每个评估指标，以及每个评估指标的整体平均结果。\n
                               11、请分别返回python脚本，以及设计整个模型方案的理由。生成完代码后，请对代码进行检查，是否符合上述所有要求，如果没有，则重新生成。\n
                               12.请按照以下格式输出：
                            
                              ## 思考理由
                               [在这里详细说明你的设计思路、技术选型理由、架构考虑等，要求简洁、明了。一定要简洁！]
                            
                             ## 模型代码
                            ```python
                            [在这里输出完整的模型代码]
                            ```"""

        return prompt

    def build_analysis_task_prompt(self, retrieved_features, database_center):
        task_goal = retrieved_features["任务目标"]
        input_features = retrieved_features["特征字段名"]
        output_label = retrieved_features["标签名"]
        output_label_type = retrieved_features["标签类型"]
        task_related_data_path = retrieved_features['任务相关数据路径']
        result_root_path = create_dir_if_not_exist(f'../data/model/任务目标({task_goal})_{database_center}')

        input_features_prompt = ""
        for feature_type in input_features:
            if input_features[feature_type] != "nan":
                input_features_prompt += f"""{feature_type}为{','.join(input_features[feature_type])} ;\n """

        if output_label != "nan":
            target = f'输出{output_label}等相关指标'
        else:
            target = f'根据医学相关知识输出相关指标'



        prompt = f"""请开发一个任务目标为{task_goal}的数据分析任务。\n
                                       分析以下特征{input_features_prompt}之间的关系，并{target}。\n
                                       1、用python代码进行实现，返回一个可行性的python脚本，要求存在 if __name__ == "__main__" 启动入口。\n
                                       2、分析数据存放路径为{task_related_data_path}。\n
                                       3、相关性分析结果存放在{result_root_path}下面，用单独的 evaluation_result.json文件保存。\n
                                       4、保持路径<"{result_root_path}">和<"{task_related_data_path}">中出现<>任务目标<\>这四个字，不要写错了。\n
                                       5、请保证引用的特征名与要求的特征是一致的，不要更改。\n
                                        注意：
                                       1、请对分析的数据进行处理，删除为空的数据。\n
                                       2、请从医学的角度，专业地分析问题和设计分析方案。\n
                                       3、请分别返回python脚本，以及设计整个数据分析方案的理由。\n
                                       4.请按照以下格式输出：

                                      ## 思考理由
                                       [在这里详细说明你的设计思路、技术选型理由、架构考虑等，要求简洁、明了。一定要简洁！]

                                     ## 模型代码
                                    ```python
                                    [在这里输出完整的模型代码]
                                    ```"""

        return prompt




    def generate_single_code(self, retrieved_features: dict= None,
                      task_related_data_path: str= None, code_save_path: str= None,
                      result_root_path: str=None, database_center: str=None) -> Optional[str]:
        """
             生成代码
             intention： 意图字典
             retrieved_features： 检索到的特征字段字典
             task_related_data_path： 特征相关数据的保存路径
             code_save_path： 代码保存路径
        """
        logger.info("=== AI代码生成器 ===")

        # 根据任务类型生成代码提示
        logger.info(f"\n正在生成{database_center}中心的代码...")
        task_type = retrieved_features['任务类型']
        if task_type == '生成任务':
            prompt = self.build_generation_task_prompt(retrieved_features=retrieved_features,
                                                       task_related_data_path=task_related_data_path,
                                                       database_center=database_center)
        elif task_type == '回归任务':
            prompt = self.build_regression_task_prompt(retrieved_features=retrieved_features,
                                              database_center=database_center)
        elif task_type == '分类任务':
            prompt = self.build_classification_task_prompt(retrieved_features=retrieved_features,
                                                           database_center=database_center)
        elif task_type == '数据分析任务':
            prompt = self.build_analysis_task_prompt(retrieved_features=retrieved_features,
                                            database_center=database_center)

        # 提取代码块
        res = self.call_llm_api_for_code(prompt)

        if res:
            # 提取代码块和推理内容
            code, reasoning = self.parse_response_to_code_reasoning(res)
            # 再次验证代码块
            clean_code = self.extract_code_blocks(code)
            feature_list = []
            for feature_type, columns in retrieved_features['特征字段名'].items():
                if columns != 'nan':
                    for col in columns:
                        # 去除<>标记
                        col_name = col.strip('<>')
                        feature_list.append(col_name)

            if retrieved_features['标签类型'] != 'nan':
                feature_list.append(retrieved_features["标签名"].strip('<>'))

            logger.info("特征合规性检查中...\n")
            flag, message = check_features_openai(code=clean_code, feature_list=feature_list, token_recorder=self.token_recorder)
            if not flag:
                logger.info(f"特征：{message}不在特征字典！")
                return None
            logger.info("代码特征合规性检查通过")

            self.save_code(clean_code, code_save_path)

            task_goal = retrieved_features['任务目标']
            tracker = ThinkTracker(file_path=f'../data/interim/任务目标({task_goal})', filename="think_records.json")
            tracker.add_record(module='模型设计与实现', sub_module=database_center,think=reasoning)

            return code

    def generate_code(self, retrieved_features: dict = None,
                      task_related_data_path: str = None, code_save_path: str = None,
                      result_root_path: str = None, database_center: str = None,
                      task_goal: str = None) -> Optional[str]:


        for i in range(3):
            logger.info(f"start generating code for {i+1} time...")
            code = self.generate_single_code(retrieved_features=retrieved_features,
                                      task_related_data_path=task_related_data_path,
                                      code_save_path=code_save_path,
                                      result_root_path=result_root_path,
                                      database_center=database_center)

            if code is not None:
                return code
            else:
                logger.info('There is an error in the code！Regenerate it!')



