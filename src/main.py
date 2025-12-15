import json
import logging
import os
import sys
import time

from intent_recognition import IntentRecognition
from data_retrieval import IntentTableFeatureExtractor
from model_design_and_implementation import CodeGenerator
from code_adjustment import CodeAdjuster
from logger import init_logger
from time_util import TimeTracker
from validation_summary import MedicalHypothesisAnalyzer
base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
if not logger.handlers:
    logger = init_logger('../data/log')

def create_dir_if_not_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def intent_recognition_tool(requirement, task_goal):
    # 意图识别 -----------------------------------------------------------------------------------------------
    logger.info(f"医学假设： {requirement}")
    interim_root_path = '../data/interim'
    logger.info('开始意图识别..')
    intent = None
    if task_goal is None:
        intent_recognition = IntentRecognition()
        intent = intent_recognition.query(requirement=requirement, intent_save_root_path=interim_root_path)
    else:
        # 指定已经识别过的意图结果，方便调试
        with open(
            f"../data/interim/任务目标({task_goal})/medical_intent_recognition.json", "r", encoding="utf-8") as f:
            intent = json.load(f)

        output = []
        output.append(f"\n任务目标: {intent['任务目标']}")
        output.append(f"\n任务类型: {intent['任务类型']}")
        output.append("\n输入特征:")

        for feature_type, feature_value in intent['输入'].items():
            output.append(f"  - {feature_type}: {feature_value}")

        output.append(f"\n输出: {intent['输出']}")
        output.append(f"\n\n思考结果: {intent['思考结果']}")
        output.append("=" * 50)
        logger.info(output)

    return intent

def feature_retrieval_tool(intent):
    # 特征检索------------------------------------------------------------------------------------------------
    # 急诊数据库文件夹目录
    emergency_database_dir = "../data/raw_processed"

    task_goal = intent['任务目标']
    retrieve_features_dir = create_dir_if_not_exist(f'../data/interim/任务目标({task_goal})')

    # 检索到的特征字段的存储路径
    retrieve_features_path = f"{retrieve_features_dir}/retrieve_features_of_table.json"

    # 特征字段对应的数据存储目录
    task_related_data_dir = retrieve_features_dir

    logger.info('开始特征检索..')

    extractor = IntentTableFeatureExtractor(emergency_database_dir=emergency_database_dir,
                                            retrieve_features_path=retrieve_features_path,
                                            task_related_data_dir=task_related_data_dir)
    multicenter_retrieved_feature_dict = extractor.retrieve(intent)

    return multicenter_retrieved_feature_dict

def model_design_and_implementation_tool(multicenter_retrieved_feature_dict, reset_token=True):
    ## 代码生成-------------------------------------------------------------------------------------------------
    task_goal = list(multicenter_retrieved_feature_dict.values())[0]['任务目标']
    logger.info(f'任务目标{task_goal}')

    generator = CodeGenerator(reset_token=reset_token, task_goal=task_goal)

    tracker = TimeTracker(file_path=f'../data/interim/任务目标({task_goal})', filename="time_records.json")

    generation_start_time = time.time()

    for center, retrieved_features in multicenter_retrieved_feature_dict.items():

        if center == '耗时（min）':
            continue

        logger.info(f'开始生成{center}中心代码')

        # 代码保存路径
        code_save_path = f"task_related_code_{task_goal}_{center}.py"

        # 指标、模型参数等结果的存放mul
        task_related_data_path = retrieved_features["任务相关数据路径"]

        code = generator.generate_code(retrieved_features=retrieved_features,
                                       database_center=center,
                                       code_save_path=code_save_path,
                                       task_related_data_path=task_related_data_path,
                                       task_goal=task_goal)

    tracker.add_record(start_time=generation_start_time, end_time=time.time(),
                       record_name=f'模型设计与实现总耗时（min）')

    return code

def code_adjuster_tool(multicenter_retrieved_feature_dict, reset_token=True):
    # 代码调整-------------------------------------------------------------------------------------------
    logger.info('进入代码调整阶段..')

    task_goal = list(multicenter_retrieved_feature_dict.values())[0]['任务目标']
    logger.info(f'任务目标{task_goal}')

    first_center = True

    result = None

    for center, retrieved_features in multicenter_retrieved_feature_dict.items():
        if center == '耗时（min）':
            continue
        logger.info(f'开始调整{center}中心代码')

        # 配置参数
        code_path = f"task_related_code_{task_goal}_{center}.py"  # 要调试的脚本路径

        if first_center and reset_token:
            reset = True
            first_center = False
        else:
            reset = False

        # 创建调试器实例
        debugger = CodeAdjuster(script_path=code_path, reset_token=reset, task_goal=task_goal)
        # 开始调试
        result = debugger.debug_loop()

        if result["修改类型"] != "成功":
            logger.info("\n😞 调试失败，请检查错误信息。")
            return {"修改类型": "错误"}

    return result


def validation_summary_tool(intent, multicenter_retrieved_feature_dict):
    # 验证结果总结-------------------------------------------------------------

    logger.info('执行验证结果总结分析...')

    task_goal = intent['任务目标']

    logger.info(f'任务目标{task_goal}')

    execution_time_path = f"../data/interim/任务目标({task_goal})/time_records.json"

    # 创建分析器实例
    analyzer = MedicalHypothesisAnalyzer(intent=intent,
                                         execution_time_path=execution_time_path,
                                         multicenter_retrieved_feature_dict=multicenter_retrieved_feature_dict)

    # 执行分析
    result = analyzer.analyze_and_summarize()

    return result


def single_validation(requirement, task_goal: str=None):
    # 意图识别
    intent = intent_recognition_tool(requirement=requirement, task_goal=task_goal)
    # 特征检索
    multicenter_retrieved_feature_dict = feature_retrieval_tool(intent=intent)

    if multicenter_retrieved_feature_dict is not None:
        # 代码设计与生成
        code=model_design_and_implementation_tool(multicenter_retrieved_feature_dict=multicenter_retrieved_feature_dict)
        # 代码调整
        task_goal = list(multicenter_retrieved_feature_dict.values())[0]['任务目标']
        restart = True

        tracker = TimeTracker(file_path=f'../data/interim/任务目标({task_goal})', filename="time_records.json")
        finetune_start_time = time.time()
        try:
            result = code_adjuster_tool(multicenter_retrieved_feature_dict=multicenter_retrieved_feature_dict)
            # logger.info(result)
            if result["修改类型"] == "失败" or result["修改类型"] == "错误":
                logger.info("调试失败，重新执行代码生成！")
                code = model_design_and_implementation_tool(
                    multicenter_retrieved_feature_dict=multicenter_retrieved_feature_dict, reset_token=False)
                result = code_adjuster_tool(multicenter_retrieved_feature_dict=multicenter_retrieved_feature_dict, reset_token=False)

        except Exception as e:
            logger.info(f"错误：{str(e)}，重新执行代码！！！")
            result = code_adjuster_tool(multicenter_retrieved_feature_dict=multicenter_retrieved_feature_dict, reset_token=False)
            if result["修改类型"] != "成功":
                logger.info("\n😞 调试失败，请检查错误信息。")


        tracker.add_record(start_time=finetune_start_time, end_time=time.time(),
                           record_name=f'模型调整总耗时（min）')

        # 验证总结
        validation_summary_tool(intent=intent, multicenter_retrieved_feature_dict=multicenter_retrieved_feature_dict)


    else:
        validation_summary_tool(intent=intent, multicenter_retrieved_feature_dict=multicenter_retrieved_feature_dict)


def main():
    logger.info("\nWelcome to the HypoAgent. Please enter your verification query (input \"exit\" to exit).：")
    while True:
        requirement = input().strip()
        if requirement == 'exit':
            break
        single_validation(requirement=requirement)

if __name__ == "__main__":
    main()



#


