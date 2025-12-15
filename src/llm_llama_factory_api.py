import os
import yaml
import subprocess
import sys
from pathlib import Path

import os
import logging
import sys
from logger import init_logger

base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
if not logger.handlers:
    logger = init_logger('../data/log')


def create_inference_yaml_config(dataset_dir='./data/dataset',
                                 eval_dataset='alpaca_zh_demo',
                                 output_dir='./data/saves/Qwen3-8B/lora/predict',
                                 model_name_or_path='./data/model/Qwen3-8B',
                                 adapter_name_or_path=None,
                                 yaml_file_path='./data/train_lora/predict.yaml',
                                 test_max_samples=100000,
                                 per_device_eval_batch_size=16,
                                 temperature=0.2):
    pass
    # 定义配置数据
    config = {
        'model_name_or_path': model_name_or_path,
        'trust_remote_code': True,
        'adapter_name_or_path': adapter_name_or_path,
        # method
        'stage': 'sft',
        'do_predict': True,
        'finetuning_type': 'lora',

        'dataset_dir': dataset_dir,
        'eval_dataset': eval_dataset,
        'template': 'qwen',
        'cutoff_len': 1024,
        'max_samples': test_max_samples,
        'overwrite_cache': True,
        'preprocessing_num_workers': 16,
        'dataloader_num_workers': 4,
        'per_device_eval_batch_size': per_device_eval_batch_size,
        'temperature': temperature,

        'output_dir': output_dir,
        'overwrite_output_dir': True,
        'predict_with_generate': True

    }

    # 确保/data/目录存在
    os.makedirs('./data', exist_ok=True)

    try:
        with open(yaml_file_path, 'w', encoding='utf-8') as f:
            # 添加注释分组
            f.write("### model\n")
            if config['adapter_name_or_path'] is not None:
                yaml.dump({
                    'model_name_or_path': config['model_name_or_path'],
                    'trust_remote_code': config['trust_remote_code'],
                    'adapter_name_or_path': config['adapter_name_or_path']
                }, f, default_flow_style=False, allow_unicode=True)

                f.write("\n### method\n")
                yaml.dump({
                    'stage': config['stage'],
                    'do_predict': config['do_predict'],
                    'finetuning_type': config['finetuning_type']
                }, f, default_flow_style=False, allow_unicode=True)
            else:
                yaml.dump({
                    'model_name_or_path': config['model_name_or_path'],
                    'trust_remote_code': config['trust_remote_code'],
                }, f, default_flow_style=False, allow_unicode=True)

                f.write("\n### method\n")
                yaml.dump({
                    'stage': config['stage'],
                    'do_predict': config['do_predict'],
                }, f, default_flow_style=False, allow_unicode=True)


            f.write("\n### dataset\n")
            yaml.dump({
                'dataset_dir': config['dataset_dir'],
                'eval_dataset': config['eval_dataset'],
                'template': config['template'],
                'cutoff_len': config['cutoff_len'],
                'max_samples': config['max_samples'],
                'overwrite_cache': config['overwrite_cache'],
                'preprocessing_num_workers': config['preprocessing_num_workers'],
                'dataloader_num_workers': config['dataloader_num_workers'],
                'per_device_eval_batch_size': config['per_device_eval_batch_size']
            }, f, default_flow_style=False, allow_unicode=True)

            f.write("\n### output\n")
            yaml.dump({
                'output_dir': config['output_dir'],
                'overwrite_output_dir': config['overwrite_output_dir'],
                'predict_with_generate': config['predict_with_generate']
            }, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"✅ 成功创建配置文件: {yaml_file_path}")
        return yaml_file_path

    except Exception as e:
        logger.info(f"❌ 创建配置文件时出错: {e}")
        return None


def create_train_yaml_config(dataset_dir='../data/dataset',  # '/mnt/d/project-hxd/pprs/data',
                             dataset='alpaca_zh_demo',
                             output_dir='./data/saves/Qwen3-8B/lora/sft',
                             # '/home/dell/LLaMA-Factory/saves/Qwen3-4B/lora/sft',
                             model_name_or_path='../data/llm/model/Qwen3-8B',  # '/home/dell/LLaMA-Factory/models/Qwen3-4B',
                             yaml_file_path='./data/train_lora/train.yaml',
                             train_max_samples=500000,
                             num_train_epochs=3):
    """
    dataset_dir: 存放dataset_info.json文件和数据集的地方
    dataset：dataset_info.json中数据集的key
    output_dir: 存放微调的参数的地方
    model_name_or_path: 存放原始大模型的地方
    yaml_file_path: 存放yaml文件的路径
    : return
    返回 test.yaml配置文件
    """
    # 定义配置数据
    config = {
        # model
        'model_name_or_path': model_name_or_path,
        'trust_remote_code': True,

        # method
        'stage': 'sft',
        'do_train': True,
        'finetuning_type': 'lora',
        'lora_rank': 8,
        'lora_target': 'all',

        # dataset (根据要求修改)
        'dataset_dir': dataset_dir,
        'dataset': dataset,
        'template': 'qwen',
        'cutoff_len': 2048,
        'max_samples': train_max_samples,
        'overwrite_cache': True,
        'preprocessing_num_workers': 16,
        'dataloader_num_workers': 4,

        # output (根据要求修改)
        'output_dir': output_dir,
        'logging_steps': 10,
        'save_steps': 500,
        'plot_loss': True,
        'overwrite_output_dir': True,
        'save_only_model': False,
        'report_to': 'none',

        # train
        'per_device_train_batch_size': 1,
        'gradient_accumulation_steps': 8,
        'learning_rate': 1.0e-4,
        'num_train_epochs': num_train_epochs,
        'lr_scheduler_type': 'cosine',
        'warmup_ratio': 0.1,
        'bf16': True,
        'ddp_timeout': 180000000,
        'resume_from_checkpoint': None,

        # eval (注释掉的部分，这里用None表示或者不包含)
        # 'eval_dataset': dataset,
        # 'val_size': 0.1,
        # 'per_device_eval_batch_size': 1,
        # 'eval_strategy': 'steps',
        # 'eval_steps': 500
    }

    # 确保/data/目录存在
    os.makedirs('./data', exist_ok=True)

    try:
        with open(yaml_file_path, 'w', encoding='utf-8') as f:
            # 添加注释分组
            f.write("### model\n")
            yaml.dump({
                'model_name_or_path': config['model_name_or_path'],
                'trust_remote_code': config['trust_remote_code']
            }, f, default_flow_style=False, allow_unicode=True)

            f.write("\n### method\n")
            yaml.dump({
                'stage': config['stage'],
                'do_train': config['do_train'],
                'finetuning_type': config['finetuning_type'],
                'lora_rank': config['lora_rank'],
                'lora_target': config['lora_target']
            }, f, default_flow_style=False, allow_unicode=True)

            f.write("\n### dataset\n")
            yaml.dump({
                'dataset_dir': config['dataset_dir'],
                'dataset': config['dataset'],
                'template': config['template'],
                'cutoff_len': config['cutoff_len'],
                'max_samples': config['max_samples'],
                'overwrite_cache': config['overwrite_cache'],
                'preprocessing_num_workers': config['preprocessing_num_workers'],
                'dataloader_num_workers': config['dataloader_num_workers']
            }, f, default_flow_style=False, allow_unicode=True)

            f.write("\n### output\n")
            yaml.dump({
                'output_dir': config['output_dir'],
                'logging_steps': config['logging_steps'],
                'save_steps': config['save_steps'],
                'plot_loss': config['plot_loss'],
                'overwrite_output_dir': config['overwrite_output_dir'],
                'save_only_model': config['save_only_model'],
                'report_to': config['report_to']
            }, f, default_flow_style=False, allow_unicode=True)

            f.write("\n# choices: [none, wandb, tensorboard, swanlab, mlflow]\n")

            f.write("\n### train\n")
            yaml.dump({
                'per_device_train_batch_size': config['per_device_train_batch_size'],
                'gradient_accumulation_steps': config['gradient_accumulation_steps'],
                'learning_rate': config['learning_rate'],
                'num_train_epochs': config['num_train_epochs'],
                'lr_scheduler_type': config['lr_scheduler_type'],
                'warmup_ratio': config['warmup_ratio'],
                'bf16': config['bf16'],
                'ddp_timeout': config['ddp_timeout'],
                'resume_from_checkpoint': config['resume_from_checkpoint']
            }, f, default_flow_style=False, allow_unicode=True)

            f.write("\n### eval\n")
            f.write("# eval_dataset: alpaca_en_demo\n")
            f.write("# val_size: 0.1\n")
            f.write("# per_device_eval_batch_size: 1\n")
            f.write("# eval_strategy: steps\n")
            f.write("# eval_steps: 500\n")

        logger.info(f"✅ 成功创建配置文件: {yaml_file_path}")
        return yaml_file_path

    except Exception as e:
        logger.info(f"❌ 创建配置文件时出错: {e}")
        return None


def run_llamafactory_train(config_path):
    """
    使用llamafactory-cli train运行训练
    """
    try:
        # 检查配置文件是否存在
        if not os.path.exists(config_path):
            logger.info(f"❌ 配置文件不存在: {config_path}")
            return False

        # 构建命令
        cmd = ['llamafactory-cli', 'train', config_path]

        logger.info(f"🚀 开始执行训练命令: {' '.join(cmd)}")
        logger.info("=" * 50)

        # 执行命令
        result = subprocess.run(
            cmd,
            capture_output=False,  # 让输出直接显示在终端
            text=True,
            check=False  # 不自动抛出异常，手动检查返回码
        )

        if result.returncode == 0:
            logger.info("=" * 50)
            logger.info("✅ 训练命令执行成功!")
            return True
        else:
            logger.info("=" * 50)
            logger.info(f"❌ 训练命令执行失败，返回码: {result.returncode}")
            return False

    except FileNotFoundError:
        logger.info("❌ 找不到 llamafactory-cli 命令，请确保已正确安装 LLaMA-Factory")
        logger.info("   可以尝试: pip install llamafactory")
        return False
    except Exception as e:
        logger.info(f"❌ 执行训练时出错: {e}")
        return False


def llm_train(dataset, output_dir, yaml_file_path, dataset_dir, model_name_or_path, train_max_samples=500000, num_train_epochs=3):
    """
    主函数
    """
    logger.info("🔧 LLaMA Factory 配置文件生成和训练脚本")
    logger.info("=" * 50)

    # 1. 创建YAML配置文件
    logger.info("📝 步骤1: 创建训练yaml配置文件...")
    config_path = create_train_yaml_config(dataset_dir=dataset_dir,
                                           dataset=dataset,
                                           output_dir=output_dir,
                                           model_name_or_path=model_name_or_path,
                                           yaml_file_path=yaml_file_path,
                                           train_max_samples=train_max_samples,
                                           num_train_epochs=num_train_epochs)  # create_train_yaml_config()

    if config_path is None:
        logger.info("❌ 配置文件创建失败，程序退出")
        sys.exit(1)

    # 显示创建的配置文件内容
    logger.info("\n📋 生成的配置文件内容:")
    logger.info("-" * 30)
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            logger.info(f.read())
    except Exception as e:
        logger.info(f"无法读取配置文件: {e}")

    logger.info("-" * 30)

    # 2. 执行训练
    logger.info("\n🚀 步骤2: 执行训练...")
    success = run_llamafactory_train(config_path)

    if success:
        logger.info("\n🎉 所有步骤完成!")
    else:
        logger.info("\n⚠️  配置文件已创建，但训练执行失败")
        logger.info(f"   您可以手动执行: llamafactory-cli train {config_path}")


def run_llamafactory_predict(config_path):
    """
    使用llamafactory-cli train运行训练
    """
    try:
        # 检查配置文件是否存在
        if not os.path.exists(config_path):
            logger.info(f"❌ 配置文件不存在: {config_path}")
            return False

        # 构建命令
        cmd = ['llamafactory-cli', 'train', config_path]

        logger.info(f"🚀 开始执行推理命令: {' '.join(cmd)}")
        logger.info("=" * 50)

        # 执行命令
        result = subprocess.run(
            cmd,
            capture_output=False,  # 让输出直接显示在终端
            text=True,
            check=False  # 不自动抛出异常，手动检查返回码
        )

        if result.returncode == 0:
            logger.info("=" * 50)
            logger.info("✅ 推理命令执行成功!")
            return True
        else:
            logger.info("=" * 50)
            logger.info(f"❌ 推理命令执行失败，返回码: {result.returncode}")
            return False

    except FileNotFoundError:
        logger.info("❌ 找不到 llamafactory-cli 命令，请确保已正确安装 LLaMA-Factory")
        logger.info("   可以尝试: pip install llamafactory")
        return False
    except Exception as e:
        logger.info(f"❌ 执行推理时出错: {e}")
        return False


def llm_predict(dataset_dir, eval_dataset, output_dir, yaml_file_path, model_name_or_path,
                adapter_name_or_path=None, test_max_samples=100000, per_device_eval_batch_size=16, temperature=0.6):
    """
    主函数
    """
    logger.info("🔧 LLaMA Factory 配置文件生成和推理脚本")
    logger.info("=" * 50)

    # 1. 创建YAML配置文件
    logger.info("📝 步骤1: 创建推理yaml配置文件...")
    config_path = create_inference_yaml_config(dataset_dir=dataset_dir,
                                               eval_dataset=eval_dataset,
                                               output_dir=output_dir,
                                               model_name_or_path=model_name_or_path,
                                               adapter_name_or_path=adapter_name_or_path,
                                               yaml_file_path=yaml_file_path,
                                               test_max_samples=test_max_samples,
                                               per_device_eval_batch_size=per_device_eval_batch_size,
                                               temperature=temperature)

    if config_path is None:
        logger.info("❌ 配置文件创建失败，程序退出")
        sys.exit(1)

    # 显示创建的配置文件内容
    logger.info("\n📋 生成的配置文件内容:")
    logger.info("-" * 30)
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            logger.info(f.read())
    except Exception as e:
        logger.info(f"无法读取配置文件: {e}")

    logger.info("-" * 30)

    # 2. 执行推理
    logger.info("\n🚀 步骤2: 执行推理...")
    success = run_llamafactory_predict(config_path)

    if success:
        logger.info("\n🎉 所有步骤完成!")
    else:
        logger.info("\n⚠️  配置文件已创建，但推理执行失败")
        logger.info(f"   您可以手动执行: llamafactory-cli train {config_path}")


if __name__ == "__main__":
    pass