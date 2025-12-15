import subprocess
import sys
import os
import re
import json
from typing import Dict, Tuple, Optional, List, Any
import openai
import logging
from logger import init_logger

import llm_api_config
from streaming_llm_handler import stream_openai_response, deepseek_stream_calling
from time_util import TimeTracker, TokenTracker
from llm_api_config import CLAUDE_KEY_API

base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

if not logger.handlers:
    logger = init_logger('../data/log')

class CodeAdjuster:
    def __init__(self, script_path: str, reset_token: bool = True, task_goal: str = None):
        """
        初始化智能Python脚本调试器

        Args:
            script_path: Python脚本路径
            api_key: 大模型API密钥
            api_url: API端点URL
            model: 使用的模型名称
        """
        self.script_path = script_path
        self.max_attempts = 4
        self.attempt_count = 0
        self.reset_token = reset_token

        self.token_recorder = TokenTracker(file_path=f'../data/interim/任务目标({task_goal})',
                                           filename="token_records.json",
                                           module='代码调整',
                                           reset=self.reset_token)

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

    def identify_error_type(self, error_message: str) -> str:
        # logger.info(error_message)
        prompt = f"""作为一个专业的深度学习专家，请根据以下报错内容，判断是环境依赖问题还是代码其他问题，如果是环境依赖问题，则输出import_error，否则输出code_error。\n
                    报错内容：{error_message} 
                    \n\n

                    请以以下json格式返回。\n
                    {{
                    {{
                    "error_type": import_error或code_error，
                    "think": 判断的依据
                    }}
                    """
        api_key_list = llm_api_config.DEEPSEEK_KEY_API

        for api_key in api_key_list:
            result = deepseek_stream_calling(api_key=api_key, prompt=prompt, model="deepseek-chat", token_recorder=self.token_recorder)

            if result is not None and result != "":
                parsed_result = self.parse_api_response(result)
                if parsed_result is not None:
                    logger.info(f"error_type: {parsed_result['error_type']}")
                    logger.info(f"think: {parsed_result['think']}")
                    return parsed_result['error_type']
                else:
                    logger.info('返回内容解析失败，尝试换用API KEY')
                    continue
            else:
                logger.info('调用失败，尝试换用API KEY')
                continue

        if result is None or result == "":
            logger.info('现有API KEY全部调用失败！！！')
            return "code_error"
        return "code_error"



    def extract_missing_modules(self, error_message: str) -> List[str]:
        """从错误信息中提取缺失的模块名"""
        modules = []

        # 匹配 No module named 'xxx'
        matches = re.findall(r"No module named ['\"]([^'\"]+)['\"]", error_message)
        modules.extend(matches)

        # 匹配 cannot import name 'xxx' from 'yyy'
        matches = re.findall(r"cannot import name.*from ['\"]([^'\"]+)['\"]", error_message)
        modules.extend(matches)

        # 去重并返回
        return list(set(modules))

    def run_script(self):
        """
        运行Python脚本，实时显示输出并捕获内容（使用Popen）

        参数:
            script_path: Python脚本路径
            *args: 传递给脚本的参数

        返回:
            (成功/失败, 输出内容)
        """
        cmd = [sys.executable, self.script_path]

        logger.info(f"执行: {' '.join(cmd)}\n{'-' * 50}")

        # 使用Popen实现实时输出
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 合并错误输出到标准输出
            text=True,
            bufsize=1
        )

        output = []
        for line in iter(process.stdout.readline, ''):
            if "warning" not in line.lower():
                print(line, end='')
                output.append(line+'\n')

        logger.info(output)

        process.wait()

        if process.returncode == 0:
            return True, output, ""
        else:
            logger.info(f"❌ 执行失败 (返回码: {process.returncode})")
            return False, output, ""

    def save_script_content(self, content: str) -> None:
        """保存修改后的脚本内容"""
        try:
            # 先备份原文件
            backup_path = f"../data/tmp_code/backup_{self.attempt_count}_{self.script_path}"
            if os.path.exists(self.script_path):
                with open(self.script_path, 'r', encoding='utf-8') as f:
                    backup_content = f.read()
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(backup_content)

            # 保存新内容
            with open(self.script_path, 'w', encoding='utf-8') as f:
                f.write(content)

        except Exception as e:
            raise Exception(f"无法保存脚本文件: {str(e)}")

    def call_claude_llm_api(self, prompt: str) -> Optional[str]:

        openai.api_key = CLAUDE_KEY_API
        openai.base_url = "https://chat.cloudapi.vip/v1/"
        openai.default_headers = {"x-foo": "true"}

        try:
            completion = openai.chat.completions.create(
                model="claude-opus-4-1-20250805",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
            res = completion.choices[0].message.content

            self.token_recorder.add_record(prompt_tokens=completion.usage.prompt_tokens,
                                              completion_tokens=completion.usage.completion_tokens,
                                              total_tokens=completion.usage.total_tokens)



            return res
        except Exception as e:
            logger.info(f"An error occurred: {e}")
            return None


    def call_llm_api_for_fix(self, prompt: str):

        new_code = stream_openai_response(base_url="https://chat.cloudapi.vip/v1/",
                                        api_key=CLAUDE_KEY_API,
                                        prompt=prompt,
                                        model="gpt-5",
                                        timeout=6000,
                                        temperature=0.3,
                                        token_recorder=self.token_recorder)
        return new_code



    def call_llm_for_install_script(self, missing_modules: List[str]) -> Optional[str]:
        """
        调用大模型生成安装脚本
        """
        prompt = f"""针对以下缺失的模块名，生成一个Python脚本来自动安装对应的库和依赖：
            {', '.join(missing_modules)}
            
            要求：
            1. 使用subprocess调用pip安装
            2. 只返回可执行的Python代码，不要有任何解释
            3. 使用"https://pypi.tuna.tsinghua.edu.cn/simple" 作为镜像源
            4. 根据操作系统确定安装命令
            5. 验证是否安装成功
            
            请生成可执行的python脚本,不要返回其他内容。"""

        try:
            res = self.call_claude_llm_api(prompt=prompt)
            return  res

        except Exception as e:
            logger.info(f"调用API时发生错误: {str(e)}")
            return None

    def code_fix(self, error_message: str, script_content: str) -> Optional[str]:
        """
        调用大模型修复代码逻辑错误
        """
        prompt = f"""修复以下Python代码中的错误。

            当前代码：
            ```python
            {script_content}
            ```
            
            错误信息：
            ```
            {error_message}
            ```
            
            \n只返回修复后的完整Python代码，不要包含任何解释或markdown标记。
            \n代码开头和结尾不要出现```python和```。
"""

        try:
            return self.call_llm_api_for_fix(prompt=prompt)

        except Exception as e:
            logger.info(f"调用API时发生错误: {str(e)}")
            return None

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

    def execute_install_script(self, install_script: str) -> bool:
        """执行安装脚本"""
        try:
            # 保存安装脚本到临时文件
            install_script = self.extract_code_blocks(response=install_script)
            temp_script = "temp_install_script.py"
            with open(temp_script, 'w', encoding='utf-8') as f:
                f.write(install_script)

            # 执行安装脚本
            result = subprocess.run(
                [sys.executable, temp_script],
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )

            # 删除临时文件
            if os.path.exists(temp_script):
                os.remove(temp_script)

            if result.returncode == 0:
                logger.info("✅ 依赖安装成功")
                return True
            else:
                logger.info(f"❌ 依赖安装失败: {result.stderr}")
                return False

        except Exception as e:
            logger.info(f"执行安装脚本时出错: {str(e)}")
            return False

    def debug_loop(self) -> Dict[str, str]:
        """
        主调试循环

        Returns:
            {
                "修改类型": "安装环境包" | "修改运行脚本" | "成功",
                "脚本": "对应的脚本内容"
            }
        """
        logger.info(f"开始调试脚本: {self.script_path}")

        while self.attempt_count < self.max_attempts:
            self.attempt_count += 1
            logger.info(f"\n尝试 #{self.attempt_count}")

            # 运行脚本
            success, stdout, stderr = self.run_script()

            if success:
                logger.info(f"✅ 脚本成功运行！")
                if stdout:
                    # logger.info(f"输出:\n{stdout}")
                    pass
                return {
                    "修改类型": "成功",
                    "脚本": self.read_script_content()
                }

            # 脚本运行失败
            logger.info(f"❌ 脚本运行失败")
            error_message = f"{{\nstderr:{stderr};\nstdout:{stdout}\n}}" if stderr or stdout else "未知错误"
            print(f"错误信息:\n{error_message}")

            # 识别错误类型
            error_type = self.identify_error_type(error_message)
            logger.info(f"错误类型: {error_type}")

            if error_type == "import_error":
                # 处理导入错误
                missing_modules = self.extract_missing_modules(error_message)
                if missing_modules:
                    logger.info(f"缺失的模块: {missing_modules}")

                    # 生成安装脚本
                    install_script = self.call_llm_for_install_script(missing_modules)
                    if install_script:
                        logger.info("已生成安装脚本")

                        # 执行安装脚本
                        if self.execute_install_script(install_script):
                            # 安装成功后继续尝试运行原脚本
                            continue
                        else:
                            logger.info("Failed to install environment package!")
                            # 重新运行
                            continue
                            # return {
                            #     "修改类型": "错误",
                            #     "脚本": "执行安装依赖脚本失败"
                            # }
                    else:
                        logger.info("No executable scripts!")
                        return {
                            "修改类型": "错误",
                            "脚本": "无可执行的依赖安装脚本"
                        }

            elif error_type == "code_error":  # code_error
                # 处理代码逻辑错误
                try:
                    script_content = self.read_script_content()
                except Exception as e:
                    return {
                        "修改类型": "错误",
                        "脚本": f"无法读取脚本: {str(e)}"
                    }

                # 调用大模型修复代码
                logger.info("正在调用大模型修复代码...")
                fixed_code = self.code_fix(error_message, script_content)

                if fixed_code:
                    # 保存修复后的代码
                    try:
                        self.save_script_content(fixed_code)
                        logger.info("已保存修复后的代码")
                    except Exception as e:
                        return {
                            "修改类型": "修改运行脚本",
                            "脚本": fixed_code
                        }
                else:
                    logger.info("无法获取修复方案")
                    return {
                        "修改类型": "错误",
                        "脚本": "无法获取修复方案"
                    }

        # 达到最大尝试次数
        return {
            "修改类型": "失败",
            "脚本": f"已达到最大尝试次数 ({self.max_attempts})，调试失败"
        }
