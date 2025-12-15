import json
import os
from typing import Optional, Generator, Dict, Any

import openai
import requests
from anthropic import Anthropic
from openai import OpenAI
import sys
import logging
from logger import init_logger
from llm_api_config import CLAUDE_KEY_API
import tiktoken
base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
if not logger.handlers:
    logger = init_logger('../data/log')

def count_tokens(text: str, model: str) -> int:
    """
    统计文本的token数
    """
    try:
        # 根据模型选择合适的编码器
        if "gpt" in model.lower():
            # GPT模型使用cl100k_base编码
            encoding = tiktoken.get_encoding("cl100k_base")
        elif "claude" in model.lower():
            # Claude模型的token计算方式
            # 注意：Claude的实际token计算可能不同，这里使用近似方法
            encoding = tiktoken.get_encoding("cl100k_base")
        else:
            # 默认编码器
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))
    except Exception as e:
        # 如果无法准确计算，使用粗略估算（1 token ≈ 4字符）
        return len(text) // 4


def stream_openai_response(base_url=None, api_key=None, model=None, prompt=None, temperature=0.7, timeout=300, maxtoken=8096, response_format=None, token_recorder=None,**kwargs):
    """
    流式接收OpenAI响应，解决返回结果较多的问题

    参数:
        api_key: OpenAI API密钥（可选，默认从环境变量读取）
        model: 使用的模型，默认 gpt-3.5-turbo
        **kwargs: 其他参数如 temperature, max_tokens 等

    返回:
        完整的响应文本
    """
    try:
        client = Anthropic(api_key=CLAUDE_KEY_API, base_url="https://chat.cloudapi.vip")
        full_response = ''
        with client.messages.stream(
                model=model,
                max_tokens=maxtoken,
                temperature=temperature,
                timeout=timeout,
                messages=[{"role": "user", "content": prompt}],

        ) as stream:
            for text in stream.text_stream:
                full_response += text
                print(text, end="", flush=True)  # 实时打印

            # 获取最终的usage信息
            final_message = stream.get_final_message()
            prompt_tokens = final_message.usage.input_tokens
            completion_tokens = final_message.usage.output_tokens

            token_recorder.add_record(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens)

            return full_response


    except Exception as e:
        print(f"错误: {e}")
        return None


    except Exception as e:
        print(f"错误: {e}")
        return None



class DeepSeekClient:
    def __init__(self, api_key: Optional[str] = None, model:str="deepseek-chat", base_url: str = "https://api.deepseek.com"):
        """
        初始化DeepSeek客户端

        Args:
            api_key: DeepSeek API密钥，如果不提供则从环境变量DEEPSEEK_API_KEY读取
            base_url: API基础URL
        """
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError(
                "API key is required. Please set DEEPSEEK_API_KEY environment variable or pass api_key parameter.")

        self.base_url = base_url.rstrip('/')
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        self.model = model


    def chat_stream(self,
                    messages: list,
                    model: str = "deepseek-chat",
                    temperature: float = 1.0,
                    max_tokens: Optional[int] = None,
                    **kwargs) -> Generator[Dict[str, Any], None, None]:
        """
        流式聊天接口

        Args:
            messages: 消息列表，格式: [{"role": "user", "content": "Hello"}]
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大令牌数
            **kwargs: 其他参数

        Yields:
            Dict: 流式响应数据
        """
        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            **kwargs
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        # 1022 todo
        # 请求在流中包含用量，兼容OpenAI风格；服务端不支持时会被忽略
        payload["stream_options"] = {"include_usage": True}

        try:
            with requests.post(url, headers=self.headers, json=payload, stream=True) as response:
                response.raise_for_status()

                for line in response.iter_lines(decode_unicode=True):
                    if line.strip():
                        if line.startswith('data: '):
                            data = line[6:]  # 移除 'data: ' 前缀

                            if data.strip() == '[DONE]':
                                break

                            try:
                                chunk = json.loads(data)
                                yield chunk
                            except json.JSONDecodeError:
                                continue

        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {e}")

    def chat_stream_text(self,
                         messages: list,
                         model: str = "deepseek-chat",
                         temperature: float = 1.0,
                         max_tokens: Optional[int] = None,
                         **kwargs) -> Generator[str, None, None]:
        """
        流式聊天接口，只返回文本内容

        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大令牌数
            **kwargs: 其他参数

        Yields:
            str: 流式文本内容
        """
        for chunk in self.chat_stream(messages, model, temperature, max_tokens, **kwargs):
            if 'choices' in chunk and chunk['choices']:
                delta = chunk['choices'][0].get('delta', {})
                if 'content' in delta and delta['content']:
                    yield delta['content']


def deepseek_stream_calling(api_key, prompt, model="deepseek-chat", temperature=0.2, max_tokens=8096, token_recorder=None, **kwargs):
    """高级流式输出示例，展示完整的响应数据"""
    client = DeepSeekClient(api_key=api_key)

    messages = [
        {"role": "system", "content": "你是一个专业的医疗需求分析助手。"},
        {"role": "user", "content": prompt}
    ]
    # print(prompt)

    # print("完整流式响应数据:")
    # print("-" * 50)

    try:
        full_content = ""
        for chunk in client.chat_stream(messages, model=model, temperature=temperature, max_tokens=max_tokens):
            # 打印完整的chunk数据（调试用）
            # print(f"Chunk: {json.dumps(chunk, ensure_ascii=False, indent=2)}")

            if 'choices' in chunk and chunk['choices']:
                choice = chunk['choices'][0]

                if 'delta' in choice and 'content' in choice['delta']:
                    content = choice['delta']['content']
                    full_content += content
                    print(content, end='', flush=True)

                # 检查是否完成
                if choice.get('finish_reason') :
                    if choice['finish_reason'] == 'stop':
                        print(f"\n\n调用成功")

            # 捕获用量（不同实现格式可能不同，这里做了多种兼容）
            # 1) 顶层 usage 字段
            if 'usage' in chunk and isinstance(chunk['usage'], dict):
                token_recorder.add_record(prompt_tokens=chunk['usage'].get('prompt_tokens'),
                            completion_tokens=chunk['usage'].get('completion_tokens'),
                            total_tokens=chunk['usage'].get('total_tokens'))

            # 2) 一些兼容OpenAI的新流式格式会有 type 字段
            #    如 type == 'response.completed' 时附带 usage
            if chunk.get('type') in ('response.completed', 'response.summary'):
                if 'usage' in chunk and isinstance(chunk['usage'], dict):
                    token_recorder.add_record(prompt_tokens=chunk['usage'].get('prompt_tokens'),
                                              completion_tokens=chunk['usage'].get('completion_tokens'),
                                              total_tokens=chunk['usage'].get('total_tokens'))

        # print(f"\n\n完整内容:\n{full_content}")
        return full_content

    except Exception as e:
        return None

