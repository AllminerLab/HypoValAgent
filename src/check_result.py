import ast
import json
import os
import re
from typing import List

from llm_api_config import CLAUDE_KEY_API
from streaming_llm_handler import stream_openai_response

from json_repair import repair_json
import json
import re

def check_features_openai(code: str, feature_list: List[str], token_recorder=None):
    """
    使用 OpenAI API 检查代码中的特征引用

    Args:
        code: Python代码字符串
        feature_list: 允许的特征列表

    Returns:
        1 if all features are in the list, 0 otherwise
    """
    # try:
    prompt = f"""请分析以下Python代码，找出所有被引用的特征变量名。
特征通常是作为数据框的列名、字典的键、或者变量名出现。

Python代码：
```python
{code}
```

允许的特征列表：{feature_list}

请以JSON格式返回结果：
\n\n
{{
    "referenced_features": "[列出代码中引用的所有特征]",
    "missing_features": "[列出不在允许列表中的特征]",
    "all_features_valid": "true/false"
}}
\n\n
只返回JSON格式结果，不要返回其他内容！
"""

    result = stream_openai_response(base_url=None,
                                     api_key=CLAUDE_KEY_API,
                                     prompt=prompt,
                                     model="claude-opus-4-1-20250805",
                                     timeout=6000,
                                     temperature=0.2,
                                    response_format={"type": "json_object"},
                                    token_recorder=token_recorder)

    # result_text = response.choices[0].message.content
    cleaned_text = result.strip()
    res = parse_llm_json_safe(cleaned_text)
    try:
        if res.get("all_features_valid"):
            if len(res.get("referenced_features")) > 0:
                return  True, None
            else:
                return False, None
        else:
            return False, res.get('missing_features')
    except Exception as e:
        print(e)
        return False, None



def parse_llm_json_safe(response: str):
    """使用 json-repair 库的安全解析"""
    try:
        # 清理 markdown
        cleaned = re.sub(r'```(?:json)?\s*', '', response)
        cleaned = re.sub(r'```\s*', '', cleaned)

        # 提取 JSON
        match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', cleaned)
        if match:
            cleaned = match.group(0)

        # 使用 json-repair 自动修复并解析
        repaired = repair_json(cleaned)
        return json.loads(repaired)

    except Exception as e:
        print(f"解析失败: {e}")
        return None