import json
import os
from datetime import datetime
from typing import Dict, List, Union, Optional
import uuid


class TimeTracker:
    """
    时间工具类 - 用于记录和管理时间跟踪数据

    特性：
    - 支持增量读取，多次初始化不会丢失历史数据
    - 在指定路径创建和管理JSON文件
    - 添加时间记录（自动计算耗时）
    - 读取和查询时间记录
    - 支持反复初始化和使用
    """

    def __init__(self, file_path: str, filename: str = "time_records.json"):
        """
        初始化时间跟踪器（支持增量加载）

        Args:
            file_path (str): JSON文件存储路径
            filename (str): JSON文件名，默认为"time_records.json"
        """
        self.file_path = file_path
        self.filename = filename
        self.full_path = os.path.join(file_path, filename)

        # 确保目录存在
        os.makedirs(file_path, exist_ok=True)

        # 初始化或加载JSON文件（增量模式）
        self._initialize_or_load_json_file()

        print(f"时间跟踪器已初始化: {self.full_path}")

    def _initialize_or_load_json_file(self) -> None:
        """初始化或加载JSON文件（增量模式，保留已有数据）"""
        if not os.path.exists(self.full_path):
            # 文件不存在，创建新文件
            initial_data = {}
            self._save_json_data(initial_data)
            print(f"已创建新的时间记录文件: {self.full_path}")
        else:
            data = self._read_json_data()
            # 更新最后访问时间
            print(f"已加载现有时间记录文件: {self.full_path}")

    def _parse_time(self, time_input: Union[str, datetime]) -> datetime:
        """
        解析时间输入，支持多种格式

        Args:
            time_input: 时间输入，可以是字符串或datetime对象

        Returns:
            datetime: 解析后的datetime对象
        """
        if isinstance(time_input, datetime):
            return time_input

        # 支持的时间格式
        time_formats = [
            "%Y-%m-%d %H:%M:%S",  # 2024-01-01 10:30:00
            "%Y-%m-%d %H:%M",  # 2024-01-01 10:30
            "%m-%d %H:%M",  # 01-01 10:30
            "%H:%M:%S",  # 10:30:00
            "%H:%M"  # 10:30
        ]

        for fmt in time_formats:
            try:
                if fmt in ["%H:%M:%S", "%H:%M"]:
                    # 对于只有时间的格式，使用今天的日期
                    time_obj = datetime.strptime(time_input, fmt)
                    return datetime.now().replace(
                        hour=time_obj.hour,
                        minute=time_obj.minute,
                        second=time_obj.second if fmt == "%H:%M:%S" else 0,
                        microsecond=0
                    )
                elif fmt == "%m-%d %H:%M":
                    # 对于没有年份的格式，使用当前年份
                    time_obj = datetime.strptime(f"{datetime.now().year}-{time_input}", "%Y-%m-%d %H:%M")
                    return time_obj
                else:
                    return datetime.strptime(time_input, fmt)
            except ValueError:
                continue

        raise ValueError(f"无法解析时间格式: {time_input}")


    def add_record(self, start_time,
                   end_time,
                   record_name: str,
                   time_reset: bool=True) -> Dict:
        """
        添加时间记录（增量添加，不影响现有记录）

        Args:
            start_time: 开始时间
            end_time: 结束时间
            record_name: 记录名称
        Returns:
            Dict: 添加的记录信息
        """
        try:
            # 解析时间


            # 验证时间逻辑
            if end_time <= start_time:
                raise ValueError("结束时间必须晚于开始时间")

            # 计算耗时（分钟）
            duration_minutes = round((end_time - start_time)/ 60, 4)

            # 读取最新数据（确保获取到其他实例可能添加的记录）
            data = self._read_json_data()

            if time_reset:
                data[record_name] = duration_minutes
            else:
                data[record_name] += duration_minutes
            # 保存数据
            self._save_json_data(data)

            print(f"✓ 成功添加记录: {record_name} (耗时: {duration_minutes}分钟)")

        except Exception as e:
            print(f"✗ 添加记录失败: {str(e)}")
            raise

    def _read_json_data(self) -> Dict:
        """读取JSON文件数据（每次都从文件读取最新数据）"""
        try:
            with open(self.full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # 如果文件被删除，重新初始化
            print("警告: JSON文件被删除，正在重新创建...")
            self._initialize_or_load_json_file()
            with open(self.full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"读取JSON文件失败: {str(e)}")
            raise

    def _save_json_data(self, data: Dict) -> None:
        """保存数据到JSON文件"""
        try:
            # 确保目录存在
            os.makedirs(self.file_path, exist_ok=True)

            # 原子性写入（先写临时文件，再重命名）
            temp_path = f"{self.full_path}.tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # 原子性替换
            os.replace(temp_path, self.full_path)

        except Exception as e:
            # 清理临时文件
            temp_path = f"{self.full_path}.tmp"
            if os.path.exists(temp_path):
                os.remove(temp_path)
            print(f"保存JSON文件失败: {str(e)}")
            raise



class ThinkTracker:
    """
    时间工具类 - 用于记录和管理时间跟踪数据

    特性：
    - 支持增量读取，多次初始化不会丢失历史数据
    - 在指定路径创建和管理JSON文件
    - 添加时间记录（自动计算耗时）
    - 读取和查询时间记录
    - 支持反复初始化和使用
    """

    def __init__(self, file_path: str, filename: str = "think_records.json"):
        """
        初始化时间跟踪器（支持增量加载）

        Args:
            file_path (str): JSON文件存储路径
            filename (str): JSON文件名，默认为"think_records.json"
        """
        self.file_path = file_path
        self.filename = filename
        self.full_path = os.path.join(file_path, filename)

        # 确保目录存在
        os.makedirs(file_path, exist_ok=True)

        # 初始化或加载JSON文件（增量模式）
        self._initialize_or_load_json_file()

        print(f"时间跟踪器已初始化: {self.full_path}")

    def _initialize_or_load_json_file(self) -> None:
        """初始化或加载JSON文件（增量模式，保留已有数据）"""
        if not os.path.exists(self.full_path):
            # 文件不存在，创建新文件
            initial_data = {}
            self._save_json_data(initial_data)
            print(f"已创建新的思考记录文件: {self.full_path}")
        else:
            data = self._read_json_data()
            # 更新最后访问时间
            print(f"已加载现有思考记录文件: {self.full_path}")

    def _parse_time(self, time_input: Union[str, datetime]) -> datetime:
        """
        解析时间输入，支持多种格式

        Args:
            time_input: 时间输入，可以是字符串或datetime对象

        Returns:
            datetime: 解析后的datetime对象
        """
        if isinstance(time_input, datetime):
            return time_input

        # 支持的时间格式
        time_formats = [
            "%Y-%m-%d %H:%M:%S",  # 2024-01-01 10:30:00
            "%Y-%m-%d %H:%M",  # 2024-01-01 10:30
            "%m-%d %H:%M",  # 01-01 10:30
            "%H:%M:%S",  # 10:30:00
            "%H:%M"  # 10:30
        ]

        for fmt in time_formats:
            try:
                if fmt in ["%H:%M:%S", "%H:%M"]:
                    # 对于只有时间的格式，使用今天的日期
                    time_obj = datetime.strptime(time_input, fmt)
                    return datetime.now().replace(
                        hour=time_obj.hour,
                        minute=time_obj.minute,
                        second=time_obj.second if fmt == "%H:%M:%S" else 0,
                        microsecond=0
                    )
                elif fmt == "%m-%d %H:%M":
                    # 对于没有年份的格式，使用当前年份
                    time_obj = datetime.strptime(f"{datetime.now().year}-{time_input}", "%Y-%m-%d %H:%M")
                    return time_obj
                else:
                    return datetime.strptime(time_input, fmt)
            except ValueError:
                continue

        raise ValueError(f"无法解析时间格式: {time_input}")


    def add_record(self,
                   module: str,
                   think,
                   sub_module=None) -> Dict:
        """
        添加时间记录（增量添加，不影响现有记录）
        """
        try:
            # 解析时间


            # 验证时间逻辑

            # 读取最新数据（确保获取到其他实例可能添加的记录）
            data = self._read_json_data()
            if sub_module is None:
                data[module] = think

            elif data.get(module, -1) == -1 and sub_module is not None:
                data[module] = {}
                data[module][sub_module] = think

            elif data.get(module, -1) != -1 and sub_module is not None:

                if isinstance(data[module], dict):
                    data[module][sub_module] = think
                else:
                    data[module] = {}
                    data[module][sub_module] = think

            # 保存数据
            self._save_json_data(data)

            print(f"✓ 成功添加 {module}({sub_module})的思考：{think} ")

        except Exception as e:
            print(f"✗ 添加记录失败: {str(e)}")
            raise

    def _read_json_data(self) -> Dict:
        """读取JSON文件数据（每次都从文件读取最新数据）"""
        try:
            with open(self.full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # 如果文件被删除，重新初始化
            print("警告: JSON文件被删除，正在重新创建...")
            self._initialize_or_load_json_file()
            with open(self.full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"读取JSON文件失败: {str(e)}")
            raise

    def _save_json_data(self, data: Dict) -> None:
        """保存数据到JSON文件"""
        try:
            # 确保目录存在
            os.makedirs(self.file_path, exist_ok=True)

            # 原子性写入（先写临时文件，再重命名）
            temp_path = f"{self.full_path}.tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # 原子性替换
            os.replace(temp_path, self.full_path)

        except Exception as e:
            # 清理临时文件
            temp_path = f"{self.full_path}.tmp"
            if os.path.exists(temp_path):
                os.remove(temp_path)
            print(f"保存JSON文件失败: {str(e)}")
            raise


class TokenTracker:
    """
    时间工具类 - 用于记录和管理时间跟踪数据

    特性：
    - 支持增量读取，多次初始化不会丢失历史数据
    - 在指定路径创建和管理JSON文件
    - 添加时间记录（自动计算耗时）
    - 读取和查询时间记录
    - 支持反复初始化和使用
    """

    def __init__(self, file_path: str, filename: str = "think_records.json", module=None, reset=True):
        """
        初始化时间跟踪器（支持增量加载）

        Args:
            file_path (str): JSON文件存储路径
            filename (str): JSON文件名，默认为"think_records.json"
        """
        self.file_path = file_path
        self.filename = filename
        self.full_path = os.path.join(file_path, filename)
        self.module = module
        self.reset = reset

        # 确保目录存在
        os.makedirs(file_path, exist_ok=True)

        # 初始化或加载JSON文件（增量模式）
        self._initialize_or_load_json_file()

        print(f"Token跟踪器已初始化: {self.full_path}")

    def _initialize_or_load_json_file(self) -> None:
        """初始化或加载JSON文件（增量模式，保留已有数据）"""
        if not os.path.exists(self.full_path):
            # 文件不存在，创建新文件
            initial_data = {self.module: {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}}
            self._save_json_data(initial_data)
            print(f"已创建新的token记录文件: {self.full_path}")
            print(f"当前 {self.module}的token数 prompt_tokens: {initial_data[self.module]['prompt_tokens']}, "
                  f"completion_tokens: {initial_data[self.module]['completion_tokens']}, "
                  f"total_tokens: {initial_data[self.module]['total_tokens']}")
        else:
            data = self._read_json_data()
            if self.reset:
                data[self.module] = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
            self._save_json_data(data)
            # 更新最后访问时间
            print(f"已加载现有token记录文件: {self.full_path}")
            print(f"加载后 {self.module}的token数 prompt_tokens: {data[self.module]['prompt_tokens']}, "
                  f"completion_tokens: {data[self.module]['completion_tokens']}, "
                  f"total_tokens: {data[self.module]['total_tokens']}")


    def _parse_time(self, time_input: Union[str, datetime]) -> datetime:
        """
        解析时间输入，支持多种格式

        Args:
            time_input: 时间输入，可以是字符串或datetime对象

        Returns:
            datetime: 解析后的datetime对象
        """
        if isinstance(time_input, datetime):
            return time_input

        # 支持的时间格式
        time_formats = [
            "%Y-%m-%d %H:%M:%S",  # 2024-01-01 10:30:00
            "%Y-%m-%d %H:%M",  # 2024-01-01 10:30
            "%m-%d %H:%M",  # 01-01 10:30
            "%H:%M:%S",  # 10:30:00
            "%H:%M"  # 10:30
        ]

        for fmt in time_formats:
            try:
                if fmt in ["%H:%M:%S", "%H:%M"]:
                    # 对于只有时间的格式，使用今天的日期
                    time_obj = datetime.strptime(time_input, fmt)
                    return datetime.now().replace(
                        hour=time_obj.hour,
                        minute=time_obj.minute,
                        second=time_obj.second if fmt == "%H:%M:%S" else 0,
                        microsecond=0
                    )
                elif fmt == "%m-%d %H:%M":
                    # 对于没有年份的格式，使用当前年份
                    time_obj = datetime.strptime(f"{datetime.now().year}-{time_input}", "%Y-%m-%d %H:%M")
                    return time_obj
                else:
                    return datetime.strptime(time_input, fmt)
            except ValueError:
                continue

        raise ValueError(f"无法解析时间格式: {time_input}")


    def add_record(self,
                   module: str=None,
                   prompt_tokens=0,
                   completion_tokens=0,
                   total_tokens=0) -> Dict:
        """
        添加时间记录（增量添加，不影响现有记录）
        """
        try:
            # 解析时间


            # 验证时间逻辑

            # 读取最新数据（确保获取到其他实例可能添加的记录）
            data = self._read_json_data()

            if module is None:
                module = self.module

            if data.get(module, -1) == -1:
                data[module] = {}

            if data[module].get('prompt_tokens', -1) == -1:
                data[module]['prompt_tokens'] = 0

            if data[module].get('completion_tokens', -1) == -1:
                data[module]['completion_tokens'] = 0

            if data[module].get('total_tokens', -1) == -1:
                data[module]['total_tokens'] = 0

            data[module]['prompt_tokens'] = data[module]['prompt_tokens'] + prompt_tokens
            data[module]['completion_tokens'] = data[module]['completion_tokens'] + completion_tokens
            data[module]['total_tokens'] = data[module]['total_tokens'] + total_tokens

            # 保存数据
            self._save_json_data(data)

            print(f"\n✓ 成功添加 {module}的token数 / 截止当前token数:  prompt_tokens: {prompt_tokens}/{data[module]['prompt_tokens']}, "
                  f"completion_tokens: {completion_tokens}/{data[module]['completion_tokens']}, "
                  f"total_tokens: total_tokens: {total_tokens}/{data[module]['total_tokens']}")

        except Exception as e:
            print(f"✗ 添加记录失败: {str(e)}")
            raise

    def _read_json_data(self) -> Dict:
        """读取JSON文件数据（每次都从文件读取最新数据）"""
        try:
            with open(self.full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # 如果文件被删除，重新初始化
            print("警告: JSON文件被删除，正在重新创建...")
            self._initialize_or_load_json_file()
            with open(self.full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"读取JSON文件失败: {str(e)}")
            raise

    def _save_json_data(self, data: Dict) -> None:
        """保存数据到JSON文件"""
        try:
            # 确保目录存在
            os.makedirs(self.file_path, exist_ok=True)

            # 原子性写入（先写临时文件，再重命名）
            temp_path = f"{self.full_path}.tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # 原子性替换
            os.replace(temp_path, self.full_path)

        except Exception as e:
            # 清理临时文件
            temp_path = f"{self.full_path}.tmp"
            if os.path.exists(temp_path):
                os.remove(temp_path)
            print(f"保存JSON文件失败: {str(e)}")
            raise


