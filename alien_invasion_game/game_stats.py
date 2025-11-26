import json
import os

class GameStats:
    """跟踪《外星人入侵》的统计信息。"""

    def __init__(self, ai_game):
        """初始化统计信息。"""
        self.settings = ai_game.settings  # 获取游戏设置
        self.reset_stats()  # 重置游戏统计数据

        # 游戏开始时处于非活动状态。
        self.game_active = False  # 游戏是否处于活动状态

        # 最高分 - 如果文件存在则从文件中读取
        self.high_score = self._load_high_score()  # 加载最高分

    def _load_high_score(self):
        """如果文件存在，则从文件中加载最高分。"""
        filepath = os.path.join(os.path.dirname(__file__), 'high_score.json')
        try:
            with open(filepath, 'r') as f:  # 打开存储最高分的文件
                return json.load(f)  # 读取并返回最高分
        except (FileNotFoundError, json.JSONDecodeError):  # 捕获文件未找到或 JSON 解码错误
            return 0  # 如果文件不存在或内容无效，则返回 0

    def save_high_score(self):
        """将最高分保存到文件。"""
        filepath = os.path.join(os.path.dirname(__file__), 'high_score.json')
        with open(filepath, 'w') as f:  # 打开文件以写入最高分
            json.dump(self.high_score, f)  # 将最高分写入文件

    def reset_stats(self):
        """初始化在游戏过程中可能变化的统计信息。"""
        self.ships_left = self.settings.ship_limit  # 剩余飞船数量
        self.score = 0  # 当前得分
        self.level = 1  # 当前等级