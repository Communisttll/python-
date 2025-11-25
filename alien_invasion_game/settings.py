class Settings:
    """一个存储《外星人入侵》所有设置的类。"""

    def __init__(self):
        """初始化游戏的静态设置。"""
        # 屏幕设置
        self.screen_width = 1200  # 屏幕宽度
        self.screen_height = 800  # 屏幕高度
        self.bg_color = (230, 230, 230)  # 背景颜色 (R, G, B)

        # 飞船设置
        self.ship_limit = 3  # 飞船生命值

        # Bullet settings
        self.bullet_speed = 2.5  # 子弹速度
        self.bullet_width = 3  # 子弹宽度
        self.bullet_height = 15  # 子弹高度
        self.bullet_color = (60, 60, 60)  # 子弹颜色
        self.bullets_allowed = 3  # 允许存在的子弹数量

        # Alien bullet settings
        self.alien_bullet_speed = 1.5  # 外星人子弹速度
        self.alien_bullet_width = 3  # 外星人子弹宽度
        self.alien_bullet_height = 15  # 外星人子弹高度
        self.alien_bullet_color = (255, 0, 0)  # 外星人子弹颜色 (红色)
        self.alien_bullets_allowed = 5  # 允许存在的外星人子弹数量
        self.alien_shoot_probability = 0.001  # 每帧外星人射击的概率

        # Alien settings
        self.alien_speed = 1.0  # 外星人速度
        self.fleet_drop_speed = 10  # 外星人群向下移动的速度
        # fleet_direction 为 1 表示向右移动；-1 表示向左移动。
        self.fleet_direction = 1  # 外星人群移动方向

        # Scoring settings
        self.alien_points = 50  # 击落一个外星人的得分

        # 游戏加速的程度
        self.speedup_scale = 1.1  # 游戏速度加快的比例
        # 外星人分数增加的程度
        self.score_scale = 1.5  # 外星人分数增加的比例

        self.initialize_dynamic_settings()

    def initialize_dynamic_settings(self):
        """初始化在游戏过程中会变化的设置。"""
        self.ship_speed = 1.5  # 飞船速度
        self.bullet_speed = 2.5  # 子弹速度
        self.alien_speed = 1.0  # 外星人速度

    def increase_speed(self):
        """提高速度设置和外星人分数。"""
        self.ship_speed *= self.speedup_scale  # 提高飞船速度
        self.bullet_speed *= self.speedup_scale  # 提高子弹速度
        self.alien_speed *= self.speedup_scale  # 提高外星人速度

        self.alien_points = int(self.alien_points * self.score_scale)  # 提高外星人分数