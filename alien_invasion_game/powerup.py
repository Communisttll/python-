import pygame
from pygame.sprite import Sprite
import random


class PowerUp(Sprite):
    """表示补给品的类"""

    def __init__(self, ai_game):
        """初始化补给品并设置其起始位置"""
        super().__init__()
        self.screen = ai_game.screen
        self.settings = ai_game.settings
        self.stats = ai_game.stats

        # 创建补给品的矩形
        self.rect = pygame.Rect(0, 0, self.settings.powerup_width,
                                self.settings.powerup_height)
        self.screen_rect = self.screen.get_rect()

        # 设置补给品的初始位置
        self.rect.x = random.randint(0, self.screen_rect.width - self.rect.width)
        self.rect.y = 0

        # 存储补给品的精确垂直位置
        self.y = float(self.rect.y)

        # 随机选择补给类型 (0: 增加生命值, 1: 增加子弹数量)
        self.powerup_type = random.randint(0, 1)
        
        # 根据类型设置颜色
        if self.powerup_type == 0:
            self.color = (0, 255, 0)  # 绿色 - 增加生命值
        else:
            self.color = (0, 0, 255)  # 蓝色 - 增加子弹数量
            
        self.speed = self.settings.powerup_speed

    def update(self):
        """向下移动补给品"""
        self.y += self.speed
        self.rect.y = int(self.y)

    def draw_powerup(self):
        """在屏幕上绘制补给品"""
        pygame.draw.rect(self.screen, self.color, self.rect)

    def is_off_screen(self):
        """检查补给品是否超出屏幕底部"""
        return self.rect.top > self.screen_rect.bottom