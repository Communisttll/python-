import pygame
from pygame.sprite import Sprite

class Bullet(Sprite):
    """一个管理飞船发射子弹的类。"""

    def __init__(self, ai_game):
        """在飞船的当前位置创建一个子弹对象。"""
        super().__init__()
        self.screen = ai_game.screen  # 获取游戏屏幕对象
        self.settings = ai_game.settings  # 获取游戏设置
        self.color = self.settings.bullet_color  # 设置子弹颜色

        # 在 (0, 0) 处创建一个子弹矩形，然后设置正确的位置。
        self.rect = pygame.Rect(0, 0, self.settings.bullet_width,
            self.settings.bullet_height)  # 创建子弹矩形
        self.rect.midtop = ai_game.ship.rect.midtop  # 将子弹的 midtop 设置为飞船的 midtop

        # 存储子弹的精确位置为浮点数。
        self.y = float(self.rect.y)  # 将子弹的 y 坐标存储为浮点数

    def update(self):
        """向上移动子弹。"""
        # 更新子弹的精确位置。
        self.y -= self.settings.bullet_speed  # 根据子弹速度更新子弹的 y 坐标
        # 更新 rect 的位置。
        self.rect.y = self.y  # 更新子弹矩形的 y 坐标

    def draw_bullet(self):
        """在屏幕上绘制子弹。"""
        pygame.draw.rect(self.screen, self.color, self.rect)  # 绘制子弹的矩形