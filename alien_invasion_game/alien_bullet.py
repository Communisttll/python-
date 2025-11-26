import pygame
from pygame.sprite import Sprite
import os


class AlienBullet(Sprite):
    """一个管理外星人发射子弹的类。"""

    def __init__(self, ai_game, alien):
        """在外星人当前位置创建一个子弹对象。"""
        super().__init__()
        self.screen = ai_game.screen  # 获取游戏屏幕对象
        self.settings = ai_game.settings  # 获取游戏设置
        self.color = self.settings.alien_bullet_color  # 设置子弹颜色

        # 在 (0, 0) 处创建一个子弹矩形，然后设置正确的位置。
        self.rect = pygame.Rect(0, 0, self.settings.alien_bullet_width,
            self.settings.alien_bullet_height)  # 创建子弹矩形
        self.rect.centerx = alien.rect.centerx  # 将子弹的中心 x 坐标设置为外星人的中心 x 坐标
        self.rect.bottom = alien.rect.bottom  # 将子弹的底部 y 坐标设置为外星人的底部 y 坐标

        # 存储子弹的精确位置为浮点数。
        self.y = float(self.rect.y)  # 将子弹的 y 坐标存储为浮点数

    def update(self):
        """向下移动子弹。"""
        # 更新子弹的精确位置。
        self.y += self.settings.alien_bullet_speed  # 根据外星人子弹速度更新子弹的 y 坐标
        # 更新 rect 的位置。
        self.rect.y = self.y  # 更新子弹矩形的 y 坐标

    def draw_bullet(self):
        """在屏幕上绘制子弹。"""
        pygame.draw.rect(self.screen, self.color, self.rect)  # 绘制子弹的矩形