import pygame
from pygame.sprite import Sprite
import os


class Alien(Sprite):
    """表示舰队中单个外星人的类。"""

    def __init__(self, ai_game):
        """初始化外星人并设置其起始位置。"""
        super().__init__()
        self.screen = ai_game.screen  # 获取游戏屏幕对象
        self.settings = ai_game.settings  # 获取游戏设置

        # 加载外星人图像并设置其 rect 属性。
        self.image = pygame.image.load(os.path.join(os.path.dirname(__file__), 'images', 'alien.bmp'))  # 加载外星人图像
        self.rect = self.image.get_rect()  # 获取外星人图像的矩形区域

        # 每个新外星人最初都在屏幕左上角附近。
        self.rect.x = self.rect.width  # 设置外星人的 x 坐标
        self.rect.y = self.rect.height  # 设置外星人的 y 坐标

        # 存储外星人的精确水平位置。
        self.x = float(self.rect.x)  # 将外星人的 x 坐标存储为浮点数

    def check_edges(self):
        """如果外星人位于屏幕边缘，则返回 True。"""
        screen_rect = self.screen.get_rect()  # 获取屏幕的矩形区域
        return (self.rect.right >= screen_rect.right) or (self.rect.left <= 0)  # 判断外星人是否碰到屏幕左右边缘

    def update(self):
        """向左或向右移动外星人。"""
        self.x += (self.settings.alien_speed * self.settings.fleet_direction)  # 根据外星人速度和舰队方向更新外星人的水平位置
        self.rect.x = self.x  # 更新外星人矩形的 x 坐标