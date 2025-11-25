import pygame
from pygame.sprite import Sprite
import os

class Ship(Sprite):
    """一个管理飞船的类。"""

    def __init__(self, ai_game):
        """初始化飞船并设置其起始位置。"""
        super().__init__()
        self.screen = ai_game.screen  # 获取游戏屏幕对象
        self.settings = ai_game.settings  # 获取游戏设置
        self.screen_rect = ai_game.screen.get_rect()  # 获取屏幕的矩形区域

        # 加载飞船图像并获取其矩形。
        self.image = pygame.image.load(os.path.join(os.path.dirname(__file__), 'images', 'ship.bmp'))  # 加载飞船图像
        self.rect = self.image.get_rect()  # 获取飞船图像的矩形区域

        # 创建护盾矩形（略大于飞船）
        self.shield_rect = pygame.Rect(0, 0, self.rect.width + 20, self.rect.height + 10)  # 创建护盾矩形
        self.shield_color = (0, 255, 255)  # 护盾的青色
        self.shield_active = True  # 护盾是否激活的标志

        # 将每艘新飞船放置在屏幕底部中央。
        self.rect.midbottom = self.screen_rect.midbottom  # 设置飞船的 midbottom 位置
        self.shield_rect.centerx = self.rect.centerx  # 设置护盾的中心 x 坐标
        self.shield_rect.centery = self.rect.centery  # 设置护盾的中心 y 坐标

        # 存储飞船精确水平位置的浮点数。
        self.x = float(self.rect.x)  # 将飞船的 x 坐标存储为浮点数

        # 移动标志；开始时飞船不移动。
        self.moving_right = False  # 飞船向右移动的标志
        self.moving_left = False  # 飞船向左移动的标志

    def center_ship(self):
        """将飞船放置在屏幕底部中央。"""
        self.rect.midbottom = self.screen_rect.midbottom  # 设置飞船的 midbottom 位置
        self.x = float(self.rect.x)  # 更新飞船的精确 x 坐标
        # 更新护盾位置
        self.shield_rect.centerx = self.rect.centerx  # 设置护盾的中心 x 坐标
        self.shield_rect.centery = self.rect.centery  # 设置护盾的中心 y 坐标

    def update(self):
        """根据移动标志更新飞船的位置。"""
        # 更新飞船的 x 值，而不是 rect。
        if self.moving_right and self.rect.right < self.screen_rect.right:
            self.x += self.settings.ship_speed  # 向右移动飞船
        if self.moving_left and self.rect.left > 0:
            self.x -= self.settings.ship_speed  # 向左移动飞船
            
        # 根据 self.x 更新 rect 对象。
        self.rect.x = self.x  # 更新飞船矩形的 x 坐标

    def draw_shield(self):
        """如果护盾激活，则绘制飞船的护盾。"""
        if self.shield_active:
            pygame.draw.rect(self.screen, self.shield_color, self.shield_rect, 2)  # 仅绘制护盾轮廓

    def blitme(self):
        """在当前位置绘制飞船。"""
        self.screen.blit(self.image, self.rect)  # 在屏幕上绘制飞船图像