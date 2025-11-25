import pygame.font
from pygame.sprite import Group

from ship import Ship


class Scoreboard:
    """一个报告得分信息的类。"""

    def __init__(self, ai_game):
        """初始化记分牌属性。"""
        self.ai_game = ai_game  # 游戏实例
        self.screen = ai_game.screen  # 屏幕对象
        self.screen_rect = self.screen.get_rect()  # 屏幕矩形
        self.settings = ai_game.settings  # 游戏设置
        self.stats = ai_game.stats  # 游戏统计信息

        # 得分信息的字体设置。
        self.text_color = (30, 30, 30)  # 文本颜色
        self.font = pygame.font.SysFont(None, 48)  # 字体和字号

        # 准备初始得分图像。
        self.prep_score()  # 准备当前得分图像
        self.prep_high_score()  # 准备最高分图像
        self.prep_level()  # 准备等级图像
        self.prep_ships()  # 准备飞船数量图像

    def prep_score(self):
        """将得分转换为渲染图像。"""
        rounded_score = round(self.stats.score, -1)  # 将得分四舍五入到最近的 10 的倍数
        score_str = f"{rounded_score:,}"  # 格式化得分字符串，添加逗号分隔符
        self.score_image = self.font.render(score_str, True,
                self.text_color, self.settings.bg_color)  # 渲染得分图像

        # 在屏幕右上角显示得分。
        self.score_rect = self.score_image.get_rect()  # 获取得分图像的矩形
        self.score_rect.right = self.screen_rect.right - 20  # 设置得分矩形的右边缘位置
        self.score_rect.top = 20  # 设置得分矩形的顶部位置

    def prep_high_score(self):
        """将最高分转换为渲染图像。"""
        high_score = round(self.stats.high_score, -1)  # 将最高分四舍五入到最近的 10 的倍数
        high_score_str = f"{high_score:,}"  # 格式化最高分字符串，添加逗号分隔符
        self.high_score_image = self.font.render(high_score_str, True,
                self.text_color, self.settings.bg_color)  # 渲染最高分图像
        
        # 将最高分居中显示在屏幕顶部。
        self.high_score_rect = self.high_score_image.get_rect()  # 获取最高分图像的矩形
        self.high_score_rect.centerx = self.screen_rect.centerx  # 将最高分矩形水平居中
        self.high_score_rect.top = self.score_rect.top  # 将最高分矩形的顶部与当前得分矩形对齐

    def prep_level(self):
        """将等级转换为渲染图像。"""
        level_str = str(self.stats.level)  # 将等级转换为字符串
        self.level_image = self.font.render(level_str, True,
                self.text_color, self.settings.bg_color)  # 渲染等级图像

        # 将等级放置在得分下方。
        self.level_rect = self.level_image.get_rect()  # 获取等级图像的矩形
        self.level_rect.right = self.score_rect.right  # 将等级矩形的右边缘与得分矩形对齐
        self.level_rect.top = self.score_rect.bottom + 10  # 将等级矩形的顶部放置在得分矩形下方 10 像素处

    def prep_ships(self):
        """显示还剩下多少艘飞船。"""
        self.ships = Group()  # 创建一个空的精灵组来存储飞船
        for ship_number in range(self.stats.ships_left):  # 遍历剩余飞船的数量
            ship = Ship(self.ai_game)  # 创建一艘新飞船
            ship.rect.x = 10 + ship_number * ship.rect.width  # 设置每艘飞船的 x 坐标，使其并排显示
            ship.rect.y = 10  # 设置飞船的 y 坐标
            self.ships.add(ship)  # 将飞船添加到精灵组中

    def check_high_score(self):
        """检查是否诞生了新的最高分。"""
        if self.stats.score > self.stats.high_score:  # 如果当前得分高于最高分
            self.stats.high_score = self.stats.score  # 更新最高分
            self.prep_high_score()  # 重新准备最高分图像

    def show_score(self):
        """在屏幕上绘制得分、等级和飞船。"""
        self.screen.blit(self.score_image, self.score_rect)  # 绘制当前得分
        self.screen.blit(self.high_score_image, self.high_score_rect)  # 绘制最高分
        self.screen.blit(self.level_image, self.level_rect)  # 绘制等级
        self.ships.draw(self.screen)  # 绘制剩余飞船
