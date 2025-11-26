import pygame.font


class Button:
    """一个为游戏构建按钮的类。"""

    def __init__(self, ai_game, msg):
        """初始化按钮的属性。"""
        self.screen = ai_game.screen  # 获取游戏屏幕对象
        self.screen_rect = self.screen.get_rect()  # 获取屏幕的矩形区域

        # 设置按钮的尺寸和其他属性。
        self.width, self.height = 200, 50  # 按钮的宽度和高度
        self.button_color = (0, 135, 0)  # 按钮的颜色（绿色）
        self.text_color = (255, 255, 255)  # 文本颜色（白色）
        self.font = pygame.font.SysFont(None, 48)  # 字体设置（默认字体，字号 48）

        # 创建按钮的 rect 对象，并使其居中。
        self.rect = pygame.Rect(0, 0, self.width, self.height)  # 创建按钮的矩形对象
        self.rect.center = self.screen_rect.center  # 将按钮居中放置在屏幕上

        # 按钮的消息只需要准备一次。
        self._prep_msg(msg)  # 准备按钮上显示的消息

    def _prep_msg(self, msg):
        """将 msg 渲染为图像，并使其在按钮上居中。"""
        self.msg_image = self.font.render(msg, True, self.text_color,
                self.button_color)  # 将消息渲染为图像
        self.msg_image_rect = self.msg_image.get_rect()  # 获取消息图像的矩形区域
        self.msg_image_rect.center = self.rect.center  # 将消息图像居中放置在按钮上

    def draw_button(self):
        """绘制一个空白按钮，然后绘制消息。"""
        self.screen.fill(self.button_color, self.rect)  # 绘制按钮的矩形背景
        self.screen.blit(self.msg_image, self.msg_image_rect)  # 在按钮上绘制消息