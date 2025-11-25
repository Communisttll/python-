# 导入必要的模块
import sys  # 用于退出游戏
import pygame  # 导入 Pygame 库，用于游戏开发
from time import sleep  # 用于暂停游戏
import os  # 用于处理文件路径
import random

# 从自定义模块中导入游戏组件
from settings import Settings  # 游戏设置
from game_stats import GameStats  # 游戏统计数据
from scoreboard import Scoreboard  # 记分牌
from button import Button  # 按钮
from ship import Ship  # 飞船
from bullet import Bullet  # 玩家子弹
from alien import Alien  # 外星人
from alien_bullet import AlienBullet  # 外星人子弹


class AlienInvasion:
    """管理游戏资源和行为的整体类。"""

    def __init__(self):
        """初始化游戏，并创建游戏资源。"""
        pygame.init()  # 初始化 Pygame 模块
        self.settings = Settings()  # 创建游戏设置实例

        # 初始化混音器用于声音播放
        pygame.mixer.init()
        self._load_sounds()  # 加载所有游戏音效

        # 设置窗口显示模式（非全屏）
        self.screen = pygame.display.set_mode((self.settings.screen_width, self.settings.screen_height))
        pygame.display.set_caption("Alien Invasion")  # 设置窗口标题

        # 创建时钟对象，用于控制游戏帧率
        self.clock = pygame.time.Clock()

        # 创建一个实例来存储游戏统计信息，并创建一个记分牌。
        self.stats = GameStats(self)  # 创建游戏统计实例
        self.sb = Scoreboard(self)  # 创建记分牌实例

        self.ship = Ship(self)  # 创建飞船实例
        self.bullets = pygame.sprite.Group()  # 创建玩家子弹编组
        self.aliens = pygame.sprite.Group()  # 创建外星人编组
        self.alien_bullets = pygame.sprite.Group()  # 创建外星人子弹编组

        self._create_fleet()  # 创建外星人群

        # 创建“Play”按钮。
        self.play_button = Button(self, "Play")

        print("游戏已初始化。")

        # 再次创建游戏统计实例和记分牌实例 (此部分代码重复，后续会优化)
        self.stats = GameStats(self)
        self.sb = Scoreboard(self)

        self.ship = Ship(self)
        self.bullets = pygame.sprite.Group()
        self.aliens = pygame.sprite.Group()
        self.alien_bullets = pygame.sprite.Group()  # 添加外星人子弹编组

        self._create_fleet()

        # 创建“Play”按钮。
        self.play_button = Button(self, "Play")

    def run_game(self):
        """开始游戏的主循环。"""
        print("进入游戏循环。")
        while True:  # 持续运行游戏循环
            self._check_events()  # 检查用户输入事件

            if self.stats.game_active:  # 如果游戏处于活动状态
                self.ship.update()  # 更新飞船位置
                self._update_bullets()  # 更新玩家子弹位置
                self._update_aliens()  # 更新外星人位置
                self._update_alien_bullets()  # 更新外星人子弹位置

            self._update_screen()  # 更新屏幕显示
            self.clock.tick(60)  # 控制游戏帧率为60FPS

    def _check_events(self):
        """响应按键和鼠标事件。"""
        for event in pygame.event.get():  # 遍历所有事件
            if event.type == pygame.QUIT:  # 如果是退出事件
                # 退出前保存最高分
                self.stats.save_high_score()
                sys.exit()  # 退出游戏
            elif event.type == pygame.KEYDOWN:  # 如果是按键按下事件
                self._check_keydown_events(event)  # 处理按键按下事件
            elif event.type == pygame.KEYUP:  # 如果是按键释放事件
                self._check_keyup_events(event)  # 处理按键释放事件
            elif event.type == pygame.MOUSEBUTTONDOWN:  # 如果是鼠标按下事件
                mouse_pos = pygame.mouse.get_pos()  # 获取鼠标位置
                self._check_play_button(mouse_pos)  # 检查是否点击了“Play”按钮

    def _check_play_button(self, mouse_pos):
        """当玩家点击"Play"时开始新游戏。"""
        button_clicked = self.play_button.rect.collidepoint(mouse_pos)  # 检查鼠标是否点击了"Play"按钮
        if button_clicked and not self.stats.game_active:  # 如果点击了按钮且游戏未激活
            # 重置游戏设置。
            self.settings.initialize_dynamic_settings()

            # 重置游戏统计数据。
            self.stats.reset_stats()
            self.sb.prep_score()  # 准备分数显示
            self.sb.prep_level()  # 准备等级显示
            self.sb.prep_ships()  # 准备飞船数量显示
            self.stats.game_active = True  # 设置游戏为活动状态

            # 清空所有剩余的子弹和外星人。
            self.bullets.empty()  # 清空玩家子弹
            self.aliens.empty()  # 清空外星人
            self.alien_bullets.empty()  # 清空外星人子弹

            # 创建新的外星人群并使飞船居中。
            self._create_fleet()  # 创建新的外星人群
            self.ship.center_ship()  # 使飞船居中

            # 隐藏鼠标光标。
            pygame.mouse.set_visible(False)

    def _check_keydown_events(self, event):
        """响应按键按下。"""
        if event.key == pygame.K_RIGHT:  # 如果按下右箭头键
            self.ship.moving_right = True  # 设置飞船向右移动
        elif event.key == pygame.K_LEFT:  # 如果按下左箭头键
            self.ship.moving_left = True  # 设置飞船向左移动
        elif event.key == pygame.K_q:  # 如果按下 'q' 键
            # 退出前保存最高分
            self.stats.save_high_score()
            sys.exit()  # 退出游戏
        elif event.key == pygame.K_SPACE:  # 如果按下空格键
            self._fire_bullet()  # 发射玩家子弹

    def _check_keyup_events(self, event):
        """响应按键释放。"""
        if event.key == pygame.K_RIGHT:  # 如果释放右箭头键
            self.ship.moving_right = False  # 停止飞船向右移动
        elif event.key == pygame.K_LEFT:  # 如果释放左箭头键
            self.ship.moving_left = False  # 停止飞船向左移动

    def _fire_bullet(self):
        """创建一颗新子弹并将其添加到子弹编组中。"""
        if len(self.bullets) < self.settings.bullets_allowed:  # 如果当前子弹数量未达到上限
            new_bullet = Bullet(self)  # 创建新子弹
            self.bullets.add(new_bullet)  # 将新子弹添加到编组中
            # 播放射击音效
            if self.shoot_sound:
                self.shoot_sound.play()

    def _update_bullets(self):
        """更新子弹位置并删除已消失的子弹。"""
        # 更新玩家子弹位置。
        self.bullets.update()

        # 删除已消失的子弹（到达屏幕顶部）。
        for bullet in self.bullets.copy():  # 遍历子弹编组的副本，以便在循环中删除元素
            if bullet.rect.bottom <= 0:  # 如果子弹到达屏幕顶部
                self.bullets.remove(bullet)  # 删除该子弹

        self._check_bullet_alien_collisions()  # 检查玩家子弹与外星人的碰撞

    def _check_bullet_alien_collisions(self):
        """响应玩家子弹与外星人的碰撞。"""
        # 检查是否有子弹击中外星人。
        # 如果是，则删除子弹和外星人。
        collisions = pygame.sprite.groupcollide(self.bullets, self.aliens, True, True)  # 检测碰撞并删除碰撞的子弹和外星人

        if collisions:
            # 为每个被击中的外星人播放爆炸音效
            if self.explosion_sound:
                for aliens in collisions.values():
                    for alien in aliens:
                        self.explosion_sound.play()
            
            for aliens in collisions.values():  # 遍历所有发生碰撞的外星人组
                self.stats.score += self.settings.alien_points * len(aliens)  # 增加分数
            self.sb.prep_score()  # 更新记分牌上的分数
            self.sb.check_high_score()  # 检查是否创建了新的最高分

        if not self.aliens:  # 如果外星人编组为空，表示所有外星人已被消灭
            # 销毁现有子弹并创建新的外星人群。
            self.bullets.empty()  # 清空玩家子弹
            self.alien_bullets.empty()  # 清空外星人子弹
            self._create_fleet()  # 创建新的外星人群
            self.settings.increase_speed()  # 提高游戏速度

            # 提高等级。
            self.stats.level += 1  # 增加游戏等级
            self.sb.prep_level()  # 更新记分牌上的等级

    def _ship_hit(self):
        """响应飞船被外星人或子弹击中。"""
        if self.stats.ships_left > 0:  # 如果还有剩余飞船
            # 飞船数量减一。
            self.stats.ships_left -= 1  # 飞船数量减1
            self.sb.prep_ships()  # 更新记分牌上的飞船数量

            # 播放爆炸音效
            if self.explosion_sound:
                self.explosion_sound.play()

            # 清空所有剩余的玩家子弹、外星人和外星人子弹。
            self.bullets.empty()  # 清空玩家子弹
            self.aliens.empty()  # 清空外星人
            self.alien_bullets.empty()  # 清空外星人子弹

            # 创建新的外星人群并使飞船居中。
            self._create_fleet()  # 创建新的外星人群
            self.ship.center_ship()  # 使飞船居中

            # 暂停。
            sleep(0.5)  # 暂停0.5秒
        else:  # 如果没有剩余飞船
            self.stats.game_active = False  # 设置游戏为非活动状态
            pygame.mouse.set_visible(True)  # 显示鼠标光标

    def _update_aliens(self):
        """检查外星人群是否在边缘，然后更新所有外星人的位置。"""
        self._check_fleet_edges()  # 检查外星人群是否到达边缘
        self.aliens.update()  # 更新所有外星人的位置

        # 随机让外星人射击
        self._alien_shooting()

        # 查找外星人与飞船的碰撞。
        if pygame.sprite.spritecollideany(self.ship, self.aliens):  # 如果有外星人与飞船碰撞
            self._ship_hit()  # 处理飞船被击中

        # 查找是否有外星人到达屏幕底部。
        self._check_aliens_bottom()  # 检查外星人是否到达屏幕底部

    def _check_aliens_bottom(self):
        """检查是否有外星人到达屏幕底部。"""
        for alien in self.aliens.sprites():  # 遍历外星人编组
            if alien.rect.bottom >= self.settings.screen_height:  # 如果外星人到达屏幕底部
                # 像飞船被击中一样处理。
                self._ship_hit()  # 处理飞船被击中
                break

    def _create_fleet(self):
        """创建外星人群。"""
        # 创建一个外星人，并不断添加外星人直到没有空间。
        # 外星人之间的间距是一个外星人的宽度和高度。
        alien = Alien(self)  # 创建一个外星人实例用于获取尺寸
        alien_width, alien_height = alien.rect.size  # 获取外星人的宽度和高度

        current_x, current_y = alien_width, alien_height  # 初始化当前位置
        while current_y < (self.settings.screen_height - 3 * alien_height):  # 确保外星人不会太靠近底部
            while current_x < (self.settings.screen_width - 2 * alien_width):  # 确保外星人不会太靠近边缘
                self._create_alien(current_x, current_y)  # 创建一个外星人
                current_x += 2 * alien_width  # 移动到下一个外星人的位置

            # 完成一行；重置 x 值，并增加 y 值。
            current_x = alien_width  # 重置 x 坐标到行首
            current_y += 2 * alien_height  # 移动到下一行

    def _create_alien(self, x_position, y_position):
        """创建一个外星人并将其放置在外星人群中。"""
        new_alien = Alien(self)  # 创建一个新的外星人实例
        new_alien.x = x_position  # 设置外星人的精确 x 坐标
        new_alien.rect.x = x_position  # 设置外星人矩形的 x 坐标
        new_alien.rect.y = y_position  # 设置外星人矩形的 y 坐标
        self.aliens.add(new_alien)  # 将新外星人添加到外星人编组中

    def _check_fleet_edges(self):
        """如果任何外星人到达边缘，则做出相应的响应。"""
        for alien in self.aliens.sprites():  # 遍历外星人编组
            if alien.check_edges():  # 如果外星人到达屏幕边缘
                self._change_fleet_direction()  # 改变外星人群的移动方向
                break

    def _change_fleet_direction(self):
        """使整个外星人群向下移动，并改变它们的方向。"""
        for alien in self.aliens.sprites():  # 遍历外星人编组
            alien.rect.y += self.settings.fleet_drop_speed  # 外星人向下移动
        self.settings.fleet_direction *= -1  # 改变外星人群的移动方向

    def _update_screen(self):
        """更新屏幕上的图像，并切换到新屏幕。"""
        self.screen.fill(self.settings.bg_color)  # 每次循环时都重绘屏幕
        for bullet in self.bullets.sprites():  # 绘制所有子弹
            bullet.draw_bullet()
        for alien_bullet in self.alien_bullets.sprites():  # 绘制所有外星人子弹
            alien_bullet.draw_bullet()
        self.ship.blitme()  # 绘制飞船
        self.aliens.draw(self.screen)  # 绘制外星人

        # 绘制得分信息。
        self.sb.show_score()  # 显示得分板

        # 如果游戏处于非活动状态，则绘制“Play”按钮。
        if not self.stats.game_active:  # 如果游戏不活跃
            self.play_button.draw_button()  # 绘制开始按钮

        pygame.display.flip()  # 让最近绘制的屏幕可见


    def _update_alien_bullets(self):
        """更新外星人子弹的位置并删除旧子弹。"""
        self.alien_bullets.update()  # 更新外星人子弹的位置

        # 删除已经消失的子弹。
        for alien_bullet in self.alien_bullets.copy():  # 遍历外星人子弹编组的副本
            if alien_bullet.rect.top >= self.settings.screen_height:  # 如果子弹到达屏幕底部
                self.alien_bullets.remove(alien_bullet)  # 删除子弹

        # 检查外星人子弹和飞船之间的碰撞。
        if pygame.sprite.spritecollideany(self.ship, self.alien_bullets):  # 如果外星人子弹与飞船碰撞
            self._ship_hit()  # 处理飞船被击中

    def _alien_shooting(self):
        """随机让外星人射击。"""
        for alien in self.aliens.sprites():  # 遍历外星人编组
            if random.randint(0, 1000) < self.settings.alien_shoot_probability * 1000:  # 根据概率判断是否射击
                self._fire_alien_bullet(alien)  # 外星人射击

    def _fire_alien_bullet(self, alien):
        """创建一个外星人子弹，并将其添加到外星人子弹编组中。"""
        if len(self.alien_bullets) < self.settings.alien_bullets_allowed:  # 检查是否达到外星人子弹数量限制
            new_bullet = AlienBullet(self, alien)  # 创建一个新的外星人子弹
            self.alien_bullets.add(new_bullet)  # 将新子弹添加到外星人子弹编组中
            if self.alien_shoot_sound:  # 如果外星人射击音效存在
                self.alien_shoot_sound.play()  # 播放外星人射击音效


    def _load_sounds(self):
        """加载游戏音效。"""
        try:
            # 加载射击音效
            shoot_path = os.path.join(os.path.dirname(__file__), 'sounds', 'shoot.wav')
            if os.path.exists(shoot_path):
                self.shoot_sound = pygame.mixer.Sound(shoot_path)  # 加载玩家射击音效
                self.shoot_sound.set_volume(0.5)  # 设置玩家射击音效的音量
            else:
                self.shoot_sound = None  # 设置射击音效为 None
                
            # 加载外星人射击音效
            alien_shoot_path = os.path.join(os.path.dirname(__file__), 'sounds', 'alien_shoot.wav')
            if os.path.exists(alien_shoot_path):
                self.alien_shoot_sound = pygame.mixer.Sound(alien_shoot_path)  # 加载外星人射击音效
                self.alien_shoot_sound.set_volume(0.5)  # 设置外星人射击音效的音量
            else:
                self.alien_shoot_sound = None  # 设置外星人射击音效为 None
                
            # 加载爆炸音效
            explosion_path = os.path.join(os.path.dirname(__file__), 'sounds', 'explosion.wav')
            if os.path.exists(explosion_path):
                self.explosion_sound = pygame.mixer.Sound(explosion_path)  # 加载爆炸音效
                self.explosion_sound.set_volume(0.7)  # 设置爆炸音效的音量
            else:
                self.explosion_sound = None  # 设置爆炸音效为 None
                
            # 加载背景音乐
            background_path = os.path.join(os.path.dirname(__file__), 'sounds', 'background.mp3')
            if os.path.exists(background_path):
                pygame.mixer.music.load(background_path)  # 加载背景音乐
                pygame.mixer.music.set_volume(0.5)  # 设置背景音乐音量
                pygame.mixer.music.play(-1)  # 循环播放背景音乐
        except pygame.error as e:
            print(f"无法加载音效文件: {e}")  # 打印加载音效文件时遇到的错误
            self.shoot_sound = None  # 设置射击音效为 None
            self.alien_shoot_sound = None  # 设置外星人射击音效为 None
            self.explosion_sound = None  # 设置爆炸音效为 None


if __name__ == '__main__':
    # Make a game instance, and run the game.
    ai = AlienInvasion()
    ai.run_game()