from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class Tag(models.Model):
    """标签模型"""
    name = models.CharField(max_length=50, unique=True, verbose_name='标签名')
    slug = models.SlugField(max_length=50, unique=True, verbose_name='标签别名')
    
    class Meta:
        verbose_name = '标签'
        verbose_name_plural = '标签'
        ordering = ['name']
    
    def __str__(self):
        return self.name


class BlogPost(models.Model):
    """博客帖子模型"""
    title = models.CharField(max_length=200, verbose_name='标题')
    text = models.TextField(verbose_name='正文')
    date_added = models.DateTimeField(default=timezone.now, verbose_name='添加日期')
    author = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name='作者')
    tags = models.ManyToManyField(Tag, blank=True, verbose_name='标签')
    
    class Meta:
        verbose_name = '博客帖子'
        verbose_name_plural = '博客帖子'
        ordering = ['-date_added']  # 按时间倒序排列
    
    def __str__(self):
        """返回模型的字符串表示"""
        return self.title[:50]


class Comment(models.Model):
    """评论模型"""
    blog_post = models.ForeignKey(BlogPost, on_delete=models.CASCADE, related_name='comments', verbose_name='博客文章')
    author = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name='作者')
    content = models.TextField(verbose_name='评论内容')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')
    is_active = models.BooleanField(default=True, verbose_name='是否激活')
    
    class Meta:
        verbose_name = '评论'
        verbose_name_plural = '评论'
        ordering = ['-created_at']
    
    def __str__(self):
        return f'{self.author.username} 在 "{self.blog_post.title}" 的评论'


class UserProfile(models.Model):
    """用户个人资料模型"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile', verbose_name="用户")
    bio = models.TextField(max_length=500, blank=True, verbose_name="个人简介")
    avatar = models.ImageField(upload_to='avatars/', blank=True, null=True, verbose_name="头像")
    website = models.URLField(blank=True, verbose_name="个人网站")
    location = models.CharField(max_length=100, blank=True, verbose_name="所在地")
    birth_date = models.DateField(null=True, blank=True, verbose_name="生日")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新时间")
    
    def __str__(self):
        return f"{self.user.username} 的个人资料"
    
    class Meta:
        verbose_name = "用户资料"
        verbose_name_plural = "用户资料"
