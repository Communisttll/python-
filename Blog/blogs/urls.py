"""定义blogs的URL模式"""
from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

app_name = 'blogs'

urlpatterns = [
    # 主页
    path('', views.index, name='index'),
    # 帖子详情
    path('post/<int:pk>/', views.post_detail, name='post_detail'),
    # 新建帖子
    path('post/new/', views.new_post, name='new_post'),
    # 编辑帖子
    path('post/<int:pk>/edit/', views.edit_post, name='edit_post'),
    # 删除帖子
    path('post/<int:pk>/delete/', views.delete_post, name='delete_post'),
    path('profile/', views.user_profile, name='user_profile'),
    path('profile/edit/', views.edit_profile, name='edit_profile'),
    path('profile/<str:username>/', views.user_profile, name='user_profile'),
    # 我的帖子
    path('my_posts/', views.my_posts, name='my_posts'),
    # 用户认证相关URL
    path('login/', auth_views.LoginView.as_view(template_name='blogs/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('register/', views.register, name='register'),
    # 搜索功能
    path('search/', views.search_posts, name='search_posts'),
    path('tag/<slug:tag_slug>/', views.posts_by_tag, name='posts_by_tag'),
]