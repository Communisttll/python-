from django.shortcuts import render, redirect, get_object_or_404
from .models import BlogPost, Tag, Comment, UserProfile
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.contrib.auth.forms import UserCreationForm
from .models import BlogPost
from .forms import BlogPostForm, CommentForm, UserProfileForm
from django.contrib.auth.models import User
from django.views.decorators.http import require_POST
from django.contrib import messages
from django.http import JsonResponse
from django.core.paginator import Paginator
from django.db.models import Q
from django.urls import reverse
from django.http import HttpResponseRedirect

def index(request):
    """博客主页，显示所有帖子"""
    posts = BlogPost.objects.all()
    
    # 分页功能
    paginator = Paginator(posts, 5)  # 每页显示5篇文章
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {'page_obj': page_obj}
    return render(request, 'blogs/index.html', context)

def post_detail(request, pk):
    """显示单个帖子的详细信息，包括评论"""
    post = get_object_or_404(BlogPost, id=pk)
    
    # 获取活跃评论
    comments = post.comments.filter(is_active=True)
    
    if request.method == 'POST':
        comment_form = CommentForm(data=request.POST)
        if comment_form.is_valid() and request.user.is_authenticated:
            new_comment = comment_form.save(commit=False)
            new_comment.blog_post = post
            new_comment.author = request.user
            new_comment.save()
            return redirect('blogs:post_detail', pk=post.id)
    else:
        comment_form = CommentForm()
    
    context = {
        'post': post,
        'comments': comments,
        'comment_form': comment_form,
    }
    return render(request, 'blogs/post_detail.html', context)


@login_required
def user_profile(request, username=None):
    """用户个人资料页面"""
    if username:
        user = get_object_or_404(User, username=username)
    else:
        user = request.user
    
    # 获取或创建用户资料
    profile, created = UserProfile.objects.get_or_create(user=user)
    
    # 获取用户的文章
    user_posts = BlogPost.objects.filter(author=user).order_by('-date_added')
    
    # 获取用户的评论
    user_comments = Comment.objects.filter(author=user).order_by('-created_at')[:10]
    
    context = {
        'profile_user': user,
        'profile': profile,
        'user_posts': user_posts,
        'user_comments': user_comments,
        'is_own_profile': request.user == user
    }
    
    return render(request, 'blogs/user_profile.html', context)


@login_required
def edit_profile(request):
    """编辑个人资料"""
    profile, created = UserProfile.objects.get_or_create(user=request.user)
    
    if request.method == 'POST':
        form = UserProfileForm(request.POST, request.FILES, instance=profile)
        if form.is_valid():
            form.save()
            messages.success(request, '个人资料已更新！')
            return redirect('blogs:user_profile', username=request.user.username)
    else:
        form = UserProfileForm(instance=profile)
    
    return render(request, 'blogs/edit_profile.html', {
        'form': form,
        'profile': profile
    })

@login_required
def new_post(request):
    """添加新帖子"""
    if request.method != 'POST':
        # 未提交数据，创建一个新表单
        form = BlogPostForm()
    else:
        # POST提交的数据，对数据进行处理
        form = BlogPostForm(data=request.POST)
        if form.is_valid():
            new_post = form.save(commit=False)
            new_post.author = request.user
            new_post.save()
            # 保存多对多关系（标签）
            form.save_m2m()
            messages.success(request, '博客发布成功！')
            return redirect('blogs:index')
    
    # 显示空表单或指出表单数据无效
    context = {'form': form}
    return render(request, 'blogs/new_post.html', context)

@login_required
def edit_post(request, pk):
    """编辑既有帖子"""
    post = get_object_or_404(BlogPost, id=pk)
    
    # 确保用户只能编辑自己的帖子
    if post.author != request.user:
        messages.error(request, '您只能编辑自己的帖子！')
        return redirect('blogs:index')
    
    if request.method != 'POST':
        # 初次请求，使用当前帖子填充表单
        form = BlogPostForm(instance=post)
    else:
        # POST提交的数据，对数据进行处理
        form = BlogPostForm(instance=post, data=request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, '博客更新成功！')
            return redirect('blogs:post_detail', pk=post.id)
    
    context = {'form': form, 'post': post}
    return render(request, 'blogs/edit_post.html', context)

@login_required
def my_posts(request):
    """显示当前用户的所有帖子"""
    posts = BlogPost.objects.filter(author=request.user).order_by('-date_added')
    
    # 分页功能
    paginator = Paginator(posts, 5)  # 每页显示5篇文章
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {'page_obj': page_obj}
    return render(request, 'blogs/my_posts.html', context)

@login_required
@require_POST
def delete_post(request, pk):
    """删除帖子"""
    post = get_object_or_404(BlogPost, id=pk)
    if post.author != request.user:
        messages.error(request, '您没有权限删除这篇文章。')
        return redirect('blogs:post_detail', pk=post.id)
    
    post.delete()
    messages.success(request, '文章已成功删除！')
    return redirect('blogs:index')

def register(request):
    """用户注册"""
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, '注册成功！欢迎加入我的博客！')
            return redirect('blogs:index')
    else:
        form = UserCreationForm()
    
    context = {'form': form}
    return render(request, 'blogs/register.html', context)

def search_posts(request):
    """搜索文章"""
    query = request.GET.get('q', '')
    if query:
        posts = BlogPost.objects.filter(
            Q(title__icontains=query) | Q(text__icontains=query)
        ).distinct()
    else:
        posts = BlogPost.objects.all()
    
    # 分页功能
    paginator = Paginator(posts, 5)  # 每页显示5篇文章
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'query': query,
        'search_results_count': posts.count()
    }
    return render(request, 'blogs/search_results.html', context)

def posts_by_tag(request, tag_slug):
    """按标签查看文章"""
    tag = get_object_or_404(Tag, slug=tag_slug)
    posts = BlogPost.objects.filter(tags=tag)
    
    # 分页功能
    paginator = Paginator(posts, 5)  # 每页显示5篇文章
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'tag': tag,
        'posts_count': posts.count()
    }
    return render(request, 'blogs/posts_by_tag.html', context)
