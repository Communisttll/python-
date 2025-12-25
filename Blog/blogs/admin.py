from django.contrib import admin
from .models import BlogPost, Tag, Comment, UserProfile

@admin.register(BlogPost)
class BlogPostAdmin(admin.ModelAdmin):
    """博客帖子管理后台"""
    list_display = ['title', 'author', 'date_added']
    list_filter = ['date_added', 'author']
    search_fields = ['title', 'text']
    date_hierarchy = 'date_added'
    ordering = ['-date_added']

@admin.register(Tag)
class TagAdmin(admin.ModelAdmin):
    list_display = ['name', 'slug']
    search_fields = ['name']
    prepopulated_fields = {'slug': ('name',)}

@admin.register(Comment)
class CommentAdmin(admin.ModelAdmin):
    list_display = ['blog_post', 'author', 'created_at', 'is_active']
    list_filter = ['is_active', 'created_at']
    search_fields = ['content', 'author__username', 'blog_post__title']
    actions = ['approve_comments']
@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'bio', 'location', 'website', 'created_at']
    list_filter = ['created_at', 'updated_at']
    search_fields = ['user__username', 'user__email', 'bio', 'location']
    readonly_fields = ['created_at', 'updated_at']
    
    fieldsets = (
        ('基本信息', {
            'fields': ('user', 'bio', 'avatar')
        }),
        ('联系信息', {
            'fields': ('website', 'location')
        }),
        ('个人信息', {
            'fields': ('birth_date',)
        }),
        ('时间信息', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def approve_comments(self, request, queryset):
        queryset.update(is_active=True)
    approve_comments.short_description = "批准选中的评论"
