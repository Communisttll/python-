from django import template
from django.db.models import Count
from ..models import Tag

register = template.Library()

@register.inclusion_tag('blogs/popular_tags.html')
def get_popular_tags(count=10):
    """获取热门标签"""
    tags = Tag.objects.annotate(
        post_count=Count('blogpost')
    ).filter(post_count__gt=0).order_by('-post_count')[:count]
    return {'tags': tags}