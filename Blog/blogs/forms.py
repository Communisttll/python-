from django import forms
from .models import BlogPost, Comment, UserProfile

class BlogPostForm(forms.ModelForm):
    """博客帖子表单"""
    tags = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': '请输入标签，用逗号分隔'
        }),
        label='标签',
        help_text='请输入标签，用逗号分隔多个标签'
    )
    
    class Meta:
        model = BlogPost
        fields = ['title', 'text']
        labels = {
            'title': '标题',
            'text': '正文'
        }
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '请输入标题'
            }),
            'text': forms.Textarea(attrs={
                'class': 'form-control',
                'placeholder': '请输入正文内容'
            })
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance and self.instance.pk:
            # 如果实例已存在，显示现有标签
            tags = self.instance.tags.all()
            self.fields['tags'].initial = ', '.join([tag.name for tag in tags])
    
    def save(self, commit=True):
        instance = super().save(commit=False)
        
        if commit:
            instance.save()
            
        # 处理标签
        if 'tags' in self.cleaned_data:
            tags_string = self.cleaned_data['tags']
            tag_names = [name.strip() for name in tags_string.split(',') if name.strip()]
            
            # 只有在实例有ID的情况下才能处理多对多关系
            if instance.pk:
                # 清除现有标签
                instance.tags.clear()
                
                # 添加新标签
                for tag_name in tag_names:
                    tag, created = Tag.objects.get_or_create(
                        name=tag_name,
                        defaults={'slug': tag_name.lower().replace(' ', '-')}
                    )
                    instance.tags.add(tag)
        
        if commit:
            instance.save()
        
        return instance


class CommentForm(forms.ModelForm):
    """评论表单"""
    class Meta:
        model = Comment
        fields = ['content']
        widgets = {
            'content': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': '写下您的评论...'
            })
        }
        labels = {
            'content': '评论内容'
        }


class UserProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['bio', 'avatar', 'website', 'location', 'birth_date']
        widgets = {
            'bio': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': '介绍一下你自己...'
            }),
            'website': forms.URLInput(attrs={
                'class': 'form-control',
                'placeholder': 'https://example.com'
            }),
            'location': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '例如：北京'
            }),
            'birth_date': forms.DateInput(attrs={
                'class': 'form-control',
                'type': 'date'
            }),
        }
        labels = {
            'content': '评论内容'
        }