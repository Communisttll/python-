# Blog 项目

这是一个基于 Django 框架开发的个人博客系统。它允许用户注册、登录、发布文章、编辑文章、删除文章，并提供了美观的用户界面和交互功能。

## 项目结构

```
.
├── Blog/                  # 主项目配置目录
├── blogs/                 # 博客应用目录
├── db.sqlite3             # SQLite 数据库文件
└── manage.py              # Django 项目管理脚本
```

### `Blog/` (主项目配置目录)

这是 Django 项目的根配置目录，包含整个项目的全局设置和 URL 路由。

-   `__init__.py`: 一个空文件，表示 `Blog` 目录是一个 Python 包。
-   `__pycache__/`: Python 解释器缓存的字节码文件，用于加快加载速度。
-   `asgi.py`: ASGI (Asynchronous Server Gateway Interface) 配置文件，用于支持异步操作，例如 WebSocket。
-   `settings.py`: 项目的全局设置文件，包括数据库配置、应用注册、静态文件路径、认证设置等。
-   `urls.py`: 项目的根 URL 配置文件，负责将 URL 模式映射到各个应用的 URL 配置。
-   `wsgi.py`: WSGI (Web Server Gateway Interface) 配置文件，用于部署 Django 应用到生产环境的 Web 服务器。

### `blogs/` (博客应用目录)

这是博客的核心应用，包含了博客功能的所有逻辑和组件。

-   `__init__.py`: 一个空文件，表示 `blogs` 目录是一个 Python 包。
-   `__pycache__/`: Python 解释器缓存的字节码文件。
-   `admin.py`: Django 管理后台的配置文件，用于注册模型，使其可以在管理后台进行管理。
-   `apps.py`: 应用的配置类，用于定义应用的名称和行为。
-   `forms.py`: 定义了用于创建和编辑文章的表单，以及用户注册和登录的表单。
-   `migrations/`: 数据库迁移文件目录。
    -   `0001_initial.py`: 第一次创建模型时生成的迁移文件，定义了数据库表的初始结构。
    -   `__init__.py`: 表示 `migrations` 目录是一个 Python 包。
    -   `__pycache__/`: Python 解释器缓存的字节码文件。
-   `models.py`: 定义了博客应用的数据模型，例如 `Post` (文章) 模型。
-   `static/`: 存放静态文件 (CSS, JavaScript, 图片) 的目录。
    -   `blogs/`: 应用特有的静态文件。
        -   `css/`: 存放 CSS 样式文件。
            -   `style.css`: 自定义全局样式，包括美化后的 UI 样式。
        -   `js/`: 存放 JavaScript 脚本文件。
            -   `script.js`: 自定义 JavaScript 脚本，用于实现交互效果。
-   `templates/`: 存放 HTML 模板文件的目录。
    -   `blogs/`: 应用特有的模板文件。
        -   `base.html`: 基础模板，定义了网站的整体布局、导航栏、页脚等公共元素。
        -   `edit_post.html`: 编辑文章的页面模板。
        - `index.html`: 博客首页，显示所有文章列表。
        -   `login.html`: 用户登录页面模板。
        -   `my_posts.html`: 显示当前用户所有文章的页面模板。
        -   `new_post.html`: 发布新文章的页面模板。
        -   `post_detail.html`: 文章详情页模板。
        -   `register.html`: 用户注册页面模板。
-   `tests.py`: 存放应用测试代码的文件。
-   `urls.py`: 博客应用的 URL 配置文件，定义了博客相关功能的 URL 路由。
-   `views.py`: 视图文件，包含了处理用户请求、渲染模板、与模型交互的逻辑。

### `db.sqlite3`

这是项目的默认 SQLite 数据库文件。在开发阶段，Django 默认使用 SQLite 数据库来存储数据。

### `manage.py`

这是一个命令行工具，用于执行各种 Django 项目管理任务，例如：
-   `python manage.py runserver`: 启动开发服务器。
-   `python manage.py makemigrations`: 根据模型的变化创建数据库迁移文件。
-   `python manage.py migrate`: 应用数据库迁移，创建或更新数据库表。
-   `python manage.py createsuperuser`: 创建一个超级用户，用于访问管理后台。

## 如何运行

1.  **克隆项目 (如果适用)**:
    ```bash
    git clone <项目仓库地址>
    cd Blog
    ```
2.  **安装依赖**:
    ```bash
    pip install -r requirements.txt # 假设存在 requirements.txt 文件
    ```
    如果 `requirements.txt` 不存在，您可能需要手动安装 Django：
    ```bash
    pip install Django
    ```
3.  **创建数据库迁移**:
    ```bash
    python manage.py makemigrations blogs
    python manage.py migrate
    ```
4.  **创建超级用户 (可选，用于访问管理后台)**:
    ```bash
    python manage.py createsuperuser
    ```
    按照提示输入用户名、邮箱和密码。
5.  **启动开发服务器**:
    ```bash
    python manage.py runserver
    ```
    然后您可以在浏览器中访问 `http://127.0.0.1:8000/` 来查看博客。

## 功能特性

-   **用户认证**: 用户注册、登录、注销。
-   **文章管理**: 发布新文章、编辑已有文章、删除文章。
-   **权限控制**: 只有登录用户才能发布和编辑自己的文章，并确保用户只能编辑自己的文章。
-   **文章搜索**: 导航栏中集成了文章搜索框，方便用户快速查找感兴趣的文章。
-   **用户个人中心**: 导航栏的用户下拉菜单中包含了“我的帖子”等链接，方便用户管理自己的文章。
-   **美观的 UI 设计**: 采用了现代化的设计风格，包括玻璃拟态效果、渐变背景和动画效果，提升了用户体验。
-   **交互增强**: 实现了文章预览、字数统计、草稿保存、键盘快捷键以及删除确认模态框等功能，使文章编辑和管理更加便捷。
-   **响应式设计**: 页面布局能够适应不同设备的屏幕尺寸，提供良好的移动端浏览体验。
