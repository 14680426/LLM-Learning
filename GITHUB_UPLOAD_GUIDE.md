# 上传项目到GitHub指南

本指南将帮助您将LLM Learning项目上传到GitHub，并跳过大型文件夹。

## 1. 安装Git（如果尚未安装）

### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install git
```

### CentOS/RHEL:
```bash
sudo yum install git
```

### macOS (使用Homebrew):
```bash
brew install git
```

## 2. 配置Git

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## 3. 创建GitHub仓库

1. 访问 [GitHub](https://github.com) 并登录
2. 点击右上角的 "+" 号，选择 "New repository"
3. 输入仓库名称（例如：llm-learning）
4. 选择公开或私有仓库
5. 不要初始化 README、.gitignore 或许可证
6. 点击 "Create repository"

## 4. 在本地项目中初始化Git并推送

```bash
# 进入项目目录
cd "/data/LLM Learning"

# 初始化Git仓库
git init

# 添加所有文件（除了.gitignore中指定的文件）
git add .

# 提交更改
git commit -m "Initial commit"

# 添加远程仓库（将URL替换为您的GitHub仓库URL）
git remote add origin https://github.com/yourusername/your-repo-name.git

# 推送到GitHub
git branch -M main
git push -u origin main
```

## 5. 配置.gitignore文件

项目中已包含.gitignore文件，它会自动忽略以下目录：
- `Models/Qwen/` - 大型模型文件目录
- `Finetune/output/` - 微调输出目录

## 6. 如果您已经创建了仓库并需要更新

```bash
# 添加更改
git add .

# 提交更改
git commit -m "Update project"

# 推送到GitHub
git push origin main
```

## 7. 处理大型文件的额外建议

对于特别大的模型文件，建议使用Git LFS（Large File Storage）：

### 安装Git LFS:
```bash
# Ubuntu/Debian
sudo apt install git-lfs
git lfs install

# 或者从官方网站下载安装
```

### 跟踪大型文件:
```bash
git lfs track "*.bin"
git lfs track "*.pt"
git lfs track "*.ckpt"
```

然后将 `.gitattributes` 文件添加到仓库中：
```bash
git add .gitattributes
```

这样可以更有效地管理大型文件，避免Git仓库变得过于庞大。