# Rainyun-Qiandao-V2 (Selenium)

## 项目概述
Rainyun-Qiandao-V2 是一个基于 Selenium 和 ddddocr 的雨云自动签到工具，通过模拟浏览器操作和验证码识别，实现雨云账户的自动每日签到以赚取积分。

## 功能特性
- 自动完成雨云账户登录
- 使用 ddddocr 进行验证码自动识别与处理
- 支持自定义随机延时（5-20秒），避免被系统识别为自动化脚本
- 支持在本地环境和 GitHub Actions 中运行
- 集成 ChromeDriver 自动匹配，提高兼容性
- 详细的日志记录，便于排查问题

## 技术栈
- Python 3.9+
- Selenium WebDriver
- ddddocr 验证码识别库
- Chrome 浏览器

## 安装步骤

### 1. 环境要求
- Python 3.9 或更高版本
- Google Chrome 浏览器

### 2. 克隆项目
```bash
git clone https://github.com/scfcn/Rainyun-Qiandao.git
cd Rainyun-Qiandao
```

### 3. 安装依赖
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. 配置 ChromeDriver
本项目已集成自动匹配 ChromeDriver 的功能，无需手动下载和配置。工具将按以下顺序尝试：
1. 使用系统路径中的 ChromeDriver
2. 使用 webdriver-manager 自动安装匹配的 ChromeDriver
3. 尝试常见的备用路径

## 使用方法

### 本地运行

#### 方法一：通过环境变量配置
```bash
# Windows
export RAINYUN_USER="您的用户名"
export RAINYUN_PASS="您的密码"

# Linux/macOS
export RAINYUN_USER="您的用户名"
export RAINYUN_PASS="您的密码"

# 运行脚本
python rainyun.py
```

#### 方法二：通过代码配置（不推荐，存在安全风险）
修改 `rainyun.py` 中的用户凭据（仅建议本地测试使用）。

### 使用 GitHub Actions 自动签到

1. Fork 本仓库
2. 进入仓库的 `Settings` > `Secrets and variables` > `Actions`
3. 添加以下密钥：
   - `RAINYUN_USER`: 您的雨云用户名
   - `RAINYUN_PASS`: 您的雨云密码
4. 工作流将每天 UTC 4 点（UTC+8 12点）自动运行，也可以手动触发

## 配置说明

### 环境变量
- `RAINYUN_USER`: 雨云用户名（必需）
- `RAINYUN_PASS`: 雨云密码（必需）
- `HEADLESS`: 是否以无头模式运行（true/false，默认false）
- `DEBUG`: 是否启用调试模式（true/false，默认false）
- `GITHUB_ACTIONS`: 在 GitHub Actions 环境中自动设置为 true，用于强制无头模式

### 关键设置
- 随机延时设置为 5-20 秒，可在代码中调整
- 超时时间设置为 15 秒，可在代码中修改

## 常见问题

### 1. Linux 系统怎么使用？
参考 [Linux 环境配置指南](https://github.com/SerendipityR-2022/Rainyun-Qiandao/issues/1#issuecomment-3096198779)。

### 2. 找不到元素或等待超时，报错 `NoSuchElementException`/`TimeoutException`
- 网页加载缓慢，尝试延长超时等待时间
- 更换连接性更好的网络环境
- 确认 Chrome 浏览器版本与系统兼容

### 3. 验证码识别失败
验证码识别率约为 48.3%，脚本会自动重试，多次尝试后通常能成功通过验证。

## GitHub Actions 优化
项目已集成 GitHub Actions 缓存功能，以加快每次运行的速度，主要缓存：
- Python 依赖
- Chrome 浏览器

## 贡献指南
欢迎提交 Issue 和 Pull Request 来改进本项目：

1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启一个 Pull Request

## 许可证
本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 免责声明
- 本工具仅用于学习和个人使用
- 使用本工具应遵守雨云官方的用户协议和相关规定
- 作者不对因使用本工具可能产生的任何后果负责

## 鸣谢
- [SerendipityR-2022](https://github.com/SerendipityR-2022) - 项目初始版本
- [ddddocr](https://github.com/sml2h3/ddddocr) - 开源的验证码识别库
- [Selenium](https://www.selenium.dev/) - 自动化测试工具
- [webdriver-manager](https://github.com/SergeyPirogov/webdriver_manager) - ChromeDriver 自动管理工具
