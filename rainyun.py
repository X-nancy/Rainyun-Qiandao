import logging
import os
import random
import re
import time

import cv2
import ddddocr
import requests
from selenium import webdriver
from selenium.common import TimeoutException
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


# 修改init_selenium函数，修复linux变量作用域问题
# 在文件开头导入webdriver-manager
# 修改这两行导入语句
# from webdriver_manager.chrome import ChromeDriverManager
# from webdriver_manager.core.utils import ChromeType

# 修改为正确的导入方式
try:
    from webdriver_manager.chrome import ChromeDriverManager
    # 尝试不同的ChromeType导入路径
    try:
        from webdriver_manager.core.utils import ChromeType
    except ImportError:
        try:
            from webdriver_manager.chrome import ChromeType
        except ImportError:
            # 如果找不到ChromeType，设置为None
            ChromeType = None
except ImportError:
    print("webdriver_manager未安装，将使用备用方式")
    ChromeDriverManager = None
    ChromeType = None
import subprocess
import sys

def init_selenium(debug=False, headless=False):
    ops = webdriver.ChromeOptions()
    
    # 无论什么环境都添加无头模式选项
    if headless or os.environ.get("GITHUB_ACTIONS", "false") == "true":
        for option in ['--headless', '--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu']:
            ops.add_argument(option)
    
    # 添加通用选项
    ops.add_argument('--window-size=1920,1080')
    ops.add_argument('--disable-blink-features=AutomationControlled')
    ops.add_argument('--no-proxy-server')
    ops.add_argument('--lang=zh-CN')
    
    # 环境变量判断是否在GitHub Actions中运行
    is_github_actions = os.environ.get("GITHUB_ACTIONS", "false") == "true"
    
    if debug and not is_github_actions:
        ops.add_experimental_option("detach", True)
    
    # 尝试不同的ChromeDriver使用策略
    # 策略1: 直接使用系统路径中的ChromeDriver（最简单可靠）
    try:
        print("尝试直接使用系统ChromeDriver...")
        # 不指定service，让Selenium自动查找系统路径中的ChromeDriver
        driver = webdriver.Chrome(options=ops)
        print("成功使用系统ChromeDriver")
        return driver
    except Exception as e:
        print(f"系统ChromeDriver失败: {e}")
    
    # 策略2: 优化webdriver-manager的使用方式
    try:
        print("尝试使用webdriver-manager...")
        if ChromeDriverManager:
            # 仅当ChromeType可用时才指定chrome_type参数
            if ChromeType and hasattr(ChromeType, 'GOOGLE'):
                manager = ChromeDriverManager(chrome_type=ChromeType.GOOGLE)
            else:
                # 在新版本中，可能不再需要指定ChromeType
                manager = ChromeDriverManager()
            
            # 获取驱动路径但不自动安装
            driver_path = manager.install()
            print(f"获取到ChromeDriver路径: {driver_path}")
            # 手动创建service并指定正确的驱动路径
            service = Service(driver_path)
            driver = webdriver.Chrome(service=service, options=ops)
            print("成功使用webdriver-manager")
            return driver
        else:
            raise ImportError("webdriver_manager未安装")
    except Exception as e:
        print(f"webdriver-manager失败: {e}")
    
    # 策略3: 作为最后的备用，尝试使用固定路径
    try:
        print("尝试使用备用ChromeDriver路径...")
        # 尝试常见的ChromeDriver路径
        common_paths = ['/usr/local/bin/chromedriver', '/usr/bin/chromedriver', './chromedriver', 'chromedriver']
        for path in common_paths:
            try:
                service = Service(path)
                driver = webdriver.Chrome(service=service, options=ops)
                print(f"成功使用备用路径: {path}")
                return driver
            except:
                continue
    except Exception as e:
        print(f"备用路径失败: {e}")
    
    # 所有策略都失败时的错误处理
    print("错误: 无法初始化ChromeDriver，请检查Chrome和ChromeDriver的安装")
    # 在GitHub Actions环境中，尝试安装ChromeDriver的备用方法
    if is_github_actions:
        print("在GitHub Actions环境中，尝试安装ChromeDriver...")
        try:
            # 使用npm的chromedriver包作为备用
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'chromedriver-binary-auto'])
            import chromedriver_binary  # 这个包会自动设置路径
            driver = webdriver.Chrome(options=ops)
            print("成功使用chromedriver-binary-auto")
            return driver
        except Exception as e:
            print(f"备用安装失败: {e}")
    
    # 彻底失败
    raise Exception("无法初始化Selenium WebDriver")

def download_image(url, filename):
    os.makedirs("temp", exist_ok=True)
    try:
        # 禁用代理以避免连接问题
        response = requests.get(url, timeout=10, proxies={"http": None, "https": None}, verify=False)
        if response.status_code == 200:
            path = os.path.join("temp", filename)
            with open(path, "wb") as f:
                f.write(response.content)
            return True
        else:
            logger.error(f"下载图片失败！状态码: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"下载图片异常: {str(e)}")
        return False


def get_url_from_style(style):
    return re.search(r'url\(["\']?(.*?)["\']?\)', style).group(1)


def get_width_from_style(style):
    return re.search(r'width:\s*([\d.]+)px', style).group(1)


def get_height_from_style(style):
    return re.search(r'height:\s*([\d.]+)px', style).group(1)


def process_captcha():
    try:
        download_captcha_img()
        if check_captcha():
            logger.info("开始识别验证码")
            captcha = cv2.imread("temp/captcha.jpg")
            with open("temp/captcha.jpg", 'rb') as f:
                captcha_b = f.read()
            bboxes = det.detection(captcha_b)
            result = dict()
            for i in range(len(bboxes)):
                x1, y1, x2, y2 = bboxes[i]
                spec = captcha[y1:y2, x1:x2]
                cv2.imwrite(f"temp/spec_{i + 1}.jpg", spec)
                for j in range(3):
                    similarity, matched = compute_similarity(f"temp/sprite_{j + 1}.jpg", f"temp/spec_{i + 1}.jpg")
                    similarity_key = f"sprite_{j + 1}.similarity"
                    position_key = f"sprite_{j + 1}.position"
                    if similarity_key in result.keys():
                        if float(result[similarity_key]) < similarity:
                            result[similarity_key] = similarity
                            result[position_key] = f"{int((x1 + x2) / 2)},{int((y1 + y2) / 2)}"
                    else:
                        result[similarity_key] = similarity
                        result[position_key] = f"{int((x1 + x2) / 2)},{int((y1 + y2) / 2)}"
            if check_answer(result):
                for i in range(3):
                    similarity_key = f"sprite_{i + 1}.similarity"
                    position_key = f"sprite_{i + 1}.position"
                    positon = result[position_key]
                    logger.info(f"图案 {i + 1} 位于 ({positon})，匹配率：{result[similarity_key]}")
                    slideBg = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="slideBg"]')))
                    style = slideBg.get_attribute("style")
                    x, y = int(positon.split(",")[0]), int(positon.split(",")[1])
                    width_raw, height_raw = captcha.shape[1], captcha.shape[0]
                    width, height = float(get_width_from_style(style)), float(get_height_from_style(style))
                    x_offset, y_offset = float(-width / 2), float(-height / 2)
                    final_x, final_y = int(x_offset + x / width_raw * width), int(y_offset + y / height_raw * height)
                    ActionChains(driver).move_to_element_with_offset(slideBg, final_x, final_y).click().perform()
                confirm = wait.until(
                    EC.element_to_be_clickable((By.XPATH, '//*[@id="tcStatus"]/div[2]/div[2]/div/div')))
                logger.info("提交验证码")
                confirm.click()
                time.sleep(5)
                result = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="tcOperation"]')))
                if result.get_attribute("class") == 'tc-opera pointer show-success':
                    logger.info("验证码通过")
                    return
                else:
                    logger.error("验证码未通过，正在重试")
            else:
                logger.error("验证码识别失败，正在重试")
        else:
            logger.error("当前验证码识别率低，尝试刷新")
        reload = driver.find_element(By.XPATH, '//*[@id="reload"]')
        time.sleep(5)
        reload.click()
        time.sleep(5)
        process_captcha()
    except TimeoutException:
        logger.error("获取验证码图片失败")


def download_captcha_img():
    if os.path.exists("temp"):
        for filename in os.listdir("temp"):
            file_path = os.path.join("temp", filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
    slideBg = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="slideBg"]')))
    img1_style = slideBg.get_attribute("style")
    img1_url = get_url_from_style(img1_style)
    logger.info("开始下载验证码图片(1): " + img1_url)
    download_image(img1_url, "captcha.jpg")
    sprite = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="instruction"]/div/img')))
    img2_url = sprite.get_attribute("src")
    logger.info("开始下载验证码图片(2): " + img2_url)
    download_image(img2_url, "sprite.jpg")


def check_captcha() -> bool:
    """改进的验证码检查函数"""
    try:
        raw = cv2.imread("temp/sprite.jpg")
        if raw is None:
            logger.error("无法读取验证码图片")
            return False
        
        # 图像质量检查
        h, w = raw.shape[:2]
        if h < 50 or w < 100:  # 检查图像尺寸是否合理
            logger.warning(f"验证码图片尺寸过小: {w}x{h}")
            return False
        
        # 图像清晰度检查
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian < 50:  # 低于阈值可能是模糊图像
            logger.warning(f"验证码图片清晰度不足: {laplacian}")
            return False
            
        # 分割和保存三个子图像
        for i in range(3):
            w_segment = w // 3
            # 添加一定的边界裕度，避免分割到边缘
            start_x = max(0, w_segment * i + 2)
            end_x = min(w, w_segment * (i + 1) - 2)
            temp = raw[:, start_x:end_x]
            cv2.imwrite(f"temp/sprite_{i + 1}.jpg", temp)
            
            # 图像识别检查
            with open(f"temp/sprite_{i + 1}.jpg", mode="rb") as f:
                temp_rb = f.read()
            try:
                result = ocr.classification(temp_rb)
                if result in ["0", "1"]:
                    logger.warning(f"发现无效验证码: sprite_{i + 1}.jpg = {result}")
                    return False
            except Exception as e:
                logger.error(f"OCR识别出错: {e}")
                return False
        
        return True
    except Exception as e:
        logger.error(f"验证码检查失败: {e}")
        return False


# 检查是否存在重复坐标，快速判断识别错误
def check_answer(d: dict) -> bool:
    """改进的答案检查函数，不仅检查重复坐标，还检查相似度阈值"""
    # 检查是否有重复值
    flipped = dict()
    for key in d.keys():
        flipped[d[key]] = key
    
    if len(d.values()) != len(flipped.keys()):
        return False
    
    # 检查相似度是否达到最低阈值
    min_similarity_threshold = 0.3  # 设置最低相似度阈值
    for i in range(3):
        similarity_key = f"sprite_{i + 1}.similarity"
        if similarity_key in d and float(d[similarity_key]) < min_similarity_threshold:
            logger.warning(f"相似度不足: {similarity_key} = {d[similarity_key]}")
            return False
    
    # 检查位置是否合理分布（避免太集中）
    positions = []
    for i in range(3):
        position_key = f"sprite_{i + 1}.position"
        if position_key in d:
            x, y = map(int, d[position_key].split(","))
            positions.append((x, y))
    
    # 检查位置分布是否合理
    if len(positions) == 3:
        # 计算x坐标的分布范围
        x_coords = [p[0] for p in positions]
        x_range = max(x_coords) - min(x_coords)
        
        # 如果三个点太集中，可能识别有误
        if x_range < 50:  # 假设阈值为50像素
            logger.warning(f"位置分布过于集中: x范围 = {x_range}")
            return False
    
    return True


def preprocess_image(image):
    """图像预处理函数，提高特征匹配准确率"""
    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # 自适应阈值二值化，增强对比度
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # 形态学操作，增强特征
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return morph

def compute_similarity(img1_path, img2_path):
    """优化的相似度计算函数"""
    # 读取图像
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    # 图像尺寸标准化
    if img1.shape[0] > 100 or img1.shape[1] > 100:
        # 如果图像太大，先缩小以提高处理速度
        scale = 100.0 / max(img1.shape)
        img1 = cv2.resize(img1, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    if img2.shape[0] > 100 or img2.shape[1] > 100:
        scale = 100.0 / max(img2.shape)
        img2 = cv2.resize(img2, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    # 图像预处理
    img1 = preprocess_image(img1)
    img2 = preprocess_image(img2)

    # 使用SIFT特征提取
    try:
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            return 0.0, 0

        # 使用FLANN匹配器，比BFMatcher更高效
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # 增加检查次数以提高准确率
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # 应用比例测试筛选好的匹配点，使用更严格的阈值0.7
        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good.append(m)

        if len(good) == 0:
            return 0.0, 0

        # 计算相似度时考虑特征点总数和匹配点比例
        # 同时考虑特征点数量的影响，避免小图像的匹配点少但比例高的问题
        feature_factor = min(1.0, len(kp1) / 100.0, len(kp2) / 100.0)  # 归一化特征点数量因子
        match_ratio = len(good) / min(len(des1), len(des2))  # 使用最小特征点数作为分母更合理
        
        # 综合相似度计算
        similarity = match_ratio * 0.7 + feature_factor * 0.3
        
        return similarity, len(good)
    except Exception as e:
        logger.error(f"相似度计算出错: {e}")
        return 0.0, 0


# 修改main函数，移除重复的变量定义
# 修改随机延时等待设置
if __name__ == "__main__":
    # 连接超时等待
    timeout = 15

    user = os.environ.get("RAINYUN_USER")
    pwd = os.environ.get("RAINYUN_PASS")
    
    # 确保有用户名和密码
    if not user or not pwd:
        print("错误: 未设置用户名或密码，请在环境变量中设置RAINYUN_USER和RAINYUN_PASS")
        exit(1)
    
    # 环境变量判断是否在GitHub Actions中运行
    is_github_actions = os.environ.get("GITHUB_ACTIONS", "false") == "true"
    # 从环境变量读取模式设置
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    headless = os.environ.get('HEADLESS', 'false').lower() == 'true'
    
    # 如果在GitHub Actions环境中，强制使用无头模式
    if is_github_actions:
        headless = True
    
    # 以下代码保持不变...
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    ver = "2.2"
    logger.info("------------------------------------------------------------------")
    logger.info(f"雨云自动签到工作流 v{ver} by 筱序二十 ~")
    logger.info("Github发布页: https://github.com/scfcn/Rainyun-Qiandao")
    logger.info("------------------------------------------------------------------")
    
    if not debug:
        delay_sec = random.randint(5, 10)
        logger.info(f"随机延时等待 {delay_sec} 秒")
        time.sleep(delay_sec)
    logger.info("初始化 ddddocr")
    ocr = ddddocr.DdddOcr(ocr=True, show_ad=False)
    det = ddddocr.DdddOcr(det=True, show_ad=False)
    logger.info("初始化 Selenium")
    # 在 main 函数中添加
    headless = os.environ.get('HEADLESS', 'false').lower() == 'true'
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    # 传递 headless 参数给 init_selenium
    driver = init_selenium(debug=debug, headless=headless)
    # 过 Selenium 检测
    with open("stealth.min.js", mode="r") as f:
        js = f.read()
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": js
    })
    logger.info("发起登录请求")
    driver.get("https://app.rainyun.com/auth/login")
    wait = WebDriverWait(driver, timeout)
    try:
        username = wait.until(EC.visibility_of_element_located((By.NAME, 'login-field')))
        password = wait.until(EC.visibility_of_element_located((By.NAME, 'login-password')))
        login_button = wait.until(EC.visibility_of_element_located((By.XPATH,
                                                                    '//*[@id="app"]/div[1]/div[1]/div/div[2]/fade/div/div/span/form/button')))
        username.send_keys(user)
        password.send_keys(pwd)
        login_button.click()
    except TimeoutException:
        logger.error("页面加载超时，请尝试延长超时时间或切换到国内网络环境！")
        exit()
    try:
        login_captcha = wait.until(EC.visibility_of_element_located((By.ID, 'tcaptcha_iframe_dy')))
        logger.warning("触发验证码！")
        driver.switch_to.frame("tcaptcha_iframe_dy")
        process_captcha()
    except TimeoutException:
        logger.info("未触发验证码")
    time.sleep(5)
    driver.switch_to.default_content()
    if driver.current_url == "https://app.rainyun.com/dashboard":
        logger.info("登录成功！")
        logger.info("正在转到赚取积分页")
        driver.get("https://app.rainyun.com/account/reward/earn")
        driver.implicitly_wait(5)
        earn = driver.find_element(By.XPATH,
                                   '//*[@id="app"]/div[1]/div[3]/div[2]/div/div/div[2]/div[2]/div/div/div/div[1]/div/div[1]/div/div[1]/div/span[2]/a')
        logger.info("点击赚取积分")
        earn.click()
        logger.info("处理验证码")
        driver.switch_to.frame("tcaptcha_iframe_dy")
        process_captcha()
        driver.switch_to.default_content()
        driver.implicitly_wait(5)
        points_raw = driver.find_element(By.XPATH,
                                         '//*[@id="app"]/div[1]/div[3]/div[2]/div/div/div[2]/div[1]/div[1]/div/p/div/h3').get_attribute(
            "textContent")
        current_points = int(''.join(re.findall(r'\d+', points_raw)))
        logger.info(f"当前剩余积分: {current_points} | 约为 {current_points / 2000:.2f} 元")
        logger.info("任务执行成功！")
    else:
        logger.error("登录失败！")
