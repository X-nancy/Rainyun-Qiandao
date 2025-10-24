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
    raw = cv2.imread("temp/sprite.jpg")
    for i in range(3):
        w = raw.shape[1]
        temp = raw[:, w // 3 * i: w // 3 * (i + 1)]
        cv2.imwrite(f"temp/sprite_{i + 1}.jpg", temp)
        with open(f"temp/sprite_{i + 1}.jpg", mode="rb") as f:
            temp_rb = f.read()
        if ocr.classification(temp_rb) in ["0", "1"]:
            return False
    return True


# 检查是否存在重复坐标，快速判断识别错误
def check_answer(d: dict) -> bool:
    flipped = dict()
    for key in d.keys():
        flipped[d[key]] = key
    return len(d.values()) == len(flipped.keys())


def compute_similarity(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0.0, 0

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = [m for m_n in matches if len(m_n) == 2 for m, n in [m_n] if m.distance < 0.8 * n.distance]

    if len(good) == 0:
        return 0.0, 0

    similarity = len(good) / len(matches)
    return similarity, len(good)


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
    logger.info(f"雨云签到工具 v{ver} by SerendipityR ~")
    logger.info("Github发布页: https://github.com/SerendipityR-2022/Rainyun-Qiandao")
    logger.info("------------------------------------------------------------------")
    
    if not debug:
        delay_sec = random.randint(5, 20)
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
