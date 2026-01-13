import cv2
import ctypes
import time
import numpy as np
from collections import deque
from openvino.runtime import Core
from PIL import Image, ImageDraw, ImageFont

# ======================= 配置区 =======================

# 模型路径
OV_MODEL_XML = r"C:\H3C_WORK\py\Boss\yolov11n-face_openvino_model\yolov11n-face.xml"

# 摄像头与应用
CAM_ID = 1                        # 默认摄像头编号
TARGET_APP = "飞书"               # 目标窗口关键字（窗口标题中包含该字符串）

# 检测逻辑
CONF_THRES = 0.55                 # 人脸检测置信度阈值
DETECT_INTERVAL = 3               # 每 N 帧推理一次，越大越省 CPU，越小越灵敏
REQUIRED_FACE_SECONDS = 0.5       # 连续检测到人脸至少多少秒才切换窗口

# 模型输入尺寸（导出的 OpenVINO 模型固定为 640x640）
INFER_W, INFER_H = 640, 640

# 按键设置
EXIT_KEY = ord('q')               # 退出键
CAM_SWITCH_KEY = ord('z')         # 切换摄像头键

# 颜色（BGR）
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)

# 中文日志字体配置
FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"   # 一种支持中文的字体文件
FONT_SIZE_LOG = 16                         # 日志字体大小
FONT_SIZE_HINT = 18                        # 提示字体大小
MAX_MSG_LINES = 6                          # 画面上最多显示几条最近日志

# ===================== 配置区结束 =====================


# ============ 初始化字体 & 日志缓冲区 ============
font_log = ImageFont.truetype(FONT_PATH, FONT_SIZE_LOG)
font_hint = ImageFont.truetype(FONT_PATH, FONT_SIZE_HINT)

log_buffer = deque(maxlen=MAX_MSG_LINES)

def add_log(text: str):
    """输出到控制台并写入画面日志缓冲区"""
    t = time.strftime("%H:%M:%S", time.localtime())
    msg = f"[{t}] {text}"
    print(msg)
    log_buffer.append(msg)

def draw_chinese_text(frame, text_list, start_xy, font, color=(255, 255, 255), line_gap=2):
    """用 Pillow 在 frame 上绘制多行中文文本"""
    if not text_list:
        return frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)
    x, y = start_xy
    for text in text_list:
        draw.text((x, y), text, font=font, fill=color)
        y += font.size + line_gap
    return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)

def draw_logs(frame):
    """左下/左侧日志区域"""
    return draw_chinese_text(
        frame,
        list(log_buffer),
        start_xy=(10, 70),
        font=font_log,
        color=COLOR_YELLOW,
        line_gap=2
    )

def draw_hints(frame, current_cam, face_time_acc):
    """左上角提示区域"""
    hints = [
        f"当前摄像头: {current_cam}",
        "Q: 退出    Z: 切换摄像头",
        f"连续检测人脸: {face_time_acc:.1f}s / {REQUIRED_FACE_SECONDS:.1f}s"
    ]
    return draw_chinese_text(
        frame,
        hints,
        start_xy=(10, 10),
        font=font_hint,
        color=COLOR_WHITE,
        line_gap=4
    )

# ================= Windows 窗口控制 =================
user32 = ctypes.WinDLL('user32', use_last_error=True)
HWND = ctypes.c_void_p
user32.FindWindowA.restype = HWND
user32.SetForegroundWindow.argtypes = [HWND]
user32.ShowWindow.argtypes = [HWND, ctypes.c_int]
user32.GetForegroundWindow.restype = HWND
user32.GetWindowTextW.argtypes = [HWND, ctypes.c_wchar_p, ctypes.c_int]
user32.IsWindowVisible.argtypes = [HWND]
user32.GetWindow.argtypes = [HWND, ctypes.c_int]

SW_RESTORE = 9
SW_MAXIMIZE = 3

def get_foreground_window_title():
    hwnd = user32.GetForegroundWindow()
    buf = ctypes.create_unicode_buffer(256)
    user32.GetWindowTextW(hwnd, buf, 256)
    return buf.value

def switch_to_app(app_name: str):
    """切换到包含 app_name 的窗口标题"""
    current_win_title = get_foreground_window_title()
    if app_name in current_win_title:
        add_log(f"当前已在【{app_name}】窗口，无需切换")
        return True

    hwnd = user32.FindWindowA(None, None)
    while hwnd:
        buf = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, buf, 256)
        win_title = buf.value
        if app_name in win_title and user32.IsWindowVisible(hwnd):
            user32.ShowWindow(hwnd, SW_RESTORE)
            user32.ShowWindow(hwnd, SW_MAXIMIZE)
            user32.SetForegroundWindow(hwnd)
            add_log(f"检测到人脸，已切换至【{app_name}】窗口")
            return True
        hwnd = user32.GetWindow(hwnd, 2)

    add_log(f"未找到包含【{app_name}】的窗口标题")
    return False

# ================= OpenVINO 模型加载 =================
core = Core()
add_log(f"加载 OpenVINO 模型: {OV_MODEL_XML}")
ov_model = core.read_model(OV_MODEL_XML)

compiled_model = core.compile_model(ov_model, "CPU")
infer_request = compiled_model.create_infer_request()
input_layer = compiled_model.inputs[0]
output_layer = compiled_model.outputs[0]

add_log(f"模型加载完成，输入: {input_layer.shape} 输出: {output_layer.shape}")

# ================= 推理后处理 =================
def ov_infer_yolo(frame_bgr):
    """
    输入: 原始 BGR 图像
    输出: [(x1, y1, x2, y2, conf, cls), ...]，坐标基于 640x640
    """
    img = cv2.resize(frame_bgr, (INFER_W, INFER_H))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    input_blob = np.transpose(img_rgb, (2, 0, 1))[None, :]

    result = infer_request.infer({input_layer.any_name: input_blob})
    output = result[output_layer]  # [1, 5, 8400]

    out = output[0]                # [5, 8400]
    cx, cy, w, h, conf = out

    keep = conf > CONF_THRES
    if not np.any(keep):
        return []

    cx = cx[keep]
    cy = cy[keep]
    w = w[keep]
    h = h[keep]
    conf_kept = conf[keep]

    x1 = np.clip(cx - w / 2, 0, INFER_W - 1)
    y1 = np.clip(cy - h / 2, 0, INFER_H - 1)
    x2 = np.clip(cx + w / 2, 0, INFER_W - 1)
    y2 = np.clip(cy + h / 2, 0, INFER_H - 1)

    max_boxes = 10
    order = np.argsort(-conf_kept)[:max_boxes]

    boxes = []
    for idx in order:
        boxes.append((
            float(x1[idx]), float(y1[idx]),
            float(x2[idx]), float(y2[idx]),
            float(conf_kept[idx]),
            0
        ))
    return boxes

# ================= 摄像头初始化 =================
cap = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    add_log("摄像头打开失败，程序退出")
    raise SystemExit

current_cam = CAM_ID
win_name = "BOSS检测器【OpenVINO版】"

frame_count = 0
last_boxes_xyxy_conf = []
last_face_detected = False

face_time_acc = 0.0       # 连续检测人脸的累计时间
triggered = False         # 是否已切过飞书
last_time = time.time()

add_log(f"程序启动，摄像头：{current_cam}，每 {DETECT_INTERVAL} 帧推理一次")
add_log(f"连续检测到人脸 {REQUIRED_FACE_SECONDS:.1f} 秒后才切换至【{TARGET_APP}】窗口")
add_log("按 Q 退出，按 Z 切换摄像头")

# ================= 主循环 =================
# ================= 主循环 =================
while True:
    now = time.time()
    dt = now - last_time
    last_time = now

    ret, frame = cap.read()
    if not ret:
        add_log("摄像头读取失败，程序退出")
        break

    frame_count += 1
    face_detected = False
    h0, w0 = frame.shape[:2]

    # 每 DETECT_INTERVAL 帧推理一次
    if frame_count % DETECT_INTERVAL == 0:
        boxes = ov_infer_yolo(frame)
        last_boxes_xyxy_conf = []
        for x1_s, y1_s, x2_s, y2_s, conf, cls in boxes:
            x1 = int(x1_s * w0 / INFER_W)
            x2 = int(x2_s * w0 / INFER_W)
            y1 = int(y1_s * h0 / INFER_H)
            y2 = int(y2_s * h0 / INFER_H)
            last_boxes_xyxy_conf.append((x1, y1, x2, y2, conf))

        face_detected = len(last_boxes_xyxy_conf) > 0
        last_face_detected = face_detected
    else:
        face_detected = last_face_detected

    # 累计“连续检测到人脸”的时间
    if face_detected:
        face_time_acc += dt
    else:
        face_time_acc = 0.0

    # 画人脸框
    for x1, y1, x2, y2, conf in last_boxes_xyxy_conf:
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_RED, 2)
        cv2.putText(
            frame,
            f"{int(conf * 100)}%",
            (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_RED,
            2
        )

    # 达到时间阈值就尝试切换到飞书
    if face_time_acc >= REQUIRED_FACE_SECONDS:
        add_log(f"已连续检测到人脸 {face_time_acc:.1f} 秒，尝试切换至【{TARGET_APP}】")
        switch_to_app(TARGET_APP)

    # 文本提示 + 日志
    frame = draw_hints(frame, current_cam, face_time_acc)
    frame = draw_logs(frame)

    cv2.imshow(win_name, frame)
    try:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)
    except:
        pass

    key = cv2.waitKey(33) & 0xFF
    if key == EXIT_KEY:
        add_log("收到退出指令，程序结束")
        break
    if key == CAM_SWITCH_KEY:
        cap.release()
        current_cam = 0 if current_cam == 1 else 1
        cap = cv2.VideoCapture(current_cam, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        add_log(f"已切换摄像头 → {current_cam}")