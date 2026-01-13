import sys
import time
import cv2
import ctypes
import numpy as np
import os
from openvino.runtime import Core
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QTextEdit,
    QVBoxLayout, QHBoxLayout, QGroupBox,
    QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor

# ===================== 全局配置区 =====================
current_dir = os.path.dirname(os.path.realpath(__file__))
OV_MODEL_XML = os.path.join(current_dir, "models", "yolov11n-face.xml")
INFER_W, INFER_H = 640, 640          
CAM_W, CAM_H = 640, 480              
DEFAULT_CONF = 0.55                  
DEFAULT_DELAY = 0.5                  
DEFAULT_INTERVAL = 3                 
LAST_SWITCH_INTERVAL = 1.0           

# ================= Windows API 窗口切换 =================
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
last_switch_time = 0.0  

def get_foreground_window_title():
    hwnd = user32.GetForegroundWindow()
    buf = ctypes.create_unicode_buffer(256)
    user32.GetWindowTextW(hwnd, buf, 256)
    return buf.value

def switch_to_app(app_name, log_cb):
    global last_switch_time
    curr_time = time.time()
    if curr_time - last_switch_time < LAST_SWITCH_INTERVAL:
        return
    if app_name in get_foreground_window_title():
        return

    hwnd = user32.FindWindowA(None, None)
    while hwnd:
        buf = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, buf, 256)
        win_title = buf.value
        if app_name in win_title and user32.IsWindowVisible(hwnd):
            user32.ShowWindow(hwnd, SW_RESTORE)
            user32.ShowWindow(hwnd, SW_MAXIMIZE)
            user32.SetForegroundWindow(hwnd)
            log_cb(f"检测到人脸，已切换至【{app_name}】窗口")
            last_switch_time = curr_time
            return
        hwnd = user32.GetWindow(hwnd, 2)

    log_cb(f"⚠️ 未找到包含【{app_name}】关键词的窗口")

# ================= 公共工具函数 =================
def get_time_str():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def log_with_timestamp(content):
    return f"[{get_time_str()}] {content}"

# ================= OpenVINO YOLOV11 人脸检测 =================
class YoloFaceOV:
    def __init__(self):
        core = Core()
        model_path = self.get_model_path()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件缺失：{model_path}，请检查models目录")
        
        print(log_with_timestamp(f"开始加载模型: {model_path}"))
        self.model = core.read_model(model_path)
        self.compiled = core.compile_model(self.model, "CPU")
        self.req = self.compiled.create_infer_request()
        self.input_layer = self.compiled.inputs[0]
        self.output_layer = self.compiled.outputs[0]

    def get_model_path(self):
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = current_dir
        model_path = os.path.join(base_path, "models", "yolov11n-face.xml")
        return model_path

    def infer(self, frame, conf_thres):
        img = cv2.resize(frame, (INFER_W, INFER_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        blob = np.transpose(img, (2, 0, 1))[None]

        res = self.req.infer({self.input_layer.any_name: blob})
        out = res[self.output_layer][0]
        
        boxes = out[:4].T
        confs = out[4].T
        keep = confs > conf_thres
        
        if not np.any(keep):
            return []
        
        boxes = boxes[keep]
        confs = confs[keep]

        cx, cy, w, h = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        order = np.argsort(-confs)[:10]
        return [(x1[i], y1[i], x2[i], y2[i], confs[i]) for i in order]

# ================= 检测线程 =================
class DetectThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)
    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = False
        self.cam_id = 0
        self.conf_thres = DEFAULT_CONF
        self.required_sec = DEFAULT_DELAY
        self.target_app = "飞书"
        self.detect_interval = DEFAULT_INTERVAL
        self.face_detected_start = 0.0
        self.detector = None
        self.cap = None
        self.retry_count = 0

    def run(self):
        self.running = True
        try:
            self.detector = YoloFaceOV()
        except Exception as e:
            self.log_signal.emit(log_with_timestamp(f"❌ 模型加载失败: {str(e)}"))
            return
        
        self.cap = cv2.VideoCapture(self.cam_id, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.cap.isOpened():
            self.log_signal.emit(log_with_timestamp(f"❌ 摄像头 {self.cam_id} 打开失败，请检查是否被占用"))
            return

        self.log_signal.emit(log_with_timestamp("✅ 人脸检测线程启动成功，开始实时检测"))
        frame_count = 0

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.retry_count += 1
                self.log_signal.emit(log_with_timestamp(f"⚠️ 摄像头读取失败，重试第{self.retry_count}次"))
                time.sleep(0.2)
                if self.retry_count >= 3:
                    self.log_signal.emit(log_with_timestamp("❌ 摄像头连续读取失败，停止检测"))
                    break
                continue
            self.retry_count = 0

            frame_count = (frame_count + 1) % self.detect_interval
            if frame_count == 0:
                h, w = frame.shape[:2]
                boxes = self.detector.infer(frame, self.conf_thres)
                face_exist = len(boxes) > 0

                scale_w = w / INFER_W
                scale_h = h / INFER_H
                for x1, y1, x2, y2, conf in boxes:
                    x1 = int(x1 * scale_w)
                    x2 = int(x2 * scale_w)
                    y1 = int(y1 * scale_h)
                    y2 = int(y2 * scale_h)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    text_y = y1 - 5 if y1 > 10 else y1 + 20
                    cv2.putText(frame, f"{int(conf*100)}%", (x1, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if face_exist:
                    if self.face_detected_start == 0:
                        self.face_detected_start = time.time()
                    elif time.time() - self.face_detected_start >= self.required_sec:
                        switch_to_app(self.target_app, self.log_signal.emit)
                        self.face_detected_start = 0.0
                else:
                    self.face_detected_start = 0.0

            self.frame_signal.emit(frame)

        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.log_signal.emit(log_with_timestamp("🔴 人脸检测线程已停止运行"))

    def stop(self):
        self.running = False
        self.wait(2000)

# ================= 调整布局后的苹果风GUI =================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BOSS 人脸检测器 • macOS")
        self.resize(1200, 780)  # 适配新布局调整窗口尺寸
        self.setFont(QFont("PingFang SC", 10, QFont.Medium))
        self._set_macos_style()

        self.worker = None
        self._init_ui()
        self._layout()
        self._bind()
        self.check_camera_available()

    def _set_macos_style(self):
        """苹果风样式（适配新布局）"""
        self.setStyleSheet('''
            QWidget {
                background-color: #f5f5f7;
                color: #1d1d1f;
            }
            QLabel#videoLabel {
                background-color: #1c1c1e;
                border-radius: 20px;
                border: none;
                padding: 4px;
                box-shadow: 0 6px 24px rgba(0,0,0,0.08);
            }
            QLabel#sectionTitle {
                font-size: 14px;
                font-weight: 600;
                color: #1d1d1f;
                margin-bottom: 8px;
            }
            QLabel#itemLabel {
                font-size: 12px;
                font-weight: 500;
                color: #6e6e73;
                margin: 8px 0 4px 0;
            }
            QTextEdit {
                background-color: rgba(255,255,255,0.7);
                border-radius: 16px;
                border: 1px solid #e2e2e7;
                padding: 12px;
                font-size: 12px;
                line-height: 1.6;
                selection-background-color: #007aff;
                selection-color: white;
            }
            QGroupBox {
                font-size: 14px;
                font-weight: 600;
                color: #1d1d1f;
                border: none;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 0;
                padding: 0;
            }
            QPushButton {
                border-radius: 12px;
                padding: 10px 0;
                font-size: 13px;
                font-weight: 500;
                border: none;
                margin: 4px 0;
                transition: background-color 0.2s ease;
            }
            QPushButton#startBtn {
                background-color: #34c759;
                color: white;
            }
            QPushButton#startBtn:hover {
                background-color: #28a745;
            }
            QPushButton#startBtn:disabled {
                background-color: #e2e2e7;
                color: #a1a1a6;
            }
            QPushButton#stopBtn {
                background-color: #ff3b30;
                color: white;
            }
            QPushButton#stopBtn:hover {
                background-color: #d92d20;
            }
            QPushButton#stopBtn:disabled {
                background-color: #e2e2e7;
                color: #a1a1a6;
            }
            QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit {
                border-radius: 12px;
                border: 1px solid #e2e2e7;
                padding: 8px 12px;
                background-color: rgba(255,255,255,0.9);
                font-size: 12px;
                height: 36px;
            }
            QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover, QLineEdit:hover {
                border-color: #c7c7cc;
            }
            QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus {
                border-color: #007aff;
                outline: none;
            }
            QComboBox::drop-down {
                border-radius: 0 12px 12px 0;
                border: none;
                background-color: transparent;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: url(:/icons/down_arrow.png);
                width: 12px;
                height: 12px;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
            QSpinBox::up-button, QSpinBox::down-button {
                width: 28px;
                border-radius: 0 12px 12px 0;
                background-color: transparent;
            }
            QDoubleSpinBox::up-arrow, QDoubleSpinBox::down-arrow,
            QSpinBox::up-arrow, QSpinBox::down-arrow {
                width: 10px;
                height: 10px;
            }
        ''')

    def _init_ui(self):
        # 顶部日志区域
        self.log_title = QLabel("系统运行日志")
        self.log_title.setObjectName("sectionTitle")
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("运行日志将自动显示在这里...")

        # 摄像头显示区
        self.video = QLabel()
        self.video.setObjectName("videoLabel")
        self.video.setFixedSize(CAM_W, CAM_H)
        self.video.setAlignment(Qt.AlignCenter)
        self.cam_subtitle = QLabel("摄像头实时画面")
        self.cam_subtitle.setObjectName("sectionTitle")

        # 配置按钮
        self.start_btn = QPushButton("▶ 启动人脸检测")
        self.start_btn.setObjectName("startBtn")
        self.stop_btn = QPushButton("⏹ 停止人脸检测")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setEnabled(False)

        # 配置项
        self.cam_label = QLabel("选择摄像头设备")
        self.cam_label.setObjectName("itemLabel")
        self.cam_box = QComboBox()
        self.update_camera_list()

        self.conf_label = QLabel("检测置信度（越高越精准）")
        self.conf_label.setObjectName("itemLabel")
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.1, 0.9)
        self.conf_spin.setValue(DEFAULT_CONF)
        self.conf_spin.setSingleStep(0.05)

        self.sec_label = QLabel("人脸持续判定秒数")
        self.sec_label.setObjectName("itemLabel")
        self.sec_spin = QDoubleSpinBox()
        self.sec_spin.setRange(0.1, 3.0)
        self.sec_spin.setValue(DEFAULT_DELAY)
        self.sec_spin.setSingleStep(0.1)

        self.app_label = QLabel("目标切换窗口名称")
        self.app_label.setObjectName("itemLabel")
        self.app_edit = QLineEdit("飞书")
        self.app_edit.setPlaceholderText("输入窗口关键词")

        self.interval_label = QLabel("检测帧间隔（越小越灵敏）")
        self.interval_label.setObjectName("itemLabel")
        self.frame_interval_spin = QSpinBox()
        self.frame_interval_spin.setRange(1, 30)
        self.frame_interval_spin.setValue(DEFAULT_INTERVAL)

    def _layout(self):
        # ========== 顶部日志布局（铺满宽度） ==========
        top_log_layout = QVBoxLayout()
        top_log_layout.addWidget(self.log_title)
        top_log_layout.addWidget(self.log_view)
        top_log_layout.setSpacing(8)
        top_log_layout.setContentsMargins(20, 20, 20, 15)
        self.log_view.setFixedHeight(180)  # 固定日志高度，避免占用过多空间

        # ========== 下方内容布局（摄像头+配置） ==========
        content_layout = QHBoxLayout()

        # 左侧摄像头区域（占60%宽度）
        cam_layout = QVBoxLayout()
        cam_layout.addWidget(self.cam_subtitle)
        cam_layout.addWidget(self.video)
        cam_layout.setSpacing(10)
        cam_layout.setContentsMargins(20, 0, 15, 20)
        # 让摄像头区域占满左侧垂直空间
        cam_layout.addStretch(1)

        # 右侧配置区域（占40%宽度，铺满垂直空间）
        config_group = QGroupBox("检测配置")
        config_layout = QVBoxLayout(config_group)
        config_layout.setSpacing(6)
        config_layout.setContentsMargins(0, 15, 0, 0)
        config_layout.addWidget(self.cam_label)
        config_layout.addWidget(self.cam_box)
        config_layout.addWidget(self.conf_label)
        config_layout.addWidget(self.conf_spin)
        config_layout.addWidget(self.sec_label)
        config_layout.addWidget(self.sec_spin)
        config_layout.addWidget(self.app_label)
        config_layout.addWidget(self.app_edit)
        config_layout.addWidget(self.interval_label)
        config_layout.addWidget(self.frame_interval_spin)
        config_layout.addSpacing(15)
        config_layout.addWidget(self.start_btn)
        config_layout.addWidget(self.stop_btn)
        # 配置项下方拉伸，让按钮贴紧上方内容
        config_layout.addStretch(1)

        content_layout.addLayout(cam_layout, stretch=6)
        content_layout.addWidget(config_group, stretch=4)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # ========== 主布局（顶部日志 + 下方内容） ==========
        main_layout = QVBoxLayout(self)
        main_layout.addLayout(top_log_layout)
        main_layout.addLayout(content_layout)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

    def _bind(self):
        self.start_btn.clicked.connect(self.start_detection)
        self.stop_btn.clicked.connect(self.stop_detection)

    def update_camera_list(self):
        self.cam_box.clear()
        available_cameras = []
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                available_cameras.append(f"摄像头 {i}")
                cap.release()
        self.cam_box.addItems(available_cameras if available_cameras else ["❌ 无可用摄像头"])

    def check_camera_available(self):
        if self.cam_box.count() == 0 or "❌" in self.cam_box.currentText():
            self.start_btn.setEnabled(False)
            self.log_view.append(log_with_timestamp("⚠️ 当前无可用摄像头设备，请检查连接"))

    def get_current_params(self):
        return {
            "cam_id": self.cam_box.currentIndex(),
            "conf_thres": self.conf_spin.value(),
            "required_sec": self.sec_spin.value(),
            "detect_interval": self.frame_interval_spin.value(),
            "target_app": self.app_edit.text().strip()
        }

    def start_detection(self):
        if self.worker and self.worker.isRunning():
            self.log_view.append(log_with_timestamp("⚠️ 检测线程已在运行中，请勿重复启动"))
            return
        
        params = self.get_current_params()
        if not params["target_app"]:
            self.log_view.append(log_with_timestamp("⚠️ 目标窗口名称不能为空"))
            return
        
        self.worker = DetectThread()
        self.worker.cam_id = params["cam_id"]
        self.worker.conf_thres = params["conf_thres"]
        self.worker.required_sec = params["required_sec"]
        self.worker.detect_interval = params["detect_interval"]
        self.worker.target_app = params["target_app"]

        self.worker.frame_signal.connect(self.update_frame)
        self.worker.log_signal.connect(self.log_view.append)
        self.worker.finished.connect(self.on_thread_finished)

        self.worker.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log_view.append(log_with_timestamp("📌 正在初始化检测线程..."))

    def stop_detection(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.log_view.append(log_with_timestamp("📌 正在停止检测线程..."))

    def on_thread_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.worker = None
        self.check_camera_available()

    def update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video.setPixmap(QPixmap.fromImage(img).scaled(
            self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

# ================= 主函数 =================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())