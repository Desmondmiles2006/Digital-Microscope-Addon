"""
main.py — PyQt5 Desktop Application for ESP32 AI Digital Microscope

Panels:
  Left  — Live MJPEG stream viewer
  Right — Tabbed: Live View controls | Gallery | AI Analysis | Settings
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from ai_engine.malaria import MalariaDetector
import logging
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap, QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QSlider, QComboBox, QTabWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QTextEdit, QLineEdit, QCheckBox, QFileDialog,
    QScrollArea, QStatusBar, QShortcut, QSplitter, QGroupBox,
    QSpinBox, QMessageBox
)

from streamer_client import MJPEGClient
from microscope_ai import MicroscopeAI
from image_tools import enhance_contrast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DARK_STYLE = """
QMainWindow, QWidget { background: #1e1e1e; color: #e0e0e0; font-family: Arial; font-size: 13px; }
QPushButton { background: #1A6B8A; color: white; border-radius: 4px; padding: 6px 14px; }
QPushButton:hover { background: #2196b0; }
QPushButton:disabled { background: #444; color: #888; }
QTabWidget::pane { border: 1px solid #333; }
QTabBar::tab { background: #2d2d2d; color: #aaa; padding: 8px 18px; }
QTabBar::tab:selected { background: #1A6B8A; color: white; }
QSlider::groove:horizontal { background: #333; height: 6px; border-radius: 3px; }
QSlider::handle:horizontal { background: #1A6B8A; width: 16px; height: 16px;
    margin: -5px 0; border-radius: 8px; }
QLineEdit, QTextEdit, QComboBox { background: #2d2d2d; border: 1px solid #444;
    border-radius: 3px; padding: 4px; color: #e0e0e0; }
QGroupBox { border: 1px solid #444; border-radius: 4px; margin-top: 8px; padding: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #1A9BC8; }
QStatusBar { background: #111; color: #888; }
"""

SAVES_DIR = Path.home() / "MicroscopeCaptures"
SAVES_DIR.mkdir(exist_ok=True)


class FrameSignal(QObject):
    frame_ready = pyqtSignal(np.ndarray)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ESP32 AI Digital Microscope")
        self.resize(1200, 700)
        self.setStyleSheet(DARK_STYLE)

        self._current_frame: Optional[np.ndarray] = None
        self._client: Optional[MJPEGClient] = None
        self._ai: Optional[MicroscopeAI] = None
        self._malaria: Optional[MalariaDetector] = None
        self._frame_signal = FrameSignal()
        self._frame_signal.frame_ready.connect(self._on_frame)

        self._build_ui()
        self._setup_shortcuts()

    # ── UI Construction ───────────────────────────────────────
    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)

        # Left: stream viewer
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self._stream_label = QLabel("No signal")
        self._stream_label.setAlignment(Qt.AlignCenter)
        self._stream_label.setMinimumSize(640, 480)
        self._stream_label.setStyleSheet("background: #000; color: #666; font-size: 18px;")
        left_layout.addWidget(self._stream_label)
        splitter.addWidget(left)

        # Right: tabs
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_live_tab(),     "Live View")
        self._tabs.addTab(self._build_gallery_tab(),  "Gallery")
        self._tabs.addTab(self._build_ai_tab(),       "AI Analysis")
        self._tabs.addTab(self._build_settings_tab(), "Settings")
        splitter.addWidget(self._tabs)
        splitter.setSizes([700, 480])

        self.setCentralWidget(splitter)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Disconnected")

    def _build_live_tab(self) -> QWidget:
        w = QWidget(); layout = QVBoxLayout(w)

        grp = QGroupBox("Camera Controls")
        g_layout = QVBoxLayout(grp)

        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Resolution:"))
        self._res_combo = QComboBox()
        self._res_combo.addItems(["320x240 (QVGA)", "160x120 (QQVGA)"])
        res_row.addWidget(self._res_combo)
        g_layout.addLayout(res_row)

        g_layout.addWidget(QLabel("Brightness"))
        self._brightness_slider = QSlider(Qt.Horizontal)
        self._brightness_slider.setRange(-4, 4); self._brightness_slider.setValue(0)
        self._brightness_slider.valueChanged.connect(self._on_brightness_change)
        g_layout.addWidget(self._brightness_slider)

        g_layout.addWidget(QLabel("Contrast"))
        self._contrast_slider = QSlider(Qt.Horizontal)
        self._contrast_slider.setRange(-4, 4); self._contrast_slider.setValue(0)
        self._contrast_slider.valueChanged.connect(self._on_contrast_change)
        g_layout.addWidget(self._contrast_slider)

        layout.addWidget(grp)

        self._capture_btn = QPushButton("📷 Capture Frame  [Space]")
        self._capture_btn.clicked.connect(self._capture_frame)
        layout.addWidget(self._capture_btn)
        layout.addStretch()
        return w

    def _build_gallery_tab(self) -> QWidget:
        w = QWidget(); layout = QVBoxLayout(w)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        self._gallery_grid_widget = QWidget()
        self._gallery_grid = QGridLayout(self._gallery_grid_widget)
        scroll.setWidget(self._gallery_grid_widget)
        layout.addWidget(scroll)
        refresh_btn = QPushButton("Refresh Gallery")
        refresh_btn.clicked.connect(self._refresh_gallery)
        layout.addWidget(refresh_btn)
        self._refresh_gallery()
        return w

    def _build_ai_tab(self) -> QWidget:
        w = QWidget(); layout = QVBoxLayout(w)
        self._ai_image_label = QLabel("No image selected")
        self._ai_image_label.setAlignment(Qt.AlignCenter)
        self._ai_image_label.setFixedHeight(200)
        self._ai_image_label.setStyleSheet("background:#000;")
        layout.addWidget(self._ai_image_label)

        btn_row = QHBoxLayout()
        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self._load_ai_image)
        analyze_btn = QPushButton("🔬 Analyze with AI")
        analyze_btn.clicked.connect(self._run_ai_analysis)
        malaria_btn = QPushButton("🦟 Malaria Detection")
        malaria_btn.clicked.connect(self._run_malaria_detection)
        btn_row.addWidget(malaria_btn)
        report_btn = QPushButton("📄 Generate PDF Report")
        report_btn.clicked.connect(self._generate_report)
        btn_row.addWidget(load_btn); btn_row.addWidget(analyze_btn); btn_row.addWidget(report_btn)
        layout.addLayout(btn_row)

        self._ai_text = QTextEdit()
        self._ai_text.setPlaceholderText("AI analysis will appear here...")
        self._ai_text.setReadOnly(True)
        layout.addWidget(self._ai_text)

        self._ai_image_path: Optional[str] = None
        return w

    def _build_settings_tab(self) -> QWidget:
        w = QWidget(); layout = QVBoxLayout(w)

        grp = QGroupBox("Connection")
        g = QVBoxLayout(grp)
        ip_row = QHBoxLayout()
        ip_row.addWidget(QLabel("ESP32 IP:"))
        self._ip_input = QLineEdit("192.168.1.100")
        ip_row.addWidget(self._ip_input)
        g.addLayout(ip_row)

        port_row = QHBoxLayout()
        port_row.addWidget(QLabel("Port:"))
        self._port_spin = QSpinBox(); self._port_spin.setRange(1, 65535); self._port_spin.setValue(80)
        port_row.addWidget(self._port_spin)
        g.addLayout(port_row)

        self._reconnect_check = QCheckBox("Auto-reconnect on disconnect")
        self._reconnect_check.setChecked(True)
        g.addWidget(self._reconnect_check)

        connect_btn = QPushButton("Connect")
        connect_btn.clicked.connect(self._connect)
        g.addWidget(connect_btn)
        layout.addWidget(grp)

        api_grp = QGroupBox("Anthropic API Key")
        ag = QVBoxLayout(api_grp)
        self._api_key_input = QLineEdit()
        self._api_key_input.setPlaceholderText("sk-ant-...")
        self._api_key_input.setEchoMode(QLineEdit.Password)
        ag.addWidget(self._api_key_input)
        layout.addWidget(api_grp)
        layout.addStretch()
        return w

    # ── Shortcuts ─────────────────────────────────────────────
    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(self._capture_frame)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self._save_current_frame)

    # ── Streaming ─────────────────────────────────────────────
    def _connect(self) -> None:
        ip = self._ip_input.text().strip()
        port = self._port_spin.value()
        if self._client:
            self._client.stop()
        self._client = MJPEGClient(
            host=ip, port=port,
            on_frame=lambda f: self._frame_signal.frame_ready.emit(f)
        )
        self._client.start()
        self._status.showMessage(f"Connecting to {ip}:{port}...")

        api_key = self._api_key_input.text().strip() or None
        self._ai = MicroscopeAI(api_key=api_key)
        self._malaria = MalariaDetector()

    def _on_frame(self, frame: np.ndarray) -> None:
        self._current_frame = frame
        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format_BGR888)
        pix = QPixmap.fromImage(qimg).scaled(
            self._stream_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._stream_label.setPixmap(pix)
        if self._client:
            self._status.showMessage(
                f"Connected — {self._ip_input.text()} — {self._client.fps} FPS"
            )

    # ── Capture & Save ─────────────────────────────────────────
    def _capture_frame(self) -> None:
        if self._client:
            frame = self._client.capture_single()
            if frame is not None:
                self._current_frame = frame
                self._save_frame(frame)
                return
        if self._current_frame is not None:
            self._save_frame(self._current_frame)

    def _save_frame(self, frame: np.ndarray) -> None:
        from datetime import datetime
        fname = SAVES_DIR / f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(str(fname), frame)
        self._status.showMessage(f"Saved: {fname.name}", 3000)
        self._refresh_gallery()

    def _save_current_frame(self) -> None:
        if self._current_frame is not None:
            self._save_frame(self._current_frame)

    # ── Gallery ───────────────────────────────────────────────
    def _refresh_gallery(self) -> None:
        for i in reversed(range(self._gallery_grid.count())):
            self._gallery_grid.itemAt(i).widget().setParent(None)

        images = sorted(SAVES_DIR.glob("*.jpg"), reverse=True)[:20]
        for idx, img_path in enumerate(images):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            thumb = cv2.resize(img, (140, 105))
            h, w, ch = thumb.shape
            qimg = QImage(thumb.data, w, h, ch * w, QImage.Format_BGR888)
            lbl = QLabel()
            lbl.setPixmap(QPixmap.fromImage(qimg))
            lbl.setToolTip(img_path.name)
            lbl.mousePressEvent = lambda e, p=img_path: self._open_gallery_image(str(p))
            self._gallery_grid.addWidget(lbl, idx // 3, idx % 3)

    def _open_gallery_image(self, path: str) -> None:
        self._ai_image_path = path
        img = cv2.imread(path)
        if img is not None:
            h, w, ch = img.shape
            scaled = cv2.resize(img, (300, int(300 * h / w)))
            qimg = QImage(scaled.data, scaled.shape[1], scaled.shape[0],
                          scaled.shape[1] * 3, QImage.Format_BGR888)
            self._ai_image_label.setPixmap(QPixmap.fromImage(qimg))
            self._tabs.setCurrentIndex(2)

    # ── Camera Controls ───────────────────────────────────────
    def _on_brightness_change(self, val: int) -> None:
        if self._client:
            import requests
            try:
                requests.post(
                    f"http://{self._ip_input.text()}/settings",
                    json={"brightness": val}, timeout=2
                )
            except Exception:
                pass

    def _on_contrast_change(self, val: int) -> None:
        if self._client:
            import requests
            try:
                requests.post(
                    f"http://{self._ip_input.text()}/settings",
                    json={"contrast": val}, timeout=2
                )
            except Exception:
                pass

    # ── AI Analysis ────────────────────────────────────────────
    def _load_ai_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", str(SAVES_DIR),
                                               "Images (*.jpg *.jpeg *.png)")
        if path:
            self._ai_image_path = path
            img = cv2.imread(path)
            if img is not None:
                scaled = cv2.resize(img, (300, 200))
                h, w, ch = scaled.shape
                qimg = QImage(scaled.data, w, h, ch * w, QImage.Format_BGR888)
                self._ai_image_label.setPixmap(QPixmap.fromImage(qimg))

    def _run_ai_analysis(self) -> None:
        if not self._ai_image_path:
            QMessageBox.warning(self, "No Image", "Please load or capture an image first.")
            return
        if not self._ai:
            QMessageBox.warning(self, "Not Connected", "Connect to ESP32 first (Settings tab).")
            return
        self._ai_text.setText("Analyzing... please wait.")
        QApplication.processEvents()
        try:
            result = self._ai.analyze_slide(self._ai_image_path)
            self._ai_text.setText(result)
        except Exception as exc:
            self._ai_text.setText(f"Error: {exc}")

    def _run_malaria_detection(self) -> None:
        if not self._ai_image_path:
            QMessageBox.warning(self, "No Image", "Please load or capture an image first.")
            return

            # Initialize detector if not done yet
        if self._malaria is None:
            try:
                self._malaria = MalariaDetector()
            except Exception as exc:
                QMessageBox.critical(self, "Model Error", str(exc))
                return

        self._ai_text.setText("Running malaria detection... please wait.")
        QApplication.processEvents()

        try:
            result = self._malaria.analyze_slide(self._ai_image_path)
            self._ai_text.setText(result)
        except Exception as exc:
            self._ai_text.setText("Error: " + str(exc))

    def _generate_report(self) -> None:
        if not self._ai_image_path or not self._ai_text.toPlainText():
            QMessageBox.warning(self, "Missing Data", "Run AI analysis first.")
            return
        if not self._ai:
            return
        try:
            pdf_path = self._ai.generate_report(
                self._ai_image_path, self._ai_text.toPlainText()
            )
            QMessageBox.information(self, "Report Saved", f"PDF saved:\n{pdf_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Report Error", str(exc))


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("ESP32 AI Microscope")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()