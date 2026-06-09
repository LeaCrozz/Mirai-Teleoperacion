import sys
import math
import random
import socket
import time
import json
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from torchvision import models, transforms
from PIL import Image

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"
import pygame

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QPushButton, QSizePolicy,
    QMessageBox, QGridLayout
)
from PyQt6.QtCore import Qt, QTimer, QDateTime, QPointF, pyqtSignal, QThread
from PyQt6.QtGui import (QPainter, QColor, QPen, QBrush, QFont,
                          QPainterPath, QImage, QPixmap)

FLOTA = {
    "GRUA":      {"ip": "192.168.10.187", "port": 5005},
    "BULLDOZER": {"ip": "192.168.10.123", "port": 5005}
}
UDP_PORT_VIDEO = 5006

sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_send.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

MODELO_PATH = os.path.join(
    os.path.expanduser("~"), "OneDrive", "Desktop",
    "modelos_mirai_v2", "mejor_modelo.pth"
)
MODELO_BULLDOZER_PATH = os.path.join(
    os.path.expanduser("~"), "OneDrive", "Desktop",
    "modelos_mirai_bulldozer", "mejor_modelo.pth"
)

class P:
    BG0    = QColor("#0d0e10")
    BG1    = QColor("#13151a")
    BG2    = QColor("#1a1d24")
    BG3    = QColor("#22262f")
    BG4    = QColor("#2a2f3a")
    BORDER = QColor("#2a2f3a")
    ACCENT = QColor("#e87c3a")
    ACCENT2= QColor("#f0a060")
    OK     = QColor("#3ecf8e")
    WARN   = QColor("#f5c542")
    DANGER = QColor("#e85555")
    AI     = QColor("#7b8cde")
    TXT1   = QColor("#e8eaf0")
    TXT2   = QColor("#8b90a0")
    TXT3   = QColor("#555a6a")

    @staticmethod
    def lat_color(ms: int) -> QColor:
        if ms < 50:  return P.OK
        if ms < 100: return P.WARN
        return P.DANGER

class MiraiNetV2(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        n_sensor = 4 + cfg["n_fases"] + 5
        backbone = models.mobilenet_v2(weights=None)
        backbone.classifier = nn.Sequential(
            nn.Dropout(cfg["dropout"]),
            nn.Linear(backbone.last_channel, 256),
            nn.ReLU(),
        )
        self.cnn = backbone
        self.sensor_enc = nn.GRU(
            input_size=n_sensor, hidden_size=128,
            num_layers=2, batch_first=True,
            dropout=cfg["dropout"],
        )
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.LayerNorm(256), nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
        )
        n_outputs = cfg.get("n_outputs", len(cfg.get("output_names", range(5))))
        self.head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(),
            nn.Dropout(cfg["dropout"] * 0.5),
            nn.Linear(64, n_outputs), nn.Tanh(),
        )

    def forward(self, imgs, sensores):
        vis_feat  = self.cnn(imgs[:, -1])
        _, h_n    = self.sensor_enc(sensores)
        sens_feat = h_n[-1]
        fused     = self.fusion(torch.cat([vis_feat, sens_feat], dim=-1))
        return self.head(fused)

class MiraiInferenceThread(QThread):
    sugerencia_signal = pyqtSignal(list)
    estado_signal     = pyqtSignal(str)

    VEHICLE_CFG = {
        "GRUA": {
            "modelo":       MODELO_PATH,
            "output_names": ["oruga_izq", "oruga_der", "brazo", "pala_rot", "rotacion_cabina"],
            "imu_keys":     ["imu_cabina", "imu_eslabon1", "imu_eslabon2", "imu_pala"],
            "cmd_size":     5,
        },
        "BULLDOZER": {
            "modelo":       MODELO_BULLDOZER_PATH,
            "output_names": ["oruga_izq", "oruga_der", "pala"],
            "imu_keys":     ["imu_chasis", "imu_chasis", "imu_pala", "imu_pala"],
            "cmd_size":     3,
        },
    }

    def __init__(self, machine: str = "GRUA"):
        super().__init__()
        import threading
        self._lock    = threading.Lock()
        self._machine = machine
        self._frame   = None
        self._imu_data = {}
        self._fase     = 0
        self._cmd_prev = [0.0] * 5

        self._seq_len = 8
        self._img_buf = deque(maxlen=self._seq_len)
        self._sen_buf = deque(maxlen=self._seq_len)
        self._pendiente_cambio = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = None
        self.cfg    = None

        self._img_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std= [0.229, 0.224, 0.225]),
        ])

    def cambiar_maquina(self, machine: str):
        with self._lock:
            self._pendiente_cambio = machine

    def actualizar(self, frame: np.ndarray, imu_data: dict,
                   fase: int, cmd_prev: list):
        with self._lock:
            self._frame    = frame.copy() if frame is not None else None
            self._imu_data = dict(imu_data)
            self._fase     = fase
            self._cmd_prev = list(cmd_prev)

    def _cargar_modelo(self, machine: str = None):
        if machine:
            self._machine = machine
        vcfg = self.VEHICLE_CFG[self._machine]
        ruta = vcfg["modelo"]
        if not os.path.exists(ruta):
            print(f"[MIRAI-AI] Modelo no encontrado: {ruta}")
            self.estado_signal.emit("SIN_MODELO")
            return False
        try:
            ckpt = torch.load(ruta, map_location=self.device, weights_only=False)
            cfg  = ckpt["config"]
            cfg.setdefault("n_outputs", len(vcfg["output_names"]))
            cfg.setdefault("output_names", vcfg["output_names"])
            self.cfg   = cfg
            self.model = MiraiNetV2(cfg).to(self.device)
            self.model.load_state_dict(ckpt["model_state"])
            self.model.eval()
            self._seq_len = cfg.get("seq_len", 8)
            self._img_buf = deque(maxlen=self._seq_len)
            self._sen_buf = deque(maxlen=self._seq_len)
            self._cmd_prev = [0.0] * vcfg["cmd_size"]
            print(f"[MIRAI-AI] {self._machine} cargado (época {ckpt.get('epoch','?')})")
            self.estado_signal.emit("OK")
            return True
        except Exception as e:
            print(f"[MIRAI-AI] Error cargando modelo: {e}")
            self.estado_signal.emit("ERROR")
            return False

    def _frame_a_tensor(self, frame_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self._img_tf(Image.fromarray(rgb))

    def _imu_a_vector(self, imu_data: dict, fase: int, cmd_prev: list) -> list:
        def ay(d, key):
            v = d.get(key, {})
            return float(v.get("ay", 0.0) if isinstance(v, dict) else float(v))

        imu_max  = self.cfg.get("imu_max", 90.0)
        n_fases  = self.cfg.get("n_fases", 5)
        imu_keys = self.VEHICLE_CFG[self._machine]["imu_keys"]
        cmd_size = self.VEHICLE_CFG[self._machine]["cmd_size"]

        imu_vals = [ay(imu_data, k) / imu_max for k in imu_keys]
        fase_oh = [0.0] * n_fases
        fase_oh[min(fase, n_fases - 1)] = 1.0
        return [*imu_vals, *fase_oh, *cmd_prev[:cmd_size]]

    def run(self):
        if not self._cargar_modelo():
            return

        while not self.isInterruptionRequested():
            self.msleep(100)

            with self._lock:
                pendiente = self._pendiente_cambio
                self._pendiente_cambio = None

            if pendiente and pendiente != self._machine:
                self.estado_signal.emit("CARGANDO...")
                self._cargar_modelo(pendiente)
                continue

            with self._lock:
                frame    = self._frame
                imu_data = self._imu_data
                fase     = self._fase
                cmd_prev = self._cmd_prev

            if frame is None or self.model is None:
                continue

            try:
                img_t = self._frame_a_tensor(frame)
                sen_v = self._imu_a_vector(imu_data, fase, cmd_prev)

                self._img_buf.append(img_t)
                self._sen_buf.append(sen_v)

                while len(self._img_buf) < self._seq_len:
                    self._img_buf.appendleft(img_t)
                    self._sen_buf.appendleft(sen_v)

                imgs_t = torch.stack(list(self._img_buf)).unsqueeze(0).to(self.device)
                sens_t = torch.tensor(
                    list(self._sen_buf), dtype=torch.float32
                ).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    pred = self.model(imgs_t, sens_t)

                self.sugerencia_signal.emit(pred.squeeze(0).cpu().tolist())
            except Exception as e:
                print(f"[MIRAI-AI] Error inferencia: {e}")

class AIAssistPanel(QWidget):
    NOMBRES_GRUA      = ["ORUGA IZQ", "ORUGA DER", "BRAZO", "PALA ROT", "ROT CAB"]
    NOMBRES_BULLDOZER = ["ORUGA IZQ", "ORUGA DER", "PALA"]

    def __init__(self):
        super().__init__()
        self.setFixedHeight(148)
        self._nombres     = self.NOMBRES_GRUA
        self._sugerencias = [0.0] * 5
        self._operador    = [0.0] * 5
        self._activo      = False
        self._estado      = "CARGANDO..."

    def set_machine(self, machine: str):
        self._nombres     = self.NOMBRES_BULLDOZER if machine == "BULLDOZER" else self.NOMBRES_GRUA
        self._sugerencias = [0.0] * len(self._nombres)
        self._operador    = [0.0] * len(self._nombres)
        self.update()

    def set_sugerencias(self, vals: list):
        n = len(self._nombres)
        self._sugerencias = (vals + [0.0] * n)[:n]
        self._activo      = True
        self.update()

    def set_operador(self, vals: list):
        self._operador = vals[:5]
        self.update()

    def set_estado(self, estado: str):
        self._estado = estado
        self._activo = (estado == "OK")
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, P.BG2)
        font_lbl = QFont("Consolas", 8)
        font_val = QFont("Consolas", 8, QFont.Weight.Bold)

        if not self._activo:
            p.setPen(QPen(P.TXT3)); p.setFont(font_lbl)
            p.drawText(8, 20, f"IA: {self._estado}")
            return

        row_h, lbl_w = 26, 64
        bar_w = w - lbl_w - 54
        bar_x = lbl_w + 4
        for i, nombre in enumerate(self._nombres):
            cy = i * row_h + 4 + row_h // 2
            p.setPen(QPen(P.TXT2)); p.setFont(font_lbl)
            p.drawText(4, cy + 4, nombre)
            p.fillRect(bar_x, cy - 2, bar_w, 4, P.BG4)
            mid = bar_x + bar_w // 2
            op = max(-1.0, min(1.0, self._operador[i]))
            fw_op = int(abs(op) * (bar_w // 2))
            fx_op = mid - fw_op if op < 0 else mid
            p.fillRect(fx_op, cy - 1, fw_op, 2, QColor(232, 124, 58, 180))
            sg = max(-1.0, min(1.0, self._sugerencias[i]))
            fw_sg = int(abs(sg) * (bar_w // 2))
            fx_sg = mid - fw_sg if sg < 0 else mid
            p.fillRect(fx_sg, cy - 3, fw_sg, 6, QColor(123, 140, 222, 200))
            p.setPen(QPen(P.TXT3, 1)); p.drawLine(mid, cy - 4, mid, cy + 4)
            p.setPen(QPen(P.AI)); p.setFont(font_val)
            p.drawText(bar_x + bar_w + 4, cy + 4, f"{sg:+.2f}")
        p.setFont(QFont("Consolas", 7))
        p.setPen(QPen(QColor(232, 124, 58, 180)))
        p.fillRect(4, h - 12, 8, 4, QColor(232, 124, 58, 180))
        p.drawText(16, h - 8, "OPERADOR")
        p.setPen(QPen(P.AI))
        p.fillRect(80, h - 12, 8, 6, QColor(123, 140, 222, 200))
        p.drawText(92, h - 8, "IA SUGIERE")

class AIOverlayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._sugerencias = [0.0] * 5
        self._activo   = False
        self._opacidad = 0.0
        self._modo     = 0

    def set_modo(self, modo: int):
        self._modo = modo
        self.update()

    def set_sugerencias(self, vals: list):
        self._sugerencias = vals[:5]
        self._activo   = True
        self._opacidad = min(1.0, self._opacidad + 0.3)
        self.update()

    def set_inactivo(self):
        self._activo   = False
        self._opacidad = max(0.0, self._opacidad - 0.1)
        self.update()

    def paintEvent(self, _):
        if self._opacidad < 0.05:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        alpha = int(self._opacidad * 200)

        panel_w, panel_h = 200, 22
        px, py = w - panel_w - 12, 10
        p.setBrush(QBrush(QColor(13, 14, 16, 160)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(px, py, panel_w, panel_h, 4, 4)
        modo_txt = {0: "● VISUAL", 1: "● BLEND", 2: "▶ OVERRIDE"}.get(self._modo, "● VISUAL")
        modo_color = {0: QColor(123,140,222,alpha),
                      1: QColor(232,124,58,alpha),
                      2: QColor(123,140,222,alpha)}.get(self._modo)
        p.setPen(QPen(modo_color))
        p.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        p.drawText(px + 8, py + 15, f"MIRAI  {modo_txt}")

        or_izq, or_der, brazo = self._sugerencias[0], self._sugerencias[1], self._sugerencias[2]

        def flecha(painter, cx, cy, magnitud, vertical=False):
            if abs(magnitud) < 0.08:
                return
            color = QColor(123, 140, 222, int(abs(magnitud) * alpha))
            painter.setPen(QPen(color, 2))
            largo = int(abs(magnitud) * 40)
            aw = 6
            if not vertical:
                dx = largo if magnitud > 0 else -largo
                painter.drawLine(cx, cy, cx + dx, cy)
                tip = cx + dx
                if magnitud > 0:
                    painter.drawLine(tip, cy, tip - aw, cy - aw)
                    painter.drawLine(tip, cy, tip - aw, cy + aw)
                else:
                    painter.drawLine(tip, cy, tip + aw, cy - aw)
                    painter.drawLine(tip, cy, tip + aw, cy + aw)
            else:
                dy = -largo if magnitud > 0 else largo
                painter.drawLine(cx, cy, cx, cy + dy)
                tip = cy + dy
                if magnitud > 0:
                    painter.drawLine(cx, tip, cx - aw, tip + aw)
                    painter.drawLine(cx, tip, cx + aw, tip + aw)
                else:
                    painter.drawLine(cx, tip, cx - aw, tip - aw)
                    painter.drawLine(cx, tip, cx + aw, tip - aw)

        margin_bot = h - 50
        flecha(p, w // 4, margin_bot, or_izq)
        flecha(p, 3 * w // 4, margin_bot, or_der)
        p.setFont(QFont("Consolas", 7))
        p.setPen(QPen(QColor(123, 140, 222, int(alpha * 0.7))))
        p.drawText(w // 4 - 20, margin_bot + 14, f"IZQ {or_izq:+.2f}")
        p.drawText(3*w//4 - 20, margin_bot + 14, f"DER {or_der:+.2f}")
        flecha(p, w // 2, h // 2, brazo, vertical=True)
        if abs(brazo) >= 0.08:
            p.drawText(w // 2 + 10, h // 2, f"BRAZO {brazo:+.2f}")


# ============================================================
# COMPONENTES UI
# ============================================================
class SmoothValue:
    def __init__(self, initial=0.0, alpha=0.15):
        self.value = initial
        self.alpha = alpha
    def update(self, target):
        self.value += (target - self.value) * self.alpha
        return self.value


class CameraView(QFrame):
    def __init__(self):
        super().__init__()
        self.latency = 32
        self.fps = 60
        self.recording = False
        self._blink = True
        self.current_frame = None
        self.label_texto = "CH-01 / FRONTAL"
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_frame(self, pixmap):
        self.current_frame = pixmap
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, P.BG0)
        if self.current_frame is not None:
            scaled = self.current_frame.scaled(
                w, h, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.FastTransformation)
            p.drawPixmap((w - scaled.width()) // 2, (h - scaled.height()) // 2, scaled)
        p.setPen(QPen(QColor(255, 255, 255, 20), 1))
        for i in range(1, 6):
            p.drawLine(int(w * i / 6), 0, int(w * i / 6), h)
        for i in range(1, 4):
            p.drawLine(0, int(h * i / 4), w, int(h * i / 4))
        cx, cy = w // 2, h // 2
        p.setPen(QPen(QColor(255, 255, 255, 160), 1.5))
        gap, arm = 8, 22
        p.drawLine(cx-arm-gap, cy, cx-gap, cy)
        p.drawLine(cx+gap, cy, cx+arm+gap, cy)
        p.drawLine(cx, cy-arm-gap, cx, cy-gap)
        p.drawLine(cx, cy+gap, cx, cy+arm+gap)
        p.drawEllipse(cx-4, cy-4, 8, 8)
        corner, margin = 16, 10
        p.setPen(QPen(QColor(255, 255, 255, 60), 1.5))
        for x, y, dx, dy in [(margin,margin,1,1),(w-margin,margin,-1,1),
                              (margin,h-margin,1,-1),(w-margin,h-margin,-1,-1)]:
            p.drawLine(x, y, x+dx*corner, y)
            p.drawLine(x, y, x, y+dy*corner)
        p.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        p.setPen(QPen(QColor(255, 255, 255, 180)))
        p.drawText(28, 22, f"{self.fps} FPS")
        p.setPen(QPen(P.lat_color(self.latency)))
        p.drawText(93, 22, f"{self.latency} ms")
        p.setPen(QPen(QColor(255, 255, 255, 60)))
        p.setFont(QFont("Consolas", 8))
        p.drawText(14, h - 10, self.label_texto)


class MiniCamera(QFrame):
    def __init__(self, label: str, show_nosignal=False):
        super().__init__()
        self.label = label
        self.nosignal = show_nosignal
        self.current_frame = None
        self.setMinimumHeight(80)

    def set_frame(self, pixmap):
        self.current_frame = pixmap
        self.nosignal = False
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, P.BG0)
        if self.current_frame is not None:
            scaled = self.current_frame.scaled(
                w, h, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.FastTransformation)
            p.drawPixmap((w - scaled.width()) // 2, (h - scaled.height()) // 2, scaled)
        p.setFont(QFont("Consolas", 8))
        if self.nosignal:
            p.setPen(QPen(P.DANGER)); p.drawText(10, 16, "NO SIGNAL")
        else:
            p.setPen(QPen(P.OK)); p.setBrush(QBrush(P.OK))
            p.drawEllipse(10, 8, 4, 4)
        fm = p.fontMetrics()
        p.fillRect(8, h-18, fm.horizontalAdvance(self.label)+4, fm.height()+4,
                   QColor(13, 14, 16, 180))
        p.setPen(QPen(QColor(255, 255, 255, 180)))
        p.drawText(10, h-8, self.label)


class TelemetryBar(QWidget):
    def __init__(self, label: str):
        super().__init__()
        self.label = label
        self.value = 0.0
        self.setFixedHeight(20)

    def set_value(self, v: float):
        self.value = max(-1.0, min(1.0, v))
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        label_w, val_w = 48, 44
        bar_x = label_w + 6
        bar_w = w - label_w - val_w - 12
        p.setFont(QFont("Consolas", 9))
        p.setPen(QPen(P.TXT2))
        p.drawText(0, h-4, self.label)
        ty = (h - 4) // 2
        p.fillRect(bar_x, ty, bar_w, 4, P.BG3)
        mid = bar_x + bar_w // 2
        fill_w = int(abs(self.value) * (bar_w // 2))
        fx = mid - fill_w if self.value < 0 else mid
        abs_v = abs(self.value)
        color = P.DANGER if abs_v > 0.8 else (P.WARN if abs_v > 0.5 else P.ACCENT)
        p.fillRect(fx, ty, fill_w, 4, color)
        p.setPen(QPen(P.TXT1))
        val_str = f"{self.value:+.2f}"
        p.drawText(w - p.fontMetrics().horizontalAdvance(val_str), h-4, val_str)


class MiniStat(QFrame):
    def __init__(self, label: str):
        super().__init__()
        self.setStyleSheet("background:#1a1d24; border-radius:7px; border:none;")
        self.setFixedHeight(52)
        ly = QVBoxLayout(self)
        ly.setContentsMargins(10, 6, 10, 6); ly.setSpacing(2)
        lbl = QLabel(label)
        lbl.setStyleSheet("font-family:Consolas; font-size:9px; color:#555a6a; letter-spacing:1px; background:transparent; border:none;")
        self.val_lbl = QLabel("—")
        self.val_lbl.setStyleSheet("font-family:Consolas; font-size:15px; color:#e8eaf0; background:transparent; border:none;")
        ly.addWidget(lbl); ly.addWidget(self.val_lbl)

    def set_value(self, text: str, color: str = "#e8eaf0"):
        self.val_lbl.setText(text)
        self.val_lbl.setStyleSheet(
            f"font-family:Consolas; font-size:15px; color:{color}; background:transparent; border:none;")


class StatusBadge(QLabel):
    def set_active(self):
        self.setText("ACTIVO")
        self.setStyleSheet("background:#0d2b1a; color:#3ecf8e; border:1px solid rgba(62,207,142,.25); border-radius:4px; padding:3px 8px; font-family:Consolas; font-size:9px; letter-spacing:1px; font-weight:bold;")
    def set_locked(self, text="BLOQUEADO"):
        self.setText(text)
        self.setStyleSheet("background:#2a1515; color:#e85555; border:1px solid rgba(232,85,85,.25); border-radius:4px; padding:3px 8px; font-family:Consolas; font-size:9px; letter-spacing:1px; font-weight:bold;")
    def set_ai_modo(self, modo: int):
        if modo == 1:
            self.setText("● BLEND")
            self.setStyleSheet("background:#2a1a05; color:#e87c3a; border:1px solid rgba(232,124,58,.55); border-radius:4px; padding:3px 8px; font-family:Consolas; font-size:9px; letter-spacing:1px; font-weight:bold;")
        elif modo == 2:
            self.setText("▶ OVERRIDE")
            self.setStyleSheet("background:#0d1a2b; color:#7b8cde; border:1px solid rgba(123,140,222,.55); border-radius:4px; padding:3px 8px; font-family:Consolas; font-size:9px; letter-spacing:1px; font-weight:bold;")
        else:
            self.set_active()


class MachineTab(QPushButton):
    STYLE_ON  = "QPushButton { background: rgba(232,124,58,0.10); color: #e87c3a; border: 1px solid rgba(232,124,58,0.55); border-radius: 6px; padding: 5px 0; font-family: Consolas; font-size: 10px; letter-spacing: 1px; }"
    STYLE_OFF = "QPushButton { background: transparent; color: #8b90a0; border: 1px solid #2a2f3a; border-radius: 6px; padding: 5px 0; font-family: Consolas; font-size: 10px; letter-spacing: 1px; } QPushButton:hover { background: #22262f; color: #e8eaf0; }"
    def set_active(self, on: bool):
        self.setStyleSheet(self.STYLE_ON if on else self.STYLE_OFF)


class FaseIndicator(QWidget):
    FASES = ["POSICIONAR", "EXCAVAR", "LEVANTAR", "GIRAR", "DESCARGAR"]
    def __init__(self):
        super().__init__()
        self.fase_actual = 0
        self.setFixedHeight(36)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    def set_fase(self, fase: int):
        self.fase_actual = max(0, min(len(self.FASES) - 1, fase))
        self.update()
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        n = len(self.FASES)
        gap = 3
        slot_w = (w - gap * (n - 1)) // n
        for i, nombre in enumerate(self.FASES):
            x = i * (slot_w + gap)
            if i == self.fase_actual:  bg, text = P.ACCENT, P.TXT1
            elif i < self.fase_actual: bg, text = P.BG3, P.OK
            else:                      bg, text = P.BG2, P.TXT3
            p.setBrush(QBrush(bg)); p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(x, 0, slot_w, h, 4, 4)
            p.setPen(QPen(text))
            p.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
            p.drawText(x + 4, 11, str(i))
            p.setFont(QFont("Consolas", 6))
            fm = p.fontMetrics()
            txt = nombre if fm.horizontalAdvance(nombre) < slot_w - 6 else nombre[:4]
            p.drawText(x + 2, h - 6, txt)


def section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-family:Consolas; font-size:9px; color:#555a6a; letter-spacing:2px; text-transform:uppercase; margin-bottom:2px;")
    return lbl


def make_button(text: str, danger=False) -> QPushButton:
    btn = QPushButton(text)
    if danger:
        style = "QPushButton { background: rgba(232,85,85,0.06); color: #e85555; border: 1px solid rgba(232,85,85,0.35); border-radius: 7px; padding: 8px 12px; font-family: Consolas; font-size: 11px; letter-spacing: 1px; } QPushButton:hover { background: rgba(232,85,85,0.15); } QPushButton:pressed { background: rgba(232,85,85,0.25); }"
    else:
        style = "QPushButton { background: #22262f; color: #e8eaf0; border: 1px solid #2a2f3a; border-radius: 7px; padding: 8px 12px; font-family: Consolas; font-size: 11px; letter-spacing: 1px; } QPushButton:hover { background: #2a2f3a; } QPushButton:pressed { background: #e87c3a; color: white; border-color: #e87c3a; }"
    btn.setStyleSheet(style)
    return btn

class VideoThread(QThread):
    frontal_signal = pyqtSignal(QPixmap)
    trasera_signal = pyqtSignal(QPixmap)
    def __init__(self, get_active_ip_callback):
        super().__init__()
        self.get_active_ip = get_active_ip_callback
        self.frames_limpios = {"frontal": None, "trasera": None}
    def run(self):
        sock_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock_video.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock_video.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
        except: pass
        sock_video.bind(("0.0.0.0", UDP_PORT_VIDEO))
        sock_video.settimeout(0.5)
        while not self.isInterruptionRequested():
            try:
                data, addr = sock_video.recvfrom(65507)
                if addr[0] != self.get_active_ip():
                    continue
                header  = data[0:1]
                payload = data[1:]
                frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                if header == b'R':
                    self.frames_limpios["trasera"] = frame.copy()
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                else:
                    self.frames_limpios["frontal"] = frame.copy()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                pix = QPixmap.fromImage(
                    QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy())
                if header == b'R': self.trasera_signal.emit(pix)
                else:              self.frontal_signal.emit(pix)
            except socket.timeout:
                continue
            except Exception:
                continue
        sock_video.close()

class TelemetriaThread(QThread):
    datos_signal = pyqtSignal(dict)
    def __init__(self, get_active_ip_callback):
        super().__init__()
        self.get_active_ip = get_active_ip_callback
    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(0.3)
        sock.bind(("0.0.0.0", 5007))
        while not self.isInterruptionRequested():
            try:
                data, addr = sock.recvfrom(1024)
                if addr[0] == self.get_active_ip():
                    self.datos_signal.emit(json.loads(data.decode()))
            except: continue
        sock.close()

class GlobalCamThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)
    def __init__(self, cam_index: int = 1):
        super().__init__()
        self.cam_index = cam_index
        self._ultimo_frame = None
    def run(self):
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv2.CAP_PROP_FPS, 15)
        if not cap.isOpened():
            print(f"[CAM GLOBAL] No se pudo abrir índice {self.cam_index}")
            return
        print(f"[CAM GLOBAL] Iniciada en índice {self.cam_index}")
        while not self.isInterruptionRequested():
            ok, frame = cap.read()
            if ok and frame is not None:
                self._ultimo_frame = frame.copy()
                self.frame_signal.emit(frame.copy())
            else:
                self.msleep(50)
        cap.release()

class StopDetector(QThread):
    """
    Corre a 5 Hz. Solo actúa en modo OVERRIDE (ai_modo == 2).
    Trackea la posición del robot por marcador de color en la cámara cenital
    y evalúa condiciones de parada según la fase activa.
    """
    parada_signal   = pyqtSignal(str)    
    tracking_signal = pyqtSignal(bool) 

    TIERRA_FRONTAL_UMBRAL = 0.18
    DESCARGA_AREA_UMBRAL  = 0.10
    RETORNO_DIST_UMBRAL   = 0.15
    CONFIRM_FRAMES        = 8

    def __init__(self):
        super().__init__()
        import threading
        self._lock           = threading.Lock()
        self._frame_frontal  = None
        self._frame_global   = None
        self._fase           = 0
        self._ai_modo        = 0
        self._pos_inicial    = None
        self._home_calibrado = False
        self._robot_pos      = (0.5, 0.5)
        self._contador       = 0
        self._ultimo_motivo  = ""
        self._hsv_lo = np.array([20,  40, 150])
        self._hsv_hi = np.array([40, 255, 255])
        self._track_ok = False

    def actualizar(self, frame_frontal, frame_global, fase, ai_modo):
        with self._lock:
            self._frame_frontal = frame_frontal.copy() if frame_frontal is not None else None
            self._frame_global  = frame_global.copy()  if frame_global  is not None else None
            self._fase    = fase
            self._ai_modo = ai_modo

    def get_robot_pos(self):
        with self._lock:
            return self._robot_pos

    def calibrar_posicion_inicial(self, pos: tuple):
        self._pos_inicial    = pos
        self._home_calibrado = True
        self._contador       = 0
        print(f"[STOP] Posición inicial calibrada: {pos}")

    # ── TRACKER por marcador de color ──────────────────────────
    def trackear_robot(self, frame):
        """Devuelve (x,y) normalizado [0-1] del robot, o None si no lo ve."""
        if frame is None:
            return None
        fh, fw = frame.shape[:2]
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._hsv_lo, self._hsv_hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contornos:
            return None
        c = max(contornos, key=cv2.contourArea)
        if cv2.contourArea(c) < 80:
            return None
        M = cv2.moments(c)
        if M["m00"] == 0:
            return None
        return ((M["m10"] / M["m00"]) / fw, (M["m01"] / M["m00"]) / fh)

    # ── Detecciones de parada ──────────────────────────────────
    def _detectar_tierra_frontal(self, frame) -> bool:
        if frame is None:
            return False
        fh, fw = frame.shape[:2]
        roi = frame[int(fh * 0.55):, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([95, 5, 60]), np.array([135, 80, 180]))
        mask2 = cv2.inRange(hsv, np.array([0, 0, 70]), np.array([180, 30, 160]))
        mask = cv2.bitwise_or(mask1, mask2)
        ratio = cv2.countNonZero(mask) / (roi.shape[0] * roi.shape[1])
        return ratio > self.TIERRA_FRONTAL_UMBRAL

    def _detectar_zona_descarga(self, frame) -> bool:
        if frame is None:
            return False
        fh, fw = frame.shape[:2]
        roi = frame[:, fw // 2:]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_descarga = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 40, 255]))
        mask_tierra = cv2.inRange(hsv, np.array([95, 5, 60]), np.array([135, 80, 180]))
        mask = cv2.bitwise_and(mask_descarga, cv2.bitwise_not(mask_tierra))
        ratio = cv2.countNonZero(mask) / (roi.shape[0] * roi.shape[1])
        return ratio > self.DESCARGA_AREA_UMBRAL

    def _detectar_retorno(self) -> bool:
        if not self._home_calibrado or self._pos_inicial is None:
            return False
        rx, ry = self._robot_pos
        ix, iy = self._pos_inicial
        dist = ((rx - ix) ** 2 + (ry - iy) ** 2) ** 0.5
        return dist < self.RETORNO_DIST_UMBRAL

    def run(self):
        while not self.isInterruptionRequested():
            self.msleep(200)

            with self._lock:
                ai_modo      = self._ai_modo
                fase         = self._fase
                frame_front  = self._frame_frontal
                frame_global = self._frame_global
            pos = self.trackear_robot(frame_global)
            track_ahora = pos is not None
            if track_ahora:
                with self._lock:
                    self._robot_pos = pos
            if track_ahora != self._track_ok:
                self._track_ok = track_ahora
                self.tracking_signal.emit(track_ahora)
            if ai_modo != 2:
                self._contador = 0
                continue

            condicion, motivo = False, ""
            if fase == 1:
                condicion, motivo = self._detectar_tierra_frontal(frame_front), "EXCAVAR"
            elif fase == 3:
                condicion, motivo = self._detectar_zona_descarga(frame_global), "GIRO"
            elif fase == 0:
                condicion, motivo = self._detectar_retorno(), "RETORNO"

            if condicion and motivo:
                self._contador += 1
                if self._contador >= self.CONFIRM_FRAMES:
                    self._contador = 0
                    if motivo != self._ultimo_motivo:
                        self._ultimo_motivo = motivo
                        print(f"[STOP] Condición cumplida: {motivo}")
                        self.parada_signal.emit(motivo)
            else:
                self._contador = 0
                if not condicion:
                    self._ultimo_motivo = ""


# ============================================================
# VENTANA PRINCIPAL
# ============================================================
class TeleopTerminal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MIRAI — Asistencia de Teleoperación")
        self.resize(1440, 860)
        self.setMinimumSize(1100, 700)

        self.machine     = "GRUA"
        self.locked      = True
        self.start_dt    = QDateTime.currentDateTime()
        self.pkt_count   = 0
        self.blink_state = True
        self.fase_actual = 0
        self.telemetria  = {}

        self._trkL = SmoothValue(0.0, 0.15)
        self._trkR = SmoothValue(0.0, 0.15)
        self._arm  = SmoothValue(0.0, 0.10)
        self._lat  = SmoothValue(32.0, 0.10)

        self._gatillo_previo = False
        self._cambio_previo  = False
        self._btn_fase_prev  = False
        self._btn_cam_prev   = False
        self.cam_invertida   = False
        self.ciclo_id        = 0

        self._cmd_actual   = [0.0, 0.0, 0.0, 0.0, 0.0]
        self._frame_global = None
        self._ultima_sug   = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.ai_modo         = 0
        self.blend_alpha     = 0.8
        self.override_umbral = 0.3
        self._btn_blend_prev = False
        self._track_ok       = False

        self.pygame_ok = False
        try:
            pygame.init()
            pygame.joystick.init()
            self.volante   = pygame.joystick.Joystick(0) if pygame.joystick.get_count() > 0 else None
            self.joy_brazo = pygame.joystick.Joystick(1) if pygame.joystick.get_count() > 1 else None
            if self.volante:   self.volante.init()
            if self.joy_brazo: self.joy_brazo.init()
            self.pygame_ok = True
        except Exception as e:
            print(f"Pygame no disponible: {e}")
            self.volante = self.joy_brazo = None

        self._setup_styles()
        self._build_ui()
        self._status_badge.set_locked()

        self._telem_thread = TelemetriaThread(self.get_active_ip)
        self._telem_thread.datos_signal.connect(lambda d: setattr(self, 'telemetria', d))
        self._telem_thread.start()

        self.vid_thread = VideoThread(self.get_active_ip)
        self.vid_thread.frontal_signal.connect(self._rutear_frontal)
        self.vid_thread.trasera_signal.connect(self._rutear_trasera)
        self.vid_thread.start()

        self._inference = MiraiInferenceThread("GRUA")
        self._inference.sugerencia_signal.connect(self._on_sugerencia)
        self._inference.estado_signal.connect(self._on_estado_ia)
        self._inference.start()

        self._global_cam = GlobalCamThread(cam_index=1)
        self._global_cam.frame_signal.connect(self._on_frame_global,
            Qt.ConnectionType.QueuedConnection)
        self._global_cam.start()

        self._stop_detector = StopDetector()
        self._stop_detector.parada_signal.connect(self._on_parada_automatica)
        self._stop_detector.tracking_signal.connect(self._on_tracking_estado)
        self._stop_detector.start()

        self._joy_timer = QTimer(self)
        self._joy_timer.timeout.connect(self._intentar_conectar_joysticks)
        self._joy_timer.start(5000)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(50)

        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._blink_tick)
        self._blink_timer.start(600)

        self._unlock_timer = QTimer(self)
        self._unlock_timer.setSingleShot(True)
        self._unlock_timer.timeout.connect(self._unlock)

        self._overlay_timer = QTimer(self)
        self._overlay_timer.timeout.connect(self._overlay_fade)
        self._overlay_timer.start(80)

    # ── Slots IA ────────────────────────────────────────────────
    def _on_sugerencia(self, cmds: list):
        self._ultima_sug = cmds
        self._ai_overlay.set_sugerencias(cmds)
        self._ai_panel.set_sugerencias(cmds)
        self._ai_panel.set_operador(self._cmd_actual)

    def _on_estado_ia(self, estado: str):
        self._ai_panel.set_estado(estado)
        color = {"OK": "#7b8cde", "SIN_MODELO": "#f5c542", "ERROR": "#e85555"}.get(estado, "#555a6a")
        self._ai_status_lbl.setText(f"IA  {estado}")
        self._ai_status_lbl.setStyleSheet(
            f"font-family:Consolas; font-size:9px; color:{color}; letter-spacing:1px;")

    def _on_tracking_estado(self, ok: bool):
        self._track_ok = ok
        if hasattr(self, "_track_lbl"):
            if ok:
                self._track_lbl.setText("● ROBOT DETECTADO")
                self._track_lbl.setStyleSheet("font-family:Consolas; font-size:9px; color:#3ecf8e; letter-spacing:1px;")
            else:
                self._track_lbl.setText("○ ROBOT NO VISIBLE")
                self._track_lbl.setStyleSheet("font-family:Consolas; font-size:9px; color:#e85555; letter-spacing:1px;")

    def _on_frame_global(self, frame: np.ndarray):
        self._frame_global = frame
        self._stop_detector.actualizar(
            self.vid_thread.frames_limpios.get("frontal"),
            frame, self.fase_actual, self.ai_modo)
        if hasattr(self, "_cam_global_mini"):
            # Dibujar marcador detectado sobre la miniatura
            disp = cv2.resize(frame, (312, 234))
            pos = self._stop_detector.get_robot_pos()
            if self._track_ok and pos:
                cx, cy = int(pos[0] * 312), int(pos[1] * 234)
                cv2.circle(disp, (cx, cy), 8, (62, 207, 142), 2)
                cv2.drawMarker(disp, (cx, cy), (62, 207, 142),
                               cv2.MARKER_CROSS, 12, 1)
            rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            pix = QPixmap.fromImage(
                QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888).copy())
            self._cam_global_mini.set_frame(pix)

    def _on_parada_automatica(self, motivo: str):
        if self.ai_modo != 2:
            return
        self.ai_modo = 0
        self._status_badge.set_ai_modo(0)
        self._ai_overlay.set_modo(0)
        self._send_stop(self.machine)
        print(f"[STOP AUTO] Parada por: {motivo} — OVERRIDE desactivado")

    def calibrar_home(self):
        """Guarda la posición REAL detectada por el tracker como home."""
        pos = self._stop_detector.trackear_robot(self._frame_global)
        if pos is None:
            QMessageBox.warning(self, "Calibrar Home",
                "No se detecta el marcador del robot en la cámara cenital.\n"
                "Verifica que el marcador de color sea visible.")
            print("[HOME] Marcador no detectado")
            return
        self._stop_detector.calibrar_posicion_inicial(pos)
        QMessageBox.information(self, "Calibrar Home",
            f"Posición home guardada:\nx={pos[0]:.3f}  y={pos[1]:.3f}")
        print(f"[HOME] Calibrado en {pos}")

    def _overlay_fade(self):
        self._ai_overlay.update()

    # ── Video ───────────────────────────────────────────────────
    def _rutear_frontal(self, pix):
        target = self._cam_rear if self.cam_invertida else self._cam_main
        target.set_frame(pix)
        frame = self.vid_thread.frames_limpios.get("frontal")
        if frame is not None:
            imu_data = self.telemetria.get("imus", {}) if isinstance(self.telemetria, dict) else {}
            self._inference.actualizar(frame, imu_data, self.fase_actual, self._cmd_actual)

    def _rutear_trasera(self, pix):
        target = self._cam_main if self.cam_invertida else self._cam_rear
        target.set_frame(pix)

    def _toggle_camaras(self):
        self.cam_invertida = not self.cam_invertida
        if self.cam_invertida:
            self._cam_main.label_texto = "CH-02 / TRASERA"
            self._cam_rear.label       = "CH-01 / FRONTAL"
        else:
            self._cam_main.label_texto = "CH-01 / FRONTAL"
            self._cam_rear.label       = "CH-02 / TRASERA"

    # ── Joysticks ───────────────────────────────────────────────
    def _intentar_conectar_joysticks(self):
        try:
            pygame.joystick.quit(); pygame.joystick.init()
            count = pygame.joystick.get_count()
            self.volante   = pygame.joystick.Joystick(0) if count > 0 else None
            self.joy_brazo = pygame.joystick.Joystick(1) if count > 1 else None
            if self.volante:   self.volante.init()
            if self.joy_brazo: self.joy_brazo.init()
            self.pygame_ok = count > 0
        except Exception as e:
            print(f"[JOY] Error: {e}")
            self.pygame_ok = False
            self.volante = self.joy_brazo = None

    def _aplicar_haptic(self, intensidad: float, duracion_ms: int = 300):
        if self.volante:
            try: self.volante.rumble(intensidad, intensidad, duracion_ms)
            except: pass

    def get_active_ip(self):
        return FLOTA[self.machine]["ip"]

    def _setup_styles(self):
        self.setStyleSheet("""
            QMainWindow, QWidget#root { background: #0d0e10; }
            QLabel { color: #e8eaf0; }
            QFrame#sidebar { background: #13151a; border-right: 1px solid #2a2f3a; }
            QFrame#topbar  { background: #13151a; border-bottom: 1px solid #2a2f3a; }
            QFrame#right   { background: #13151a; border-left: 1px solid #2a2f3a; }
            QFrame#divider { background: #2a2f3a; max-height: 1px; }
        """)

    def _build_ui(self):
        root = QWidget(); root.setObjectName("root")
        self.setCentralWidget(root)
        root_ly = QVBoxLayout(root)
        root_ly.setContentsMargins(0, 0, 0, 0); root_ly.setSpacing(0)
        root_ly.addWidget(self._build_topbar())
        body = QWidget()
        body_ly = QHBoxLayout(body)
        body_ly.setContentsMargins(0, 0, 0, 0); body_ly.setSpacing(0)
        body_ly.addWidget(self._build_sidebar())
        body_ly.addWidget(self._build_main(), stretch=1)
        body_ly.addWidget(self._build_right())
        root_ly.addWidget(body, stretch=1)

    def _build_topbar(self) -> QFrame:
        bar = QFrame(); bar.setObjectName("topbar"); bar.setFixedHeight(46)
        ly = QHBoxLayout(bar); ly.setContentsMargins(18, 0, 18, 0); ly.setSpacing(16)
        logo = QLabel("MIRAI")
        logo.setStyleSheet("font-family:Consolas; font-size:13px; font-weight:bold; color:#e87c3a; letter-spacing:3px;")
        ly.addWidget(logo); ly.addWidget(self._sep())
        for dot_attr, text, color in [("_dot_grua", "GRÚA", "#3ecf8e"),
                                       ("_dot_bull", "BULLDOZER", "#2a2f3a")]:
            dot = QLabel(); dot.setFixedSize(8, 8)
            dot.setStyleSheet(f"background:{color}; border-radius:4px;")
            setattr(self, dot_attr, dot)
            row = QWidget(); row_ly = QHBoxLayout(row)
            row_ly.setContentsMargins(0,0,0,0); row_ly.setSpacing(6)
            row_ly.addWidget(dot)
            lbl = QLabel(text); lbl.setStyleSheet("font-family:Consolas; font-size:11px; color:#8b90a0;")
            row_ly.addWidget(lbl); ly.addWidget(row)
        self._ai_status_lbl = QLabel("IA  CARGANDO...")
        self._ai_status_lbl.setStyleSheet("font-family:Consolas; font-size:9px; color:#555a6a; letter-spacing:1px;")
        ly.addWidget(self._sep()); ly.addWidget(self._ai_status_lbl)
        # Estado del tracker
        self._track_lbl = QLabel("○ ROBOT NO VISIBLE")
        self._track_lbl.setStyleSheet("font-family:Consolas; font-size:9px; color:#e85555; letter-spacing:1px;")
        ly.addWidget(self._sep()); ly.addWidget(self._track_lbl)
        ly.addStretch()
        self._uptime_lbl = QLabel("00:00:00")
        self._uptime_lbl.setStyleSheet("font-family:Consolas; font-size:11px; color:#8b90a0; font-variant-numeric:tabular-nums;")
        self._clock_lbl = QLabel("--:--:--")
        self._clock_lbl.setStyleSheet("font-family:Consolas; font-size:11px; color:#e8eaf0; font-variant-numeric:tabular-nums;")
        ly.addWidget(self._uptime_lbl); ly.addWidget(self._sep()); ly.addWidget(self._clock_lbl)
        return bar

    def _build_sidebar(self) -> QFrame:
        sb = QFrame(); sb.setObjectName("sidebar"); sb.setFixedWidth(52)
        ly = QVBoxLayout(sb); ly.setContentsMargins(8,14,8,14); ly.setSpacing(6)
        def sb_btn(symbol, tip, active=False, danger=False):
            b = QPushButton(symbol)
            color = "#e85555" if danger else ("#e87c3a" if active else "#8b90a0")
            bghov = "#2a1515" if danger else "#22262f"
            b.setStyleSheet(
                f"QPushButton {{ background: {'#22262f' if active else 'transparent'}; "
                f"color: {color}; border: {'1px solid #e87c3a' if active else 'none'}; "
                f"border-radius: 8px; font-size: 16px; padding: 0; "
                f"min-width:36px; max-width:36px; min-height:36px; max-height:36px; }}"
                f"QPushButton:hover {{ background: {bghov}; }}")
            b.setToolTip(tip)
            return b
        ly.addWidget(sb_btn("⬡", "Cámara", active=True), alignment=Qt.AlignmentFlag.AlignHCenter)
        ly.addWidget(sb_btn("◎", "Mapa"), alignment=Qt.AlignmentFlag.AlignHCenter)
        ly.addWidget(sb_btn("⎍", "Telemetría"), alignment=Qt.AlignmentFlag.AlignHCenter)
        ly.addWidget(self._hsep())
        ly.addWidget(sb_btn("⚙", "Config"), alignment=Qt.AlignmentFlag.AlignHCenter)
        ly.addStretch()
        exit_btn = sb_btn("⏻", "Salir", danger=True)
        exit_btn.clicked.connect(self.close)
        ly.addWidget(exit_btn, alignment=Qt.AlignmentFlag.AlignHCenter)
        return sb

    def _build_main(self) -> QWidget:
        w = QWidget(); w.setStyleSheet("background: #0d0e10;")
        main_ly = QVBoxLayout(w)
        main_ly.setContentsMargins(0,0,0,0); main_ly.setSpacing(0)
        self.cam_container = QWidget()
        self.cam_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.cam_container.setStyleSheet("background: #000000; border: none;")
        cont_ly = QVBoxLayout(self.cam_container)
        cont_ly.setContentsMargins(0,0,0,0); cont_ly.setSpacing(0)
        self._cam_main = CameraView()
        self._cam_main.setStyleSheet("border: none; background: #0d0e10;")
        cont_ly.addWidget(self._cam_main, stretch=1)
        self._cam_rear = MiniCamera("CH-02 / TRASERA", show_nosignal=False)
        self._cam_rear.setFixedSize(240, 160)
        self._cam_rear.setStyleSheet("MiniCamera { background: #1a1d24; border: 2px solid #2a2f3a; border-radius: 8px; }")
        self._cam_rear.setParent(self.cam_container)
        self._cam_rear.raise_()
        self._ai_overlay = AIOverlayWidget(self.cam_container)
        self._ai_overlay.raise_()
        main_ly.addWidget(self.cam_container, stretch=1)
        main_ly.addWidget(self._create_hud_bar())
        self.cam_container.installEventFilter(self)
        return w

    def _create_hud_bar(self) -> QWidget:
        bar = QWidget(); bar.setFixedHeight(42)
        bar.setStyleSheet("""
            QWidget { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                          stop:0 #13151a,stop:1 #0d0e10);
                      border-top: 1px solid #2a2f3a; }
            QLabel  { font-family:'Consolas',monospace; font-size:12px;
                      color:#e8eaf0; background:transparent; }
        """)
        ly = QHBoxLayout(bar); ly.setContentsMargins(16,0,16,0); ly.setSpacing(24)
        def hud_pair(lbl_txt, val_txt):
            row = QWidget(); row_ly = QHBoxLayout(row)
            row_ly.setContentsMargins(0,0,0,0); row_ly.setSpacing(6)
            lbl = QLabel(lbl_txt); lbl.setStyleSheet("color:#555a6a;")
            val = QLabel(val_txt); val.setStyleSheet("color:#e87c3a; font-weight:bold;")
            row_ly.addWidget(lbl); row_ly.addWidget(val)
            return row, val
        r1, self.hud_speed  = hud_pair("SPEED",  "—")
        r2, self.hud_height = hud_pair("HEIGHT", "—")
        ly.addWidget(r1); ly.addWidget(r2)
        ly.addStretch()
        r3, self.hud_flight_time = hud_pair("UPTIME", "00:00")
        self.hud_flight_time.setStyleSheet("color:#3ecf8e; font-weight:bold;")
        ly.addWidget(r3)
        return bar

    def eventFilter(self, obj, event):
        if obj == self.cam_container and event.type() == event.Type.Resize:
            self._position_overlays()
        return super().eventFilter(obj, event)

    def _position_overlays(self):
        if hasattr(self, '_cam_rear') and self.cam_container:
            margin = 16
            pw, ph = self.cam_container.width(), self.cam_container.height()
            rw, rh = self._cam_rear.width(), self._cam_rear.height()
            self._cam_rear.setGeometry(pw-rw-margin, ph-rh-margin, rw, rh)
        if hasattr(self, '_ai_overlay') and self.cam_container:
            pw, ph = self.cam_container.width(), self.cam_container.height()
            self._ai_overlay.setGeometry(0, 0, pw, ph)

    def showEvent(self, event):
        super().showEvent(event)
        self._position_overlays()

    def _build_right(self) -> QFrame:
        panel = QFrame(); panel.setObjectName("right"); panel.setFixedWidth(340)
        ly = QVBoxLayout(panel); ly.setContentsMargins(0,0,0,0); ly.setSpacing(0)
        veh = self._section_widget(); vly = veh.layout()
        vly.addWidget(section_label("UNIDAD ACTIVA"))
        header_row = QHBoxLayout()
        self._veh_name = QLabel("GRÚA")
        self._veh_name.setStyleSheet("font-family:Consolas; font-size:16px; font-weight:bold; letter-spacing:2px; color:#e8eaf0;")
        self._status_badge = StatusBadge()
        header_row.addWidget(self._veh_name); header_row.addStretch()
        header_row.addWidget(self._status_badge); vly.addLayout(header_row)
        tabs_row = QHBoxLayout(); tabs_row.setSpacing(6)
        self._tab_grua = MachineTab("GRÚA"); self._tab_bull = MachineTab("BULLDOZER")
        self._tab_grua.set_active(True); self._tab_bull.set_active(False)
        self._tab_grua.clicked.connect(lambda: self._switch_machine("GRUA"))
        self._tab_bull.clicked.connect(lambda: self._switch_machine("BULLDOZER"))
        tabs_row.addWidget(self._tab_grua); tabs_row.addWidget(self._tab_bull)
        vly.addLayout(tabs_row); ly.addWidget(veh); ly.addWidget(self._divider())

        stats_w = QWidget(); stats_w.setStyleSheet("background:transparent;")
        stats_grid = QGridLayout(stats_w); stats_grid.setContentsMargins(14,12,14,12); stats_grid.setSpacing(6)
        self._stat_lat = MiniStat("LATENCIA"); self._stat_sig = MiniStat("SEÑAL")
        self._stat_fps = MiniStat("FPS"); self._stat_pkt = MiniStat("PAQUETES")
        self._stat_sig.set_value("94%", "#3ecf8e"); self._stat_fps.set_value("60")
        stats_grid.addWidget(self._stat_lat, 0, 0); stats_grid.addWidget(self._stat_sig, 0, 1)
        stats_grid.addWidget(self._stat_fps, 1, 0); stats_grid.addWidget(self._stat_pkt, 1, 1)
        ly.addWidget(stats_w); ly.addWidget(self._divider())

        telem = self._section_widget(); tly = telem.layout()
        tly.addWidget(section_label("TELEMETRÍA — OPERADOR"))
        self._bar_trkL = TelemetryBar("TRK-L"); self._bar_trkR = TelemetryBar("TRK-R")
        self._bar_arm = TelemetryBar("BRAZO")
        for bar in [self._bar_trkL, self._bar_trkR, self._bar_arm]:
            tly.addWidget(bar)
        self._imu_label = QLabel("IMU: —")
        self._imu_label.setStyleSheet("font-family:Consolas; font-size:11px; color:#8b90a0; margin-top:4px;")
        tly.addWidget(self._imu_label)
        self._fase_indicator = FaseIndicator()
        tly.addWidget(self._fase_indicator)
        ly.addWidget(telem); ly.addWidget(self._divider())

        ai_sec = self._section_widget(); ai_ly = ai_sec.layout()
        ai_ly.addWidget(section_label("ASISTENCIA MIRAI — SUGERENCIAS IA"))
        self._ai_panel = AIAssistPanel()
        ai_ly.addWidget(self._ai_panel)
        hint = QLabel("azul = IA sugiere  |  naranja = comando actual")
        hint.setStyleSheet("font-family:Consolas; font-size:8px; color:#555a6a; margin-top:2px;")
        ai_ly.addWidget(hint)
        ly.addWidget(ai_sec); ly.addWidget(self._divider())

        cam_sec = self._section_widget(); cam_ly = cam_sec.layout()
        cam_ly.addWidget(section_label("CÁMARA GLOBAL — TRACKING"))
        self._cam_global_mini = MiniCamera("CENITAL", show_nosignal=True)
        self._cam_global_mini.setFixedSize(312, 180)
        cam_ly.addWidget(self._cam_global_mini, alignment=Qt.AlignmentFlag.AlignHCenter)
        ly.addWidget(cam_sec); ly.addWidget(self._divider())

        act = self._section_widget(); aly = act.layout()
        aly.addWidget(section_label("ACCIONES"))
        btn_home = make_button("📍 CALIBRAR HOME")
        btn_home.clicked.connect(self.calibrar_home)
        aly.addWidget(btn_home)
        btn_rth = make_button("RETORNO (RTH)", danger=True)
        btn_rth.clicked.connect(self._confirm_rth)
        aly.addWidget(btn_rth)
        ly.addWidget(act); ly.addWidget(self._divider())
        ly.addStretch()
        return panel

    def _section_widget(self) -> QWidget:
        w = QWidget(); w.setStyleSheet("background:transparent;")
        ly = QVBoxLayout(w); ly.setContentsMargins(14,12,14,12); ly.setSpacing(8)
        return w

    def _sep(self) -> QFrame:
        f = QFrame(); f.setFrameShape(QFrame.Shape.VLine)
        f.setStyleSheet("background:#2a2f3a; max-width:1px; border:none;")
        return f

    def _hsep(self) -> QFrame:
        f = QFrame(); f.setFrameShape(QFrame.Shape.HLine)
        f.setStyleSheet("background:#2a2f3a; max-height:1px; border:none; margin:2px 4px;")
        return f

    def _divider(self) -> QFrame:
        f = QFrame(); f.setObjectName("divider"); f.setFixedHeight(1)
        return f

    def _send_stop(self, machine):
        ip, port = FLOTA[machine]["ip"], FLOTA[machine]["port"]
        try:
            if machine == "GRUA":
                sock_send.sendto(json.dumps({
                    "brazo_superior": 0, "pala": 0, "pala_rot": 0,
                    "rotacion_cabina": 0, "oruga_izq": 0, "oruga_der": 0,
                    "bloqueo": True
                }).encode(), (ip, port))
            elif machine == "BULLDOZER":
                sock_send.sendto("0".encode(), (ip, port))
        except Exception as e:
            print(f"Stop error: {e}")

    def _switch_machine(self, name: str):
        if self.machine == name: return
        self._inference.cambiar_maquina(name)
        self._ai_panel.set_machine(name)
        try:
            sock_send.sendto("VIDEO:OFF".encode(),
                (FLOTA[self.machine]["ip"], FLOTA[self.machine]["port"]))
        except: pass
        self.machine = name
        self.locked = True
        self._veh_name.setText(name)
        self._status_badge.set_locked()
        self._tab_grua.set_active(name == "GRUA")
        self._tab_bull.set_active(name == "BULLDOZER")
        grua_color = "#3ecf8e" if name == "GRUA" else "#2a2f3a"
        bull_color = "#3ecf8e" if name == "BULLDOZER" else "#2a2f3a"
        self._dot_grua.setStyleSheet(f"background:{grua_color}; border-radius:4px;")
        self._dot_bull.setStyleSheet(f"background:{bull_color}; border-radius:4px;")
        try:
            sock_send.sendto("VIDEO:ON".encode(), (FLOTA[name]["ip"], FLOTA[name]["port"]))
        except: pass
        self._unlock_timer.start(1500)

    def _unlock(self):
        self.locked = False
        self._status_badge.set_ai_modo(self.ai_modo)

    def _confirm_rth(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("Confirmar RTH")
        msg.setText("¿Iniciar retorno al punto home?\nSe detendrán todos los movimientos.")
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
        msg.setDefaultButton(QMessageBox.StandardButton.Cancel)
        msg.setStyleSheet("QMessageBox { background:#13151a; color:#e8eaf0; font-family:Consolas; } QPushButton { background:#22262f; color:#e8eaf0; border:1px solid #2a2f3a; border-radius:6px; padding:6px 18px; font-family:Consolas; } QPushButton:hover { background:#2a2f3a; }")
        if msg.exec() == QMessageBox.StandardButton.Yes:
            self.locked = True
            self._status_badge.set_locked("RTH ACTIVO")
            self._send_stop(self.machine)

    def _blink_tick(self):
        self.blink_state = not self.blink_state
        self._cam_main._blink = self.blink_state
        self._cam_main.update()

    def _tick(self):
        now = QDateTime.currentDateTime()
        secs = self.start_dt.secsTo(now)
        self._clock_lbl.setText(now.toString("HH:mm:ss"))
        self._uptime_lbl.setText(f"{secs//3600:02d}:{(secs%3600)//60:02d}:{secs%60:02d}")

        if self.pygame_ok:
            try: pygame.event.pump()
            except: self.pygame_ok = False

        try:
            if self.joy_brazo:
                btn0 = self.joy_brazo.get_button(0)
                if btn0 and not self._gatillo_previo:
                    self.locked = not self.locked
                    if self.locked:
                        self._status_badge.set_locked()
                        self._send_stop(self.machine)
                    else:
                        self._status_badge.set_active()
                self._gatillo_previo = btn0

                btn2 = self.joy_brazo.get_button(2)
                if btn2 and not self._cambio_previo:
                    self._switch_machine("BULLDOZER" if self.machine == "GRUA" else "GRUA")
                self._cambio_previo = btn2

                btn_fase = self.joy_brazo.get_button(3)
                if btn_fase and not self._btn_fase_prev:
                    self.fase_actual = (self.fase_actual + 1) % 5
                    self._fase_indicator.set_fase(self.fase_actual)
                self._btn_fase_prev = btn_fase

                btn_cam = self.joy_brazo.get_button(1)
                if btn_cam and not self._btn_cam_prev:
                    self._toggle_camaras()
                self._btn_cam_prev = btn_cam

                btn_blend = self.joy_brazo.get_button(6)
                if btn_blend and not self._btn_blend_prev:
                    self.ai_modo = (self.ai_modo + 1) % 3
                    if not self.locked:
                        self._status_badge.set_ai_modo(self.ai_modo)
                    self._ai_overlay.set_modo(self.ai_modo)
                    nombres = ["OFF", "BLEND", "OVERRIDE"]
                    print(f"[IA MODO] {nombres[self.ai_modo]}")
                self._btn_blend_prev = btn_blend

            v_izq = v_der = 0.0
            v_brazo = v_pala = v_rot = v_esl2 = 0.0
            tl = tr = ar = 0.0

            if not self.locked:
                if self.volante:
                    giro = self.volante.get_axis(0) * -1.0
                    gas_raw = self.volante.get_axis(1)
                    freno_raw = self.volante.get_axis(2)
                    gas = 1.0 if gas_raw == 0.0 else (gas_raw - 1) / -2
                    freno = 1.0 if freno_raw == 0.0 else (freno_raw - 1) / -2
                    impulso = gas - freno
                    v_izq = float(np.clip(impulso + giro, -1.0, 1.0))
                    v_der = float(np.clip(impulso - giro, -1.0, 1.0))

                tl = self._trkL.update(v_izq)
                tr = self._trkR.update(v_der)
                ip_dest = FLOTA[self.machine]["ip"]
                port_dest = FLOTA[self.machine]["port"]

                def _aplicar_modo(op_val: float, ai_idx: int) -> float:
                    sug = self._ultima_sug[ai_idx] if ai_idx < len(self._ultima_sug) else 0.0
                    if self.ai_modo == 0:
                        return op_val
                    elif self.ai_modo == 1:
                        if abs(op_val) > 0.1:
                            return op_val
                        return float(np.clip(
                            self.blend_alpha * op_val + (1 - self.blend_alpha) * sug, -1.0, 1.0))
                    else:
                        if abs(op_val) > self.override_umbral:
                            return op_val
                        return float(np.clip(sug, -1.0, 1.0))

                if self.machine == "GRUA":
                    if self.joy_brazo:
                        v_brazo = round(-self.joy_brazo.get_axis(1), 2)
                        v_pala = round(self.joy_brazo.get_axis(0), 2)
                        try:
                            hat = self.joy_brazo.get_hat(0)
                            v_rot = float(hat[0])
                            v_esl2 = float(hat[1])
                        except: pass
                    b_tl = _aplicar_modo(tl, 0)
                    b_tr = _aplicar_modo(tr, 1)
                    b_brazo = _aplicar_modo(v_brazo, 2)
                    b_pala = _aplicar_modo(v_pala, 3)
                    b_rot = _aplicar_modo(v_rot, 4)
                    paquete = {
                        "brazo_superior": round(b_brazo, 2), "pala": v_esl2,
                        "pala_rot": round(b_pala, 2), "rotacion_cabina": round(b_rot, 2),
                        "oruga_izq": round(b_tl, 2), "oruga_der": round(b_tr, 2),
                        "bloqueo": False
                    }
                    sock_send.sendto(json.dumps(paquete).encode(), (ip_dest, port_dest))
                    tl, tr, v_brazo, v_pala, v_rot = b_tl, b_tr, b_brazo, b_pala, b_rot

                elif self.machine == "BULLDOZER":
                    v_pala_bd = 0.0
                    if self.joy_brazo:
                        raw = self.joy_brazo.get_axis(1)
                        if abs(raw) < 0.20: raw = 0.0
                        if raw < 0: raw *= 1.4
                        v_pala_bd = round(float(np.clip(raw, -1.0, 1.0)), 2)
                    ar = self._arm.update(v_pala_bd)
                    b_tr_bd = _aplicar_modo(-tr, 0)
                    b_tl_bd = _aplicar_modo(-tl, 1)
                    b_ar = _aplicar_modo(ar, 2)
                    sock_send.sendto(f"ORUGAS:{round(b_tr_bd,2)}:{round(b_tl_bd,2)}".encode(),
                                     (ip_dest, port_dest))
                    sock_send.sendto(f"PALA:{round(b_ar,2)}".encode(), (ip_dest, port_dest))
                    ar = b_ar

                self._cmd_actual = [round(tl,2), round(tr,2), v_brazo, v_pala, v_rot]
                self._bar_trkL.set_value(tl)
                self._bar_trkR.set_value(tr)
                self._bar_arm.set_value(v_brazo if self.machine == "GRUA" else ar)
                self._ai_panel.set_operador(self._cmd_actual)
            else:
                self._cmd_actual = [0.0, 0.0, 0.0, 0.0, 0.0]
                if self.machine == "GRUA":
                    sock_send.sendto(json.dumps({
                        "brazo_superior": 0, "pala": 0, "pala_rot": 0,
                        "rotacion_cabina": 0, "oruga_izq": 0, "oruga_der": 0,
                        "bloqueo": True
                    }).encode(), (FLOTA[self.machine]["ip"], FLOTA[self.machine]["port"]))
                self._bar_trkL.set_value(0.0)
                self._bar_trkR.set_value(0.0)
                self._bar_arm.set_value(0.0)
        except Exception:
            import traceback; traceback.print_exc()

        self._cam_main.latency = round(self._lat.update(random.uniform(22, 80)))
        self._cam_main.fps = random.randint(58, 62)
        self._cam_main.update()
        lat = self._cam_main.latency
        self._stat_lat.set_value(f"{lat}ms",
            "#3ecf8e" if lat < 50 else ("#f5c542" if lat < 100 else "#e85555"))
        self._stat_fps.set_value(str(self._cam_main.fps))
        self.pkt_count += 1
        self._stat_pkt.set_value(str(self.pkt_count), "#8b90a0")

        try:
            def _ay(d, key):
                v = d.get(key, 0)
                return v.get("ay", 0.0) if isinstance(v, dict) else float(v)
            imus = self.telemetria.get("imus", {}) if isinstance(self.telemetria, dict) else {}
            self._imu_label.setText(
                f"Pala:{_ay(imus,'imu_pala'):.1f}°  "
                f"Esl1:{_ay(imus,'imu_eslabon1'):.1f}°  "
                f"Cab:{_ay(imus,'imu_cabina'):.1f}°")
        except RuntimeError: pass

        self.hud_flight_time.setText(f"{int(secs//60):02d}:{int(secs%60):02d}")

    def closeEvent(self, e):
        for timer in [self._timer, self._blink_timer, self._unlock_timer,
                      self._joy_timer, self._overlay_timer]:
            timer.stop()
        for hilo in [self._telem_thread, self.vid_thread, self._inference,
                     self._global_cam, self._stop_detector]:
            if hilo.isRunning():
                hilo.requestInterruption()
                hilo.wait(2000)
        self._send_stop(self.machine)
        if self.pygame_ok: pygame.quit()
        e.accept()


# ============================================================
if __name__ == "__main__":
    app = QApplication.instance() or QApplication(sys.argv)
    window = TeleopTerminal()
    window.show()
    app.exec()