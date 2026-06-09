import socket
import json
import cv2
import threading
import time
from gpiozero import PWMOutputDevice, DigitalOutputDevice
from smbus2 import SMBus
from mpu6050 import mpu6050

# ==========================================
# 1. CONFIGURACIÓN DE PINES (MOTORES)
# ==========================================
motores = {
    "esl1":  {"pwm": PWMOutputDevice(17), "in1": DigitalOutputDevice(27), "in2": DigitalOutputDevice(22)},
    "esl2":  {"pwm": PWMOutputDevice(15), "in1": DigitalOutputDevice(16), "in2": DigitalOutputDevice(20)},
    "pala":  {"pwm": PWMOutputDevice(4),  "in1": DigitalOutputDevice(19), "in2": DigitalOutputDevice(26)},
    "rot":   {"pwm": PWMOutputDevice(23), "in1": DigitalOutputDevice(24), "in2": DigitalOutputDevice(25)},
    "o_izq": {"pwm": PWMOutputDevice(12), "in1": DigitalOutputDevice(21), "in2": DigitalOutputDevice(9)},
    "o_der": {"pwm": PWMOutputDevice(13), "in1": DigitalOutputDevice(5),  "in2": DigitalOutputDevice(6)}
}

# ==========================================
# 2. CONFIGURACIÓN DE RED
# ==========================================
LAPTOP_IP        = "192.168.10.200"
UDP_PORT_CONTROL = 5005
UDP_PORT_VIDEO   = 5006
UDP_PORT_TELEM   = 5007   # Puerto dedicado para telemetría → laptop

sock_control = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_control.bind(("0.0.0.0", UDP_PORT_CONTROL))

sock_telem = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # solo envío

enviar_video = True

# ==========================================
# 3. CONFIGURACIÓN DE IMUS (MULTIPLEXOR)
# ==========================================
bus           = SMBus(1)
DIRECCION_IMU = 0x68
_imu_lock     = threading.Lock()

def seleccionar_canal(c):
    try:
        bus.write_byte(0x70, 1 << c)
        time.sleep(0.005)
    except:
        pass

canales_activos = [2, 3, 4, 5]
sensores        = {}
mapeo_imus      = {
    5: "imu_cabina",
    4: "imu_eslabon1",
    3: "imu_eslabon2",
    2: "imu_pala"
}

print("[SISTEMA] Inicializando matriz de IMUs...")
for canal in canales_activos:
    seleccionar_canal(canal)
    try:
        sensores[canal] = mpu6050(DIRECCION_IMU)
        bus.write_byte_data(DIRECCION_IMU, 0x6B, 0x00)
        print(f" -> IMU Canal {canal}: OK")
    except:
        sensores[canal] = None
        print(f" -> IMU Canal {canal}: DESCONECTADO")

# Estado compartido de telemetría (hilo IMU escribe, hilo control lee)
telemetria_actual = {
    "imu_cabina":   {"ax": 0.0, "ay": 0.0, "az": 0.0, "gx": 0.0, "gy": 0.0, "gz": 0.0},
    "imu_eslabon1": {"ax": 0.0, "ay": 0.0, "az": 0.0, "gx": 0.0, "gy": 0.0, "gz": 0.0},
    "imu_eslabon2": {"ax": 0.0, "ay": 0.0, "az": 0.0, "gx": 0.0, "gy": 0.0, "gz": 0.0},
    "imu_pala":     {"ax": 0.0, "ay": 0.0, "az": 0.0, "gx": 0.0, "gy": 0.0, "gz": 0.0},
}

# ==========================================
# 4. HILO DEDICADO PARA IMUS  ← NUEVO
# ==========================================
def hilo_imus():
    """Lee las 4 IMUs continuamente y envía telemetría a la laptop
    sin bloquear el bucle de control."""
    while True:
        datos = {}
        for canal, nombre in mapeo_imus.items():
            sensor = sensores.get(canal)
            if sensor is None:
                continue
            try:
                with _imu_lock:
                    seleccionar_canal(canal)
                    accel = sensor.get_accel_data()
                    ax, ay, az = accel['x'], accel['y'], accel['z']

                    if ax == 0.0 and ay == 0.0 and az == 0.0:
                        bus.write_byte_data(DIRECCION_IMU, 0x6B, 0x00)
                        continue

                    gyro = sensor.get_gyro_data()
                    datos[nombre] = {
                        "ax": round(ax, 3), "ay": round(ay, 3), "az": round(az, 3),
                        "gx": round(gyro['x'], 3), "gy": round(gyro['y'], 3), "gz": round(gyro['z'], 3),
                    }
            except:
                pass

        if datos:
            with _imu_lock:
                telemetria_actual.update(datos)
            try:
                payload = json.dumps({"imus": telemetria_actual}).encode()
                sock_telem.sendto(payload, (LAPTOP_IP, UDP_PORT_TELEM))
            except:
                pass

        time.sleep(0.05)   # 20 Hz — más que suficiente para telemetría

# ==========================================
# 5. FUNCIÓN DE MOVIMIENTO DE MOTORES
# ==========================================
def accionar_motor(m, velocidad):
    zona_muerta = 0.05
    val = max(-1.0, min(1.0, velocidad))
    if val > zona_muerta:
        m["in1"].on()
        m["in2"].off()
        m["pwm"].value = val
    elif val < -zona_muerta:
        m["in1"].off()
        m["in2"].on()
        m["pwm"].value = abs(val)
    else:
        m["in1"].off()
        m["in2"].off()
        m["pwm"].value = 0

# ==========================================
# 6. MOTOR DE STREAMING DE VIDEO
# ==========================================
def stream_camera(cam_id, header_byte, target_ip, port, w=320, h=240):
    global enviar_video
    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 15)
    sock_v = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        if enviar_video:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (w, h))
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
                if len(buf) < 65507:
                    sock_v.sendto(header_byte + buf.tobytes(), (target_ip, port))
        else:
            cap.grab()
        time.sleep(0.02)

def iniciar_streaming():
    print(f"[VIDEO] Iniciando cámaras hacia {LAPTOP_IP}...")
    threading.Thread(target=stream_camera, args=(0, b'F', LAPTOP_IP, UDP_PORT_VIDEO, 480, 320), daemon=True).start()
    threading.Thread(target=stream_camera, args=(2, b'R', LAPTOP_IP, UDP_PORT_VIDEO, 320, 240), daemon=True).start()

# ==========================================
# 7. BUCLE PRINCIPAL DE CONTROL
# ==========================================
def main():
    global enviar_video
    print("=========================================")
    print("   SISTEMA MIRAI ONLINE - RASPBERRY PI   ")
    print("=========================================")

    iniciar_streaming()

    # Hilo IMU completamente separado del control
    threading.Thread(target=hilo_imus, daemon=True).start()
    print(f"[TELEM] Enviando telemetría a {LAPTOP_IP}:{UDP_PORT_TELEM}")

    while True:
        try:
            data, addr = sock_control.recvfrom(1024)
            mensaje    = data.decode('utf-8')

            # --- CASO 1: MOVIMIENTO (JSON) ---
            if mensaje.startswith("{"):
                cmds = json.loads(mensaje)

                if not cmds.get("bloqueo", True):
                    accionar_motor(motores["esl1"],  cmds.get("brazo_superior",   0))
                    accionar_motor(motores["esl2"],  cmds.get("pala",             0))
                    accionar_motor(motores["pala"],  cmds.get("pala_rot",         0))
                    accionar_motor(motores["rot"],   cmds.get("rotacion_cabina",  0))
                    accionar_motor(motores["o_izq"], cmds.get("oruga_izq",        0))
                    accionar_motor(motores["o_der"], cmds.get("oruga_der",        0))
                else:
                    for m in motores.values():
                        accionar_motor(m, 0)

            # --- CASO 2: COMANDOS DE SISTEMA ---
            elif mensaje.startswith("VIDEO:"):
                enviar_video = "ON" in mensaje

        except json.JSONDecodeError:
            pass
        except Exception:
            pass

# ==========================================
# ARRANQUE DEL SCRIPT
# ==========================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[APAGADO] Deteniendo motores...")
        for m in motores.values():
            m["pwm"].value = 0
            m["in1"].off()
            m["in2"].off()