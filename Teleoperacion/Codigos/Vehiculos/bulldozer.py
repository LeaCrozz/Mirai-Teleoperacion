import socket
import json
import cv2
import threading
import time
from gpiozero import PWMOutputDevice, Motor, DigitalOutputDevice
from smbus2 import SMBus
from mpu6050 import mpu6050

# ==========================================
# 1. RED
# ==========================================
LAPTOP_IP = "192.168.10.200"
UDP_PORT_CONTROL = 5005
UDP_PORT_VIDEO    = 5006
enviar_video = True

# ==========================================
# 2. HARDWARE
# ==========================================
pwm_izq  = PWMOutputDevice(12, frequency=2000)
motor_izq = Motor(forward=17, backward=27, pwm=pwm_izq)
pwm_der  = PWMOutputDevice(13, frequency=2000)
motor_der = Motor(forward=24, backward=23, pwm=pwm_der)

pala_en  = PWMOutputDevice(19, frequency=1000)
pala_in3 = DigitalOutputDevice(25)
pala_in4 = DigitalOutputDevice(16)

def mover_pala(valor):
    potencia = abs(valor)
    if potencia < 0.10:
        pala_in3.off(); pala_in4.off(); pala_en.value = 0
        return
    if valor < 0:
        pala_in3.off(); pala_in4.on()
        pala_en.value = min(1.0, potencia * 1.2)  # boost subida
    else:
        pala_in3.on(); pala_in4.off()
        pala_en.value = potencia

def mover_orugas(izq, der):
    if izq > 0:   motor_izq.forward(izq)
    elif izq < 0: motor_izq.backward(abs(izq))
    else:         motor_izq.stop()
    if der > 0:   motor_der.forward(der)
    elif der < 0: motor_der.backward(abs(der))
    else:         motor_der.stop()

def stop_all():
    motor_izq.stop(); motor_der.stop(); mover_pala(0)

# ==========================================
# 3. IMUs 6D — dos sensores por dirección I2C
# ==========================================
DIR_IMU_CHASIS = 0x68
DIR_IMU_PALA   = 0x69
bus_bd = SMBus(1)
sensores_bd = {}

print("[BULLDOZER] Inicializando IMUs 6D...")
for nombre, dir_i2c in [("imu_chasis", DIR_IMU_CHASIS), ("imu_pala", DIR_IMU_PALA)]:
    try:
        s = mpu6050(dir_i2c)
        bus_bd.write_byte_data(dir_i2c, 0x6B, 0x00)  # despertar
        sensores_bd[nombre] = (s, dir_i2c)
        print(f" -> {nombre} (0x{dir_i2c:02X}): OK")
    except:
        sensores_bd[nombre] = None
        print(f" -> {nombre} (0x{dir_i2c:02X}): DESCONECTADO")

IMU_VACIO = {"ax":0.0,"ay":0.0,"az":0.0,"gx":0.0,"gy":0.0,"gz":0.0}

def leer_imus_6d() -> dict:
    resultado = {"imu_chasis": IMU_VACIO.copy(), "imu_pala": IMU_VACIO.copy()}
    for nombre, datos in sensores_bd.items():
        if datos is None:
            continue
        sensor, dir_i2c = datos
        try:
            accel = sensor.get_accel_data()
            ax, ay, az = accel['x'], accel['y'], accel['z']
            if ax == 0.0 and ay == 0.0 and az == 0.0:
                bus_bd.write_byte_data(dir_i2c, 0x6B, 0x00)
            else:
                gyro = sensor.get_gyro_data()
                resultado[nombre] = {
                    "ax": round(ax,  3), "ay": round(ay,  3), "az": round(az,  3),
                    "gx": round(gyro['x'], 3), "gy": round(gyro['y'], 3), "gz": round(gyro['z'], 3),
                }
        except:
            pass
    return resultado

# ==========================================
# 4. VIDEO
# ==========================================
def stream_camera(cam_id, header_byte, target_ip, port, w=320, h=240):
    global enviar_video
    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
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
    threading.Thread(target=stream_camera,
        args=(0, b'F', LAPTOP_IP, UDP_PORT_VIDEO, 640, 480), daemon=True).start()

# ==========================================
# 5. BUCLE PRINCIPAL
# ==========================================
LAPTOP_IP_TELEM = "192.168.10.200"
UDP_PORT_TELEM  = 5007
sock_telem_bd   = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def hilo_imus_bd():
    while True:
        try:
            imus = leer_imus_6d()
            sock_telem_bd.sendto(
                json.dumps({"imus": imus}).encode(),
                (LAPTOP_IP_TELEM, UDP_PORT_TELEM))
        except:
            pass
        time.sleep(0.05)  # 20 Hz

def main():
    global enviar_video
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT_CONTROL))
    print("🚜 Bulldozer Online — IMUs 6D activos")
    iniciar_streaming()

    ultimo_addr = None

    threading.Thread(target=hilo_imus_bd, daemon=True).start()
    print("[TELEM] Enviando IMUs al puerto 5007")

    try:
        while True:
            data, addr = sock.recvfrom(1024)
            ultimo_addr = addr
            msg = data.decode().strip()

            if msg.startswith("PALA:"):
                _, val = msg.split(":")
                mover_pala(float(val))

            elif msg.startswith("ORUGAS:"):
                _, izq, der = msg.split(":")
                mover_orugas(float(izq), float(der))

            elif msg.startswith("VIDEO:"):
                enviar_video = "ON" in msg

            elif msg == "0":
                stop_all()
            pass
            # Enviar telemetría 6D tras cada comando
            # if ultimo_addr:
            #     imus = leer_imus_6d()
            #     sock.sendto(json.dumps({"imus": imus}).encode(), ultimo_addr)

    except KeyboardInterrupt:
        print("\n Apagando...")
    finally:
        stop_all()
        sock.close()

if __name__ == "__main__":
    main()