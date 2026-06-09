# MIRAI — Sistema de Teleoperación Asistida

Sistema de teleoperación para maquinaria de construcción a escala (excavadora y bulldozer) con asistencia por aprendizaje por imitación. Una laptop actúa como estación de control y cada vehículo lleva una Raspberry Pi que ejecuta el video, los sensores y el control de motores.

---

## Arquitectura

```
   [ Laptop / Estación de control ]
   - Terminal de teleoperación (laptop.py)
   - Joystick / volante
   - Modelo  
   - Cámara cenital USB (tracking de posición)
          │
          │  WiFi / red local (UDP)
          │
   ┌──────┴───────┐
   │              │
[ RPi excabadora ]   [ RPi BULLDOZER ]
- Video         - Video
- IMUs          - IMUs
- Motores       - Motores
```

La laptop y las Raspberry deben estar en la **misma red local**.

---

## Requisitos

**En la laptop:**
- Python 3.10+
- GPU NVIDIA recomendada 
- Paquetes: `torch`, `torchvision`, `opencv-python`, `numpy`, `pillow`, `pygame`, `PyQt6`
- El modelo entrenado `mejor_modelo.pth` en la carpeta de modelos 
- Una cámara cenital USB que vea toda el área de trabajo
- Joystick/volante conectados

**En cada Raspberry Pi:**
- Python 3 y las dependencias de los scripts del vehículo
- IP conocida (deben coincidir con las del script de la laptop)

---

## IPs de la flota

Verifica que las IPs en laptop.py coincidan con las reales de los Raspberry:

```python
FLOTA = {
    "GRUA":      {"ip": "192.168.10.187", "port": 5005},
    "BULLDOZER": {"ip": "192.168.10.123", "port": 5005}
}
```

Si las Raspberry tienen otras IPs, edítarlas en.

---

## Pasos para correr el sistema

### 1. Arrancar las Raspberry Pi (vehículos)

Para **cada** vehículo que vayas a usar:

1. Enciender la Raspberry y verificar que esta conectada a la red local.
2. Carga/transferir los scripts del vehículo a la Raspberry.
3. Arranca el script del vehículo:

   ```bash
   # En la Raspberry de la GRÚA
   python3 [script_grua.py]

   # En la Raspberry del BULLDOZER
   python3 [script_bulldozer.py]
   ```

4. Confirmar que la Raspberry empieza a transmitir.

### 2. Preparar la laptop (estación de control)

1. Conecta el joystick/volante y la cámara cenital USB.
2. Verifica que el modelo entrenado esté en su ruta:

   ```
   ...\modelos_mirai_v2\mejor_modelo.pth
   ```

3. (Opcional) Si la cámara cenital no es el índice 1, ajusta `GlobalCamThread(cam_index=1)` en el script — cambiar a 0, 1 o 2.

### 3. Arrancar la terminal

```bash
python mirai_teleop_tracker.py
```

La ventana de control debería abrirse. En la barra superior se vera el estado de la IA y del tracking del robot.

---

## Uso de la terminal

### Controles del joystick (mando del brazo)

| Botón | Acción |
|---|---|
| 0 | Bloquear / desbloquear movimiento |
| 1 | Cambiar cámara (frontal ↔ trasera) |
| 2 | Cambiar de vehículo (grúa ↔ bulldozer) |
| 3 | Avanzar fase de la tarea |
| 6 | Cambiar modo IA: OFF → BLEND → OVERRIDE → OFF |

### Modos de asistencia IA

- **OFF (visual):** la IA solo muestra sugerencias en pantalla; el usuario controla todo.
- **BLEND:** cuando el usuario suelta el control, la IA lo completa suavemente. Si el usuario mueve, mandas el usuario.
- **OVERRIDE:** la IA controla el vehículo; el usuario interrumpe moviendo el joystick por encima del umbral.

### Tracking y retorno (cámara cenital)

1. Coloca el vehículo en su posición de inicio.
2. Verifica que en la barra superior diga **"● ROBOT DETECTADO"** (verde). En la miniatura cenital se vera un círculo verde sobre el marcador del vehículo.
3. Pulsa **"CALIBRAR HOME"** para guardar esa posición como punto de retorno.
4. Durante la fase 0 en modo OVERRIDE, cuando el vehículo regrese cerca del home, el sistema detiene el control automáticamente.


## Apagado

1. Cierra la ventana de la terminal (botón ⏻ o cerrar). El sistema envía STOP a los vehículos automáticamente.
2. Detén los scripts en cada Raspberry (Ctrl+C).
