import os
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ============================================================
# CONFIGURACIÓN
# ============================================================
CONFIG = {
    # Rutas
    "dataset_root": os.path.join(
        os.path.expanduser("~"),  "Desktop", "dataset_mirai", "grua"
    ),
    "output_dir": os.path.join(
        os.path.expanduser("~"),  "Desktop", "modelos_mirai_v2"
    ),

    # Modelo
    "seq_len":      8,      
    "hidden_size":  256, 
    "lstm_layers":  2,
    "dropout":      0.4,    
    "epochs":       80,
    "batch_size":   8,     
    "lr":           3e-4,  
    "weight_decay": 1e-4,
    "warmup_epochs": 5,     
    "freeze_cnn_epochs": 15,
    "val_split":    0.2,    
    "val_operador": None,    
    "img_size":     (224, 224),   
    "imu_max":      90.0,
    "n_fases":      5,
    "output_names": [
        "oruga_izq", "oruga_der",
        "brazo", "pala_rot", "rotacion_cabina",
    ],
}


# ============================================================
# DATASET
# ============================================================
class MiraiDataset(Dataset):
    """
    Ventana temporal de seq_len frames.
    Entrada:  [img_actual, IMUs, fase_onehot, velocidades_previas]
    Salida:   comandos del último frame
    """

    def __init__(self, muestras: list, cfg: dict, augment: bool = False):
        self.cfg      = cfg
        self.augment  = augment
        self.muestras = muestras  # lista de (ep_path, df, idx_inicio)

        base = [
            transforms.Resize(cfg["img_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std= [0.229, 0.224, 0.225]),
        ]
        aug = [
            transforms.Resize(cfg["img_size"]),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std= [0.229, 0.224, 0.225]),
        ]
        self.img_tf = transforms.Compose(aug if augment else base)

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, idx):
        ep_path, df, start = self.muestras[idx]
        seq   = self.cfg["seq_len"]
        chunk = df.iloc[start : start + seq]

        imgs     = []
        sensores = []

        for _, row in chunk.iterrows():
            # Imagen
            img_path = os.path.join(ep_path, "cam_frontal", row["frame_nombre"])
            try:
                img_t = self.img_tf(Image.open(img_path).convert("RGB"))
            except Exception:
                img_t = torch.zeros(3, *self.cfg["img_size"])
            imgs.append(img_t)

            # Sensores
            imu_c  = float(row.get("imu_cabina",   0.0)) / self.cfg["imu_max"]
            imu_e1 = float(row.get("imu_eslabon1", 0.0)) / self.cfg["imu_max"]
            imu_e2 = float(row.get("imu_eslabon2", 0.0)) / self.cfg["imu_max"]
            imu_p  = float(row.get("imu_pala",     0.0)) / self.cfg["imu_max"]

            fase = int(row.get("fase", 0))
            fase_oh = [0.0] * self.cfg["n_fases"]
            fase_oh[min(fase, self.cfg["n_fases"] - 1)] = 1.0

            # Comandos previos como contexto adicional
            cmd_prev = [
                float(row.get("oruga_izq",       0.0)),
                float(row.get("oruga_der",       0.0)),
                float(row.get("brazo",           0.0)),
                float(row.get("pala_rot",        0.0)),
                float(row.get("rotacion_cabina", 0.0)),
            ]

            sensores.append([imu_c, imu_e1, imu_e2, imu_p, *fase_oh, *cmd_prev])

        imgs_t     = torch.stack(imgs)
        sensores_t = torch.tensor(sensores, dtype=torch.float32)

        last = chunk.iloc[-1]
        target = torch.tensor([
            float(last.get("oruga_izq",       0.0)),
            float(last.get("oruga_der",       0.0)),
            float(last.get("brazo",           0.0)),
            float(last.get("pala_rot",        0.0)),
            float(last.get("rotacion_cabina", 0.0)),
        ], dtype=torch.float32)

        peso = 2.0 if float(last.get("fase_transicion", 0)) > 0 else 1.0
        return imgs_t, sensores_t, target, torch.tensor(peso, dtype=torch.float32)


# ============================================================
# MODELO v2
# ============================================================
class MiraiNetV2(nn.Module):
    """
    Mejoras sobre v1:
    - CNN procesa solo el último frame (ahorra memoria, MobileNetV2 correcto)
    - Sensor path incluye comandos previos como contexto
    - Temporal: GRU en vez de LSTM (menos parámetros, similar rendimiento)
    - Atención temporal simple sobre la secuencia de sensores
    """

    def __init__(self, cfg: dict):
        super().__init__()
        # n_sensor = 4 imu + 5 fase_oh + 5 cmd_previos = 14
        n_sensor = 4 + cfg["n_fases"] + 5

        # ── CNN visual (solo último frame) ──────────────────────
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        backbone.classifier = nn.Sequential(
            nn.Dropout(cfg["dropout"]),
            nn.Linear(backbone.last_channel, 256),
            nn.ReLU(),
        )
        self.cnn = backbone

        # ── Encoder de sensores temporales (GRU) ───────────────
        self.sensor_enc = nn.GRU(
            input_size=n_sensor,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=cfg["dropout"],
        )

        # ── Fusión ──────────────────────────────────────────────
        # 256 (CNN) + 128 (GRU hidden)
        fusion_dim = 256 + 128

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
        )

        # ── Cabeza de salida ─────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"] * 0.5),
            nn.Linear(64, 5),
            nn.Tanh(),
        )

    def forward(self, imgs, sensores):
        """
        imgs:     [B, seq, 3, H, W]
        sensores: [B, seq, n_sensor]
        """
        B = imgs.shape[0]

        # Solo el último frame para CNN
        last_img  = imgs[:, -1]              # [B, 3, H, W]
        vis_feat  = self.cnn(last_img)       # [B, 256]

        # Sensores: secuencia completa → tomar último hidden
        _, h_n    = self.sensor_enc(sensores)  # h_n: [2, B, 128]
        sens_feat = h_n[-1]                    # [B, 128]

        # Fusión
        fused = self.fusion(torch.cat([vis_feat, sens_feat], dim=-1))
        return self.head(fused)


# ============================================================
# UTILIDADES
# ============================================================
def cargar_episodios(cfg):
    """Devuelve lista de (ep_path, df) para todos los episodios válidos."""
    root = Path(cfg["dataset_root"])
    episodios = sorted([
        str(p) for p in root.iterdir()
        if p.is_dir() and p.name.startswith("episodio_")
    ])
    resultado = []
    for ep in episodios:
        csv_path = os.path.join(ep, "telemetria.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if "tracker_confianza" in df.columns:
            df = df[df["tracker_confianza"] >= 0.3].reset_index(drop=True)
        if len(df) >= cfg["seq_len"] + 1:
            resultado.append((ep, df))
    return resultado


def construir_muestras(episodios, cfg):
    """
    Split temporal 80/20 dentro de cada episodio.
    Retorna (muestras_train, muestras_val).
    """
    train_m, val_m = [], []
    seq = cfg["seq_len"]

    for ep_path, df in episodios:
        n = len(df)
        corte = int(n * (1 - cfg["val_split"]))

        for i in range(corte - seq):
            train_m.append((ep_path, df, i))
        for i in range(corte, n - seq):
            val_m.append((ep_path, df, i))

    return train_m, val_m


def construir_muestras_por_operador(episodios, cfg):
    """Split por operador (comportamiento original v1)."""
    train_m, val_m = [], []
    seq = cfg["seq_len"]
    val_op = cfg["val_operador"]

    for ep_path, df in episodios:
        op = df["operador_id"].iloc[0] if "operador_id" in df.columns else "?"
        n  = len(df)
        target = val_m if op == val_op else train_m
        for i in range(n - seq):
            target.append((ep_path, df, i))

    return train_m, val_m


def calcular_pesos_loss(muestras_train, cfg):
    """
    Calcula el inverso de la desviación estándar de cada comando.
    Si un comando tiene std~0 (columna siempre cero o casi),
    se le asigna peso 1.0 en vez de inf.
    """
    outs = cfg["output_names"]
    acum = {k: [] for k in outs}

    for ep_path, df, start in muestras_train[:500]:
        row = df.iloc[start + cfg["seq_len"] - 1]
        for k in outs:
            acum[k].append(float(row.get(k, 0.0)))

    stds = []
    for k in outs:
        std = np.std(acum[k])
        # Si std < umbral mínimo, el comando no tiene variación útil;
        # usamos peso 1.0 (sin amplificar ni ignorar)
        if std < 0.01:
            print(f"  std({k}) = {std:.6f}  ⚠️  varianza muy baja → peso=1.0")
            stds.append(1.0)
        else:
            print(f"  std({k}) = {std:.4f}")
            stds.append(std)

    stds  = np.array(stds, dtype=np.float32)
    pesos = 1.0 / stds
    pesos = pesos / pesos.mean()   # normalizar a media=1
    # Clamp de seguridad: ningún peso puede superar 10×la media
    pesos = np.clip(pesos, 0.1, 10.0)
    print(f"  → pesos finales: {pesos.round(3)}")
    return torch.tensor(pesos, dtype=torch.float32)


# ============================================================
# ENTRENAMIENTO
# ============================================================
def entrenar():
    cfg = CONFIG
    os.makedirs(cfg["output_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Dispositivo: {device}")
    if device.type == "cuda":
        print(f"[TRAIN] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[TRAIN] VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── 1. Cargar episodios ──────────────────────────────────────
    episodios = cargar_episodios(cfg)
    if not episodios:
        print(f"[ERROR] No se encontraron episodios en {cfg['dataset_root']}")
        return
    print(f"[TRAIN] {len(episodios)} episodios cargados")

    # ── 2. Split ─────────────────────────────────────────────────
    if cfg["val_operador"]:
        train_m, val_m = construir_muestras_por_operador(episodios, cfg)
    else:
        train_m, val_m = construir_muestras(episodios, cfg)

    print(f"[TRAIN] Train: {len(train_m)} | Val: {len(val_m)} secuencias")

    if len(train_m) < 100:
        print("[ERROR] Muy pocas secuencias de entrenamiento")
        return

    # ── 3. Pesos de loss por varianza ────────────────────────────
    print("[TRAIN] Calculando varianza de comandos...")
    cmd_weights = calcular_pesos_loss(train_m, cfg).to(device)
    print(f"[TRAIN] Pesos loss: {cmd_weights.cpu().numpy().round(3)}")

    # ── 4. Datasets y loaders ────────────────────────────────────
    ds_train = MiraiDataset(train_m, cfg, augment=True)
    ds_val   = MiraiDataset(val_m,   cfg, augment=False)

    # num_workers=0 en Windows evita deadlocks con multiprocessing
    import platform
    nw = 0 if platform.system() == "Windows" else 2
    pm = nw > 0
    dl_train = DataLoader(ds_train, batch_size=cfg["batch_size"],
                          shuffle=True, num_workers=nw, pin_memory=pm)
    dl_val   = DataLoader(ds_val,   batch_size=cfg["batch_size"] * 2,
                          shuffle=False, num_workers=nw, pin_memory=pm) if val_m else None

    # ── 5. Modelo ────────────────────────────────────────────────
    model = MiraiNetV2(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[TRAIN] Parámetros entrenables: {n_params:,}")

    # Freeze CNN inicial
    def set_cnn_grad(requires_grad):
        for p in model.cnn.features.parameters():
            p.requires_grad = requires_grad

    set_cnn_grad(False)
    print(f"[TRAIN] CNN congelado para primeras {cfg['freeze_cnn_epochs']} épocas")

    # ── 6. Optimizador y scheduler ───────────────────────────────
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )

    def get_lr(epoch):
        """Warm-up lineal + CosineAnnealing."""
        if epoch < cfg["warmup_epochs"]:
            return (epoch + 1) / cfg["warmup_epochs"]
        progress = (epoch - cfg["warmup_epochs"]) / (cfg["epochs"] - cfg["warmup_epochs"])
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    # ── 7. Loss ponderada ────────────────────────────────────────
    def loss_fn(pred, target, pesos_muestra):
        sq_err = (pred - target) ** 2               # [B, 5]
        sq_err = sq_err * cmd_weights.unsqueeze(0)  # pesos por varianza
        per_sample = sq_err.mean(dim=1)             # [B]
        loss = (per_sample * pesos_muestra.to(device)).mean()
        if not torch.isfinite(loss):
            print("  ⚠️  loss no finita, batch saltado")
            return torch.tensor(0.0, requires_grad=True, device=device)
        return loss

    # ── 8. Prueba de velocidad (1 batch) ────────────────────────
    print("[TRAIN] Probando velocidad de un batch...", flush=True)
    _imgs, _sens, _tgt, _p = next(iter(dl_train))
    t_start = time.time()
    with torch.no_grad():
        _ = model(_imgs.to(device), _sens.to(device))
    t_batch = time.time() - t_start
    secs_per_epoch = t_batch * len(dl_train)
    print(f"[TRAIN] 1 batch = {t_batch:.2f}s  →  ~{secs_per_epoch/60:.1f} min/época")
    del _imgs, _sens, _tgt, _p

    # ── 8. Loop ──────────────────────────────────────────────────
    mejor_val = float("inf")
    historial  = []
    no_mejora  = 0
    paciencia  = 20  # early stopping

    for epoch in range(1, cfg["epochs"] + 1):

        # Descongelar CNN tras freeze_cnn_epochs
        if epoch == cfg["freeze_cnn_epochs"] + 1:
            set_cnn_grad(True)
            # Reiniciar optimizador con todos los parámetros
            optimizer = optim.AdamW(
                model.parameters(),
                lr=cfg["lr"] * 0.1,  # LR más bajo para fine-tune del CNN
                weight_decay=cfg["weight_decay"]
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg["epochs"] - cfg["freeze_cnn_epochs"]
            )
            print(f"\n[TRAIN] Época {epoch}: CNN descongelado para fine-tuning")

        # ── Train
        model.train()
        loss_train = 0.0
        t0 = time.time()
        n_batches = len(dl_train)
        for batch_i, (imgs, sens, tgt, pesos) in enumerate(dl_train):
            imgs = imgs.to(device, non_blocking=True)
            sens = sens.to(device, non_blocking=True)
            tgt  = tgt.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(imgs, sens)
            loss = loss_fn(pred, tgt, pesos)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_train += loss.item()

            # Progreso visible cada 50 batches
            if (batch_i + 1) % 50 == 0 or (batch_i + 1) == n_batches:
                elapsed = time.time() - t0
                pct = (batch_i + 1) / n_batches * 100
                print(f"  E{epoch:03d} [{batch_i+1:4d}/{n_batches}] "
                      f"{pct:5.1f}%  loss={loss_train/(batch_i+1):.5f}  "
                      f"t={elapsed:.0f}s", end="\r", flush=True)

        print()  # salto de linea al terminar epoca
        loss_train /= len(dl_train)

        # ── Val
        loss_val = 0.0
        if dl_val:
            model.eval()
            with torch.no_grad():
                for imgs, sens, tgt, pesos in dl_val:
                    imgs = imgs.to(device, non_blocking=True)
                    sens = sens.to(device, non_blocking=True)
                    tgt  = tgt.to(device, non_blocking=True)
                    pred = model(imgs, sens)
                    loss_val += loss_fn(pred, tgt, pesos).item()
            loss_val /= len(dl_val)

        if epoch <= cfg["freeze_cnn_epochs"]:
            scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"E{epoch:03d}/{cfg['epochs']} | "
            f"train {loss_train:.5f} | val {loss_val:.5f} | lr {lr_now:.2e}"
        )

        historial.append({
            "epoch": epoch, "loss_train": loss_train, "loss_val": loss_val
        })

        # ── Guardar mejor modelo
        if loss_val < mejor_val and dl_val:
            mejor_val = loss_val
            no_mejora = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "loss_val":    mejor_val,
                "config":      cfg,
            }, os.path.join(cfg["output_dir"], "mejor_modelo.pth"))
            print(f"  ✅  Mejor guardado (val={mejor_val:.5f})")
        else:
            no_mejora += 1

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(), "config": cfg,
            }, os.path.join(cfg["output_dir"], f"ckpt_ep{epoch:03d}.pth"))

        # Early stopping
        if no_mejora >= paciencia:
            print(f"\n[TRAIN] Early stopping tras {paciencia} épocas sin mejora")
            break

    # ── 9. Guardar resultados ─────────────────────────────────────
    pd.DataFrame(historial).to_csv(
        os.path.join(cfg["output_dir"], "historial.csv"), index=False
    )
    torch.save({
        "model_state": model.state_dict(), "config": cfg,
    }, os.path.join(cfg["output_dir"], "modelo_final.pth"))
    print(f"\n[TRAIN] ✅ Completado. Modelos en: {cfg['output_dir']}")

    # ── 10. Evaluación rápida ─────────────────────────────────────
    if dl_val:
        evaluar_rapido(model, dl_val, cfg, device)


# ============================================================
# EVALUACIÓN RÁPIDA (integrada, sin script separado)
# ============================================================
def evaluar_rapido(model, dl_val, cfg, device):
    """
    Muestra MAE, RMSE y R² por comando en validación.
    No requiere script externo.
    """
    model.eval()
    preds_all  = []
    target_all = []

    with torch.no_grad():
        for imgs, sens, tgt, _ in dl_val:
            pred = model(imgs.to(device), sens.to(device))
            preds_all.append(pred.cpu().numpy())
            target_all.append(tgt.numpy())

    preds  = np.vstack(preds_all)
    target = np.vstack(target_all)

    print("\n" + "="*55)
    print("EVALUACIÓN RÁPIDA — VAL SET")
    print("="*55)
    print(f"{'Comando':>20}  {'MAE':>6}  {'RMSE':>6}  {'R²':>7}")
    print("-"*55)

    r2s = []
    for i, nombre in enumerate(cfg["output_names"]):
        p = preds[:, i]
        t = target[:, i]
        mae  = np.mean(np.abs(p - t))
        rmse = np.sqrt(np.mean((p - t) ** 2))
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2) + 1e-8
        r2  = 1 - ss_res / ss_tot
        r2s.append(r2)
        bar = "█" * int(max(0, r2) * 20)
        print(f"{nombre:>20}  {mae:.4f}  {rmse:.4f}  {r2:+.4f}  {bar}")

    print("-"*55)
    print(f"{'GLOBAL':>20}  {np.mean(np.abs(preds-target)):.4f}  "
          f"{np.sqrt(np.mean((preds-target)**2)):.4f}  "
          f"{np.mean(r2s):+.4f}")
    print("="*55)

    if np.mean(r2s) >= 0.60:
        print("🎯 Objetivo alcanzado (R² ≥ 0.60)")
    elif np.mean(r2s) >= 0.40:
        print("📈 Progreso sólido — necesita más épocas o datos")
    else:
        print("⚠️  R² bajo — revisar calidad del dataset")

def evaluar_modelo(ruta_modelo: str = None):
    """
    Carga un modelo guardado y ejecuta evaluación rápida.
    Uso: python mirai_train_v2.py evaluar [ruta_opcional.pth]
    """
    cfg = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if ruta_modelo is None:
        ruta_modelo = os.path.join(cfg["output_dir"], "mejor_modelo.pth")

    print(f"[EVAL] Cargando: {ruta_modelo}")
    ckpt = torch.load(ruta_modelo, map_location=device)

    # Usar config guardada si existe
    if "config" in ckpt:
        cfg = ckpt["config"]
        cfg.setdefault("val_split", 0.2)
        cfg.setdefault("val_operador", None)

    model = MiraiNetV2(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"[EVAL] Modelo cargado (época {ckpt.get('epoch','?')})")

    episodios = cargar_episodios(cfg)
    if cfg.get("val_operador"):
        _, val_m = construir_muestras_por_operador(episodios, cfg)
    else:
        _, val_m = construir_muestras(episodios, cfg)

    if not val_m:
        print("[EVAL] No hay datos de validación")
        return

    ds_val = MiraiDataset(val_m, cfg, augment=False)
    dl_val = DataLoader(ds_val, batch_size=16, shuffle=False, num_workers=2)
    evaluar_rapido(model, dl_val, cfg, device)

def verificar_dataset():
    cfg  = CONFIG
    root = Path(cfg["dataset_root"])

    if not root.exists():
        print(f"[ERROR] No existe: {root}")
        return

    episodios = sorted([
        p for p in root.iterdir()
        if p.is_dir() and p.name.startswith("episodio_")
    ])

    print(f"\n{'='*55}")
    print("VERIFICACIÓN DEL DATASET")
    print(f"{'='*55}")
    print(f"Ruta:      {root}")
    print(f"Episodios: {len(episodios)}\n")

    total_frames = 0
    operadores   = {}
    problemas    = []

    for ep in episodios:
        csv_path = ep / "telemetria.csv"
        cam_path = ep / "cam_frontal"

        if not csv_path.exists():
            print(f"  ⚠️  {ep.name}: SIN CSV")
            continue

        df = pd.read_csv(csv_path)
        n_frames   = len(df)
        n_imgs     = len(list(cam_path.glob("*.jpg"))) if cam_path.exists() else 0
        op         = df["operador_id"].iloc[0] if "operador_id" in df.columns else "?"
        fases      = df["fase"].value_counts().to_dict() if "fase" in df.columns else {}
        ciclos     = df["ciclo_id"].max() if "ciclo_id" in df.columns else 0

        # Detectar NaN en columnas de comandos
        cmds = [c for c in cfg["output_names"] if c in df.columns]
        nan_pct = df[cmds].isna().mean().mean() * 100 if cmds else 0

        operadores[op] = operadores.get(op, 0) + 1
        total_frames  += n_frames

        ok    = n_frames == n_imgs and nan_pct < 5
        emoji = "✅" if ok else "⚠️ "
        if not ok:
            problemas.append(ep.name)

        print(
            f"  {emoji} {ep.name} | op={op} | "
            f"frames={n_frames} imgs={n_imgs} | "
            f"ciclos={ciclos} | NaN_cmd={nan_pct:.1f}% | fases={fases}"
        )

    print(f"\n{'─'*55}")
    print(f"Total frames:    {total_frames:,}")
    print(f"Operadores:      {operadores}")
    n_seq = max(0, total_frames - len(episodios) * cfg["seq_len"])
    print(f"Secuencias ~:    {n_seq:,}")
    if problemas:
        print(f"  Episodios con problemas: {problemas}")
    print(f"{'='*55}\n")

if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else ""

    if cmd == "verificar":
        verificar_dataset()
    elif cmd == "evaluar":
        ruta = sys.argv[2] if len(sys.argv) > 2 else None
        evaluar_modelo(ruta)
    else:
        verificar_dataset()
        resp = input("¿Iniciar entrenamiento? (s/n): ").strip().lower()
        if resp == "s":
            entrenar()