"""
Visualizador low poly controlado por 4 señales de audio.

Dependencias:  pip install numpy pygame sounddevice soundfile

Uso (una entrada -c por canal, cada una es un archivo o un micrófono):
  python visualizador.py -c bajo.wav -c bateria.wav -c mic -c mic:2 \
                         -p altura -p rotacion -p color -p escala

Fuentes:
  ruta/al/archivo.wav        archivo (se reproduce en bucle)
  mic                        micrófono por defecto
  mic:DISPOSITIVO            micrófono (número o nombre parcial)
  mic:DISPOSITIVO:CANAL      un canal concreto (0, 1, 2...) de una interfaz multicanal

Parámetros disponibles: altura, rotacion, color, escala, ruido

Controles (solo canales con archivo):
  Botones Pausar / Reanudar / -5 s por canal, y barra de progreso clicable para saltar.
  Teclas: P pausar todo | R reanudar todo | FLECHA IZQ. retroceder todo 5 s
          F pantalla completa | ESPACIO mostrar/ocultar interfaz | ESC salir

Cómo funciona, en resumen:
  1. Cada señal de audio (archivo o micrófono) se convierte en un "nivel" entre 0 y 1
     (qué tan fuerte suena en ese instante).
  2. Cada nivel se asigna a un parámetro visual (altura, rotación, color, escala o ruido).
  3. Cada fotograma se deforma una malla de triángulos según esos niveles y se dibuja
     con sombreado plano, lo que da el aspecto low poly.
"""
import argparse
import colorsys
import math
import sys

import numpy as np
import pygame
import sounddevice as sd
import soundfile as sf

SR = 44100  # frecuencia de muestreo con la que trabaja todo el programa (muestras por segundo)
PARAMS = ["altura", "rotacion", "color", "escala", "ruido"]  # parámetros visuales disponibles


def mmss(seg):
    """Convierte segundos a texto 'm:ss' (para mostrar el tiempo)."""
    return f"{int(seg // 60)}:{int(seg % 60):02d}"


class Canal:
    """Una señal: archivo o micrófono. Expone `nivel` (0..1) normalizado."""

    def __init__(self, fuente, parametro):
        self.fuente = fuente        # texto recibido por línea de comandos
        self.parametro = parametro  # parámetro visual que controla esta señal
        self.nivel = 0.0            # volumen actual suavizado, entre 0 y 1
        self.pico = 1e-4            # volumen máximo reciente (sirve para normalizar)
        self.datos = None           # muestras del archivo (None si es micrófono)
        self.pos = 0                # posición de lectura dentro del archivo, en muestras
        self.pausado = False
        self.stream = None          # entrada de audio abierta (solo micrófonos)
        # Según el texto, la fuente es un micrófono o un archivo
        if fuente.startswith("mic"):
            self._abrir_mic(fuente)
        else:
            self._cargar_archivo(fuente)

    def _actualizar(self, rms):
        """Convierte el volumen medido (rms) en un nivel estable entre 0 y 1."""
        # Auto-ganancia: el pico sube al instante con un sonido fuerte y baja muy despacio.
        # Así el nivel es relativo al volumen de cada señal y no hay que ajustar ganancias.
        self.pico = max(rms, self.pico * 0.9995, 1e-4)
        objetivo = min(1.0, rms / self.pico)
        # Suavizado: sube rápido (ataque) y baja despacio (caída) para que no parpadee
        k = 0.5 if objetivo > self.nivel else 0.08
        self.nivel += (objetivo - self.nivel) * k

    # ---- archivo ----
    def _cargar_archivo(self, ruta):
        """Lee el archivo completo, lo pasa a mono y lo remuestrea a SR si hace falta."""
        x, sr = sf.read(ruta, dtype="float32", always_2d=True)
        x = x.mean(axis=1)  # mezcla estéreo -> mono
        if sr != SR:
            # Remuestreo simple por interpolación lineal
            n = int(len(x) * SR / sr)
            x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype("float32")
        self.datos = x

    def leer_bloque(self, n):
        """Devuelve n muestras del archivo (en bucle) y actualiza el nivel."""
        if self.pausado:
            self._actualizar(0.0)  # en pausa el nivel cae a cero
            return np.zeros(n, dtype="float32")
        # El módulo (%) hace que al llegar al final vuelva al principio (bucle)
        idx = (self.pos + np.arange(n)) % len(self.datos)
        bloque = self.datos[idx]
        self.pos = (self.pos + n) % len(self.datos)
        # rms = raíz de la media de los cuadrados: medida estándar del volumen
        self._actualizar(float(np.sqrt(np.mean(bloque ** 2))))
        return bloque

    # ---- transporte ----
    def pausar(self):
        self.pausado = True

    def reanudar(self):
        self.pausado = False

    def retroceder(self, seg=5):
        """Mueve la posición hacia atrás `seg` segundos (sin pasar del inicio)."""
        if self.datos is not None:
            self.pos = max(0, self.pos - int(seg * SR))

    def buscar(self, frac):
        """Salta a una posición del archivo dada como fracción (0 = inicio, 1 = final)."""
        if self.datos is not None:
            self.pos = int(min(max(frac, 0.0), 0.999) * len(self.datos))

    def progreso(self):
        """Fracción del archivo que ya sonó (para dibujar la barra)."""
        return self.pos / len(self.datos) if self.datos is not None else 0.0

    # ---- micrófono ----
    def _abrir_mic(self, fuente):
        """Abre una entrada de audio. Formato: mic[:dispositivo[:canal]]"""
        partes = fuente.split(":")
        disp = partes[1] if len(partes) > 1 and partes[1] != "" else None
        if disp is not None and disp.isdigit():
            disp = int(disp)  # número de dispositivo; si no, se busca por nombre
        canal = int(partes[2]) if len(partes) > 2 else 0

        # sounddevice llama a esta función cada vez que llega un bloque de audio
        def cb(indata, frames, t, status):
            self._actualizar(float(np.sqrt(np.mean(indata[:, canal] ** 2))))

        # Se abren canal+1 canales para poder leer el que se pidió
        self.stream = sd.InputStream(device=disp, channels=canal + 1, samplerate=SR,
                                     blocksize=1024, callback=cb)
        self.stream.start()


def crear_malla(cols=24, filas=15, semilla=7):
    """Crea la malla de puntos y triángulos. Devuelve (puntos, triángulos, altura_base)."""
    rng = np.random.default_rng(semilla)  # semilla fija: la malla es igual en cada ejecución
    xs = np.linspace(-1.6, 1.6, cols)
    ys = np.linspace(-1.0, 1.0, filas)
    gx, gy = np.meshgrid(xs, ys)  # cuadrícula regular de puntos
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    # Se desplazan los vértices al azar: la irregularidad da el aspecto low poly
    gx = gx + rng.uniform(-0.35, 0.35, gx.shape) * dx
    gy = gy + rng.uniform(-0.35, 0.35, gy.shape) * dy
    # Relieve base: combinación de ondas seno/coseno. Luego se escala con el parámetro "altura"
    base = (np.sin(gx * 2.1) * np.cos(gy * 2.6) + 0.5 * np.sin(gx * 4.3 + gy * 3.1)) * 0.5
    pts = np.stack([gx.ravel(), gy.ravel(), base.ravel()], axis=1)  # lista de puntos (x, y, z)
    # Cada celda de la cuadrícula se divide en 2 triángulos (a, b, c, d = sus 4 esquinas).
    # La diagonal alterna según (i+j) para que la malla no tenga un patrón demasiado regular.
    tris = []
    for j in range(filas - 1):
        for i in range(cols - 1):
            a, b = j * cols + i, j * cols + i + 1
            c, d = (j + 1) * cols + i, (j + 1) * cols + i + 1
            tris += [(a, b, c), (b, d, c)] if (i + j) % 2 else [(a, b, d), (a, d, c)]
    return pts, np.array(tris), base.ravel()


GRIS = (200, 200, 200)  # único color de la interfaz


def dibujar_ui(pantalla, fuente_ui, canales):
    """Dibuja la interfaz y devuelve la lista de zonas clicables: (rect, funcion, usa_posicion)."""
    zonas = []
    for i, c in enumerate(canales):
        y = 10 + i * 26  # cada canal ocupa una fila
        pantalla.blit(fuente_ui.render(f"{i + 1} {c.parametro}", True, GRIS), (10, y + 3))
        if c.datos is not None:
            # Solo los archivos tienen botones y barra de progreso
            x = 120
            botones = (("Pausar", c.pausar, 70), ("Reanudar", c.reanudar, 80),
                       ("-5 s", lambda c=c: c.retroceder(5), 50))
            for nombre, accion, ancho in botones:
                r = pygame.Rect(x, y, ancho, 20)
                pygame.draw.rect(pantalla, GRIS, r, 1)  # borde del botón
                t = fuente_ui.render(nombre, True, GRIS)
                pantalla.blit(t, t.get_rect(center=r.center))  # texto centrado
                zonas.append((r, accion, False))  # se guarda para detectar clics
                x += ancho + 6
            # Barra de progreso: contorno + relleno proporcional a lo que ya sonó
            barra = pygame.Rect(x + 6, y + 5, 200, 10)
            pygame.draw.rect(pantalla, GRIS, barra, 1)
            pygame.draw.rect(pantalla, GRIS, (barra.x, barra.y, int(barra.w * c.progreso()), barra.h))
            # La barra recibe la posición del ratón y salta a esa fracción del archivo
            zonas.append((barra, lambda pos, c=c, b=barra: c.buscar((pos[0] - b.x) / b.w), True))
            texto = f"{mmss(c.pos / SR)} / {mmss(len(c.datos) / SR)}"
            pantalla.blit(fuente_ui.render(texto, True, GRIS), (barra.right + 8, y + 3))
        else:
            pantalla.blit(fuente_ui.render("mic", True, GRIS), (120, y + 3))
        # Medidor de nivel (para archivos y micrófonos)
        nivel = pygame.Rect(690, y + 5, 80, 10)
        pygame.draw.rect(pantalla, GRIS, nivel, 1)
        pygame.draw.rect(pantalla, GRIS, (nivel.x, nivel.y, int(nivel.w * c.nivel), nivel.h))
    return zonas


def main():
    # ---- Argumentos de línea de comandos ----
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--canal", action="append", required=True, help="archivo o mic (x4)")
    ap.add_argument("-p", "--param", action="append", choices=PARAMS, help="parámetro por canal")
    a = ap.parse_args()
    if len(a.canal) != 4:
        sys.exit("Necesitas exactamente 4 canales (-c).")
    params = a.param or PARAMS[:4]  # por defecto: altura, rotacion, color, escala
    if len(params) != 4:
        sys.exit("Necesitas 4 parámetros (-p), uno por canal.")

    # ---- Audio ----
    canales = [Canal(f, p) for f, p in zip(a.canal, params)]
    archivos = [c for c in canales if c.datos is not None]
    if archivos:
        # Un único stream de salida reproduce la mezcla de todos los archivos.
        # Al pedir el bloque de cada canal también se actualiza su nivel, así el
        # análisis queda sincronizado con lo que se escucha.
        def salida(outdata, frames, t, status):
            mezcla = sum(c.leer_bloque(frames) for c in archivos) * 0.5  # 0.5 evita saturar
            outdata[:] = mezcla[:, None]
        sd.OutputStream(samplerate=SR, channels=1, blocksize=1024, callback=salida).start()

    # ---- Ventana ----
    pygame.init()
    pantalla = pygame.display.set_mode((1000, 640), pygame.RESIZABLE)
    pygame.display.set_caption("Visualizador")
    fuente_ui = pygame.font.SysFont("monospace", 14)
    reloj = pygame.time.Clock()
    pts0, tris, base_z = crear_malla()
    angulo, mostrar, zonas = 0.0, True, []  # zonas: áreas clicables del fotograma anterior

    # ---- Bucle principal: un fotograma por vuelta ----
    while True:
        # Eventos de teclado y ratón
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_f:
                    pygame.display.toggle_fullscreen()
                elif e.key == pygame.K_SPACE:
                    mostrar = not mostrar
                elif e.key == pygame.K_p:
                    [c.pausar() for c in archivos]
                elif e.key == pygame.K_r:
                    [c.reanudar() for c in archivos]
                elif e.key == pygame.K_LEFT:
                    [c.retroceder(5) for c in archivos]
            if e.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION) and mostrar:
                # Solo interesa el botón izquierdo (al hacer clic o al arrastrar)
                if e.type == pygame.MOUSEMOTION and not e.buttons[0]:
                    continue
                if e.type == pygame.MOUSEBUTTONDOWN and e.button != 1:
                    continue
                for rect, fn, usa_pos in zonas:
                    if rect.collidepoint(e.pos):
                        if usa_pos:
                            fn(e.pos)  # barra: saltar (también arrastrando)
                        elif e.type == pygame.MOUSEBUTTONDOWN:
                            fn()  # botón: solo al hacer clic

        # Valor actual de cada parámetro visual (0 si ningún canal lo controla)
        v = {p: 0.0 for p in PARAMS}
        for c in canales:
            v[c.parametro] = c.nivel
        t = pygame.time.get_ticks() / 1000  # tiempo en segundos, para animar el "ruido"

        # ---- Deformar la malla según los parámetros ----
        angulo += 0.003 + v["rotacion"] * 0.05  # gira siempre un poco; el audio la acelera
        pts = pts0.copy()
        pts[:, 2] = base_z * (0.2 + 2.2 * v["altura"])  # altura: relieve más o menos marcado
        # ruido: los vértices vibran en x e y con una onda que depende del tiempo
        pts[:, 0] += np.sin(t * 6 + pts0[:, 1] * 5) * v["ruido"] * 0.08
        pts[:, 1] += np.cos(t * 5 + pts0[:, 0] * 5) * v["ruido"] * 0.08
        pts *= 1.0 + v["escala"] * 0.8  # escala: agranda toda la malla

        # ---- Rotación 3D ----
        # 1) giro alrededor del eje vertical (el ángulo que crece con el tiempo)
        ca, sa = math.cos(angulo), math.sin(angulo)
        x = pts[:, 0] * ca - pts[:, 1] * sa
        y = pts[:, 0] * sa + pts[:, 1] * ca
        z = pts[:, 2]
        # 2) inclinación fija hacia la cámara para ver la malla desde arriba en perspectiva
        tilt = 1.0
        y2 = y * math.cos(tilt) - z * math.sin(tilt)
        z2 = y * math.sin(tilt) + z * math.cos(tilt)
        rot = np.stack([x, y2, z2], axis=1)

        # ---- Proyección a 2D con perspectiva simple (lo lejano se ve más pequeño) ----
        w, h = pantalla.get_size()
        f = min(w, h) * 0.55
        prof = 1.0 / (4.0 - rot[:, 1] * 0.8)
        sx = w / 2 + rot[:, 0] * f * prof * 3.2
        sy = h / 2 - rot[:, 2] * f * prof * 3.2

        # ---- Sombreado plano: cada triángulo tiene un único brillo ----
        A, B, C = rot[tris[:, 0]], rot[tris[:, 1]], rot[tris[:, 2]]  # vértices de cada triángulo
        n = np.cross(B - A, C - A)  # normal de cada cara
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        luz = np.array([0.3, -0.5, 0.8])
        luz /= np.linalg.norm(luz)
        br = 0.25 + 0.75 * np.abs(n @ luz)  # más alineada con la luz = más brillo
        # Algoritmo del pintor: se dibujan primero los triángulos más lejanos
        orden = np.argsort(-(A[:, 1] + B[:, 1] + C[:, 1]))

        # ---- Dibujo ----
        pantalla.fill((12, 12, 16))
        matiz = (0.55 + v["color"] * 0.45) % 1.0  # color: el audio desplaza el matiz (azul -> rojo)
        for k in orden:
            r, g, b = colorsys.hsv_to_rgb((matiz + br[k] * 0.08) % 1.0, 0.65, min(1.0, br[k]))
            poli = [(sx[i], sy[i]) for i in tris[k]]
            pygame.draw.polygon(pantalla, (int(r * 255), int(g * 255), int(b * 255)), poli)

        # La interfaz se dibuja encima; `zonas` queda listo para los clics del siguiente fotograma
        zonas = dibujar_ui(pantalla, fuente_ui, canales) if mostrar else []

        pygame.display.flip()
        reloj.tick(60)  # limita a 60 fotogramas por segundo


if __name__ == "__main__":
    main()
