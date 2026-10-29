"""
Visualizador low poly controlado por 4 señales de audio.

Dependencias:  pip install numpy pygame sounddevice soundfile

Uso (una entrada -c por canal, cada una es un archivo o un micrófono):
  python visualizador.py -c bajo.wav -c bateria.wav -c mic -c mic:2 \
                         -p altura -p rotacion -p color -p escala

Ejemplos con 4 micrófonos:
  python visualizador.py --listar                       (muestra dispositivos y sus entradas)
  # Interfaz multicanal (una entrada física por señal):
  python visualizador.py -c mic:2:0 -c mic:2:1 -c mic:2:2 -c mic:2:3
  # Cuatro dispositivos distintos (USB, integrado...), por número o por parte del nombre:
  python visualizador.py -c mic:1 -c mic:3 -c mic:Scarlett -c mic:USB
  # Mezcla de archivos y micrófonos:
  python visualizador.py -c bajo.wav -c mic:2:1 -c mic:2:2 -c batería.wav

Baja latencia con interfaz de audio (ASIO, solo Windows):
  python visualizador.py --asio --bloque 128 -c mic::0 -c mic::1 -c mic::2 -c mic::3
  python visualizador.py --asio "Focusrite" -c mic::0 -c mic::1 -c bajo.wav -c mic::3
  - Con --asio, la interfaz se indica ahí (nada, número o parte del nombre) y en -c solo se
    pone la entrada física: mic::ENTRADA. Todas las señales pasan por la misma interfaz.
  - Los archivos suenan por las salidas 1 y 2 de esa misma interfaz.
  - Un bloque más pequeño baja la latencia pero exige más a la CPU; si oyes cortes, súbelo
    (64, 128, 256, 512). Ajústalo también en el panel del driver ASIO de tu interfaz.
  - Sin --asio se usan las APIs normales del sistema, también con latencia baja (--bloque).

Fuentes:
  ruta/al/archivo.wav        archivo (se reproduce en bucle)
  mic                        micrófono por defecto
  mic:DISPOSITIVO            micrófono (número o nombre parcial)
  mic:DISPOSITIVO:CANAL      un canal concreto (0, 1, 2...) de una interfaz multicanal

Parámetros disponibles: altura, rotacion, color, escala, ruido, fragmentos

Características del sonido: cada parámetro puede seguir algo distinto del sonido, no solo
el volumen. Se indica así:  -p parametro:caracteristica   (sin característica = amplitud)
  amplitud   volumen general
  graves     energía en las frecuencias bajas (20-250 Hz): bombo, bajo
  medios     energía en las frecuencias medias (250-2000 Hz): voces, guitarras
  agudos     energía en las frecuencias altas (2-10 kHz): platillos, siseos
  pulso      golpes y ataques: salta con cada transitorio (percusión) y cae enseguida
  Ejemplo:  -p fragmentos:pulso -p altura:graves -p color:medios -p rotacion:agudos

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
import os
import sys

# El soporte ASIO de sounddevice (Windows) se activa con esta variable, que debe fijarse
# antes de importar la librería. Por eso se mira sys.argv directamente.
if any(x.startswith("--asio") for x in sys.argv):
    os.environ.setdefault("SD_ENABLE_ASIO", "1")

import numpy as np
import pygame
import sounddevice as sd
import soundfile as sf

SR = 44100  # frecuencia de muestreo con la que trabaja todo el programa (muestras por segundo)
PARAMS = ["altura", "rotacion", "color", "escala", "ruido", "fragmentos"]  # parámetros visuales disponibles


CARACT = ["amplitud", "graves", "medios", "agudos", "pulso"]  # características del sonido
BANDAS = {"graves": (20, 250), "medios": (250, 2000), "agudos": (2000, 10000)}  # límites en Hz
TAM_FFT = 2048                                  # muestras que se analizan (~46 ms): da resolución en graves
VENTANA = np.hanning(TAM_FFT)                   # ventana que suaviza los bordes antes de la FFT
FREQS = np.fft.rfftfreq(TAM_FFT, 1 / SR)        # frecuencia en Hz de cada "casilla" del espectro
MASCARAS = {n: (FREQS >= lo) & (FREQS < hi) for n, (lo, hi) in BANDAS.items()}
ESCALA_RMS = 2.0 / (TAM_FFT ** 2 * np.mean(VENTANA ** 2))  # pasa potencia espectral a volumen (rms)


class Normalizador:
    """Convierte un valor cualquiera en un nivel estable entre 0 y 1.

    Rango dinámico adaptativo: el nivel se mide entre el valor más bajo y el más alto
    recientes de ESTA señal, así recorre de verdad de 0 a 1 aunque la señal sea constante
    o esté comprimida. `piso` es el valor mínimo que cuenta como sonido: por debajo, el
    ruido de fondo no se amplifica.
    """

    def __init__(self, frac=0.6, piso=1e-4):
        self.techo, self.suelo, self.nivel = piso, 0.0, 0.0
        self.frac, self.piso = frac, piso

    def paso(self, x, e):
        """x: valor nuevo. e: tiempo transcurrido (1 = 1024 muestras), para que las
        constantes de tiempo no dependan del tamaño de bloque ni de los fotogramas."""
        self.techo = max(x, self.techo * 0.998 ** e, self.piso)
        # El suelo nunca sube más de `frac` del techo: una señal estable da nivel alto, no cero
        self.suelo = min(x, self.suelo + (self.frac * self.techo - self.suelo) * (1 - 0.997 ** e))
        self.suelo = min(self.suelo, self.frac * self.techo)
        objetivo = min(1.0, max(0.0, (x - self.suelo) / (self.techo - self.suelo + 1e-9)))
        # Suavizado: sube rápido (ataque) y baja despacio (caída) para que no parpadee
        self.nivel += (objetivo - self.nivel) * (1 - (0.5 if objetivo > self.nivel else 0.88) ** e)
        return self.nivel


def mmss(seg):
    """Convierte segundos a texto 'm:ss' (para mostrar el tiempo)."""
    return f"{int(seg // 60)}:{int(seg % 60):02d}"


class Canal:
    """Una señal: archivo o micrófono. Expone `nivel` (0..1) normalizado."""

    def __init__(self, fuente, parametro, caracteristica="amplitud"):
        self.fuente = fuente        # texto recibido por línea de comandos
        self.parametro = parametro  # parámetro visual que controla esta señal
        self.caracteristica = caracteristica  # qué del sonido mueve el parámetro (ver CARACT)
        self.nivel = 0.0            # nivel de amplitud suavizado, entre 0 y 1
        self.norm_amp = Normalizador(0.6, 1e-4)
        self.buf = np.zeros(TAM_FFT, dtype="float32")   # últimas muestras recibidas, para el análisis de frecuencias
        self.car = {k: 0.0 for k in CARACT}             # valor actual (0..1) de cada característica
        self.norm = {b: Normalizador(0.6, 2e-3) for b in BANDAS}  # un normalizador por banda
        self.banda_prev = {b: 0.0 for b in BANDAS}      # energía de cada banda en el fotograma anterior
        self.banda_media = {b: 0.0 for b in BANDAS}     # energía habitual reciente de cada banda (para detectar golpes)
        self.datos = None           # muestras del archivo (None si es micrófono)
        self.pos = 0                # posición de lectura dentro del archivo, en muestras
        self.pausado = False
        self.mic = None             # (dispositivo, entrada) si es micrófono; se abre después, compartido
        # Según el texto, la fuente es un micrófono o un archivo
        if fuente.startswith("mic"):
            self.mic = parsear_mic(fuente)
        else:
            self._cargar_archivo(fuente)

    def recibir(self, x):
        """Guarda un bloque de muestras nuevas (de archivo o micrófono) para analizarlas."""
        n = len(x)
        # Se conservan las últimas TAM_FFT muestras; el análisis se hace una vez por fotograma
        self.buf = x[-TAM_FFT:].copy() if n >= TAM_FFT else np.concatenate((self.buf[n:], x))

    def analizar(self, dt):
        """Calcula todas las características del sonido a partir de las últimas muestras.
        Se llama una vez por fotograma; dt es la duración del fotograma en segundos."""
        e = dt * SR / 1024
        # Amplitud: volumen general (rms de las últimas muestras)
        rms = float(np.sqrt(np.mean(self.buf ** 2)))
        self.nivel = self.norm_amp.paso(rms, e)
        mag2 = np.abs(np.fft.rfft(self.buf * VENTANA)) ** 2  # potencia en cada frecuencia
        hay_senal = rms > 1e-4                                # ~ -80 dB: por debajo es silencio

        # Bandas: cuánta energía hay en graves, medios y agudos (cada una con su propio rango)
        crudas = {}  # energía de cada banda sin normalizar (se usa también para detectar golpes)
        for nombre, mascara in MASCARAS.items():
            crudas[nombre] = float(np.sqrt(ESCALA_RMS * mag2[mascara].sum()))
            self.car[nombre] = self.norm[nombre].paso(crudas[nombre], e)

        # Pulso: cuánto ha SUBIDO de golpe la energía de alguna banda desde el fotograma anterior,
        # comparado con lo habitual en esa banda. Un ataque (bombo, caja, platillo) sube muy por
        # encima de lo normal -> 1. Un sonido estable casi no sube -> 0.
        subida = 0.0
        for b in BANDAS:
            if self.banda_media[b] == 0.0:  # primer fotograma: no hay historia, no hay golpe
                self.banda_media[b] = self.banda_prev[b] = crudas[b]
            subida = max(subida, max(0.0, crudas[b] - self.banda_prev[b]) / (self.banda_media[b] + 2e-3))
            self.banda_media[b] += (crudas[b] - self.banda_media[b]) * (1 - 0.97 ** e)
            self.banda_prev[b] = crudas[b]
        objetivo = min(1.0, max(0.0, (subida - 0.35) / 0.5)) if hay_senal else 0.0
        p = self.car["pulso"]
        self.car["pulso"] = p + (objetivo - p) * (1 - (0.15 if objetivo > p else 0.85) ** e)

        self.car["amplitud"] = self.nivel

    def valor(self):
        """Valor (0..1) de la característica que controla el parámetro de este canal."""
        return self.car[self.caracteristica]

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
            silencio = np.zeros(n, dtype="float32")
            self.recibir(silencio)  # en pausa todo cae a cero
            return silencio
        # El módulo (%) hace que al llegar al final vuelva al principio (bucle)
        idx = (self.pos + np.arange(n)) % len(self.datos)
        bloque = self.datos[idx]
        self.pos = (self.pos + n) % len(self.datos)
        # rms = raíz de la media de los cuadrados: medida estándar del volumen
        self.recibir(bloque)
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


def parsear_mic(fuente):
    """Interpreta 'mic[:dispositivo[:entrada]]' y devuelve (dispositivo, entrada)."""
    partes = fuente.split(":")
    disp = partes[1] if len(partes) > 1 and partes[1] != "" else None  # None = dispositivo por defecto
    if disp is not None and disp.isdigit():
        disp = int(disp)  # número de dispositivo; si no, se busca por nombre
    entrada = int(partes[2]) if len(partes) > 2 else 0  # 0 = primera entrada del dispositivo
    return disp, entrada


class EntradaMic:
    """Un dispositivo de entrada abierto UNA sola vez y compartido por varios canales.

    Abrir el mismo dispositivo varias veces suele fallar, así que se abre un único stream
    con todas las entradas necesarias y cada canal lee la suya.
    """

    def __init__(self, disp, oyentes, bloque=1024):
        self.oyentes = oyentes  # lista de (numero_de_entrada, Canal)
        n = max(k for k, _ in oyentes) + 1  # entradas que hay que abrir
        try:
            disponibles = sd.query_devices(disp, "input")["max_input_channels"]
        except Exception as e:
            sys.exit(f"No se encontró el dispositivo de entrada {disp!r}: {e}\nUsa --listar para verlos.")
        if n > disponibles:
            sys.exit(f"El dispositivo {disp!r} solo tiene {disponibles} entrada(s) y pediste la {n} "
                     f"(se numeran desde 0). Usa --listar para ver sus entradas.")
        self.stream = sd.InputStream(device=disp, channels=n, samplerate=SR, blocksize=bloque,
                                     latency="low", callback=self._cb)
        self.stream.start()

    def _cb(self, indata, frames, t, status):
        # sounddevice llama aquí con cada bloque de audio; indata[:, k] es la entrada k
        for k, canal in self.oyentes:
            canal.recibir(indata[:, k])


def abrir_entradas(canales, bloque=1024):
    """Agrupa los canales de micrófono por dispositivo y abre cada dispositivo una vez."""
    grupos = {}
    for c in canales:
        if c.mic is not None:
            disp, entrada = c.mic
            grupos.setdefault(disp, []).append((entrada, c))
    return [EntradaMic(disp, oyentes, bloque) for disp, oyentes in grupos.items()]


def indice_asio(disp):
    """Devuelve el índice del dispositivo ASIO pedido: número, parte del nombre o el predeterminado."""
    apis = sd.query_hostapis()
    asio = next((i for i, h in enumerate(apis) if "ASIO" in h["name"].upper()), None)
    if asio is None:
        sys.exit("No hay soporte ASIO. Solo existe en Windows, con el driver ASIO de tu interfaz "
                 "instalado (o ASIO4ALL) y sounddevice instalado con pip. Usa --listar para ver "
                 "qué dispositivos y APIs detecta.")
    if isinstance(disp, int):
        if sd.query_devices(disp)["hostapi"] != asio:
            sys.exit(f"El dispositivo {disp} no es ASIO. Usa --listar y elige uno marcado como ASIO.")
        return disp
    if disp:  # texto: se busca entre los dispositivos ASIO por parte del nombre
        for i, d in enumerate(sd.query_devices()):
            if d["hostapi"] == asio and disp.lower() in d["name"].lower():
                return i
        sys.exit(f"No se encontró un dispositivo ASIO que contenga {disp!r}. Usa --listar.")
    for clave in ("default_input_device", "default_output_device"):
        if apis[asio][clave] >= 0:
            return apis[asio][clave]
    sys.exit("No hay un dispositivo ASIO predeterminado. Indica uno con --asio NOMBRE.")


def abrir_asio(canales, dispositivo, bloque):
    """Abre la interfaz por ASIO con UN solo stream.

    ASIO solo permite abrir un driver una vez, así que todas las entradas de micrófono y la
    reproducción de los archivos pasan por el mismo stream (entrada y salida a la vez).
    """
    dev = indice_asio(dispositivo)
    info = sd.query_devices(dev)
    mics = [(c.mic[1], c) for c in canales if c.mic is not None]  # (entrada física, canal)
    archivos = [c for c in canales if c.datos is not None]
    entradas = sorted({k for k, _ in mics})   # solo se abren las entradas físicas que se usan
    col = {k: i for i, k in enumerate(entradas)}  # entrada física -> columna dentro de indata
    if entradas and entradas[-1] >= info["max_input_channels"]:
        sys.exit(f"{info['name']} tiene {info['max_input_channels']} entradas y pediste la "
                 f"{entradas[-1]} (se numeran desde 0).")
    n_out = min(2, info["max_output_channels"])
    if archivos and n_out == 0:
        sys.exit(f"{info['name']} no tiene salidas para reproducir los archivos.")

    def leer_entradas(indata, frames):
        for k, c in mics:
            c.recibir(indata[:, col[k]])

    def mezclar(outdata, frames):
        mezcla = sum(c.leer_bloque(frames) for c in archivos) * 0.5  # 0.5 evita saturar
        outdata[:] = mezcla[:, None]  # el mismo audio mono en las salidas 1 y 2

    sel_in = sd.AsioSettings(channel_selectors=entradas) if entradas else None
    sel_out = sd.AsioSettings(channel_selectors=list(range(n_out))) if archivos else None
    comun = dict(device=dev, samplerate=SR, blocksize=bloque, latency="low")
    if mics and archivos:  # entrada y salida simultáneas
        def cb(indata, outdata, frames, t, status):
            leer_entradas(indata, frames)
            mezclar(outdata, frames)
        stream = sd.Stream(channels=(len(entradas), n_out), extra_settings=(sel_in, sel_out),
                           callback=cb, **comun)
    elif mics:  # solo micrófonos
        stream = sd.InputStream(channels=len(entradas), extra_settings=sel_in,
                                callback=lambda indata, frames, t, status: leer_entradas(indata, frames),
                                **comun)
    else:  # solo archivos
        stream = sd.OutputStream(channels=n_out, extra_settings=sel_out,
                                 callback=lambda outdata, frames, t, status: mezclar(outdata, frames),
                                 **comun)
    stream.start()
    lat = stream.latency
    lat = max(lat) if isinstance(lat, (tuple, list)) else lat
    print(f"ASIO: {info['name']} | bloque {bloque} muestras | latencia ~{lat * 1000:.1f} ms")
    return stream


def abrir_audio(canales, a):
    """Abre todo el audio: por ASIO (un stream) o por la API normal del sistema."""
    bloque = a.bloque or (256 if a.asio else 1024)
    if a.asio:
        disp = "" if a.asio is True else (int(a.asio) if a.asio.isdigit() else a.asio)
        return [abrir_asio(canales, disp, bloque)]
    streams = list(abrir_entradas(canales, bloque))  # micrófonos: un stream por dispositivo
    archivos = [c for c in canales if c.datos is not None]
    if archivos:
        # Un único stream de salida reproduce la mezcla de todos los archivos.
        # Al pedir el bloque de cada canal también se actualiza su nivel, así el
        # análisis queda sincronizado con lo que se escucha.
        def salida(outdata, frames, t, status):
            mezcla = sum(c.leer_bloque(frames) for c in archivos) * 0.5  # 0.5 evita saturar
            outdata[:] = mezcla[:, None]
        dev = None if a.salida is None else (int(a.salida) if a.salida.isdigit() else a.salida)
        out = sd.OutputStream(device=dev, samplerate=SR, channels=1, blocksize=bloque,
                              latency="low", callback=salida)
        out.start()
        streams.append(out)
    return streams


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
        pantalla.blit(fuente_ui.render(f"{i + 1} {c.parametro}:{c.caracteristica}", True, GRIS), (10, y + 3))
        if c.datos is not None:
            # Solo los archivos tienen botones y barra de progreso
            x = 200
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
            pantalla.blit(fuente_ui.render("mic", True, GRIS), (200, y + 3))
        # Medidor de nivel (para archivos y micrófonos)
        nivel = pygame.Rect(760, y + 5, 80, 10)
        pygame.draw.rect(pantalla, GRIS, nivel, 1)
        pygame.draw.rect(pantalla, GRIS, (nivel.x, nivel.y, int(nivel.w * c.valor()), nivel.h))
    return zonas


def parsear_param(texto):
    """Interpreta 'parametro' o 'parametro:caracteristica' (por ejemplo altura:graves)."""
    p, _, car = texto.partition(":")
    car = car or "amplitud"
    if p not in PARAMS:
        raise argparse.ArgumentTypeError(f"parámetro desconocido {p!r}; opciones: {', '.join(PARAMS)}")
    if car not in CARACT:
        raise argparse.ArgumentTypeError(f"característica desconocida {car!r}; opciones: {', '.join(CARACT)}")
    return p, car


def main():
    # ---- Argumentos de línea de comandos ----
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--canal", action="append", help="archivo o mic (x4)")
    ap.add_argument("--listar", action="store_true", help="lista los dispositivos de audio y sale")
    ap.add_argument("--asio", nargs="?", const=True, default=False, metavar="DISPOSITIVO",
                    help="usa ASIO (Windows): sin valor = driver ASIO predeterminado; o número/nombre")
    ap.add_argument("--bloque", type=int, default=None,
                    help="tamaño de bloque en muestras: menor = menos latencia (por defecto 256 con ASIO, 1024 sin él)")
    ap.add_argument("--salida", default=None,
                    help="dispositivo de salida para los archivos (número o nombre); no aplica con --asio")
    ap.add_argument("-p", "--param", action="append", type=parsear_param,
                    help="parámetro[:característica] por canal, p. ej. altura:graves")
    a = ap.parse_args()
    if a.listar:  # ayuda para saber qué número de dispositivo y entrada usar en mic:DISP:ENTRADA
        print(sd.query_devices())
        return
    if len(a.canal or []) != 4:
        sys.exit("Necesitas exactamente 4 canales (-c).")
    params = a.param or [(p, "amplitud") for p in PARAMS[:4]]  # por defecto: altura, rotacion, color, escala
    if len(params) != 4:
        sys.exit("Necesitas 4 parámetros (-p), uno por canal.")

    # ---- Audio ----
    canales = [Canal(f, p, car) for f, (p, car) in zip(a.canal, params)]
    streams = abrir_audio(canales, a)  # se guarda la lista para que los streams sigan abiertos
    archivos = [c for c in canales if c.datos is not None]  # los que tienen transporte (pausa, etc.)

    # ---- Ventana ----
    pygame.init()
    pantalla = pygame.display.set_mode((1000, 640), pygame.RESIZABLE)
    pygame.display.set_caption("Visualizador")
    fuente_ui = pygame.font.SysFont("monospace", 14)
    reloj = pygame.time.Clock()
    pts0, tris, base_z = crear_malla()
    # Cada triángulo se separa una distancia distinta (entre 0.4x y 1.6x), fija para toda la ejecución
    variacion = 0.4 + 1.2 * np.random.default_rng(3).random(len(tris))
    angulo, mostrar, zonas, dt = 0.0, True, [], 1 / 60  # zonas: áreas clicables del fotograma anterior

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
            c.analizar(dt)               # espectro, pulso...
            v[c.parametro] = c.valor()   # la característica elegida mueve el parámetro
        t = pygame.time.get_ticks() / 1000  # tiempo en segundos, para animar el "ruido"

        # ---- Deformar la malla según los parámetros ----
        angulo += 0.004 + v["rotacion"] * 0.08  # gira siempre un poco; el audio la acelera
        giro = angulo + v["rotacion"] * 0.8     # además, empuja el giro al instante (se nota el pulso)
        pts = pts0.copy()
        pts[:, 2] = base_z * (0.2 + 2.2 * v["altura"])  # altura: relieve más o menos marcado
        # ruido: los vértices vibran en x e y con una onda que depende del tiempo
        pts[:, 0] += np.sin(t * 6 + pts0[:, 1] * 5) * v["ruido"] * 0.08
        pts[:, 1] += np.cos(t * 5 + pts0[:, 0] * 5) * v["ruido"] * 0.08
        pts *= 0.65 + v["escala"] * 0.6  # escala: de 0.65x (silencio) a 1.25x (nivel máximo)

        # ---- Rotación 3D ----
        # 1) giro alrededor del eje vertical (el ángulo que crece con el tiempo)
        ca, sa = math.cos(giro), math.sin(giro)
        x = pts[:, 0] * ca - pts[:, 1] * sa
        y = pts[:, 0] * sa + pts[:, 1] * ca
        z = pts[:, 2]
        # 2) inclinación fija hacia la cámara para ver la malla desde arriba en perspectiva
        tilt = 1.0
        y2 = y * math.cos(tilt) - z * math.sin(tilt)
        z2 = y * math.sin(tilt) + z * math.cos(tilt)
        rot = np.stack([x, y2, z2], axis=1)

        # ---- Sombreado plano: cada triángulo tiene un único brillo ----
        A, B, C = rot[tris[:, 0]], rot[tris[:, 1]], rot[tris[:, 2]]  # vértices de cada triángulo
        n = np.cross(B - A, C - A)  # normal de cada cara
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        luz = np.array([0.3, -0.5, 0.8])
        luz /= np.linalg.norm(luz)
        br = 0.25 + 0.75 * np.abs(n @ luz)  # más alineada con la luz = más brillo

        # ---- Fragmentos: cada triángulo se despega de la figura; a más volumen, más lejos ----
        vf = v["fragmentos"]
        if vf > 0:
            G = (A + B + C) / 3                          # centro de cada triángulo
            d = G - rot.mean(axis=0)                     # vector desde el centro de la figura hasta el triángulo
            u = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)  # misma dirección, largo 1
            arriba = n * np.where(n[:, 2:3] < 0, -1.0, 1.0)  # normal orientada hacia arriba
            # Tres componentes: empuje hacia afuera con distancia propia de cada triángulo (variacion),
            # empuje a lo largo de su normal (separa también en profundidad) y una expansión
            # proporcional a la distancia al centro (los más lejanos se alejan más).
            off = (u * 0.2 + arriba * 0.25) * (vf * variacion)[:, None] + d * 0.25 * vf
            A, B, C = A + off, B + off, C + off          # los 3 vértices se mueven juntos: el triángulo no se deforma

        # ---- Proyección a 2D con perspectiva simple (lo lejano se ve más pequeño) ----
        w, h = pantalla.get_size()
        f = min(w, h) * 0.55

        def proyectar(P):
            prof = 1.0 / (4.0 - P[:, 1] * 0.8)
            return np.stack([w / 2 + P[:, 0] * f * prof * 2.4, h / 2 - P[:, 2] * f * prof * 2.4], axis=1)

        SA, SB, SC = proyectar(A), proyectar(B), proyectar(C)  # cada triángulo se proyecta por separado
        # Algoritmo del pintor: se dibujan primero los triángulos más lejanos
        orden = np.argsort(-(A[:, 1] + B[:, 1] + C[:, 1]))

        # ---- Dibujo ----
        pantalla.fill((12, 12, 16))
        matiz = (0.6 + v["color"] * 0.9) % 1.0  # color: el audio recorre casi toda la rueda de colores
        for k in orden:
            r, g, b = colorsys.hsv_to_rgb((matiz + br[k] * 0.08) % 1.0, 0.65, min(1.0, br[k]))
            poli = [tuple(SA[k]), tuple(SB[k]), tuple(SC[k])]
            pygame.draw.polygon(pantalla, (int(r * 255), int(g * 255), int(b * 255)), poli)

        # La interfaz se dibuja encima; `zonas` queda listo para los clics del siguiente fotograma
        zonas = dibujar_ui(pantalla, fuente_ui, canales) if mostrar else []

        pygame.display.flip()
        dt = max(reloj.tick(60), 1) / 1000  # limita a 60 fps y guarda lo que duró el fotograma (en segundos)


if __name__ == "__main__":
    main()
