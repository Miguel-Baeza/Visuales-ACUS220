# Visualizador Audio (ACUS220)

Un visualizador 3D reactivo al audio desarrollado en Python. Genera en tiempo real una malla poligonal, mediante el análisis dinámico de hasta **4 señales de entrada de audio independientes** (archivos `.wav`, micrófonos o interfaces multicanal).

---

## Características principales

- **Control multicanal modular:** Recibe exactamente 4 entradas de audio independientes de manera simultánea.
- **Fuentes híbridas:** Admite combinar archivos de audio local (`.wav`), el micrófono por defecto del sistema, o canales específicos de interfaces de audio multicanal.
- **Rango dinámico adaptativo:** Normalización automática del nivel RMS (ataque rápido / caída suave) para garantizar que la reactividad visual se adapte dinámicamente al volumen de cada entrada.
- **Malla 3D personalizada:** Generación de malla irregular mediante trigonometría, proyección en perspectiva 2D y renderizado con sombreado plano.
- **Interfaz GUI integrada:** Controles de transporte interactivos para archivos (pausa, reanudación, salto de tiempo), medidores de nivel dinámicos y soporte para pantalla completa.

