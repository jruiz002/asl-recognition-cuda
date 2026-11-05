"""Proyecto 3 (CC3069): aceleración con CUDA del Laboratorio 3 de Data Science (ASL).

Paquete nuevo que NO modifica el código original del laboratorio (`src/`,
`entrenar_modelos.py`, ...). Se organiza en tres subpaquetes:

- ``common``     utilidades compartidas: configuración, rutas, tiempos, hardware y monitores.
- ``sequential`` versión secuencial = código original del lab, solo instrumentado con timers.
- ``parallel``   versión optimizada: FASE 1 en CPU multinúcleo + FASE 2 en GPU (CUDA).
"""
