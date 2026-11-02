# Seminario: Tetris

Implementación de un juego de Tetris en Python usando Pygame.

## Requisitos

- Python 3.10 o superior
- Pygame

## Instalación

Se recomienda usar un entorno virtual:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

## Ejecución

```powershell
py tetris.py
```

## Controles

- Flechas izquierda/derecha: mover la pieza
- Flecha abajo: bajar la pieza
- Flecha arriba: rotar la pieza
- Espacio: caída instantánea
- `P`: pausar
- `Esc`: salir

El código del juego se basa en el ejemplo de Tetris de Al Sweigart,
publicado bajo una licencia Simplified BSD, cuya atribución se conserva
en [`tetris.py`](./tetris.py).
