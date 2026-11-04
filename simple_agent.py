"""Agente simple de Tetris: greedy de una pieza con las seis features de Dellacherie.

Basado en Thiery y Scherrer (2009), "Building Controllers for Tetris", seccion 2.1.
Es un controlador de una pieza (one-ply): solo usa el tablero y la pieza actual.
Para cada colocacion posible evalua el tablero resultante con la funcion lineal

    -(Landing height) + (Eroded piece cells) - (Row transitions)
    -(Column transitions) - 4*(Holes) - (Cumulative wells)

y juega la mejor. Mientras juega registra las seis features de cada jugada elegida
y al final entrega estadisticas (media, desviacion, minimo y maximo).

Reglas del juego usadas (simplificadas, como en el articulo, seccion 3.1):
    - Tablero de 10 x 20; el agente solo decide rotacion y columna (caida directa).
    - Una colocacion que deja alguna celda de la pieza sobre el borde superior
      lleva a Game Over y no se considera. La partida termina solo si todas las
      colocaciones llevan a Game Over (articulo, seccion 3.3).

Uso:
    python simple_agent.py --games 10 --pieces 1000 --seed 1
"""

import argparse
import csv
import datetime
import os
import random
import statistics

from tetris import (BLANK, BOARDHEIGHT, BOARDWIDTH, PIECES, TEMPLATEHEIGHT,
                    TEMPLATEWIDTH, getBlankBoard, getNewPiece, isValidPosition)

FEATURES = ['landing_height', 'eroded_piece_cells', 'row_transitions',
            'column_transitions', 'holes', 'cumulative_wells']

# Pesos de Dellacherie (articulo, seccion 2.1).
WEIGHTS = {
    'landing_height': -1.0,
    'eroded_piece_cells': 1.0,
    'row_transitions': -1.0,
    'column_transitions': -1.0,
    'holes': -4.0,
    'cumulative_wells': -1.0,
}


def piece_cells(piece):
    """Celdas (x, y) absolutas de la pieza; y = 0 es la fila superior."""
    template = PIECES[piece['shape']][piece['rotation']]
    return [(piece['x'] + dx, piece['y'] + dy)
            for dy in range(TEMPLATEHEIGHT)
            for dx in range(TEMPLATEWIDTH)
            if template[dy][dx] != BLANK]


def final_placements(board, piece):
    """Todas las colocaciones finales (rotacion y columna) tras la caida directa."""
    placements = []
    for rotation in range(len(PIECES[piece['shape']])):
        for x in range(-2, BOARDWIDTH):
            candidate = dict(piece, rotation=rotation, x=x, y=-2)
            if not isValidPosition(board, candidate):
                continue
            while isValidPosition(board, candidate, adjY=1):
                candidate['y'] += 1
            placements.append(candidate)
    return placements


def place(board, piece):
    """Fija la pieza y elimina filas. Devuelve (tablero_nuevo, lineas, celdas_erosionadas).

    Celdas erosionadas = (filas eliminadas) x (celdas de la pieza que estaban en
    esas filas), como define el articulo (Tabla 1).
    """
    new_board = [column[:] for column in board]
    cells = piece_cells(piece)
    for x, y in cells:
        new_board[x][y] = piece['color']

    full_rows = {y for y in range(BOARDHEIGHT)
                 if all(new_board[x][y] != BLANK for x in range(BOARDWIDTH))}
    lines = len(full_rows)
    eroded = lines * sum(1 for _, y in cells if y in full_rows)

    if lines:
        for x in range(BOARDWIDTH):
            kept = [new_board[x][y] for y in range(BOARDHEIGHT) if y not in full_rows]
            new_board[x] = [BLANK] * (BOARDHEIGHT - len(kept)) + kept
    return new_board, lines, eroded


def _filled(board, x, y):
    """Las paredes laterales y el suelo cuentan como celdas llenas."""
    if x < 0 or x >= BOARDWIDTH or y >= BOARDHEIGHT:
        return True
    return board[x][y] != BLANK


def compute_features(board, cells, eroded):
    """Las seis features sobre el tablero resultante.

    `cells` son las celdas de la pieza antes de eliminar filas y `eroded` las
    celdas erosionadas de esa jugada.

    - landing_height: altura (fila inferior = 1) del punto medio de la pieza.
    - eroded_piece_cells: ver `place`.
    - row_transitions: cambios lleno/vacio al recorrer cada fila de pared a pared.
    - column_transitions: cambios lleno/vacio en cada columna, con el suelo lleno.
    - holes: celdas vacias con alguna celda llena encima en su columna.
    - cumulative_wells: por cada pozo (celdas vacias alcanzables desde arriba con
      vecinos llenos a ambos lados) de profundidad d, suma 1 + 2 + ... + d.
    """
    landing_height = BOARDHEIGHT - sum(y for _, y in cells) / len(cells)

    row_transitions = 0
    for y in range(BOARDHEIGHT):
        for x in range(-1, BOARDWIDTH):
            if _filled(board, x, y) != _filled(board, x + 1, y):
                row_transitions += 1

    column_transitions = 0
    holes = 0
    cumulative_wells = 0
    for x in range(BOARDWIDTH):
        previous_filled = True  # el suelo
        for y in range(BOARDHEIGHT - 1, -1, -1):
            current = board[x][y] != BLANK
            if current != previous_filled:
                column_transitions += 1
            previous_filled = current

        top = next((y for y in range(BOARDHEIGHT) if board[x][y] != BLANK), BOARDHEIGHT)
        holes += sum(1 for y in range(top, BOARDHEIGHT) if board[x][y] == BLANK)

        depth = 0
        for y in range(top):  # celdas vacias alcanzables desde arriba
            if _filled(board, x - 1, y) and _filled(board, x + 1, y):
                depth += 1
                cumulative_wells += depth
            else:
                depth = 0

    return {
        'landing_height': landing_height,
        'eroded_piece_cells': eroded,
        'row_transitions': row_transitions,
        'column_transitions': column_transitions,
        'holes': holes,
        'cumulative_wells': cumulative_wells,
    }


class SimpleGreedyAgent:
    """Greedy de una pieza: elige la colocacion con mayor puntaje lineal."""

    def __init__(self, weights=None):
        self.weights = dict(WEIGHTS if weights is None else weights)

    def options(self, board, piece):
        """Colocaciones validas de `piece` como tuplas (puntaje, colocacion, tablero, lineas, features).

        Se descartan las colocaciones que dejan celdas sobre el borde superior
        (llevan a Game Over).
        """
        result = []
        for candidate in final_placements(board, piece):
            cells = piece_cells(candidate)
            if any(y < 0 for _, y in cells):
                continue
            new_board, lines, eroded = place(board, candidate)
            features = compute_features(new_board, cells, eroded)
            score = sum(self.weights[name] * features[name] for name in FEATURES)
            result.append((score, candidate, new_board, lines, features))
        return result

    def best_one_ply(self, options):
        """La opcion de mayor puntaje; en empate, la primera (menor rotacion, mas a la izquierda)."""
        best = None
        for option in options:
            if best is None or option[0] > best[0]:
                best = option
        return best

    def choose(self, board, piece, next_piece=None):
        """Devuelve (colocacion, tablero_resultante, lineas, features) o None si hay Game Over.

        `next_piece` se ignora: este agente solo usa la pieza actual.
        """
        best = self.best_one_ply(self.options(board, piece))
        return None if best is None else best[1:]


def play_game(agent, max_pieces, game_id):
    """Juega una partida. Devuelve (piezas, lineas, jugadas) con una fila por jugada."""
    board = getBlankBoard()
    moves = []
    total_lines = 0
    # Se sortea una pieza por adelantado para que todos los agentes vean la misma
    # secuencia con la misma semilla, y los que usan lookahead conozcan la siguiente.
    piece = getNewPiece()
    next_piece = getNewPiece()
    for turn in range(1, max_pieces + 1):
        choice = agent.choose(board, piece, next_piece)
        if choice is None:
            return turn - 1, total_lines, moves, True  # Game Over
        placement, board, lines, features = choice
        total_lines += lines
        moves.append({'game': game_id, 'turn': turn, 'shape': piece['shape'],
                      'rotation': placement['rotation'], 'x': placement['x'],
                      'lines': lines, **features})
        piece, next_piece = next_piece, getNewPiece()
    return max_pieces, total_lines, moves, False


def describe(values):
    """Media, desviacion estandar, minimo y maximo de una lista de valores."""
    return (statistics.mean(values), statistics.pstdev(values), min(values), max(values))


def parse_args(description, extra=None):
    parser = argparse.ArgumentParser(description=description)
    if extra is not None:
        extra(parser)
    parser.add_argument('--games', type=int, default=10, help='numero de partidas')
    parser.add_argument('--pieces', type=int, default=1000, help='piezas maximas por partida')
    parser.add_argument('--seed', type=int, default=1, help='semilla base (partida i usa seed + i)')
    parser.add_argument('--out', default=None, help='carpeta de salida (por defecto runs/simple_<fecha>)')
    return parser.parse_args()


def run_experiment(agent, args, name):
    """Juega las partidas con `agent`, imprime las estadisticas y guarda moves.csv."""
    all_moves = []
    games = []
    for game in range(1, args.games + 1):
        random.seed(args.seed + game)
        pieces, lines, moves, game_over = play_game(agent, args.pieces, game)
        all_moves.extend(moves)
        games.append((pieces, lines, game_over))
        estado = 'Game Over' if game_over else 'llego al limite'
        print(f'Partida {game}: piezas={pieces} lineas={lines} ({estado})')

    print(f'\nJugadas analizadas: {len(all_moves)} en {args.games} partidas')
    print(f'{"feature":22s} {"media":>9s} {"desv.":>9s} {"min":>8s} {"max":>8s}')
    for name in FEATURES:
        mean, std, low, high = describe([move[name] for move in all_moves])
        print(f'{name:22s} {mean:9.2f} {std:9.2f} {low:8.1f} {high:8.1f}')

    survived = sum(1 for _, _, over in games if not over)
    lines_mean, lines_std, _, _ = describe([lines for _, lines, _ in games])
    print(f'\nLineas por partida: {lines_mean:.1f} (desv. {lines_std:.1f})')
    print(f'Partidas que llegaron al limite de {args.pieces} piezas: {survived}/{args.games}')

    out_dir = args.out or os.path.join(
        'runs', name + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'moves.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['game', 'turn', 'shape', 'rotation', 'x',
                                               'lines'] + FEATURES)
        writer.writeheader()
        writer.writerows(all_moves)
    print(f'Jugadas guardadas en {os.path.join(out_dir, "moves.csv")}')


def main():
    args = parse_args('Agente greedy simple con estadisticas de features.')
    run_experiment(SimpleGreedyAgent(), args, 'simple')


if __name__ == '__main__':
    main()
