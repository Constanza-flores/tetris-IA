"""Agente de Tetris basado en heuristica de Dellacherie y busqueda con lookahead.

Este modulo no depende de Pygame en tiempo de ejecucion: reutiliza las piezas y
las funciones de validacion de tetris.py, pero solo calcula el tablero en memoria.
Ejecutar `python agent.py` juega varias partidas headless y muestra estadisticas.

Convenciones (heredadas de tetris.py):
    board[x][y] con x en columnas (0..BOARDWIDTH-1) e y en filas (0 = arriba).
    BLANK marca una celda vacia.
"""

import argparse
import json
import random
import statistics

from tetris import (BLANK, BOARDHEIGHT, BOARDWIDTH, PIECES, TEMPLATEHEIGHT,
                    TEMPLATEWIDTH, getBlankBoard, getNewPiece, isValidPosition)

# Pesos de Dellacherie reportados por Algorta y Simsek (2019), seccion 3.2.
# Las features se combinan como: -4*holes - wells - row_trans - col_trans
#                                -landing + eroded
WEIGHTS = {
    'holes': -4.0,
    'wells': -1.0,
    'row_transitions': -1.0,
    'column_transitions': -1.0,
    'landing_height': -1.0,
    'eroded_cells': 1.0,
}


def piece_cells(piece):
    """Devuelve las celdas (x, y) absolutas ocupadas por la pieza."""
    template = PIECES[piece['shape']][piece['rotation']]
    cells = []
    for dy in range(TEMPLATEHEIGHT):
        for dx in range(TEMPLATEWIDTH):
            if template[dy][dx] != BLANK:
                cells.append((piece['x'] + dx, piece['y'] + dy))
    return cells


def legal_placements(board, piece):
    """Genera todas las colocaciones finales (hard drop) de la pieza.

    Cada colocacion es una copia de la pieza con rotation, x e y definidos.
    """
    placements = []
    for rotation in range(len(PIECES[piece['shape']])):
        for x in range(-2, BOARDWIDTH + 2):
            candidate = dict(piece, rotation=rotation, x=x, y=-2)
            if not isValidPosition(board, candidate):
                continue
            while isValidPosition(board, candidate, adjY=1):
                candidate['y'] += 1
            placements.append(candidate)
    return placements


def place_piece(board, piece):
    """Devuelve (nuevo_tablero, lineas_eliminadas, celdas_erosionadas).

    El tablero original no se modifica. Las celdas erosionadas siguen la
    variante usada habitualmente: lineas eliminadas * celdas de la pieza en esas lineas.
    """
    new_board = [column[:] for column in board]
    cells = piece_cells(piece)
    for x, y in cells:
        if y >= 0:
            new_board[x][y] = piece['color']

    full_rows = [y for y in range(BOARDHEIGHT)
                 if all(new_board[x][y] != BLANK for x in range(BOARDWIDTH))]
    lines = len(full_rows)
    piece_cells_in_full_rows = sum(1 for _, y in cells if y in full_rows)
    eroded = lines * piece_cells_in_full_rows

    if lines:
        for x in range(BOARDWIDTH):
            kept = [new_board[x][y] for y in range(BOARDHEIGHT) if y not in full_rows]
            new_board[x] = [BLANK] * (BOARDHEIGHT - len(kept)) + kept

    return new_board, lines, eroded


def _filled(board, x, y):
    """Las paredes laterales y el fondo cuentan como celdas llenas."""
    if x < 0 or x >= BOARDWIDTH or y >= BOARDHEIGHT:
        return True
    return board[x][y] != BLANK


def board_features(board, piece, lines, eroded):
    """Calcula las features de Dellacherie sobre el tablero resultante."""
    cells = piece_cells(piece)
    landing_height = BOARDHEIGHT - sum(y for _, y in cells) / len(cells)

    row_transitions = 0
    for y in range(BOARDHEIGHT):
        for x in range(-1, BOARDWIDTH):
            if _filled(board, x, y) != _filled(board, x + 1, y):
                row_transitions += 1

    column_transitions = 0
    holes = 0
    wells = 0
    for x in range(BOARDWIDTH):
        # Transiciones verticales: el fondo cuenta como lleno, el borde superior no.
        previous_filled = True
        for y in range(BOARDHEIGHT - 1, -1, -1):
            current = board[x][y] != BLANK
            if current != previous_filled:
                column_transitions += 1
            previous_filled = current

        # Agujeros: celdas vacias con al menos una llena arriba en la misma columna.
        seen_filled = False
        for y in range(BOARDHEIGHT):
            if board[x][y] != BLANK:
                seen_filled = True
            elif seen_filled:
                holes += 1

        # Pozos: celdas vacias con vecinos laterales llenos; cada tramo de largo d
        # aporta 1 + 2 + ... + d (pozos cumulativos).
        depth = 0
        for y in range(BOARDHEIGHT):
            is_well = (board[x][y] == BLANK and _filled(board, x - 1, y)
                       and _filled(board, x + 1, y))
            if is_well:
                depth += 1
                wells += depth
            else:
                depth = 0

    return {
        'holes': holes,
        'wells': wells,
        'row_transitions': row_transitions,
        'column_transitions': column_transitions,
        'landing_height': landing_height,
        'eroded_cells': eroded,
    }


def evaluate(board, piece, placed_board, lines, eroded, weights=WEIGHTS):
    """Puntaje lineal de la heuristica para la colocacion de `piece` sobre `board`."""
    features = board_features(placed_board, piece, lines, eroded)
    return sum(weights[name] * value for name, value in features.items())


class DellacherieAgent:
    """Elige la colocacion con mejor puntaje, con lookahead opcional de la pieza siguiente.

    Con lookahead, para cada colocacion de la pieza actual se busca la mejor
    colocacion de la siguiente pieza y se suman ambos puntajes. Solo se expanden
    las `beam_width` mejores colocaciones de la pieza actual para controlar el costo.
    """

    def __init__(self, lookahead=True, beam_width=5, weights=None):
        self.lookahead = lookahead
        self.beam_width = beam_width
        self.weights = dict(WEIGHTS if weights is None else weights)

    def _score_placements(self, board, piece):
        scored = []
        for candidate in legal_placements(board, piece):
            new_board, lines, eroded = place_piece(board, candidate)
            value = evaluate(board, candidate, new_board, lines, eroded, self.weights)
            scored.append((value, candidate, new_board, lines))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    def choose(self, board, piece, next_piece=None):
        """Devuelve la colocacion elegida (pieza con rotation, x, y) o None si no hay jugadas."""
        scored = self._score_placements(board, piece)
        if not scored:
            return None

        if not (self.lookahead and next_piece is not None):
            return scored[0][1]

        best_candidate = None
        best_total = float('-inf')
        for value, candidate, new_board, _ in scored[:self.beam_width]:
            next_scored = self._score_placements(new_board, next_piece)
            next_value = next_scored[0][0] if next_scored else -1000.0
            total = value + next_value
            if total > best_total:
                best_total = total
                best_candidate = candidate
        return best_candidate


def count_holes(board):
    """Celdas vacias con al menos una celda llena encima en la misma columna."""
    holes = 0
    for x in range(BOARDWIDTH):
        seen_filled = False
        for y in range(BOARDHEIGHT):
            if board[x][y] != BLANK:
                seen_filled = True
            elif seen_filled:
                holes += 1
    return holes


def max_height(board):
    """Altura de la columna mas alta (0 si el tablero esta vacio)."""
    height = 0
    for x in range(BOARDWIDTH):
        for y in range(BOARDHEIGHT):
            if board[x][y] != BLANK:
                height = max(height, BOARDHEIGHT - y)
                break
    return height


def reward(lines, delta_holes, delta_height, game_over):
    """Recompensa del Taller 4: 10*L + 1 - 2*dHuecos - 0.5*dAlturaMax, -100 si termina."""
    if game_over:
        return -100.0
    return 10 * lines + 1 - 2 * delta_holes - 0.5 * delta_height


def play_game(agent, max_pieces=1000, log=None, game_id=0):
    """Juega una partida headless. Devuelve (piezas colocadas, lineas eliminadas).

    Si `log` es una lista, se le agrega un registro por turno con el estado,
    la accion, la recompensa y el resultado (transicion s, a, r, s').
    """
    board = getBlankBoard()
    piece = getNewPiece()
    next_piece = getNewPiece()
    total_lines = 0
    pieces_placed = 0

    while pieces_placed < max_pieces:
        if not isValidPosition(board, piece):
            break  # Game Over: la pieza no cabe al aparecer.

        choice = agent.choose(board, piece, next_piece)
        if choice is None:
            break  # Sin jugadas legales.

        board_before = [column[:] for column in board]
        holes_before = count_holes(board_before)
        height_before = max_height(board_before)

        board, lines, _ = place_piece(board, choice)
        total_lines += lines
        pieces_placed += 1

        # s = (tablero, pieza actual, pieza siguiente); la pieza que entra despues es nueva.
        new_piece = getNewPiece()
        game_over = not isValidPosition(board, next_piece)
        r = reward(lines,
                   count_holes(board) - holes_before,
                   max_height(board) - height_before,
                   game_over)

        if log is not None:
            log.append({
                'game': game_id,
                'turn': pieces_placed,
                'board': [''.join('1' if cell != BLANK else '0' for cell in column)
                          for column in board_before],
                'piece': piece['shape'],
                'next_piece': next_piece['shape'],
                'action': {'rotation': choice['rotation'], 'x': choice['x']},
                'lines': lines,
                'reward': r,
                'done': game_over,
            })

        if game_over:
            break
        piece, next_piece = next_piece, new_piece

    return pieces_placed, total_lines


def main():
    parser = argparse.ArgumentParser(description='Juega Tetris headless con el agente de Dellacherie.')
    parser.add_argument('--games', type=int, default=10, help='numero de partidas')
    parser.add_argument('--pieces', type=int, default=1000, help='piezas maximas por partida')
    parser.add_argument('--beam', type=int, default=5, help='colocaciones expandidas en lookahead')
    parser.add_argument('--no-lookahead', action='store_true', help='usar solo la pieza actual')
    parser.add_argument('--seed', type=int, default=None, help='semilla para reproducir partidas')
    parser.add_argument('--out', default=None,
                        help='archivo JSONL donde guardar cada turno (s, a, r, s\')')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    agent = DellacherieAgent(lookahead=not args.no_lookahead, beam_width=args.beam)
    log = [] if args.out else None
    results = []
    for game in range(1, args.games + 1):
        pieces, lines = play_game(agent, max_pieces=args.pieces, log=log, game_id=game)
        results.append((pieces, lines))
        print(f'Partida {game}: piezas={pieces} lineas={lines}')

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            for record in log:
                f.write(json.dumps(record) + '\n')
        print(f'Turnos guardados: {len(log)} en {args.out}')

    lines_list = [lines for _, lines in results]
    pieces_list = [pieces for pieces, _ in results]
    print('---')
    print(f'Lineas promedio: {statistics.mean(lines_list):.1f} (desv. {statistics.pstdev(lines_list):.1f})')
    print(f'Piezas promedio: {statistics.mean(pieces_list):.1f}')


if __name__ == '__main__':
    main()
