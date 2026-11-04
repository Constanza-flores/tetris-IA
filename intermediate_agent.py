"""Agente intermedio de Tetris: igual al simple, pero con lookahead de la pieza siguiente.

Es el controlador de dos piezas (two-ply) descrito en Thiery y Scherrer (2009),
seccion 2 y figura 2b. Mismas features y mismos pesos que el agente simple
(simple_agent.py); lo unico que cambia es como se decide:

    1. Para cada colocacion de la pieza actual se obtiene un tablero.
    2. Sobre ese tablero se prueba cada colocacion de la pieza siguiente y se
       evalua el tablero resultante con la funcion lineal.
    3. Cada colocacion de la pieza actual vale lo que su mejor continuacion.
       Se juega la colocacion con mayor valor.

Reglas de Game Over (articulo, seccion 3.3): una colocacion de la pieza actual
sin ninguna continuacion posible de la siguiente se descarta. Si todas estan en
ese caso, se juega la mejor segun la evaluacion de una pieza y la partida
termina en el turno siguiente.

Las features registradas por jugada son las de la colocacion de la pieza actual
(la jugada que realmente se ejecuta), igual que en el agente simple.

Uso:
    python intermediate_agent.py --games 10 --pieces 1000 --seed 1
"""

from simple_agent import SimpleGreedyAgent, parse_args, run_experiment


class LookaheadAgent(SimpleGreedyAgent):
    """Greedy de dos piezas con los mismos pesos que el agente simple."""

    def __init__(self, weights=None, beam=None):
        super().__init__(weights)
        self.beam = beam  # si no es None, solo se expanden las mejores `beam` colocaciones

    def choose(self, board, piece, next_piece=None):
        """Devuelve (colocacion, tablero_resultante, lineas, features) o None si hay Game Over."""
        options = self.options(board, piece)
        if not options:
            return None
        if next_piece is None:
            return self.best_one_ply(options)[1:]

        candidates = options
        if self.beam is not None:
            candidates = sorted(options, key=lambda option: option[0], reverse=True)[:self.beam]

        best = None
        best_value = float('-inf')
        for option in candidates:
            continuations = self.options(option[2], next_piece)
            if not continuations:
                continue  # esta colocacion lleva a Game Over con la pieza siguiente
            value = max(continuation[0] for continuation in continuations)
            if value > best_value:
                best_value = value
                best = option

        if best is None:  # todas llevan a Game Over: se juega la mejor de una pieza
            best = self.best_one_ply(options)
        return best[1:]


def main():
    def extra(parser):
        parser.add_argument('--beam', type=int, default=None,
                            help='expandir solo las mejores N colocaciones (por defecto, todas)')

    args = parse_args('Agente intermedio (greedy con lookahead) con estadisticas de features.', extra)
    run_experiment(LookaheadAgent(beam=args.beam), args, 'intermediate')


if __name__ == '__main__':
    main()
