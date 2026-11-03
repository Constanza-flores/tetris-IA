"""Entrena los pesos de la heuristica del agente con el metodo de entropia cruzada (CEM).

Cada candidato es un vector de pesos. Se evalua jugando partidas headless con
DellacherieAgent y se mide la cantidad media de lineas. Los candidatos elite
definen la nueva distribucion de muestreo. Todos los candidatos de una
iteracion juegan con las mismas semillas, asi la comparacion es justa.

Ejemplo:
    python train.py --iters 10 --pop 20 --games 3 --pieces 500 --seed 1

Al final guarda en runs/<fecha>/:
    history.json      mejor y promedio de cada iteracion
    best_weights.json pesos con mejor desempeno
    best_turns.jsonl  turnos (s, a, r, s') de una partida con los mejores pesos
"""

import argparse
import datetime
import json
import os
import random
import statistics

from agent import WEIGHTS, DellacherieAgent, play_game

FEATURES = list(WEIGHTS.keys())


def fitness(weights, games, max_pieces, seed, lookahead):
    """Lineas promedio sobre `games` partidas con semillas fijas."""
    agent = DellacherieAgent(lookahead=lookahead, weights=weights)
    total = 0
    for game in range(games):
        random.seed(seed + game)
        _, lines = play_game(agent, max_pieces=max_pieces)
        total += lines
    return total / games


def sample_candidates(mean, std, pop):
    return [[random.gauss(m, s) for m, s in zip(mean, std)] for _ in range(pop)]


def to_dict(vector):
    return dict(zip(FEATURES, vector))


def train(args):
    random.seed(args.seed)
    mean = [WEIGHTS[name] for name in FEATURES]
    std = [args.init_std] * len(FEATURES)
    history = []

    print(f'Pesos iniciales: {WEIGHTS}')
    baseline = fitness(WEIGHTS, args.games, args.pieces, args.seed, args.lookahead)
    print(f'Linea base (Dellacherie): {baseline:.1f} lineas promedio\n')

    best_vector, best_score = None, float('-inf')
    for iteration in range(1, args.iters + 1):
        candidates = sample_candidates(mean, std, args.pop)
        scored = []
        for vector in candidates:
            score = fitness(to_dict(vector), args.games, args.pieces, args.seed, args.lookahead)
            scored.append((score, vector))
        scored.sort(key=lambda item: item[0], reverse=True)

        iter_best_score, iter_best_vector = scored[0]
        if iter_best_score > best_score:
            best_score, best_vector = iter_best_score, iter_best_vector

        mean_score = statistics.mean(score for score, _ in scored)
        history.append({'iteration': iteration,
                        'best': iter_best_score,
                        'mean': mean_score,
                        'best_weights': to_dict(iter_best_vector)})
        print(f'Iteracion {iteration:2d}: mejor={iter_best_score:.1f} promedio={mean_score:.1f}')

        elite = [vector for _, vector in scored[:args.elite]]
        mean = [statistics.mean(v[i] for v in elite) for i in range(len(FEATURES))]
        std = [max(statistics.pstdev(v[i] for v in elite), args.min_std)
               for i in range(len(FEATURES))]

    return to_dict(best_vector), best_score, baseline, history


def save_results(best_weights, best_score, baseline, history, args):
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join('runs', f'train_{stamp}')
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'history.json'), 'w', encoding='utf-8') as f:
        json.dump({'baseline': baseline, 'iterations': history}, f, indent=2)
    with open(os.path.join(out_dir, 'best_weights.json'), 'w', encoding='utf-8') as f:
        json.dump({'weights': best_weights, 'lines_mean': best_score}, f, indent=2)

    # Partidas con los mejores pesos, con log de turnos para analisis posterior.
    agent = DellacherieAgent(lookahead=args.lookahead, weights=best_weights)
    log = []
    for game in range(1, args.final_games + 1):
        random.seed(args.seed + 10_000 + game)
        play_game(agent, max_pieces=args.pieces, log=log, game_id=game)
    with open(os.path.join(out_dir, 'best_turns.jsonl'), 'w', encoding='utf-8') as f:
        for record in log:
            f.write(json.dumps(record) + '\n')
    return out_dir


def main():
    parser = argparse.ArgumentParser(description='Entrena los pesos de Dellacherie con CEM.')
    parser.add_argument('--iters', type=int, default=10, help='iteraciones de CEM')
    parser.add_argument('--pop', type=int, default=20, help='candidatos por iteracion')
    parser.add_argument('--elite', type=int, default=5, help='candidatos elite por iteracion')
    parser.add_argument('--games', type=int, default=3, help='partidas por candidato')
    parser.add_argument('--pieces', type=int, default=500, help='piezas maximas por partida')
    parser.add_argument('--final-games', type=int, default=3, help='partidas para el log final')
    parser.add_argument('--init-std', type=float, default=1.0, help='desviacion inicial de los pesos')
    parser.add_argument('--min-std', type=float, default=0.05, help='desviacion minima (evita colapso)')
    parser.add_argument('--lookahead', action='store_true', help='usar lookahead (mas lento)')
    parser.add_argument('--seed', type=int, default=1, help='semilla base')
    args = parser.parse_args()

    best_weights, best_score, baseline, history = train(args)
    print('\nMejores pesos encontrados:')
    for name, value in best_weights.items():
        print(f'  {name:20s} {value:+.3f}')
    print(f'Lineas promedio con mejores pesos: {best_score:.1f} (linea base {baseline:.1f})')

    out_dir = save_results(best_weights, best_score, baseline, history, args)
    print(f'Resultados guardados en {out_dir}')


if __name__ == '__main__':
    main()
