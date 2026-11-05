"""Opcions comunes als entrenaments dels controladors."""

import argparse
import math


def parser_entrenament(description, *, neat=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--entrenar', action='store_true', help='entrena el controlador')
    parser.add_argument('--visual', action='store_true', help="mostra l'entrenament")
    parser.add_argument('--velocitat', type=float, default=4,
                        help='velocitat visual relativa (per defecte: 4)')
    parser.add_argument('--poblacio', type=int, default=12)
    parser.add_argument('--generacions', type=int, default=8,
                        help='generacions després de la població inicial (per defecte: 8)')
    parser.add_argument('--fotogrames', type=int, default=1800)
    parser.add_argument('--proves', type=int, default=1,
                        help="nombre de llavors d'entrenament i de validació (per defecte: 1)")
    parser.add_argument('--llavor', type=int, default=None)
    parser.set_defaults(neat=neat)
    return parser


def valida_opcions(parser, args):
    if args.poblacio < 3 or args.generacions < 1 or args.fotogrames < 1 or args.proves < 1:
        parser.error('Calen almenys 3 individus, 1 generació, 1 prova i 1 fotograma')
    if not math.isfinite(args.velocitat) or args.velocitat <= 0:
        parser.error('La velocitat ha de ser un nombre positiu')
    if args.visual and not args.entrenar:
        parser.error('--visual requereix --entrenar')
