"""Motor NEAT d'AITK amb els genomes de neat-python 2.

AITK 3.1 usa internament classes de neat-python, però els seus genomes
inclosos no són compatibles amb els números d'innovació de la versió 2.
"""

from aitk.algorithms import neat as aitk_neat
import neat as neat_python

Config = neat_python.Config
config = neat_python.config
reporting = neat_python.reporting
DefaultGenome = neat_python.DefaultGenome
DefaultReproduction = neat_python.DefaultReproduction
DefaultSpeciesSet = neat_python.DefaultSpeciesSet
DefaultStagnation = neat_python.DefaultStagnation
StdOutReporter = neat_python.StdOutReporter
StatisticsReporter = neat_python.StatisticsReporter
Checkpointer = neat_python.Checkpointer
Population = aitk_neat.Population
nn = aitk_neat.nn
