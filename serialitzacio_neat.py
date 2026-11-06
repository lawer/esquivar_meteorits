"""Conversió dels genomes NEAT al format JSON del projecte."""

import neat_aitk as neat

from joc_comu import OBSERVATION_VERSION


def genome_to_data(genome):
    def attributes(gene):
        return {attribute.name: getattr(gene, attribute.name)
                for attribute in gene._gene_attributes}
    return {
        'observation_version': OBSERVATION_VERSION,
        'key': genome.key,
        'fitness': genome.fitness,
        'nodes': [{'key': key, **attributes(node)} for key, node in genome.nodes.items()],
        'connections': [{'key': list(key), 'innovation': connection.innovation,
                         **attributes(connection)}
                        for key, connection in genome.connections.items()],
    }


def genome_from_data(data, config):
    if data.get('observation_version') != OBSERVATION_VERSION:
        raise ValueError('El model NEAT usa observacions incompatibles')
    genome = neat.DefaultGenome(data['key'])
    genome.fitness = data['fitness']
    for entry in data['nodes']:
        node = config.genome_config.node_gene_type(entry['key'])
        for attribute in node._gene_attributes:
            setattr(node, attribute.name, entry[attribute.name])
        genome.nodes[node.key] = node
    for entry in data['connections']:
        connection = config.genome_config.connection_gene_type(
            tuple(entry['key']), innovation=entry['innovation'])
        for attribute in connection._gene_attributes:
            setattr(connection, attribute.name, entry[attribute.name])
        genome.connections[connection.key] = connection
    return genome
