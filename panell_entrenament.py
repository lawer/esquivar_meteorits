"""Panell gràfic compartit pels entrenaments visuals."""

import arcade

from joc_comu import HEIGHT, SCALE, WIDTH

WHITE = (236, 241, 250)
BLUE = (80, 180, 255)
GREEN = (92, 222, 151)
GOLD = (255, 211, 95)


class TrainingPanel:
    def __init__(self):
        self.left = WIDTH - 310 * SCALE
        self.texts = {}

    def label(self, key, text, x, y, size=12, color=WHITE):
        if key not in self.texts:
            self.texts[key] = arcade.Text('', x, y, color, size * SCALE)
        label = self.texts[key]
        if label.text != text:
            label.text = text
        label.draw()

    def rect(self, x, y, width, height, color):
        arcade.draw_lbwh_rectangle_filled(x, y, width, height, color)

    def curve(self, key, x, top, title, limit, value, maximum, color):
        """Dibuixa la funció triangular real i el punt de l'observació."""
        scale = SCALE
        width, height = 230 * scale, 60 * scale
        bottom = top - height
        self.label(key + '_title', title, x, top + 7*scale, 10)
        self.rect(x, bottom, width, height, (27, 40, 56, 245))
        steps = 40
        for i in range(steps):
            v1, v2 = maximum * i / steps, maximum * (i + 1) / steps
            y1 = max(0, 1 - v1 / limit)
            y2 = max(0, 1 - v2 / limit)
            arcade.draw_line(x + width*i/steps, bottom + height*y1,
                             x + width*(i+1)/steps, bottom + height*y2, color, 2)
        if value is not None:
            value = max(0, min(maximum, value))
            membership = max(0, 1 - value / limit)
            arcade.draw_circle_filled(x + width*value/maximum,
                                      bottom + height*membership, 4*scale, WHITE)
            degree = f'grau {membership:.2f}'
        else:
            degree = 'sense dada'
        self.label(key + '_zero', '0', x, bottom - 15*scale, 9)
        self.label(key + '_degree', degree, x + 82*scale, bottom - 15*scale, 9)
        self.label(key + '_max', f'{maximum:g}', x + width - 20*scale, bottom - 15*scale, 9)
        return bottom - 32*scale

    def network(self, x, top, snapshot):
        """Diagrama de les connexions reals; NEAT pot afegir nodes ocults."""
        s = SCALE
        if snapshot['kind'] == 'neural':
            weights = snapshot['parameters']
            inputs = list(range(-1, -8, -1))
            hidden = list(range(100, 108))
            outputs = [200, 201, 202]
            edges = [(inputs[i], hidden[j], weights[i*8+j])
                     for i in range(7) for j in range(8)]
            edges += [(hidden[i], outputs[j], weights[64+i*3+j])
                      for i in range(8) for j in range(3)]
            total_hidden = 8
        else:
            inputs = list(range(-1, -8, -1))
            all_hidden = [node for node in snapshot['network_nodes'] if node != 0]
            hidden = all_hidden[:8]
            outputs = [0]
            edges = snapshot['network_edges']
            total_hidden = len(all_hidden)
        self.label('net_title', 'PESOS: blau (-) / groc (+); gruix = valor', x, top, 10)
        self.label('net_layers', 'Entrades                 Ocultes                 Sortides',
                   x, top - 17*s, 9)
        bottom, height = top - 180*s, 145*s
        columns = [(inputs, x+15*s), (hidden, x+125*s), (outputs, x+245*s)]
        positions = {}
        for nodes, nx in columns:
            for i, node in enumerate(nodes):
                ny = bottom + height/2 if len(nodes) == 1 else bottom + height * (1-i/(len(nodes)-1))
                positions[node] = (nx, ny)
        for source, target, weight in edges:
            if source not in positions or target not in positions:
                continue
            color = (255, 202, 75, 160) if weight >= 0 else (70, 160, 255, 160)
            arcade.draw_line(*positions[source], *positions[target], color,
                             max(0.5, min(2.5, abs(weight))) * s)
        for node, (nx, ny) in positions.items():
            arcade.draw_circle_filled(nx, ny, 4*s, WHITE if node in inputs else GREEN)
        if snapshot['kind'] == 'neat':
            info = f"Nodes ocults: {total_hidden} | connexions actives: {snapshot['total_connections']}"
            output = f"Sortida: {snapshot['output']:+.2f}"
        else:
            info = '7 entrades | 8 ocultes | 3 sortides (E / Q / D)'
            values = snapshot['outputs']
            output = f"Sortides E / Q / D: {values[0]:+.2f}  {values[1]:+.2f}  {values[2]:+.2f}"
        self.label('net_info', info, x, bottom - 18*s, 9)
        if total_hidden > 8:
            self.label('net_limit', 'Mostrant 8 nodes ocults com a maxim', x, bottom - 33*s, 9)
        self.label('net_output', output, x, bottom - 48*s, 10)

    def draw(self, game):
        s, x = SCALE, self.left
        self.rect(x, 0, WIDTH - x, HEIGHT, (10, 17, 29, 230))
        x += 15 * s
        y = HEIGHT - 27 * s
        self.label('title', 'ENTRENAMENT GENETIC   [P]', x, y, 14, GOLD)
        y -= 26 * s
        self.label('generation', game.training_status, x, y, 10)
        y -= 25 * s
        self.label('graph_title', 'Fitness: mitjana / millor', x, y, 12)
        gx, gy, gw, gh = x, y - 100 * s, 260 * s, 85 * s
        self.rect(gx, gy, gw, gh, (27, 40, 56, 245))
        history = game.training_history
        if history:
            maximum = max(1, *(max(mean, best) for mean, best in history))
            for column, color in ((0, BLUE), (1, GOLD)):
                for i in range(1, len(history)):
                    x1 = gx + gw * (i - 1) / max(1, len(history) - 1)
                    x2 = gx + gw * i / max(1, len(history) - 1)
                    arcade.draw_line(x1, gy + gh * history[i-1][column] / maximum,
                                     x2, gy + gh * history[i][column] / maximum, color, 2)
                if len(history) == 1:
                    arcade.draw_circle_filled(gx + 3*s, gy + gh * history[0][column] / maximum, 3*s, color)
            mean, best = history[-1]
            unit = getattr(game, 'training_fitness_unit', 'fot.')
            text = f'Ultima gen.: mitjana {mean:.0f} | millor {best:.0f} {unit}'
        else:
            text = 'Dades disponibles al final de la generacio 0'
        self.label('graph_value', text, x, gy - 18*s, 10)
        snapshot = game.training_snapshot
        y = gy - 48*s
        self.label('controller_title', 'CONTROLADOR: NAU AMB CERCLE', x, y, 12, GOLD)
        y -= 22*s
        if not snapshot:
            self.label('controller_info', 'Esperant una nau...', x, y, 10)
            return
        identifier = snapshot.get('identifier', snapshot.get('index', 0) + 1)
        self.label('controller_info', f"Individu {identifier} | accio: {snapshot['action']}", x, y, 11)
        y -= 23*s
        values = snapshot['observations']
        self.label('observation', f"Posicio {values[0]:+.2f} | entrades normalitzades", x, y, 10)
        y -= 19*s
        for i in range(3):
            lateral, height = values[1 + 2*i:3 + 2*i]
            text = (f"M{i+1}  lluny o absent" if height >= 1 else
                    f"M{i+1}  lateral {lateral:+.2f}   altura {height:.2f}")
            self.label(f'threat{i}', text, x, y, 10)
            y -= 18*s
        if snapshot['kind'] == 'fuzzy':
            params = snapshot['parameters']
            self.label('fuzzy_params', f'Reaccio {params[0]:.0f}  marge {params[1]:.0f}  llindar {params[2]:.2f}', x, y, 10)
            threat = snapshot['threat']
            selected = 'Punt blanc: amenaca de major risc' if threat else 'Sense amenaces proximes'
            self.label('fuzzy_selected', selected, x, y - 19*s, 9)
            y = self.curve('near', x, y - 40*s, 'PROXIMITAT / distancia vertical normalitzada',
                           snapshot['proximity_limit'], threat['height'] if threat else None, 1, BLUE)
            y = self.curve('aligned', x, y - 20*s, 'ALINEACIO / distancia lateral normalitzada',
                           snapshot['lateral_limit'], threat['offset'] if threat else None, 1, GOLD)
            self.label('fuzzy_result', f"Forca E {snapshot['left']:.2f} | D {snapshot['right']:.2f} | resultat {snapshot['result']:+.2f}", x, y, 10)
        else:
            self.network(x, y - 3*s, snapshot)
