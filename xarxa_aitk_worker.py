"""Procés separat per executar les xarxes AITK sense carregar TensorFlow a Arcade."""

import contextlib
import json
import sys
import traceback

# El protocol usa stdout; qualsevol missatge de les biblioteques va a stderr.
with contextlib.redirect_stdout(sys.stderr):
    from aitk.networks import SimpleNetwork

INPUTS, HIDDEN, OUTPUTS = 7, 8, 3


def new_network():
    return SimpleNetwork(INPUTS, HIDDEN, (OUTPUTS, "linear"))


def main():
    networks = {}
    for line in sys.stdin:
        try:
            request = json.loads(line)
            command = request["command"]
            with contextlib.redirect_stdout(sys.stderr):
                if command == "count":
                    result = len(new_network().get_weights(flat=True))
                elif command == "spec":
                    network = new_network()
                    result = {"shapes": [list(w.shape) for w in network.get_weights()],
                              "activations": [layer.activation.__name__
                                              for layer in network._model.layers
                                              if hasattr(layer, "activation")]}
                elif command == "reset":
                    networks.clear()
                    result = True
                elif command == "register":
                    network = new_network()
                    network.set_weights(request["weights"])
                    networks[request["id"]] = network
                    result = True
                elif command == "predict":
                    result = networks[request["id"]].propagate(request["inputs"])
                else:
                    raise ValueError(f"Ordre desconeguda: {command}")
            response = {"result": result}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = {"error": str(exc)}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
