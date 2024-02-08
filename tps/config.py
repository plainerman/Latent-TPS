import tps.states


class PathSetup:
    def __init__(self, start: 'tps.states.State', stop: 'tps.states.State'):
        assert start != stop, "Start and stop states must be different"

        self.start = start
        self.stop = stop

    def __repr__(self):
        return f"PathInfo({self.start.name}[{self.start.current_level}] -> {self.stop.name}[{self.stop.current_level}])"