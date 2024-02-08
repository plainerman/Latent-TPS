import openpathsampling as paths


class TwoWayShootingMoveScheme(paths.MoveScheme):
    def __init__(self, network, selector=None, ensembles=None, engine=None):
        super(TwoWayShootingMoveScheme, self).__init__(network)
        modifier = paths.NoModification()
        # temperature = 300
        # modifier = paths.RandomVelocities(beta = 1.0/temperature, engine=engine)

        self.append(paths.strategies.TwoWayShootingStrategy(modifier=modifier, engine=engine, selector=selector,
                                                            ensembles=ensembles))
        self.append(paths.strategies.OrganizeByMoveGroupStrategy())