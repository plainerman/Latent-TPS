import torch
import datasets.single_mol_dataset


class State:
    def __init__(self, name: str, center: torch.tensor, levels: list[float], current_level: int = 0):
        """
        Create a state in the phi-psi space.

        Parameters
        ----------
        name : str
            Name of the state
        center : torch.tensor
            Center of the state in phi-psi space
        levels : list[float]
            the radius of the states, with increasing tolerance
        current_level : int
            the current level of the state
        """
        self.name = name
        self.center = center
        self.levels = levels
        self.current_level = current_level

    def radius(self) -> float:
        return self.levels[self.current_level]

    def is_in_state(self, points: torch.tensor, dataset: 'datasets.single_mol_dataset.SingleMolDataset' = None,
                    period=2 * torch.pi) -> torch.tensor:
        """
        Returns a boolean mask of the points that are within the current specified interface level.

        Parameters
        ----------
        points : np.array
            The points to check. Assumed to be the same dimension as the center
        dataset : SingleMolDataset
            The dataset to check the points against. If None, the points are assumed to be already in the correct space
        period : float
            The period of the space (default: 2*pi for phi psi)
        """
        if dataset is not None:
            phis, psis = dataset.phis_psis(points)
            points = torch.stack([torch.tensor(phis), torch.tensor(psis)], dim=1).to(points.device)
        delta = torch.abs(self.center.to(points.device) - points)
        delta = torch.where(delta > period / 2, delta - period, delta)

        return torch.hypot(delta[:, 0], delta[:, 1]) < self.radius()

    def __repr__(self):
        return f"'{self.name}': center: {self.center} + {self.radius()}"


deg = 180.0 / torch.pi

ALDP_STATES = [
    State('A', torch.tensor([-150, 150]) / deg, torch.tensor([20, 45, 65, 80]) / deg),
    State('B', torch.tensor([-70, 135]) / deg, torch.tensor([20, 45, 65, 75]) / deg),
    State('C', torch.tensor([-150, -65]) / deg, torch.tensor([20, 45, 60]) / deg),
    State('D', torch.tensor([-70, -50]) / deg, torch.tensor([20, 45, 60]) / deg),
    State('E', torch.tensor([50, -100]) / deg, torch.tensor([20, 45, 65, 80]) / deg),
    State('F', torch.tensor([40, 65]) / deg, torch.tensor([20, 45, 65, 80]) / deg),
]
