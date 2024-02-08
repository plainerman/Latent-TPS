import torch
from normflows.flows import Flow


class UniformGaussianFlow(Flow):
    """
    Transforms a standard normal distribution to a mixture between a normal and a uniform distribution.
    See normflows.distributions.UniformGaussian for details.
    """
    def __init__(self, ndim, ind, scale):
        """Constructor

        Args:
          ndim: Int, number of dimensions
          ind: Iterable, indices of uniformly distributed entries
          scale: Iterable, standard deviation of Gaussian or width of uniform distribution
        """
        super().__init__()

        self.ndim = ndim
        self.ind = ind
        self.scale = scale

        self._norm = torch.distributions.normal.Normal(loc=0, scale=1)

    def forward(self, z):
        out = z.clone()
        out[..., self.ind] = self._norm.cdf(out[..., self.ind]) - 0.5
        return out * self.scale.to(z.device), 0

    def inverse(self, z):
        out = z.clone() / self.scale.to(z.device)
        out[..., self.ind] = self._norm.icdf(out[..., self.ind] + 0.5)
        return out, 0
