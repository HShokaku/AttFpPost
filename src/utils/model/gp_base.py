import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

class Standard_GPModel(ApproximateGP):
    def __init__(self, x):
        variational_distribution = CholeskyVariationalDistribution(x.size(0))
        variational_strategy = VariationalStrategy(
            self, x, variational_distribution, learn_inducing_locations=True
        )
        super(Standard_GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred