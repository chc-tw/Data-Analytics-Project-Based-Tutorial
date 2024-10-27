import torch
from torch.distributions import MultivariateNormal

class GMM:
    def __init__(self, n_components, n_features, n_iter, device='cuda'):
        self.n_components = n_components
        self.n_features = n_features
        self.device = device

        self.means = torch.randn(n_components, n_features, device=device, requires_grad=True)
        self.covariances = torch.stack([torch.eye(n_features, device=device) for _ in range(n_components)])
        self.weights = torch.ones(n_components, device=device) / n_components
        self.n_iter = n_iter
    def e_step(self, X):
        n_samples = X.shape[0]
        responsibilities = torch.zeros(n_samples, self.n_components, device=self.device)

        for k in range(self.n_components):
            dist = MultivariateNormal(self.means[k], self.covariances[k])
            responsibilities[:, k] = dist.log_prob(X) + torch.log(self.weights[k])

        responsibilities = torch.nn.functional.softmax(responsibilities, dim=1)
        return responsibilities

    def m_step(self, X, responsibilities):
        Nk = responsibilities.sum(dim=0)


        self.means = (responsibilities.T @ X) / Nk.unsqueeze(1)

        epsilon = 1e-6  # Regularization
        for k in range(self.n_components):
            diff = X - self.means[k]
            weighted_sum = (responsibilities[:, k, None] * diff).T @ diff
            self.covariances[k] = weighted_sum / Nk[k] + epsilon * torch.eye(self.n_features, device=self.device)

        self.weights = Nk / Nk.sum()

    def fit(self, X):
        X = X.to(self.device)

        for _ in range(self.n_iter):
            responsibilities = self.e_step(X)  # E step
            self.m_step(X, responsibilities)  # M step

    def predict(self, X):
        X = X.to(self.device)
        responsibilities = self.e_step(X)
        return responsibilities.argmax(dim=1)

    def log_likelihood(self, X):
        X = X.to(self.device)
        log_likelihood = torch.zeros(X.shape[0], device=self.device)

        for k in range(self.n_components):
            dist = MultivariateNormal(self.means[k], self.covariances[k])
            log_likelihood += self.weights[k] * torch.exp(dist.log_prob(X))

        return torch.sum(torch.log(log_likelihood))

    def bic(self, X):
        n_samples, n_features = X.shape

        n_params = (
            self.n_components * n_features
            + self.n_components * n_features * (n_features + 1) / 2
            + self.n_components - 1
        )

        # 計算對數似然
        log_likelihood = self.log_likelihood(X)

        # 計算 BIC
        bic_value = n_params * torch.log(torch.tensor(n_samples, device=self.device)) - 2 * log_likelihood
        return bic_value

# Test Example
if __name__ == "__main__":

    X = torch.randn(5000, 12) 

    gmm = MiniBatchGMM(n_components=5, n_features=12)
    gmm.fit(X, n_iter=5)
    print("Training Complete")
