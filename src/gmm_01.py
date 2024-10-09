import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
from keras.datasets import mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()

for i in range(9):  
    plt.subplot(330 + 1 + i)
    plt.imshow(train_X[i], cmap=plt.get_cmap('gray'))

plt.show()

# Parameters
cov_type = 'spherical'
cov_type_2 = 'diag'

n_components = 2
n_components_2 = 5

# Get example dataset
wine = datasets.load_wine()
X = wine.data[:, :2]

# Initialize GMM model
gmm_model = GaussianMixture(n_components=n_components, covariance_type=cov_type)
gmm_model_2 = GaussianMixture(n_components=n_components_2, covariance_type=cov_type_2)

# Fit the GMM model
fitted_gmm_model = gmm_model.fit(X)
fitted_gmm_model_2 = gmm_model_2.fit(X)

# Function to plot the GMM components as ellipses
def plot_gmm(gmm, ax, color='blue'):
    for i, (mean, covar) in enumerate(zip(gmm.means_, gmm.covariances_)):
        if gmm.covariance_type == 'spherical':
            covar = np.eye(len(mean)) * covar
        elif gmm.covariance_type == 'diag':
            covar = np.diag(covar)

        # Eigen decomposition for the ellipse orientation and radii
        v, w = np.linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)  # scale for the 95% confidence interval
        u = w[0] / np.linalg.norm(w[0])

        angle = np.arctan2(u[1], u[0])
        angle = np.degrees(angle)
        ell = Ellipse(xy=mean, width=v[0], height=v[1], angle=angle, edgecolor=color, facecolor='none', lw=2)
        ax.add_patch(ell)

        # Plot the mean
        ax.scatter(mean[0], mean[1], c=color, s=100, marker='x', lw=2)

# Create plot
fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()

ax.scatter(X[:, 0], X[:, 1], s=40, marker='H', edgecolor='k', c='r')
ax2.scatter(X[:, 0], X[:, 1], s=40, marker='H', edgecolor='k', c='g')

plot_gmm(fitted_gmm_model_2, ax2, color='red')
plot_gmm(fitted_gmm_model, ax, color='green')

# Generate a meshgrid for plotting the ensembled probabilities
x = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
y = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
X_grid, Y_grid = np.meshgrid(x, y)
XY_grid = np.array([X_grid.ravel(), Y_grid.ravel()]).T

# Compute ensembled probabilities on the grid
gmm_models = [fitted_gmm_model, fitted_gmm_model_2]
log_likelihoods_grid = np.array([gmm.score_samples(XY_grid) for gmm in gmm_models])
average_log_likelihood_grid = np.mean(log_likelihoods_grid, axis=0)
averaged_probabilities_grid = np.exp(average_log_likelihood_grid).reshape(X_grid.shape)

# Plot the ensembled probabilities as a contour plot in fig3
contour = ax3.contourf(X_grid, Y_grid, averaged_probabilities_grid, levels=20, cmap='viridis')
fig3.colorbar(contour)

# Overlay the data points on the ensembled probabilities plot
ax3.scatter(X[:, 0], X[:, 1], s=40, marker='H', edgecolor='k', c='g')

# Set titles and labels
ax.set_title(f'GMM with {n_components} components and {cov_type} covariance type')
ax2.set_title(f'GMM with {n_components_2} components and {cov_type_2} covariance type')
ax3.set_title('Ensembled Probabilities')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax3.set_xlabel('Feature 1')
ax3.set_ylabel('Feature 2')

plt.show()
