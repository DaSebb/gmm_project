import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist
from matplotlib.patches import Ellipse

def create_XY_table(img):
    result = []
    for i in range(28):
        for j in range(28):
            if(img[i][j] > 0):
                result.append([j, i])
    return np.array(result)

def plot_gmm(gmm, ax, color='blue'):
    for i in range(gmm.n_components):
        mean = gmm.means_[i]
        cov = gmm.covariances_[i]

        v, w = np.linalg.eigh(cov)
        v = 2. * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])

        angle = np.degrees(np.arctan2(u[1], u[0]))
        ell = Ellipse(xy=mean, width=v[0], height=v[1], angle=angle, edgecolor=color, facecolor='none', lw=2)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.6)
        ax.add_patch(ell)

def sub_plot_gmm_probabilities_in_subplot(fig, ax, train_data, fitted_gmms):
    XY_grid, X_grid, Y_grid = generate_mesh_grid(train_data)
    log_likelihoods_grid = np.array([gmm.score_samples(XY_grid) for gmm in fitted_gmms])
    average_log_likelihood_grid = np.mean(log_likelihoods_grid, axis=0)
    averaged_probabilities_grid = np.exp(average_log_likelihood_grid).reshape(X_grid.shape)

    contour = ax.contourf(X_grid, Y_grid, averaged_probabilities_grid, levels=20, cmap='viridis')
    fig.colorbar(contour)
    ax.scatter(train_data[:, 0], train_data[:, 1], s=40, marker='H', edgecolor='k', c='g')
    ax.invert_yaxis()

def generate_mesh_grid(train_data):
    x = np.linspace(train_data[:, 0].min() - 1, train_data[:, 0].max() + 1, 100)
    y = np.linspace(train_data[:, 1].max() + 1, train_data[:, 1].min() - 1, 100)
    X_grid, Y_grid = np.meshgrid(x, y)
    XY_grid = np.array([X_grid.ravel(), Y_grid.ravel()]).T
    return XY_grid, X_grid, Y_grid

# overlays all the digits in the input array over another and averages all the points in the digit 2D array
def overlay_mnistdigits(digit_imgs):
    result = np.zeros((28,28))
    for i in range(28):
        for j in range(28):
            total_pixle_value = 0
            for digit in digit_imgs:
                total_pixle_value = total_pixle_value + digit[i][j]
            result[i][j] = total_pixle_value/255

                

# Step 1: Load and preprocess MNIST data
(X_train, Y_train), (_, _) = mnist.load_data()  # X_train are the images of numbers represented in 2D arrays and Y_train is an array of numbers representing the lables of on the images

digit = 5
X_digit = X_train[Y_train == digit]

X_digit_img = X_digit[10]

type(X_digit_img)

digits_to_overlay = X_digit[2:4]

XY_table = create_XY_table(X_digit_img)

# Train GMM on Table
n_components = 10
n_components_2 = 12

gmm = GaussianMixture(n_components=n_components, covariance_type='full')
gmm.fit(XY_table)

gmm2 = GaussianMixture(n_components=n_components_2, covariance_type='full')
gmm2.fit(XY_table)

fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()

plot_gmm(gmm, ax)
plot_gmm(gmm2, ax2)
#ax.scatter(XY_table[:, 0], XY_table[:, 1], s=40, marker='H', edgecolor='k', c='r')
sub_plot_gmm_probabilities_in_subplot(fig3, ax3, XY_table, [gmm, gmm2])

ax.imshow(X_digit_img, cmap='gray')
ax.set_title(f'GMM Component 1 Overlaid on MNIST Digit "{digit}"')

ax2.imshow(X_digit_img, cmap='gray')
ax2.set_title(f'GMM Component 2 Overlaid on MNIST Digit "{digit}"')

ax3.set_title('Ensembled probabilities')

plt.show()