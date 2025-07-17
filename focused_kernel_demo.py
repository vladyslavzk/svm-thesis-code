import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
np.random.seed(42)


def create_simple_xor(n_samples=100):
    n_per_cluster = n_samples // 4

    X1_1 = np.random.normal([1.5, 1.5], 0.3, (n_per_cluster, 2))
    X1_2 = np.random.normal([-1.5, -1.5], 0.3, (n_per_cluster, 2))
    X1 = np.vstack([X1_1, X1_2])
    y1 = np.ones(len(X1))

    X0_1 = np.random.normal([-1.5, 1.5], 0.3, (n_per_cluster, 2))
    X0_2 = np.random.normal([1.5, -1.5], 0.3, (n_per_cluster, 2))
    X0 = np.vstack([X0_1, X0_2])
    y0 = np.zeros(len(X0))

    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])

    return X, y


def simple_kernel_demo():
    X_xor, y_xor = create_simple_xor()

    cancer = datasets.load_breast_cancer()
    pca = PCA(n_components=2)
    X_cancer_pca = pca.fit_transform(cancer.data)

    indices_0 = np.where(cancer.target == 0)[0][:40]
    indices_1 = np.where(cancer.target == 1)[0][:40]
    cancer_indices = np.concatenate([indices_0, indices_1])

    X_cancer = X_cancer_pca[cancer_indices]
    y_cancer = cancer.target[cancer_indices]

    datasets_info = [
        ("XOR Problem", X_xor, y_xor),
        ("Breast Cancer", X_cancer, y_cancer)
    ]

    kernels = [
        ('Linear', svm.SVC(kernel='linear', C=1.0)),
        ('Polynomial (d=2)', svm.SVC(kernel='poly', degree=2, C=1.0)),
        ('RBF', svm.SVC(kernel='rbf', C=1.0, gamma='scale'))
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for row, (dataset_name, X, y) in enumerate(datasets_info):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for col, (kernel_name, model) in enumerate(kernels):
            ax = axes[row, col]

            cv_scores = cross_val_score(model, X_scaled, y, cv=3)
            accuracy = cv_scores.mean()

            model.fit(X_scaled, y)

            margin = 0.5
            x_min, x_max = X_scaled[:, 0].min() - margin, X_scaled[:, 0].max() + margin
            y_min, y_max = X_scaled[:, 1].min() - margin, X_scaled[:, 1].max() + margin

            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                 np.linspace(y_min, y_max, 100))

            Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            max_abs_z = np.max(np.abs(Z))
            vmin, vmax = -max_abs_z, max_abs_z

            ax.contourf(xx, yy, Z, levels=15, alpha=0.3, cmap='RdBu',
                        vmin=vmin, vmax=vmax)
            ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=3)

            ax.set_facecolor('white')
            class_0_idx = (y == 0)
            class_1_idx = (y == 1)

            ax.scatter(X_scaled[class_0_idx, 0], X_scaled[class_0_idx, 1],
                       c='red', marker='o', s=50, alpha=0.8,
                       edgecolors='darkred', linewidths=1, label='Class 0')

            ax.scatter(X_scaled[class_1_idx, 0], X_scaled[class_1_idx, 1],
                       c='blue', marker='o', s=50, alpha=0.8,
                       edgecolors='darkblue', linewidths=1, label='Class 1')

            if hasattr(model, 'support_vectors_') and len(model.support_vectors_) > 0:
                ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                           s=70, facecolors='none', edgecolors='black',
                           linewidths=2, marker='o', alpha=1.0)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_title(f'{kernel_name}\nAccuracy: {accuracy:.3f}',
                         fontsize=12, fontweight='bold')

            ax.set_xticks([])
            ax.set_yticks([])

            if col == 0:
                ax.set_ylabel(dataset_name, fontsize=14, fontweight='bold')

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=8, label='Class 0', markeredgecolor='darkred', markeredgewidth=1),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
               markersize=8, label='Class 1', markeredgecolor='darkblue', markeredgewidth=1),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markersize=10, label='Support Vectors', markeredgecolor='black', markeredgewidth=2),
        Line2D([0], [0], color='black', linewidth=3, label='Decision Boundary')
    ]

    plt.figlegend(handles=legend_elements, loc='lower center', ncol=4,
                  bbox_to_anchor=(0.5, 0.02), fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig('clear_kernel_comparison.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()


if __name__ == "__main__":
    simple_kernel_demo()