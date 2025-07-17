import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import validation_curve, train_test_split

plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)


def c_parameter_analysis():
    cancer = datasets.load_breast_cancer()
    X, y = cancer.data, cancer.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    c_values = np.logspace(-3, 3, 20)

    train_scores, val_scores = validation_curve(
        svm.SVC(kernel='linear'), X_train, y_train,
        param_name='C', param_range=c_values,
        cv=5, scoring='accuracy'
    )

    margins = []
    support_vector_counts = []
    test_accuracies = []

    for c in c_values:
        model = svm.SVC(kernel='linear', C=c)
        model.fit(X_train, y_train)

        margin = 1 / np.linalg.norm(model.coef_[0])
        margins.append(margin)

        support_vector_counts.append(len(model.support_))

        test_acc = model.score(X_test, y_test)
        test_accuracies.append(test_acc)

    # Bias-Variance Trade-off
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    ax1.semilogx(c_values, train_mean, 'g-', label='Training', linewidth=2)
    ax1.fill_between(c_values, train_mean - train_std, train_mean + train_std, alpha=0.1, color='g')

    ax1.semilogx(c_values, val_mean, 'b-', label='Validation', linewidth=2)
    ax1.fill_between(c_values, val_mean - val_std, val_mean + val_std, alpha=0.1, color='b')

    ax1.semilogx(c_values, test_accuracies, 'r--', label='Test', linewidth=2)

    best_c_idx = np.argmax(val_mean)
    best_c = c_values[best_c_idx]
    ax1.axvline(best_c, color='red', linestyle=':', alpha=0.7, label=f'Optimal C={best_c:.3f}')

    ax1.set_xlabel('C Parameter')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Bias-Variance Trade-off')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.85, 1.0)

    plt.tight_layout()
    plt.savefig('bias_variance_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.show()

    # C vs Margin and Support Vectors
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax2 = axes[0]
    ax2.semilogx(c_values, margins, 'purple', marker='o', linewidth=2)
    ax2.axvline(best_c, color='red', linestyle=':', alpha=0.7)
    ax2.set_xlabel('C Parameter')
    ax2.set_ylabel('Margin Size (1/||w||)')
    ax2.set_title('C vs Margin Size')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1]
    ax3.semilogx(c_values, support_vector_counts, 'orange', marker='s', linewidth=2)
    ax3.axvline(best_c, color='red', linestyle=':', alpha=0.7)
    ax3.set_xlabel('C Parameter')
    ax3.set_ylabel('Number of Support Vectors')
    ax3.set_title('C vs Support Vectors')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('c_parameter_effects.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    c_parameter_analysis()