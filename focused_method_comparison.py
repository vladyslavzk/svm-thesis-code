import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import pandas as pd
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

def focused_method_comparison():
    classifiers = {
        'SVM (Linear)': svm.SVC(kernel='linear', C=1.0),
        'SVM (RBF)': svm.SVC(kernel='rbf', C=1.0, gamma='scale'),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'K-NN': KNeighborsClassifier(n_neighbors=5)
    }

    datasets_info = [
        ('Linearly Separable', datasets.make_classification(
            n_samples=500, n_features=20, n_redundant=0, n_informative=20,
            n_clusters_per_class=1, class_sep=1.0, random_state=42
        )),
        ('Non-linear', datasets.make_moons(n_samples=500, noise=0.1, random_state=42)),
        ('High-dimensional', datasets.load_breast_cancer())
    ]

    results = []

    for dataset_name, dataset in datasets_info:
        if isinstance(dataset, tuple):
            X, y = dataset
        else:
            X, y = dataset.data, dataset.target

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        dataset_results = {'Dataset': dataset_name}

        for clf_name, clf in classifiers.items():
            cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')
            dataset_results[clf_name] = cv_scores.mean()

        results.append(dataset_results)

    df = pd.DataFrame(results)

    create_heatmap_visualization(df, classifiers)
    create_barplot_visualization(df, classifiers)

    return df

def create_heatmap_visualization(df, classifiers):
    clf_names = list(classifiers.keys())
    dataset_names = df['Dataset'].tolist()

    accuracy_data = []
    for dataset in dataset_names:
        row = []
        for clf_name in clf_names:
            acc = df[df['Dataset'] == dataset][clf_name].iloc[0]
            row.append(acc)
        accuracy_data.append(row)

    accuracy_data = np.array(accuracy_data)

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(accuracy_data,
                xticklabels=clf_names,
                yticklabels=dataset_names,
                annot=True,
                fmt='.3f',
                cmap='RdYlGn',
                center=0.8,
                ax=ax,
                cbar_kws={'label': 'CV Accuracy'})

    ax.set_title('Cross-Validation Accuracy\nby Method and Dataset', fontweight='bold', fontsize=14)
    ax.set_xlabel('Classification Method', fontsize=12)
    ax.set_ylabel('Dataset Type', fontsize=12)

    plt.tight_layout()
    plt.savefig('method_comparison_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_barplot_visualization(df, classifiers):
    clf_names = list(classifiers.keys())

    avg_scores = df[clf_names].mean().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(range(len(avg_scores)), avg_scores.values, alpha=0.7)
    ax.set_yticks(range(len(avg_scores)))
    ax.set_yticklabels(avg_scores.index)
    ax.set_xlabel('Average Accuracy', fontsize=12)
    ax.set_title('Average Performance\nAcross All Datasets', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')

    for bar, score in zip(bars, avg_scores.values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', ha='left', va='center', fontsize=11)

    ax.set_xlim(0.9, 1.0)

    plt.tight_layout()
    plt.savefig('method_comparison_barplot.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    focused_method_comparison()