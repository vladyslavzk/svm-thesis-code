import matplotlib.pyplot as plt
from sklearn import datasets

X, y = datasets.make_moons(n_samples=500, noise=0.1, random_state=42)

plt.figure(figsize=(8, 6))
plt.scatter(X[y==0, 0], X[y==0, 1], c='red', marker='o', s=50, alpha=0.7, label='Class 0')
plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', marker='o', s=50, alpha=0.7, label='Class 1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Non-linear Dataset: Moons')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('moons_dataset.png', dpi=300, bbox_inches='tight')
plt.show()