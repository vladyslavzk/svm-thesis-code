# SVM Thesis Code

Python implementation of Support Vector Machine experiments for bachelor's thesis.

## Files

- `run_experiments.py` - **Runs all experiments automatically**
- `focused_kernel_demo.py` - Compares Linear, Polynomial, and RBF kernels
- `focused_hyperparameter.py` - Analyzes regularization parameter C effects
- `focused_method_comparison.py` - Compares SVM with other ML methods
- `moons.py` - Visualizes the moons dataset (non-linear classification problem)

## Usage

Install all necessary python packages:
```bash
pip install numpy scikit-learn matplotlib seaborn pandas
```

Run all experiments:
```bash
python run_experiments.py
```

Run individual experiments:
```bash
python focused_kernel_demo.py         # Kernel comparison  
python focused_hyperparameter.py      # Parameter analysis
python focused_method_comparison.py   # Method comparison
```
Run moons dataset visualization:
```bash
python moons.py                       
```

Generates figures and results used in the thesis chapters 4.2-4.4.
