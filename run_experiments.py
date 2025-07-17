#!/usr/bin/env python3
from moons import *
from focused_kernel_demo import simple_kernel_demo
from focused_hyperparameter import c_parameter_analysis  
from focused_method_comparison import focused_method_comparison

if __name__ == "__main__":
    print("Running all thesis experiments...")
    
    # Run all experiments
    simple_kernel_demo()
    c_parameter_analysis()
    focused_method_comparison()
    
    print("All experiments completed!")