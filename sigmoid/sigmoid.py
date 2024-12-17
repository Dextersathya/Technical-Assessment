import numpy as np

# Test with a single value
input_value = 2
output_value = 1 / (1 + np.exp(-input_value))
print(f"Sigmoid applied to {input_value} gives: {output_value:.4f} (Output ranges between 0 and 1)")

# Test with a numpy array
input_array = np.array([1, -1, 0, 5, -5])
output_array = 1 / (1 + np.exp(-input_array))
print(f"Sigmoid applied to the array {input_array} gives: {output_array.round(4)} (Each value is mapped between 0 and 1)")


"""
sample_output:
Sigmoid applied to 2 gives: 0.8808 (Output ranges between 0 and 1)
Sigmoid applied to the array [ 1 -1  0  5 -5] gives: [0.7311 0.2689 0.5    0.9933 0.0067] (Each value is mapped between 0 and 1)
"""
