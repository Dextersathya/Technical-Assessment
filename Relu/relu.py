import numpy as np

"""
Apply the ReLU (Rectified Linear Unit) activation function.
    
    Parameters:
    x : float or numpy array
        Input value or array of values to which ReLU function will be applied.

    Returns:
    numpy array
        Output after applying ReLU element-wise.
"""

# Test with a single value
input_value = -5
output_value = np.maximum(0, input_value)
print(f"ReLU({input_value}) = {output_value}")

# Test with a numpy array
input_array = np.array([-5, 3, -2, 7])
output_array = np.maximum(0, input_array)
print(f"ReLU({input_array}) = {output_array}")

"""
sample_output:

ReLU(-5) = 0
ReLU([-5  3 -2  7]) = [0 3 0 7]

"""

#formula: f(x)=max(0,x)