import numpy as np

# Create arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([10, 20, 30, 40, 50])

print("Array 1:", arr1)
print("Array 2:", arr2)

# Element-wise addition
print("Addition:", arr1 + arr2)

# Broadcasting example
arr3 = np.array([1, 2, 3])
print("Broadcasting (arr3 + 10):", arr3 + 10)

# Mathematical operations
print("Square root of arr1:", np.sqrt(arr1))
print("Mean of arr2:", np.mean(arr2))

# Reshape an array
matrix = np.arange(1, 10).reshape(3, 3)
print("3x3 Matrix:\n", matrix)
