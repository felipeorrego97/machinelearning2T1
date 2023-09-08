import numpy as np

# Function to generate a random rectangular matrix
def generate_random_matrix(rows, cols):
    return np.random.rand(rows, cols)

# Generate a random rectangular matrix A
rows = 4  # Change this to the number of rows you want
cols = 3  # Change this to the number of columns you want
A = generate_random_matrix(rows, cols)

# Calculate the rank of A
rank_A = np.linalg.matrix_rank(A)

# Calculate the trace of A
trace_A = np.trace(A)

# Calculate the dominant eigenvalue of A
eigenvalues, eigenvectors = np.linalg.eig(A @ A.T)
dominant_eigenvalue = np.max(np.real(eigenvalues))

# Check if A is invertible and find its inverse
if rows == cols:
    if rank_A == min(rows, cols):
        A_inverse = np.linalg.inv(A)
else:
    A_inverse = None

# Calculate the eigenvalues and eigenvectors of A'A
eigenvalues_AAT, eigenvectors_AAT = np.linalg.eig(A.T @ A)

# Calculate the eigenvalues and eigenvectors of AA'
eigenvalues_AAT, eigenvectors_AAT = np.linalg.eig(A @ A.T)

# Print the results
print("Matrix A:")
print(A)
print("Rank of A:", rank_A)
print("Trace of A:", trace_A)
print("Dominant eigenvalue of A:", dominant_eigenvalue)
print("Inverse of A:")
print(A_inverse)
print("Eigenvalues of A'A:")
print(eigenvalues_AAT)
print("Eigenvectors of A'A:")
print(eigenvectors_AAT)
print("Eigenvalues of AA':")
print(eigenvalues_AAT)
print("Eigenvectors of AA':")
print(eigenvectors_AAT)

#Eigenvectors and eigenvalues of AAT and ATA are equals