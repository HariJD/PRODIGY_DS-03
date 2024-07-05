import matplotlib.pyplot as plt
import numpy as np

# Generate simulated data
np.random.seed(0)  # for reproducibility
# Assuming a normal distribution of ages around mean 40 and standard deviation 10
ages = np.random.normal(loc=40, scale=10, size=1000)

# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.hist(ages, bins=30, edgecolor='black', alpha=0.7)
plt.title('Distribution of Ages in a Population Over 30 Years')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
