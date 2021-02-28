from algorithms import uniform_hypersphere
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

points = uniform_hypersphere(3, 1000)
count = {}
permutations=[]
for x in points:
    permutations.append(str(np.argsort(x)))

sns.histplot(data=permutations)
plt.show()

