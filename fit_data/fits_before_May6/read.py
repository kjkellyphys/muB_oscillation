import numpy as np
user_input = input("File name: ")
data = np.load(user_input)
print(data)
print('Array size: '+str(data.shape))
