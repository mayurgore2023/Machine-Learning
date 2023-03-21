# Diffrant ways to write data in series:

import pandas as pd
import numpy as np

Data = np.array(['a','b','c','d'])
Series1= pd.Series(Data) # put dat in series
print(Series1[0])
print(Series1)
print()

# by giving manual indexing:
Data = np.array(['a','b','c','d'])
Series2 = pd.Series(Data,index=[101,102,103,104])
print(Series2[101])
print(Series2)

# by using Dictionary:
Data ={'a':0.1,'b':1.1}
Series3 =pd.Series(Data)
print(Series3)