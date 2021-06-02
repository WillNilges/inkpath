import pandas as pd
data = pd.read_csv('lots_of_rasters_full_05.txt',sep='\s+',header=None)
data = pd.DataFrame(data)

import matplotlib.pyplot as plt
x = data[0]
y = data[1]
plt.plot(x, y,'ro')
plt.savefig('plot.png')
