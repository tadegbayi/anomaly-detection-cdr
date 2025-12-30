import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.plot([1,2,3],[1,4,9])
plt.savefig('test_plot.png')
print('WROTE test_plot.png')
