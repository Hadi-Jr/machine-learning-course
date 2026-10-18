import matplotlib.pyplot as plt
from plt_overfit import overfit_example, output
plt.style.use('./deeplearning.mplstyle')

plt.close("all")
ofit = overfit_example(False)
plt.show()