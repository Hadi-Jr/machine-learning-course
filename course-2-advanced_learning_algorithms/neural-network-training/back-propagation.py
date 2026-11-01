from sympy import *
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from matplotlib.widgets import Button
import ipywidgets as widgets
from lab_utils_backprop import *

plt.close("all")
plt_network(config_nw0, "./images/C2_W2_BP_network0.PNG")
# plt.show()

# Forward Propagation
w = 3
a = 2 + 3 * w
J = a**2
print(f"a = {a}, J = {J}")

# Backprop

# Arithmetically
a_epsilon = a + 0.001
J_epsilon = a_epsilon**2
k = (J_epsilon - J) / 0.001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_da ~= k = {k} ")

# Symbolically
sw, sJ, sa = symbols('w,J,a')
sJ = sa**2
print(sJ)
print(sJ.subs([(sa, a)]))
dJ_da = diff(sJ, sa)
print(dJ_da)

# Arithmetically
w_epislon = w + 0.001
a_epsilon = 2 + 3*w_epislon
k = (a_epsilon - a) / 0.001
print(f"a = {a}, a_epsilon = {a_epsilon}, da_dw ~= k = {k} ")

# Symbolically
sa = 2 + 3*sw
print(sa)
da_dw = diff(sa, sw)
print(da_dw)

dJ_dw = da_dw * dJ_da
print(dJ_dw)
w_epislon = w + 0.001
a_epsilon = 2 + 3 * w_epislon
J_epsilon = a_epsilon**2
k = (J_epsilon - J) / 0.001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")


# Example 2
plt.close("all")
plt_network(config_nw1, "./images/C2_W2_BP_network1.PNG")

# Forward propagation
x = 2
w = -2
b = 8
y = 1

c = w * x
a = c + b
d = a - y
J = d**2/2
print(f"J={J}, d={d}, a={a}, c={c}")

# Arithmetically
d_epsilon = d + 0.001
J_epsilon = d_epsilon**2/2
k = (J_epsilon - J)/0.001   # difference divided by epsilon
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dd ~= k = {k} ")

# Symbolically
sx,sw,sb,sy,sJ = symbols('x,w,b,y,J')
sa, sc, sd = symbols('a,c,d')
sJ = sd**2/2
sJ.subs([(sd,d)])
dJ_dd = diff(sJ, sd)


a_epsilon = a + 0.001         # a  plus a small value
d_epsilon = a_epsilon - y
k = (d_epsilon - d)/0.001   # difference divided by epsilon
print(f"d = {d}, d_epsilon = {d_epsilon}, dd_da ~= k = {k} ")
sd = sa - sy
dd_da = diff(sd,sa)
dJ_da = dd_da * dJ_dd

a_epsilon = a + 0.001
d_epsilon = a_epsilon - y
J_epsilon = d_epsilon**2/2
k = (J_epsilon - J)/0.001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_da ~= k = {k} ")

sa = sc + sb

da_dc = diff(sa,sc)
da_db = diff(sa,sb)
print(da_dc, da_db)

dJ_dc = da_dc * dJ_da
dJ_db = da_db * dJ_da
print(f"dJ_dc = {dJ_dc},  dJ_db = {dJ_db}")

sc = sw * sx
dc_dw = diff(sc,sw)

dJ_dw = dc_dw * dJ_dc
print(f"dJ_dw = {dJ_dw.subs([(sd,d),(sx,x)])}")

J_epsilon = ((w+0.001)*x+b - y)**2/2
k = (J_epsilon - J)/0.001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")


