import sympy

J, w = sympy.symbols('J,w')
J = w ** 2
print(J)

dj_dw = sympy.diff(J,w)
print(dj_dw)

print(dj_dw.subs([(w,2)]))

J = (3) ** 2
J_epsilon = (3 + 0.001)**2
k = (J_epsilon - J) / 0.001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k:0.6f} ")