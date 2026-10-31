import sympy

J, w = sympy.symbols('J,w')
J = w ** 2
print(J)

dj_dw = sympy.diff(J,w)
print(dj_dw)

print(dj_dw.subs([(w,2)]))