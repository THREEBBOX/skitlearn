import numpy as np
x=3+np.sqrt(13)
x=x/2
a=(x**6+2*x**5+3*x**4+4*x**3+3*x**2+2*x+1)/(x**5+x**3+x)
print(a)
a=a-6
for i in range(200):
    print(i*a,i)
