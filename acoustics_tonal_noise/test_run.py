import numpy as np 

for i in range(10):
    globals()[f'x{i}'] = i

dict ={}
for i in range(10):
    key= str('x'+str(i))
    dict[key] = i

# print(dict)
array = np.arange(1,11)
# print(np.array_split(array))

def func(*args):
    for s in args:
        return s

print((a for a in dict))

# print()