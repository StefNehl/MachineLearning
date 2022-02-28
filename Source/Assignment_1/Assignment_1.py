#imports
from Fibonacci import  Fibonacci
import matplotlib.pyplot as plt


number_of_calculations = 30

fibonacci = Fibonacci()
result = fibonacci.get_fibonacci_numbers(number_of_calculations)

if type(result) == int:
    print('wrong values')
    quit(0)

fig = plt.figure()
fig.suptitle('Fibonacci')
plt.xlabel('Number')
plt.ylabel('Value')

x = range(1, number_of_calculations+1)
y = result
plt.plot(x, y)
plt.show()

print("Finished")