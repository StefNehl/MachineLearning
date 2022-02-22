#imports
from Fibonacci import  Fibonacci
import matplotlib.pyplot as plt


number_of_calculations = 30

fibonacci = Fibonacci()
result = fibonacci.get_fibonacci_numbers(number_of_calculations)

fig = plt.figure()
fig.suptitle('Fibonacci')
plt.xlabel('Number')
plt.ylabel('Value')

x = range(number_of_calculations + 1)
y = result
plt.plot(x, y)
plt.show()

print(fibonacci.calculate_fibonacci(number_of_calculations))

print("Finished")