import numpy as np

class Fibonacci:

    def get_fibonacci_numbers(self, n):
        fibonacci_numbers = np.zeros(n)

        if n < 0:
            print("Input can't be lower then zero")
        elif n == 0:
            return 0
        elif n == 1:
            return 1

        for x in range(0, n):
            calculatedValue = -1
            if x == 0:
                calculatedValue = 1
            elif x == 1:
                calculatedValue = 1
            else:
                calculatedValue = fibonacci_numbers[x-1] + fibonacci_numbers[x-2]

            fibonacci_numbers[x] = calculatedValue
            print('it: ' + str(x + 1) + ' v: ' + str(calculatedValue))

        return fibonacci_numbers




