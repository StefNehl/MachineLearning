import numpy as np

class Fibonacci:

    def calculate_fibonacci(self, n):

        if n < 0:
            print("Input can't be lower then zero")
        elif n == 0:
            return 0
        elif n == 1 or n == 2:
            return 1

        else:
            return self.calculate_fibonacci(n - 1) + self.calculate_fibonacci(n - 2)

    def get_fibonacci_numbers(self, n):

        # Iteration from 0 - n
        n_with_zero = n + 1
        fibonacci_numbers = np.zeros(n_with_zero)

        if n < 0:
            print("Input can't be lower then zero")
        elif n == 0:
            return 0
        elif n == 1:
            return 1


        for x in range(n_with_zero):
            calculatedValue = -1
            if x == 0:
                calculatedValue = 0
            elif x == 1:
                calculatedValue = 1
            else:
                calculatedValue = fibonacci_numbers[x-1] + fibonacci_numbers[x-2]

            fibonacci_numbers[x] = calculatedValue
            print('it: ' + str(x) + 'v: ' + str(calculatedValue))

        return fibonacci_numbers




