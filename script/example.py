def fibonacci(n):
    """ A function that returns the nth value in the
    Fibonacci sequence using recursion . """
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
