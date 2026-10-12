import math
from typing import List


def find_primes(n: int) -> List[int]:
    """
    This function finds all prime numbers between 1 and n, inclusive.

    Args:
        n (int): The upper limit for the range of prime numbers to search.

    Returns:
        list: A list of all prime numbers found in the given range.
    """
    # Create a boolean array to store whether each number is prime or not
    primes = [True] * (n + 1)

    # Initialize the first two primes (2 and 3) as non-prime
    primes[0] = primes[1] = False

    # Iterate over all numbers from 2 to sqrt(n)
    for i in range(2, int(math.sqrt(n)) + 1):
        # If the current number is prime, mark all its multiples as composite
        if primes[i]:
            for j in range(i * i, n + 1, i):
                primes[j] = False

    # Return the list of prime numbers
    return [num for num in range(2, n + 1) if primes[num]]


if __name__ == '__main__':
    print(find_primes(30))
    print(find_primes(100))
