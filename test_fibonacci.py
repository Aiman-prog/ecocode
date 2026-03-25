def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]

    sequence = [0, 1]
    for i in range(2, n):
        next_val = sequence[i-1] + sequence[i-2]  
        sequence.append(next_val)

    return sequence


if __name__ == "__main__":
    print(fibonacci(10))