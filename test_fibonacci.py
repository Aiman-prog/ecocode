def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]

    sequence = [0, 1]
    for i in range(2, n+1):  # range stops before the end value by default
        if i > 1: # for i > 1 you have to access element at index i-2 and i-1 in the sequence list
            next_val = sequence[i-1] + sequence[i-2]
        else:
            next_val = sequence[0] 
        sequence.append(next_val)

    return sequence


if __name__ == "__main__":
    print(fibonacci(10))