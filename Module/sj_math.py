
def round_down(value, decimals):
    factor = 1 / (10 ** decimals)
    return (value // factor) * factor

def digit_length(n):
    return int(math.log10(n)) + 1 if n else 0

if __name__ == "__main__":
    round_down(0.011, 2)

    digit_length(30)
    