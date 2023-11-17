def prime_factorization_display(n):
    factors = []
    divisor = 2
    composite_num = n

    while n > 1:
        while n % divisor == 0:
            factors.append(divisor)
            n //= divisor
        divisor += 1

    result = f"{composite_num} = {'*'.join(map(str, factors))}"
    return result

# 输入合数
composite_number = int(input("请输入一个合数："))

# 显示质因数相乘的形式
result = prime_factorization_display(composite_number)
print(result)
