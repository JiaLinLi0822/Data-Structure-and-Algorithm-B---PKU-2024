n = 1000000
a = [1] * n # 用于判断是否是T-prime
s = set() 

#directly add the square of prime into a set, then check if num_input is in set.
for i in range(2,n):
    if a[i]: # 如果a[i]为1
        s.add(i*i)
        for j in range(i*i,n,i): # 从i*i开始，步长为i，将a中的数置为0，即i的倍数都不是T-prime
            a[j] = 0

input()
for x in map(int,input().split()):
    print(["NO","YES"][x in s]) # 通过布尔索引判断x是否在s中