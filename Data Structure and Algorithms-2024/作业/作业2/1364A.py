def prefix_sum(nums):
    """计算给定数组的前缀和。

    参数：
        nums：包含整数的列表。

    返回：
        一个列表，表示给定数组的前缀和。

    """
    prefix = []  # 存储前缀和
    total = 0    # 当前总和
    for num in nums:
        total += num
        prefix.append(total)
    return prefix

def suffix_sum(nums):
    """计算给定数组的后缀和。

    参数：
        nums：包含整数的列表。

    返回：
        一个列表，表示给定数组的后缀和。

    """
    suffix = []  # 存储后缀和
    total = 0    # 当前总和
    # 首先将列表反转
    reversed_nums = nums[::-1]
    for num in reversed_nums:
        total += num
        suffix.append(total)
    # 将结果反转回来
    suffix.reverse()
    return suffix

t = int(input())  # 输入测试案例数量
for _ in range(t):
    N, x = map(int, input().split())  # 输入数组长度和不喜欢的数字
    a = [int(i) for i in input().split()]  # 输入数组元素
    aprefix_sum = prefix_sum(a)  # 计算数组的前缀和
    asuffix_sum = suffix_sum(a)  # 计算数组的后缀和

    left = 0       # 左指针
    right = N - 1  # 右指针
    if right == 0:
        if a[0] % x != 0:  # 如果数组仅有一个元素且不能被 x 整除
            print(1)        # 输出1
        else:
            print(-1)       # 否则输出-1
        continue

    leftmax = 0    # 左侧最大长度
    rightmax = 0   # 右侧最大长度
    while left != right:
        total = asuffix_sum[left]  # 计算左侧子数组的后缀和
        if total % x != 0:         # 如果不能被 x 整除
            leftmax = right - left + 1  # 更新左侧最大长度
            break
        else:
            left += 1   # 否则左指针右移

    left = 0
    right = N - 1
    while left != right:
        total = aprefix_sum[right]  # 计算右侧子数组的前缀和
        if total % x != 0:          # 如果不能被 x 整除
            rightmax = right - left + 1  # 更新右侧最大长度
            break
        else:
            right -= 1  # 否则右指针左移
    
    if leftmax == 0 and rightmax == 0:
        print(-1)   # 如果没有满足条件的子数组，输出-1
    else:
        print(max(leftmax, rightmax))  # 输出最大长度
