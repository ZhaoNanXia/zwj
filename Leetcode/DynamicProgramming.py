def can_partition(nums):
    """ 416.分割等和子集：0-1背包问题 """
    total = sum(nums)
    if total % 2 != 0:
        return False
    target = total // 2
    dp = [False] * (target + 1)  # 是否存在和为i的子集
    dp[0] = True  # 初始化，存在空集和为0
    for num in nums:
        for i in range(target, num - 1, -1):
            dp[i] = dp[i] or dp[i - num]  # 已经存在和为i的子集或者存在和为i-num的子集，加上当前数值便可组成和为i的子集
    return dp[target]


# lists = [1, 5, 11, 5]
# print(can_partition(lists))


def last_stone_weight(stones):
    """ 1049.最后一块石头的重量II: 0-1背包问题 """
    total = sum(stones)
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    for weight in stones:
        for i in range(target, weight - 1, -1):
            dp[i] = dp[i] or dp[i - weight]

    for i in range(target, -1, -1):
        if dp[i]:
            return total - 2 * i


# lists = [2, 7, 4, 1, 8, 1]
# print(last_stone_weight(lists))

def find_target_sum_ways(nums, target):
    """ 494.目标和：0-1背包问题"""
    total = sum(nums)
    # 假设选择了若干个数要令其为负数，和为neg；那选择的正数的和为total-neg;
    # 则total-neg-neg=total-2*neg=target——>可得neg=(total-neg)//2
    # 该问题转化为数组中和为neg的子集有多少个
    if (total - target) % 2 != 0 or total < abs(target):
        return 0
    new_target = (total - target) // 2
    dp = [0] * (new_target + 1)
    dp[0] = 1  # 初始化和为0的子集有一个
    for num in nums:
        for i in range(new_target, num - 1, -1):
            dp[i] += dp[i - num]
    return dp[new_target]


# lists = [1, 1, 1, 1, 1]
# t = 3
# print(find_target_sum_ways(lists, t))


def find_max_form(strs, m, n):
    """ 474.一和零: 0-1背包问题"""
    # 每个二进制字符串可以看作一个物品
    # m和n可以看作是背包容量
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    # 表示最多使用i个0和j个1的情况下最大子集的大小
    for c in strs:
        ones = c.count('1')
        zeros = c.count('0')
        for i in range(m, zeros - 1, -1):
            for j in range(n, ones - 1, -1):
                dp[i][j] = max(dp[i][j], dp[i - zeros][j - ones] + 1)
    return dp[m][n]


# s = ["10", "0001", "111001", "1", "0"]
# x = 5
# y = 3
# print(find_max_form(s, x, y))
