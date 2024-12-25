"""
前缀和相关问题
概念：通过预先计算并存储序列中每个元素之前所有元素的和，使得在需要时可以快速计算出序列中任意区间的元素和
步骤：初始化一个与原序列N长度相同的前缀和序列S，序列中第一个元素S[0]为原序列的第一个元素N[0]，对于第二个到第n个元素，则有S[i]=S[i-1]+N[i]
"""
import collections


def subarray_sum(nums, k):
    """ 560.和为k的子数组 """
    count = 0  # 和为k的子数组的数量
    prefix_sum_frequency = collections.defaultdict(int)  # 存储前缀和出现的次数
    prefix_sum = 0  # 记录当前位置的前缀和，初始为0
    prefix_sum_frequency[0] = 1  # 初始化前缀和0出现的次数为1，即空数组
    for num in nums:
        prefix_sum += num
        # 已知当前位置i处的前缀和为prefix_sum，目标和为k，如果在这之前存在一个位置j的前缀和x，恰好有prefix_sum-x=k，
        # 那么区间[i,j]所囊括的元素组成的子数组和为k，即在存在前缀和x=prefix_sum-k时和为k的子数组数量+1
        if prefix_sum_frequency[prefix_sum - k]:
            count += prefix_sum_frequency[prefix_sum - k]
        prefix_sum_frequency[prefix_sum] += 1  # 更新前缀和出现的次数
    return count


# lists = [1, -1, 0]
# K = 0
# print(subarray_sum(lists, K))


def product_except_self(nums):
    """ 238.除自身以外数组的乘积"""
    n = len(nums)
    answer = [0] * n
    answer[0] = 1
    suffix_product = 1
    # 计算每个元素之前所有元素的乘积
    for i in range(1, n):
        answer[i] = answer[i - 1] * nums[i - 1]
    for i in range(n - 2, -1, -1):
        suffix_product *= nums[i + 1]  # 计算当前元素之后的元素乘积，相当于后缀积
        answer[i] *= suffix_product  # 当前元素之前的元素乘积（前缀积） ✖ 当前元素之后的元素乘积（后缀积）
    return answer


# lists = [1, 2, 3, 4]
# print(product_except_self(lists))


""" N数之和问题 """


def two_sum(nums, target):
    """ 1.两数之和 """
    number_hash = collections.defaultdict(int)
    for i, num in enumerate(nums):
        if number_hash and target - num in number_hash:
            return [number_hash[target - num], i]
        number_hash[num] = i


# lists = [2, 7, 11, 15]
# t = 9
# print(two_sum(lists, t))

def three_sum(nums):
    """ 15.三数之和 """
    n = len(nums)
    nums.sort()
    res = []
    for i in range(n):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        j, k = i + 1, n - 1
        while j < k:
            total = nums[i] + nums[j] + nums[k]
            if total == 0:
                res.append([nums[i], nums[j], nums[k]])
                while j < k and nums[k - 1] == nums[k]:
                    k -= 1
                while j < k and nums[j + 1] == nums[j]:
                    j += 1
                j += 1
                k -= 1
            elif total < 0:
                j += 1
            else:
                k -= 1
    return res


# lists = [-1, 0, 1, 2, -1, -4]
# print(three_sum(lists))


""" 合并区间问题 """


def merge(intervals):
    """ 56.合并区间 """
    n = len(intervals)
    intervals = sorted(intervals, key=lambda x: x[0])
    res = [intervals[0]]
    for interval in intervals:
        if res[-1][1] < interval[0]:
            res.append(interval)
        if interval[0] <= res[-1][1] <= interval[1]:  # 只有前一个区间的末端处于包含于后一个区间时才需要合并
            res[-1][1] = interval[1]
    return res


# lists = [[1, 3], [2, 6], [8, 10], [15, 18]]
# print(merge(lists))


""" 矩阵问题 """


def set_zeros(matrix):
    """ 73.矩阵置零 """
    m, n = len(matrix), len(matrix[0])
    row = [False] * m
    col = [False] * n
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 0:
                row[i] = True
                col[j] = True

    for i in range(m):
        for j in range(n):
            if row[i] or col[j]:
                matrix[i][j] = 0

    return matrix


# matrix = [[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]]
# print(set_zeros(matrix))


def spiral_order(matrix):
    """ 54.螺旋矩阵 """
    m, n = len(matrix), len(matrix[0])
    upper, low, left, right = 0, m - 1, 0, n - 1
    res = []
    while True:
        # 从左到右
        for i in range(left, right + 1):
            res.append(matrix[upper][i])
        upper += 1
        if upper > low:
            break
        # 从上到下
        for i in range(upper, low + 1):
            res.append(matrix[i][right])
        right -= 1
        if right < left:
            break
        # 从右往左
        for i in range(right, left - 1, -1):
            res.append(matrix[low][i])
        low -= 1
        if upper > low:
            break
        # 从下到上
        for i in range(low, upper - 1, -1):
            res.append(matrix[i][left])
        left += 1
        if right < left:
            break

    return res


# matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
# print(spiral_order(matrix))


def rotate(matrix):
    """ 48.旋转图像 """
    m = len(matrix)
    for i in range(m):
        for j in range(i, m):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for i in range(m):
        matrix[i] = matrix[i][::-1]
    return matrix


matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(rotate(matrix))
