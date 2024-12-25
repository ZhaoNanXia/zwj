from BinaryTree import TreeNode, BinaryTreePrint

"""-------------------0-1背包问题--------------------"""


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

"""-------------------多重背包问题--------------------"""


def coin_change(coins, amount):
    """ 322.零钱兑换：多重背包问题，硬币是物品，金额为背包"""
    dp = [float('inf')] * (amount + 1)  # dp数组：凑成金额i所需的最少硬币数量
    dp[0] = 0  # 凑成金额0可以不选择任何硬币
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1


# lists = [12]
# t = 11
# print(coin_change(lists, t))

def combination_sum4(nums, target):
    """ 377.组合总和Ⅳ：多重背包问题，元素是物品，和为背包"""
    dp = [0] * (target + 1)  # dp数组：组成和为i的正整数的组合数量
    dp[0] = 1  # 组成和为0的数可以不选择任何元素，即有一种组合方式
    for i in range(1, target + 1):
        for num in nums:
            if i >= num:  # 如果num小于i,则num无法形成i，无需更新
                dp[i] += dp[i - num]
        print(dp)
    return dp[target]


# lists = [1, 2, 3]
# t = 4
# print(combination_sum4(lists, t))


def num_squares(n):
    """ 279.完全平方数：多重背包问题，完全平方数是物品，和n是背包容量"""
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    for i in range(1, n + 1):
        j = 1
        while j * j <= i:
            dp[i] = min(dp[i], dp[i - j * j] + 1)
            j += 1
    return dp[-1]


# x = 12
# print(num_squares(x))


def word_break(s, word_dict):
    """ 139.单词拆分: 多重背包问题，单词就是物品，字符串就是背包"""
    n = len(s)
    dp = [False] * (n + 1)  # 前i个字符是否能被单词组成
    dp[0] = True
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j: i] in word_dict:
                dp[i] = True
                break
    return dp[n]


# x = "leetcode"
# wordDict = ["leet", "code"]
# print(word_break(x, wordDict))


class RobProblem:
    """ 打家劫舍问题 """
    @staticmethod
    def rob(nums):
        """ 198.打家劫舍 """
        n = len(nums)
        if n == 0:
            return 0
        elif n == 1:
            return nums[0]
        dp = [0] * n
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, n):
            dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])
        return dp[-1]

    @staticmethod
    def rob_1(nums):
        """ 213。打家劫舍Ⅱ """
        n = len(nums)
        if n == 0:
            return 0
        elif n == 1:
            return nums[0]
        else:
            return max(RobProblem.rob(nums[1:]), RobProblem.rob(nums[:-1]))

    @staticmethod
    def rob_2(root):
        """ 337.打家劫舍Ⅲ """
        def rob_helper(node):
            if not node:
                return [0, 0]

            left = rob_helper(node.left)
            right = rob_helper(node.right)

            rob = node.value + left[1] + right[1]
            not_rob = max(left) + max(right)
            return [rob, not_rob]

        return max(rob_helper(root))


RP = RobProblem()
# lists = [1, 2, 3, 4]
# print(RP.rob(lists))
# print(RP.rob_1(lists))

# nums = [3, 2, 3, None, 3, None, 1]
# root = BinaryTreePrint.array_to_binarytree(nums)
# print(RP.rob_2(root))


class StockProblem:
    """ 股票问题 """
    @staticmethod
    def max_profit(prices):
        """ 121.买卖股票的最佳时机：只能交易一次 """
        max_profit = float('-inf')
        min_price = float('inf')
        for price in prices:
            min_price = min(min_price, price)
            max_profit = max(max_profit, price - min_price)
        return max_profit

    @staticmethod
    def max_profit_1(prices):
        """ 122. 买卖股票的最佳时机II：可以交易多次，但每次最多只能持有一股 """
        n = len(prices)
        dp = [[0] * 2 for _ in range(n)]
        dp[0][0] = 0
        dp[0][1] = -prices[0]
        for i in range(1, n):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
        return dp[-1][0]

    @staticmethod
    def max_profit_2(prices):
        """ 123. 买卖股票的最佳时机III：最多交易两次 """
        n = len(prices)
        buy_1 = buy_2 = -prices[0]
        sell_1 = sell_2 = 0
        for i in range(1, n):
            buy_1 = max(buy_1, -prices[i])
            sell_1 = max(sell_1, buy_1 + prices[i])
            buy_2 = max(buy_2, sell_1 - prices[i])
            sell_2 = max(sell_2, buy_2 + prices[i])

        return sell_2

    @staticmethod
    def max_profit_3(prices, k):
        """ 188. 买卖股票的最佳时机IV：最多可以交易K次，最多同时交易一笔 """
        print('输入prices数组：', prices, '最多交易次数k:', k)
        print('-' * 50)
        n = len(prices)
        # k次交易便有k次买入和k次卖出,则每一天都有交易（2k种可能）和不交易（1种可能）两种情况，即2k+1
        # 索引0代表不交易，奇数索引代表买入，偶数索引代表卖出
        dp = [0] * (2 * k + 1)
        print('定义的dp数组：', dp)
        print('-' * 50)
        for i in range(1, 2 * k + 1, 2):  # 初始化每一次买入操作
            dp[i] = -prices[0]
        print('初始化买入操作之后的dp数组：', dp)
        print('-' * 50)
        for i in range(1, n):
            for j in range(0, 2 * k - 1, 2):
                dp[j + 1] = max(dp[j + 1], dp[j] - prices[i])  # 买入：对比前一次买入后的收益和此次买入后的收益
                dp[j + 2] = max(dp[j + 2], dp[j + 1] + prices[i])  # 卖出：对比前一次卖出后的收益和此次卖出后的收益
                print(f'遍历到元素 {prices[i]} 时进行第 {j//2+1} 次交易时的dp数组：', dp)
            print('-' * 50)
        return dp[-1]

    @staticmethod
    def max_profit_4(prices):
        """ 309. 买卖股票的最佳时机含冷冻期：尽可能多交易，但卖出后第二天无法买入 """
        n = len(prices)
        # 每一天都有三种可能的状态：持有股票、未持有股票且当天没有卖出操作（处于冷冻期）、未持有股票且当天刚卖出
        dp = [[0] * 3 for _ in range(n)]
        dp[0][2] = -prices[0]
        for i in range(1, n):
            # 不持有股票且当天未卖出：前一天不持有也没卖出 或者 前一天不持有但是卖出了导致今天是冷冻期无法持有
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1])
            # 不持有股票且当天卖出：基于前一天持有的基础上今天卖出
            dp[i][1] = dp[i - 1][2] + prices[i]
            # 持有股票：前一天就持有 或者 从前一天的未持有股票状态（非当天卖出的未持有，即不会导致今天是冷冻期）买入
            dp[i][2] = max(dp[i - 1][2], dp[i - 1][0] - prices[i])
            [print(x, end='\n') for x in dp[:i]]
            print('-' * 10)
        return max(dp[-1][0], dp[-1][1])

    @staticmethod
    def max_profit_5(prices, fee):
        """ 714. 买卖股票的最佳时机含手续费：不限交易次数，但是每交易一次都要付手续费 """
        n = len(prices)
        dp = [[0] * 2 for _ in range(n)]
        dp[0][0] = 0
        dp[0][1] = -prices[0]
        for i in range(1, n):
            # 不持有股票：前一天就不持有 或者 前一天持有但是今天卖出了（需要付手续费）
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i] - fee)
            # 持有股票：前一天就持有 或者 前一天不持有但是今天买入了
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
        return dp[-1][0]


SP = StockProblem()
# nums = [7, 1, 5, 3, 6, 4]
# print(SP.max_profit(nums))
# print(SP.max_profit_1(nums))
# nums = [3, 3, 5, 0, 0, 3, 1, 4]
# print(SP.max_profit_2(nums))
# nums = [3, 2, 6, 5, 0, 3]
# k = 2
# print(SP.max_profit_3(nums, k))
# nums = [1, 2, 3, 0, 2]
# print(SP.max_profit_4(nums))
# nums = [1, 3, 2, 8, 4, 9]
# fee = 2
# print(SP.max_profit_5(nums, fee))


class SubSequenceProblem:
    """ 子序列问题 """
    @staticmethod
    def length_of_lis(nums):
        """ 300.最长递增子序列：单调栈解法 """
        n = len(nums)
        stack = [nums[0]]
        for i in range(1, n):
            if nums[i] > stack[-1]:
                stack.append(nums[i])
            else:
                left, right, loc = 0, len(stack) - 1, 0
                while left <= right:
                    mid = (left + right) // 2
                    if stack[mid] >= nums[i]:
                        loc = mid
                        right = mid - 1
                    else:
                        left = mid + 1
                stack[loc] = nums[i]
            print(stack)
        return len(stack)

    @staticmethod
    def length_of_lis_1(nums):
        """ 300.最长递增子序列：动态规划解法 """
        n = len(nums)
        dp = [1] * n
        for i in range(1, n):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)

        return dp[-1]

    @staticmethod
    def find_length_of_lis(nums):
        """ 674.最长连续递增序列 """
        n = len(nums)
        dp = [1] * n
        for i in range(1, n):
            if nums[i] > nums[i - 1]:
                dp[i] = max(dp[i], dp[i - 1] + 1)
        return max(dp)

    @staticmethod
    def find_length(nums1, nums2):
        """ 718.最长重复子数组 """
        n1, n2 = len(nums1), len(nums2)
        # nums1的前i个数字组成的子数组 与 nums2的前j个数字组成的子数组 的最长重复子数组的长度
        # 需要注意：前i,j个数字，对应到数组当中就是以索引i-1、j-1结尾的子数组
        dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]
        max_length = 0
        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                if nums1[i - 1] == nums2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    max_length = max(max_length, dp[i][j])
        return max_length

    @staticmethod
    def longest_common_subsequence(text1, text2):
        """ 1143. 最长公共子序列 """
        n1, n2 = len(text1), len(text2)
        dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]
        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[-1][-1]

    @staticmethod
    def max_subarray(nums):
        """ 53.最大子数组和 """
        n = len(nums)
        for i in range(1, n):
            nums[i] = max(nums[i], nums[i - 1] + nums[i])
        return max(nums)

    @staticmethod
    def is_subsequence(s, t):
        """ 392.判断子序列：动态规划解法 """
        n1, n2 = len(s), len(t)
        # s的前i个字符 与 t的前j个字符 所匹配的字符数量
        # 如果两个字符串从前往后所匹配的字符数量恰好等于字符串s的长度则说明s是t的子序列
        dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]
        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                if s[i - 1] == t[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1  # 两个字符相等情况下，匹配数量+1
                else:
                    dp[i][j] = dp[i][j - 1]  # 两个字符不相等情况下忽略t的第j个字符
        return dp[-1][-1] == n1

    @staticmethod
    def is_subsequence_1(s, t):
        """ 392.判断子序列：双指针解法 """
        n1, n2 = len(s), len(t)
        i = j = 0
        while i < n1 and j < n2:
            if s[i] == t[j]:
                i += 1
            j += 1
        return i == n1

    @staticmethod
    def num_distinct(s, t):
        """ 115.不同的子序列：只能对字符串s进行删除操作 """
        n1, n2 = len(s), len(t)
        if n1 < n2:
            return 0
        # t的前j个字符在s的前i个字符中出现的次数
        dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]
        for i in range(n1 + 1):
            dp[i][0] = 1  # 初始化：空字符串始终是一个s的子序列
        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                if s[i - 1] == t[j - 1]:
                    # 如果两个字符相等，则t的前j个字符在s的前i个字符中出现的次数=
                    # t的前j-1个字符在s的前i-1个字符中出现的次数+t的前j个字符在s的前i-1个字符中出现的次数
                    dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]
                else:
                    # 如果两个字符不相等，则t的前j个字符在s的前i个字符中出现的次数=t的前j个字符在s的前i-1个字符中出现的次数
                    # 忽略掉s的第i个字符
                    dp[i][j] = dp[i - 1][j]
        return dp[-1][-1]

    @staticmethod
    def min_distance(word1, word2):
        """ 583.两个字符串的删除操作：只能对两个字符串进行删除操作 """
        n1, n2 = len(word1), len(word2)
        # 使word1的前i个字符 与 word2的前j个字符相同 所需的最少步数
        dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]
        # 初始化：使word1的前i个字符 与 空字符串相同 所需的步数
        for i in range(n1 + 1):
            dp[i][0] = i
        # 初始化：使word2的前j个字符 与 空字符串相同 所需的步数
        for j in range(n2 + 1):
            dp[0][j] = j
        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                if word1[i - 1] == word2[j - 1]:
                    # 如果当前字符相等，则与不需要两个字符的情况等同
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    # 如果当前字符不相等，则可以删除掉word1的第i个字符或者删除掉word2的第j个字符
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + 1
        return dp[-1][-1]

    @staticmethod
    def min_distance_1(word1, word2):
        """ 72.编辑距离：可以删除、插入、替换 """
        n1, n2 = len(word1), len(word2)
        # 使word1的前i个字符 转换成 word2的前j个字符 所需的最少步数
        dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]
        # 初始化：使word1的前i个字符 转换成 空字符串相同 所需的步数
        for i in range(n1 + 1):
            dp[i][0] = i
        # 初始化：使word2的前j个字符 转换成 空字符串相同 所需的步数
        for j in range(n2 + 1):
            dp[0][j] = j
        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                if word1[i - 1] == word2[j - 1]:
                    # 如果当前字符相等，则与不需要两个字符的情况等同
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    # 如果当前字符不相等，则可以删除掉word1的第i个字符或者删除掉word2的第j个字符、
                    # 可以对其中一个字符进行替换操作获插入一个相等的字符
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
        return dp[-1][-1]


SSP = SubSequenceProblem()
# lists = [10, 9, 2, 5, 3, 7, 101, 18]
# lists = [0, 1, 0, 3, 2, 3]
# print(SSP.length_of_lis(lists))
# print(SSP.length_of_lis_1(lists))
# lists = [1, 3, 5, 4, 7]
# print(SSP.find_length_of_lis(lists))
# lists1 = [1, 2, 3, 2, 1]
# lists2 = [3, 2, 1, 4, 7]
# print(SSP.find_length(lists1, lists2))
# t1 = 'abcde'
# t2 = 'ace'
# print(SSP.longest_common_subsequence(t1, t2))
# lists = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
# print(SSP.max_subarray(lists))
# s = 'abc'
# t = 'ahbgdc'
# print(SSP.is_subsequence_1(s, t))
# s = 'rabbbit'
# t = 'rabbit'
# print(SSP.num_distinct(s, t))
# word1 = "leetcode"
# word2 = "etco"
# print(SSP.min_distance(word1, word2))
word1 = "horse"
word2 = "ros"
print(SSP.min_distance_1(word1, word2))
