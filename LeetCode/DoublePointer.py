"""
双指针法：一种常用的算法技巧，涉及数组、链表、字符串等相关问题，其核心思想是使用两个指针来遍历数据结构以达到优化时间复杂度的目的，可主要分为以下几类：
1、快慢指针：一个指针在前另一个指针在后 或者 快指针一次走两步，慢指针一次走一步，如判断链表是否有环，找链表中点问题
2、对撞指针：使用两个指针从两端向中间移动，如两数之和、判断回文字符串问题
3、滑动窗口：使用两个指针维护一个窗口，一个指针向右扩展窗口，另一个指针在左边向右收缩窗口，如字符串匹配问题
4、归并问题：使用两个指针分别指向两个数据结构，依次比较后合并，如归并排序问题
以及一些交叉问题
"""
import collections

""" 快慢指针问题 """


def move_zeroes(nums):
    """ 283.移动零 """
    n = len(nums)
    i = j = 0
    while j < n:
        if nums[j] != 0:
            nums[j], nums[i] = nums[i], nums[j]
            i += 1
        j += 1
    return nums


# lists = [0, 1, 0, 3, 12]
# print(move_zeroes(lists))


""" 对撞指针问题 """


def max_area(height):
    """ 11.盛最多水的容器 """
    n = len(height)
    i, j = 0, n - 1
    max_left = max_right = 0
    area = 0
    while i <= j:
        max_left, max_right = max(max_left, height[i]), max(max_right, height[j])
        if max_left < max_right:
            area = max(area, max_left * (j - i))
            i += 1
        else:
            area = max(area, max_right * (j - i))
            j -= 1
    return area


# lists = [1, 8, 6, 2, 5, 4, 8, 3, 7]
# print(max_area(lists))


""" 滑动窗口问题 """


def length_of_longest_substring(s):
    """ 3.无重复字符的最长子串 """
    n = len(s)
    start_index = 0
    max_length = 0
    windows = {}
    for i in range(n):
        # 为什么要windows[s[i]] >= start_index？因为字符处于windows中并不一定意味着其就是重复字符了，
        # 一定要同时处于滑动窗口中才能说明其与滑动窗口中的字符重复了，如输入"tmmzuxt"
        if s[i] in windows and windows[s[i]] >= start_index:
            # 为什么不是start_index+=1?因为遇到的重复字符不一定是start_index位置处的字符，可能是窗口中的任何一个字符
            start_index = windows[s[i]] + 1
        else:
            # 为什么是i - start_index + 1？因为遍历下标是从0开始的
            max_length = max(max_length, i - start_index + 1)
        windows[s[i]] = i
    return max_length


# s1 = "tmmzuxt"
# print(length_of_longest_substring(s1))


def min_window(s, t):
    """ 76.最小覆盖子串 """
    m, n = len(s), len(t)
    if m < n:
        return ''
    # 使用哈希表来保存遍历过程中遇到的每个字符，键为字符，值为索引
    character = collections.defaultdict(int)
    count_t = 0  # 记录当前窗口内字符串t中字符的数量
    for c in t:
        character[c] += 1
        count_t += 1
    start_index = 0  # 窗口的起始位置
    res = (start_index, float('inf'))  # 记录当前最小的包括字符串t的窗口的起始和结束索引
    for i, c in enumerate(s):
        if character[c] > 0:
            count_t -= 1
        character[c] -= 1
        # 当count_t等于0的时候说明当前窗口已经囊括了字符串t
        while count_t == 0:
            if i - start_index < res[1] - res[0]:  # 更新最小囊括t的窗口起始位置
                res = (start_index, i)
            # 移动窗口起始位置，如果起始位置对应的字符恰好是t中的字符，移掉一个就说明窗口内又缺了一个
            if character[s[start_index]] == 0:
                count_t += 1
            character[s[start_index]] += 1  # 同时更新哈希表
            start_index += 1
    return '' if res[1] > m else s[res[0]: res[1] + 1]


# s = "ADOBECODEBANC"
# t = "ABC"
# print(min_window(s, t))


def max_sliding_window(nums, k):
    """ 239.滑动窗口最大值 """
    n = len(nums)
    deque = collections.deque()  # 使用双端队列储存窗口内元素索引
    res = []
    for i in range(n):
        # 若窗口右端位置为i，则左端位置为i-k+1，所以窗口中元素索引一定要大于等于i-k+1，若小于则要移除
        if deque and deque[0] < i - k + 1:
            deque.popleft()
        # 若当前元素小于队列末尾索引对应的元素则不管直接加入；
        # 若大于则剔除队列末尾索引往前查找，要保证队列中索引对应的元素是单调递减的才能使当前窗口的最大值索引始终位于队列头部
        while deque and nums[deque[-1]] < nums[i]:
            deque.pop()
        deque.append(i)
        # 遍历到第一个窗口之后，之后每次都要往结果中添加最大值
        if i >= k - 1:
            res.append(nums[deque[0]])
    return res


lists = [1, 3, -1, -3, 5, 3, 6, 7]
K = 3
print(max_sliding_window(lists, K))


