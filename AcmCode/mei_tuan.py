import collections
import io
import sys
"""美团2025秋招第一场笔试第一题"""
# 模拟输入的字符串
# input_data = '''2
# 1
# 4
# '''

# 创建一个StringIO对象并将其设置为标准输入
# sys.stdin = io.StringIO(input_data)
#
# n = int(input())
#
# nums = []
# for line in sys.stdin:
#     number = [int(x) for x in line.split()]
#     nums.append(number)
#
# print(nums)
#
# for num in nums:
#     if num[0] % 2 == 0:
#         print('YES')
#     else:
#         print('NO')


"""美团2025秋招第一场笔试第二题"""
# 模拟输入的字符串
input_data = '''4
ab
abc
ab
ac
ac
'''

# 创建一个StringIO对象并将其设置为标准输入
sys.stdin = io.StringIO(input_data)

n = int(input())
print(n)
ans = input().split()[0]
print(ans)
length_ans = len(ans)
print(length_ans)

nums = []
for line in sys.stdin:
    number = line.split()[0]
    nums.append(number)

nums = set(nums)
print(nums)
nums_sort = sorted(nums, key=lambda x: (len(x), x))
print(nums_sort)

hash_table = collections.defaultdict(int)
for num in nums_sort:
    hash_table[len(num)] += 1
print(hash_table)

min_step, max_step = 0, 0
for i, num in enumerate(nums_sort):
    if len(num) != length_ans:
        continue
    else:
        min_step = i + 1
        max_step = i + hash_table.get(len(num))
        break

print(f'min_step: {min_step}, max_step: {max_step}')




