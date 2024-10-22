import io
import sys

# 模拟输入的字符串
input_data = '''5
1 2 3 
1 2 3 4 5
7 8 9 10
'''

# 创建一个StringIO对象并将其设置为标准输入
sys.stdin = io.StringIO(input_data)

# 使用input方法读取首行数字
n = int(input())
# 使用input方法每次单独读取一行数字
# number = list(map(int, input().split()))

# 循环读取每行数据，并加入到一个数组中
nums = []
# for line in sys.stdin:
#     numbers = list(map(int, line.split()))
#     nums.append(numbers)

for line in sys.stdin:
    numbers = [int(x) for x in line.split()]
    nums.append(numbers)

print('n:', n)
print('nums:', nums)
