class SortAlgorithm:

    def merge_sort(self, nums):
        """ 归并排序：递归地将数组切分到最小单位，再逆向逐步合并 """
        n = len(nums)
        if n <= 1:
            return nums

        mid = n // 2
        left = nums[:mid]
        right = nums[mid:]

        left = self.merge_sort(left)
        right = self.merge_sort(right)

        res = [0] * n
        i, j, k = 0, 0, 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                res[k] = left[i]
                i += 1
            else:
                res[k] = right[j]
                j += 1
            k += 1

        while i < len(left):
            res[k] = left[i]
            i += 1
            k += 1

        while j < len(right):
            res[k] = right[j]
            j += 1
            k += 1

        return res

    def quick_sort(self, nums):
        """ 快速排序：每次都要选择一个基准，将数组分为两部分，一部分都比基准小，一部分都比基准大 """
        return self.quick_sort_helper(nums, 0, len(nums) - 1)

    def quick_sort_helper(self, nums, first, last):
        """ 快速排序辅助函数 """
        if first < last:
            pivot = nums[first]

            i, j = first + 1, last
            while True:
                while i <= j and nums[i] <= pivot:
                    i += 1
                while i <= j and nums[i] >= pivot:
                    j -= 1
                if i <= j:
                    nums[i], nums[j] = nums[j], nums[i]
                else:
                    break
            nums[first], nums[j] = nums[j], nums[first]

            self.quick_sort_helper(nums, first, j - 1)
            self.quick_sort_helper(nums, j + 1, last)
        return nums

    @staticmethod
    def bubble_sort(nums):
        """ 冒泡排序：每一轮都交换相邻位置的两个数字，较大的数字位置靠后 """
        n = len(nums)
        exchange = True
        i = 0
        while i < n and exchange:  # 如果某一轮遍历过程当中没有发生交换操作则说明数组此时已经完成排序
            exchange = False
            for j in range(n-i-1):
                if nums[j] > nums[j + 1]:
                    nums[j], nums[j + 1] = nums[j + 1], nums[j]
                    exchange = True
            i += 1
        return nums

    @staticmethod
    def select_sort(nums):
        """ 选择排序：每次都选择一个最大的数字放在最后一个位置上 """
        n = len(nums)
        for i in range(n - 1, 0, -1):
            max_pos = 0
            for j in range(1, i + 1):
                if nums[j] > nums[max_pos]:
                    max_pos = j
            nums[i], nums[max_pos] = nums[max_pos], nums[i]
        return nums

    @staticmethod
    def insert_sort(nums):
        """ 插入排序：从第二个数字开始从后往前寻找每一个数字应该插入的正确位置"""
        n = len(nums)
        for i in range(1, n):
            current_value = nums[i]
            pos = i
            while pos > 0 and nums[pos - 1] > current_value:
                nums[pos] = nums[pos - 1]
                pos -= 1
            nums[pos] = current_value
        return nums


lists = [1, 4, 2, 7, 5, 6]
sort = SortAlgorithm()
# print(sort.merge_sort(lists))
# print(sort.quick_sort(lists))
# print(sort.bubble_sort(lists))
# print(sort.select_sort(lists))
print(sort.insert_sort(lists))











