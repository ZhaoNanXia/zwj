class SortAlgorithm:

    def merge_sort(self, nums):
        """ 归并排序 """
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
        """ 快速排序 """
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


lists = [1, 4, 2, 7, 5, 6]
sort = SortAlgorithm()
# print(sort.merge_sort(lists))
print(sort.quick_sort(lists))











