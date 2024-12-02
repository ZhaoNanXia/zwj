class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class LinkedList:
    def __init__(self):
        self.head = None  # 初始化链表为空

    def append(self, val):
        """在链表尾部添加节点"""
        new_node = ListNode(val)
        if not self.head:  # 链表为空
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def display(self):
        """打印链表"""
        current = self.head
        while current:
            print(current.val, end=" -> ")
            current = current.next
        print("None")

    @staticmethod
    def array_to_linked_list(nums):
        """
        将一个数组转换成单向链表
        :param nums: 输入数组
        :return: 链表的头节点
        """
        if not nums:
            return None  # 空数组返回空链表
        dummy = ListNode(0)
        current = dummy
        for val in nums:
            current.next = ListNode(val)
            current = current.next
        return dummy.next

    def arrays_to_linked_lists(self, nums):
        """
        将二维数组转换为链表列表
        :param nums: 二维数组
        :return: 链表列表
        """
        return [self.array_to_linked_list(num) for num in nums]

    @staticmethod
    def linked_list_to_array(head):
        """
        将链表转换为数组
        :param head: 链表的头节点
        :return: 包含链表值的数组
        """
        array = []
        current = head
        while current:
            array.append(current.val)  # 将当前节点的值添加到数组中
            current = current.next  # 移动到下一个节点
        return array


LinkList = LinkedList()


def merge_k_lists(lists):
    """ 合并k个升序链表 """
    if not lists:
        return
    while len(lists) > 1:
        i, j = 0, len(lists) - 1
        while i < j:
            l1, l2 = lists[i], lists[j]
            dummy = ListNode(0)
            current_node = dummy
            while l1 and l2:
                if l1.val < l2.val:
                    current_node.next = l1
                    l1 = l1.next
                else:
                    current_node.next = l2
                    l2 = l2.next
                current_node = current_node.next
            current_node.next = l1 if l1 else l2
            lists[i] = dummy.next
            lists.pop()
            i += 1
            j -= 1
    return lists[0]


link_lists = [[1, 4, 5], [1, 3, 4], [2, 6]]
# print(LinkList.arrays_to_linked_lists(link_lists))
# res = merge_k_lists(LinkList.arrays_to_linked_lists(link_lists))
# print(LinkList.linked_list_to_array(res))


def is_palindrome(head):
    """ 是否回文链表 """
    if not head:
        return True
    node = []
    while head:
        node.append(head.val)
        head = head.next
    i, j = 0, len(node) - 1
    while i < j:
        if node[i] != node[j]:
            return False
        i += 1
        j -= 1
    return True


# linklist = [1, 2, 2, 1, 3]
# res = is_palindrome(LinkList.array_to_linked_list(linklist))
# print(res)


def reverse_list(head):
    """ 反转链表 """
    if not head:
        return
    current_node = head
    prev_node = None
    while current_node:
        temp_node = current_node.next
        current_node.next = prev_node
        prev_node = current_node
        current_node = temp_node
    return prev_node


linklist = [1, 2, 3, 4, 5]
res = reverse_list(LinkList.array_to_linked_list(linklist))
print(LinkList.linked_list_to_array(res))
