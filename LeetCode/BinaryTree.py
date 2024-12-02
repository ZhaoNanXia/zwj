class TreeNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


class BinaryTreePrint:

    def print_tree_pretty(self, node, indent=0):
        """ 打印二叉树 """
        if not node:
            return

        # 打印右子树
        self.print_tree_pretty(node.right, indent + 4)

        # 打印当前节点
        print(" " * indent + str(node.value))

        # 打印左子树
        self.print_tree_pretty(node.left, indent + 4)

    @staticmethod
    def array_to_binarytree(nums):
        """ 将数组转换为二叉树 """
        if not nums:
            return None

        root = TreeNode(nums[0])
        queue = [root]
        i = 1

        while i < len(nums):
            current_node = queue.pop(0)

            if nums[i] is not None:
                current_node.left = TreeNode(nums[i])
                queue.append(current_node.left)
            i += 1

            if i < len(nums) and nums[i] is not None:
                current_node.right = TreeNode(nums[i])
                queue.append(current_node.right)
            i += 1

        return root


class TraversalBinaryTree:
    """ 二叉树的遍历：递归法和迭代法 """
    def preorder_traversal(self, node):
        """ 前序遍历：根节点——>左子树——>右子树 """
        if not node:
            return []
        return [node.value] + self.preorder_traversal(node.left) + self.preorder_traversal(node.right)

    @staticmethod
    def preorder_traversal_iterative(node):
        """ 前序遍历：根节点——>左子树——>右子树 """
        if not node:
            return []
        stack = [node]
        result = []
        while stack:
            current_node = stack.pop()
            result.append(current_node.value)
            if current_node.right:
                stack.append(current_node.right)
            if current_node.left:
                stack.append(current_node.left)
        return result

    def inorder_traversal(self, node):
        """ 中序遍历：左子树——>根节点——>右子树 """
        if not node:
            return []
        return self.inorder_traversal(node.left) + [node.value] + self.inorder_traversal(node.right)

    @staticmethod
    def inorder_traversal_iterative(node):
        """ 迭代法中序遍历：左子树——>根节点——>右子树 """
        if not node:
            return []
        stack = []
        result = []
        current_node = node
        while current_node or stack:
            if current_node:
                stack.append(current_node)
                current_node = current_node.left
            else:
                current_node = stack.pop()
                result.append(current_node.value)
                current_node = current_node.right
        return result

    def postorder_traversal(self, node):
        """ 后序遍历：左子树——>右子树——>根节点 """
        if not node:
            return []
        return self.postorder_traversal(node.left) + self.postorder_traversal(node.right) + [node.value]

    @staticmethod
    def postorder_traversal_iterative(node):
        """ 迭代法后序遍历：左子树——>右子树——>根节点 """
        if not node:
            return []
        stack = [node]
        result = []
        while stack:
            current_node = stack.pop()
            result.append(current_node.value)
            if current_node.left:
                stack.append(current_node.left)
            if current_node.right:
                stack.append(current_node.right)
        return result[::-1]

    @staticmethod
    def level_order_traversal(node):
        """ 层序遍历：逐层遍历，利用队列实现 """
        if not node:
            return []
        queue = [node]
        result = []
        while queue:
            current_node = queue.pop(0)
            result.append(current_node.value)
            if current_node.left:
                queue.append(current_node.left)
            if current_node.right:
                queue.append(current_node.right)
        return result


root = TreeNode('A')
root.left = TreeNode('B', TreeNode('D'), TreeNode('E'))
root.right = TreeNode('C', TreeNode('F'))

# BinaryTree = TraversalBinaryTree()

# print('前序遍历：', BinaryTree.preorder_traversal(root))
# print('中序遍历：', BinaryTree.inorder_traversal(root))
# print('后序遍历：', BinaryTree.postorder_traversal(root))
# print('层序遍历：', BinaryTree.level_order_traversal(root))
#
# print('前序遍历(迭代法)：', BinaryTree.preorder_traversal_iterative(root))
# print('中序遍历(迭代法)：', BinaryTree.inorder_traversal_iterative(root))
# print('后序遍历(迭代法)：', BinaryTree.postorder_traversal_iterative(root))


class RecursiveTraversal:
    """ 将递归遍历的过程和中间结果打印出来 """
    def __init__(self):
        self.result = []

    # 前序遍历 (根 -> 左 -> 右)
    def preorder(self, node, path="Root", result=None):
        if result is None:
            result = []
        if not node:
            print(f"{path} -> None | Current Result: {result}")
            return result
        result.append(node.value)
        print(f"{path} -> Visit: {node.value} | Current Result: {result}")
        self.preorder(node.left, path + f" -> L({node.value})", result)
        self.preorder(node.right, path + f" -> R({node.value})", result)
        return result

    # 中序遍历 (左 -> 根 -> 右)
    def inorder(self, node, path="Root", result=None):
        if result is None:
            result = []
        if not node:
            print(f"{path} -> None | Current Result: {result}")
            return result
        self.inorder(node.left, path + f" -> L({node.value})", result)
        result.append(node.value)
        print(f"{path} -> Visit: {node.value} | Current Result: {result}")
        self.inorder(node.right, path + f" -> R({node.value})", result)
        return result

    # 后序遍历 (左 -> 右 -> 根)
    def postorder(self, node, path="Root", result=None):
        if result is None:
            result = []
        if not node:
            print(f"{path} -> None | Current Result: {result}")
            return result
        self.postorder(node.left, path + f" -> L({node.value})", result)
        self.postorder(node.right, path + f" -> R({node.value})", result)
        result.append(node.value)
        print(f"{path} -> Visit: {node.value} | Current Result: {result}")
        return result


# traversal = RecursiveTraversal()

# 打印前序遍历的递归过程
# print("前序遍历过程:")
# traversal.preorder(root)

# # 打印中序遍历的递归过程
# print("\n中序遍历过程:")
# traversal.inorder(root)
#
# # 打印后序遍历的递归过程
# print("\n后序遍历过程:")
# traversal.postorder(root)


class BinaryTreeAttribute:
    """ 二叉树属性问题： """

    def invert_tree(self, node):
        """ 翻转二叉树 """
        if not node:
            return

        node.left, node.right = node.right, node.left

        self.invert_tree(node.left)
        self.invert_tree(node.right)

        return node

    @staticmethod
    def is_symmetric(node):
        """ 是否是对称二叉树 """
        if not node:
            return True

        def helper(left, right):
            if not left and not right:
                return True
            if not left or not right or left.value != right.value:
                return False
            return helper(left.left, right.right) and helper(left.right, right.left)

        return helper(node.left, node.right)

    def max_depth(self, node):
        """ 二叉树的最大深度 """
        if not node:
            return 0
        left_max = self.max_depth(node.left)
        right_max = self.max_depth(node.right)
        return max(left_max, right_max) + 1


# print_tree_pretty(root)

BinaryTree = BinaryTreeAttribute()
# print_tree_pretty(BinaryTree.invert_tree(root))
# print(BinaryTree.is_symmetric(root))
# print(BinaryTree.max_depth(root))


class BinaryTreeSimulate:
    """ 模拟打印二叉树的递归过程 """

    def max_depth(self, node, depth=0):
        """ 二叉树的最大深度，并打印递归过程 """
        if not node:
            print(f"{'  ' * depth}Reached leaf node, return 0")
            return 0

        print(f"{'  ' * depth}Visit Node {node.value}")

        # 递归访问左子树
        left_max = self.max_depth(node.left, depth + 1)
        print(f"{'  ' * (depth + 1)}Left max depth of Node {node.value}: {left_max}")

        # 递归访问右子树
        right_max = self.max_depth(node.right, depth + 1)
        print(f"{'  ' * (depth + 1)}Right max depth of Node {node.value}: {right_max}")

        # 计算当前节点的最大深度
        current_max = max(left_max, right_max) + 1

        print(f"{'  ' * depth}Return max depth {current_max} for Node {node.value}")

        return current_max


# BinaryTreeSim = BinaryTreeSimulate()
# print(BinaryTreeSim.max_depth(root))
