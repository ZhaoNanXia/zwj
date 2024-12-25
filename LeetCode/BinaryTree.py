from collections import deque
import matplotlib.pyplot as plt
import networkx as nx


class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 定义一个二叉树
# root = TreeNode(1)
# root.left = TreeNode(2, TreeNode(4), TreeNode(5))
# root.right = TreeNode(3, TreeNode(6))


class BinaryTreePrint:
    def print_binary_tree(self, node, indent=0):
        """ 打印二叉树 """
        if not node:
            return
        # 打印右子树
        self.print_binary_tree(node.right, indent + 4)
        # 打印当前节点
        print(" " * indent + str(node.val))
        # 打印左子树
        self.print_binary_tree(node.left, indent + 4)

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


class DrawBinaryTree:
    """ 绘制二叉树：未调试好 """
    def get_positions(self, root, positions, depth=0, pos_x=0.5):
        """递归地为每个节点分配位置"""
        if root:
            positions[root.val] = (depth, pos_x)
            self.get_positions(root.left, positions, depth + 1, pos_x - 0.2)
            self.get_positions(root.right, positions, depth + 1, pos_x + 0.2)

    def plot_binary_tree(self, root):
        if root is None:
            return

        # 创建一个有向图
        G = nx.DiGraph()

        # 创建一个字典来保存每个节点的位置
        positions = {}

        # 为每个节点分配位置
        self.get_positions(root, positions)

        # 将节点添加到图中
        for node in positions:
            G.add_node(node)

        # 将边添加到图中
        queue = [root]
        while queue:
            node = queue.pop(0)
            if node.left:
                G.add_edge(node.val, node.left.val)
                queue.append(node.left)
            if node.right:
                G.add_edge(node.val, node.right.val)
                queue.append(node.right)

        # 绘制图形
        nx.draw(G, pos=positions, with_labels=True, node_size=1500, node_color="lightblue", font_size=10,
                font_weight="bold", arrows=True, arrowstyle="->", arrowsize=20)

        # 显示图形
        plt.show()


DBT = DrawBinaryTree()


class TraversalBinaryTree:
    """ 二叉树的遍历：递归法和迭代法 """
    def preorder_traversal(self, root):
        """ 前序遍历：根节点——>左子树——>右子树 """
        if not root:
            return []
        return [root.val] + self.preorder_traversal(root.left) + self.preorder_traversal(root.right)

    @staticmethod
    def preorder_traversal_iterative(root):
        """ 前序遍历：根节点——>左子树——>右子树 """
        if not root:
            return []
        stack = [root]
        result = []
        while stack:
            current_root = stack.pop()
            result.append(current_root.val)
            if current_root.right:
                stack.append(current_root.right)
            if current_root.left:
                stack.append(current_root.left)
        return result

    def inorder_traversal(self, root):
        """ 中序遍历：左子树——>根节点——>右子树 """
        if not root:
            return []
        return self.inorder_traversal(root.left) + [root.val] + self.inorder_traversal(root.right)

    @staticmethod
    def inorder_traversal_iterative(root):
        """ 迭代法中序遍历：左子树——>根节点——>右子树 """
        if not root:
            return []
        stack = []
        result = []
        current_node = root
        while current_node or stack:
            if current_node:
                stack.append(current_node)
                current_node = current_node.left
            else:
                current_node = stack.pop()
                result.append(current_node.val)
                current_node = current_node.right
        return result

    def postorder_traversal(self, root):
        """ 后序遍历：左子树——>右子树——>根节点 """
        if not root:
            return []
        return self.postorder_traversal(root.left) + self.postorder_traversal(root.right) + [root.val]

    @staticmethod
    def postorder_traversal_iterative(root):
        """ 迭代法后序遍历：左子树——>右子树——>根节点 """
        if not root:
            return []
        stack = [root]
        result = []
        while stack:
            current_node = stack.pop()
            result.append(current_node.val)
            if current_node.left:
                stack.append(current_node.left)
            if current_node.right:
                stack.append(current_node.right)
        return result[::-1]

    @staticmethod
    def level_order_traversal(root):
        """ 层序遍历：逐层遍历，利用队列实现 """
        if not root:
            return []
        queue = [root]
        result = []
        while queue:
            current_node = queue.pop(0)
            result.append(current_node.val)
            if current_node.left:
                queue.append(current_node.left)
            if current_node.right:
                queue.append(current_node.right)
        return result


# BinaryTree = TraversalBinaryTree()

# print('前序遍历：', BinaryTree.preorder_traversal(root))
# print('中序遍历：', BinaryTree.inorder_traversal(root))
# print('后序遍历：', BinaryTree.postorder_traversal(root))
# print('层序遍历：', BinaryTree.level_order_traversal(root))

# print('前序遍历(迭代法)：', BinaryTree.preorder_traversal_iterative(root))
# print('中序遍历(迭代法)：', BinaryTree.inorder_traversal_iterative(root))
# print('后序遍历(迭代法)：', BinaryTree.postorder_traversal_iterative(root))


class BinaryTreeAttribute:
    """ 二叉树属性问题： """

    def invert_tree(self, root):
        """ 226.翻转二叉树 """
        if not root:
            return

        root.left, root.right = root.right, root.left
        self.invert_tree(root.left)
        self.invert_tree(root.right)
        return root

    @staticmethod
    def is_symmetric(root):
        """ 101.是否是对称二叉树 """
        if not root:
            return True

        def helper(left, right):
            if not left and not right:
                return True
            if not left or not right or left.val != right.val:
                return False
            return helper(left.left, right.right) and helper(left.right, right.left)

        return helper(root.left, root.right)

    def max_depth(self, root):
        """ 104.二叉树的最大深度 """
        if not root:
            return 0
        left_max = self.max_depth(root.left)
        right_max = self.max_depth(root.right)
        return max(left_max, right_max) + 1

    def min_depth(self, root):
        """ 111.二叉树的最小深度 """
        if not root:
            return 0
        if not root.left:
            return self.min_depth(root.right) + 1
        if not root.right:
            return self.min_depth(root.left) + 1
        return min(self.min_depth(root.left), self.min_depth(root.right)) + 1

    def count_nodes(self, root):
        """ 222.完全二叉树的节点数量（节点按照从上到下、从左到右的顺序填满） """
        if not root:
            return 0
        left_count = self.count_nodes(root.left)
        right_count = self.count_nodes(root.right)
        return left_count + right_count + 1

    def is_balanced(self, root):
        """ 110.平衡二叉树：任意节点的左右子树高度差不超过1 """
        def get_height(root):
            if not root:
                return 0
            left_height = get_height(root.left)
            right_height = get_height(root.right)
            # 递归过程中如果左子树或者右子树已经被标记为不平衡状态了 或者 该节点左右高度差大于1 直接返回-1
            if left_height == -1 or right_height == -1 or abs(left_height - right_height) > 1:
                return -1
            else:
                return max(left_height, right_height) + 1
        return get_height(root) != -1

    def binarytree_paths(self, root):
        """ 257.二叉树的所有路径 """
        if not root:
            return []
        path = [root.val]
        res = []

        def backtrack(root):
            # 只有当该节点的左右子节点都为空才说明该节点是叶子节点
            if not root.left and not root.right:
                res.append('->'.join(map(str, path)))
                return
            if root.left:
                path.append(root.left.val)
                backtrack(root.left)
                path.pop()
            if root.right:
                path.append(root.right.val)
                backtrack(root.right)
                path.pop()
        backtrack(root)
        return res

    def sum_of_left_leaves(self, root):
        """ 404.左叶子之和 """
        if not root:
            return 0

        def get_sum(root):
            if not root:
                return 0

            res = 0
            if root.left and not root.left.left and not root.left.right:
                res += root.left.val
            else:
                res += get_sum(root.left)
            res += get_sum(root.right)
            return res

        return get_sum(root)

    def find_bottom_left_value(self, root):
        """ 513.找树左下角的值 """
        queue = deque([root])
        res = 0
        while queue:
            node = queue.popleft()
            if node.right:
                queue.append(node.right)
            if node.left:
                queue.append(node.left)
            res = node.val
        return res

    def path_sum(self, root, target):
        """ 112.路径总和 """
        if not root:
            return False
        if not root.left and not root.right:
            return target == root.val
        return self.path_sum(root.left, target - root.val) or self.path_sum(root.right, target - root.val)

    def lowest_common_ancestor(self, root, p, q):
        """ 236.二叉树的最近公共祖先 """
        if not root or p == root or q == root:
            return root
        left = self.lowest_common_ancestor(root.left, p, q)
        right = self.lowest_common_ancestor(root.right, p, q)
        if not left:
            return right
        if not right:
            return left
        return root

    def merge_trees(self, root1, root2):
        """ 617.合并二叉树 """
        if not root1:
            return root2
        if not root2:
            return root1
        root = TreeNode(root1.val + root2.val)
        root.left = self.merge_trees(root1.left, root2.left)
        root.right = self.merge_trees(root1.right, root2.right)
        return root

    def build_tree(self, inorder, postorder):
        """ 106.从中序与后序遍历序列构造二叉树 """
        if not postorder:
            return
        root_val = postorder[-1]
        root = TreeNode(root_val)
        index = inorder.index(root_val)
        root.left = self.build_tree(inorder[:index], postorder[:len(inorder[:index])])
        root.right = self.build_tree(inorder[index + 1:], postorder[len(inorder[:index]): len(postorder) - 1])
        return root

    def construct_max_binarytree(self, nums):
        """ 654.最大二叉树 """
        if not nums:
            return
        root_val = max(nums)
        root = TreeNode(root_val)
        index = nums.index(root_val)
        root.left = self.construct_max_binarytree(nums[:index])
        root.right = self.construct_max_binarytree(nums[index+1:])
        return root

    def search_bst(self, root, val):
        """ 700.二叉搜索树中的搜索 """
        if not root:
            return
        if root.val == val:
            return root
        elif root.val < val:
            return self.search_bst(root.right, val)
        else:
            return self.search_bst(root.left, val)

    def isvalid_bst(self, root):
        """ 98.验证二叉搜索树：左子树的值都小于根节点，右子树的值都大于根节点 """
        def helper(node, lower=float('-inf'), upper=float('inf')):
            if not node:
                return True
            if not (lower < node.val < upper):
                return False
            return helper(node.left, lower, node.val) and helper(node.right, node.val, upper)
        return helper(root)

    def diameter_of_binarytree(self, root):
        """ 543.二叉树的直径 """
        res = 0

        def helper(node):
            nonlocal res
            if not node:
                return 0
            left_depth = helper(node.left)
            right_depth = helper(node.right)
            res = max(res, left_depth + right_depth)
            return max(left_depth, right_depth) + 1

        helper(root)
        return res


root1 = TreeNode(1)
root1.left = TreeNode(2, TreeNode(4), TreeNode(5))
root1.right = TreeNode(3, TreeNode(6))

root2 = TreeNode(1)
root2.left = TreeNode(2, TreeNode(4), TreeNode(5))
root2.right = TreeNode(3)

# 创建一个二叉搜索树
root3 = TreeNode(4)
root3.left = TreeNode(2, TreeNode(1), TreeNode(3))
root3.right = TreeNode(7)

BTP = BinaryTreePrint()
# BTP.print_binary_tree(root1)

BT = BinaryTreeAttribute()
# BTP.print_binary_tree(BT.invert_tree(root1))
# print(BT.is_symmetric(root1))
# print(BT.max_depth(root1))
# print(BT.min_depth(root1))
# print(BT.count_nodes(root1))
# print(BT.is_balanced(root1))
# print(BT.binarytree_paths(root1))
# print(BT.sum_of_left_leaves(root1))
# print(BT.find_bottom_left_value(root1))
# print(BT.path_sum(root1, 8))
# print(BT.lowest_common_ancestor(root1, root1.left.left, root1.left.right))
# print(BT.merge_trees(root1, root2))
# BTP.print_binary_tree(BT.merge_trees(root1, root2))
# inorder = [9, 3, 15, 20, 7]
# postorder = [9, 15, 7, 20, 3]
# BTP.print_binary_tree(BT.build_tree(inorder, postorder))
# lists = [3, 2, 1, 6, 0, 5]
# BTP.print_binary_tree(BT.construct_max_binarytree(lists))
# BTP.print_binary_tree(BT.search_bst(root3, 2))
# print(BT.isvalid_bst(root3))
print(BT.diameter_of_binarytree(root1))
