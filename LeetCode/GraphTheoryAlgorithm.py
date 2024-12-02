"""
有向图——>加权有向图
    入度：指向该节点，出度：从该节点出发的边数
无向图——>加权无向图
连通性：节点的连通情况
    无向图中若任何两个节点都能到达称之为连通图；有向图中若任何两个节点可以互相到达(注意方向)称之为强连通图。
    无向图中的极大连通子图称之为连通分量；有向图极大强连通子图称之为强连通分量。
图的构造
    邻接矩阵：二维矩阵，索引代表节点，值代表边的权重。
    邻接表：数组＋链表的方式，从边的数量出发，有多少边该节点对应的链表就有多大。
图的遍历方式
    深度优先搜索：沿一个方向搜索，不到黄河不回头
    广度优先搜索：四面八方搜索
"""


class IslandProblem:
    def __init__(self):
        self.res = []

    """----------------------------------岛屿数量--------------------------------------------"""
    def nums_islands_dfs(self, grid, i, j):
        """ 岛屿数量问题：深度优先搜索方法 """
        if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] != '1':
            return
        ''' 第一种写法 '''
        grid[i][j] = '2'
        self.res.append([r[:] for r in grid])
        self.nums_islands_dfs(grid, i + 1, j)
        self.nums_islands_dfs(grid, i - 1, j)
        self.nums_islands_dfs(grid, i, j - 1)
        self.nums_islands_dfs(grid, i, j + 1)

        ''' 第二种写法 '''
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        grid[i][j] = '2'
        self.res.append([r[:] for r in grid])  # 这句代码只为保存结果，验证使用
        for di, dj in directions:
            self.nums_islands_bfs(grid, i + di, j + dj)

    def nums_islands_bfs(self, grid, i, j):
        """ 岛屿数量问题：广度优先搜索方法 """
        queue = [(i, j)]  # 借助队列存储每个陆地周围的所有陆地
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        grid[i][j] = '2'
        self.res.append([r[:] for r in grid])

        while queue:
            x, y = queue.pop(0)
            for di, dj in directions:
                nx, ny = x + di, y + dj
                if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == '1':
                    queue.append((nx, ny))  # 每次遍历将每一块陆地四周的所有陆地都添加到队列当中
                    grid[nx][ny] = '2'
                    self.res.append([r[:] for r in grid])

    def nums_islands(self, grid):
        """
        200.岛屿数量
        时间复杂度：O(m × n)，因为要遍历区域内所有位置，两种方法一致。
        空间复杂度：O(m × n)，递归的深度最大可能是整个网格的大小，因此最大可能使用 O(m×n) 的栈空间；
                    同时广度优先搜索里，栈最大可能存储m×n块陆地，故二者也一致。
        """
        m, n = len(grid), len(grid[0])
        count = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    # self.nums_islands_dfs(grid, i, j)  # 调用深度优先搜索方法
                    self.nums_islands_bfs(grid, i, j)  # 调用广度优先搜索方法
                    count += 1

        # 打印每一步遍历的输出变化情况便于观察
        for g in self.res:
            for r in g:
                print(r)
            print()
        return count

    """----------------------------------岛屿周长--------------------------------------------"""
    @staticmethod
    def perimeter_islands_iteration(grid, i, j):
        """ 岛屿的周长问题：迭代法，只需要判断每块陆地周围是否有水域或是否是边界"""
        if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] == 0:
            return 1
        else:
            return 0

    def perimeter_islands_bfs(self, grid, i, j):
        """ 岛屿的周长问题：广度优先搜索方法 """
        perimeter = 0
        queue = [(i, j)]
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        grid[i][j] = 2
        self.res.append([r[:] for r in grid])

        while queue:
            x, y = queue.pop(0)
            for di, dj in directions:
                nx, ny = x + di, y + dj
                if nx < 0 or nx >= len(grid) or ny < 0 or ny >= len(grid[0]) or grid[nx][ny] == 0:
                    perimeter += 1  # 超过边界或遇到水域周长加一
                elif grid[nx][ny] == 1:
                    queue.append((nx, ny))
                    grid[nx][ny] = 2
                    self.res.append([r[:] for r in grid])
                else:
                    perimeter += 0  # 再次遍历过程中遇到遍历过的陆地加0
        return perimeter

    def perimeter_islands_dfs(self, grid, i, j):
        """ 岛屿的周长问题：深度优先搜索方法 """
        if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] == 0:
            return 1
        elif grid[i][j] == 1:
            grid[i][j] = 2  # 对于访问过的陆地，标记为2
            self.res.append([r[:] for r in grid])
            return (self.perimeter_islands_dfs(grid, i + 1, j) + self.perimeter_islands_dfs(grid, i - 1, j) +
                    self.perimeter_islands_dfs(grid, i, j - 1) + self.perimeter_islands_dfs(grid, i, j + 1))
        else:
            return 0  # 遇到访问过的陆地时返回0

    def perimeter_islands(self, grid):
        """
        463.岛屿的周长
        时间复杂度：O(m × n)，因为要遍历区域内所有位置，两种方法一致。
        空间复杂度：O(m × n)，递归的深度最大可能是整个网格的大小，因此最大可能使用 O(m×n) 的栈空间；
                    同时广度优先搜索里，栈最大可能存储m×n块陆地，故二者也一致。
        """
        m, n = len(grid), len(grid[0])
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    # perimeter += self.perimeter_islands_iteration(grid, i + 1, j)
                    # perimeter += self.perimeter_islands_iteration(grid, i - 1, j)
                    # perimeter += self.perimeter_islands_iteration(grid, i, j - 1)
                    # perimeter += self.perimeter_islands_iteration(grid, i, j + 1)  # 调用迭代法

                    perimeter = self.perimeter_islands_dfs(grid, i, j)  # 调用深度优先搜索方法

                    # perimeter = self.perimeter_islands_bfs(grid, i, j)  # 调用广度优先搜索方法

                    for g in self.res:
                        for r in g:
                            print(r)
                        print()

                    return perimeter

    """----------------------------------岛屿最大面积--------------------------------------------"""
    def max_area_islands_dfs(self, grid, i, j):
        """ 岛屿的最大面积问题：深度优先搜索方法 """
        if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] == 0:
            return 0
        else:
            area = 1
            grid[i][j] = 0
            area += self.max_area_islands_dfs(grid, i - 1, j)
            area += self.max_area_islands_dfs(grid, i + 1, j)
            area += self.max_area_islands_dfs(grid, i, j - 1)
            area += self.max_area_islands_dfs(grid, i, j + 1)
        return area

    def max_area_islands_bfs(self, grid, i, j):
        """ 岛屿的最大面积问题：广度优先搜索方法 """
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        area = 1
        queue = [(i, j)]
        grid[i][j] = 2
        while queue:
            x, y = queue.pop(0)
            for di, dj in directions:
                nx, ny = x + di, y + dj
                if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == 1:
                    area += 1
                    queue.append((nx, ny))  # 每次遍历将每一块陆地四周的所有陆地都添加到队列当中
                    grid[nx][ny] = 2
                    self.res.append([r[:] for r in grid])

        return area

    def max_area_islands(self, grid):
        """
        695.岛屿的最大面积
        时间复杂度：O(m × n)，因为要遍历区域内所有位置，两种方法一致。
        空间复杂度：O(m × n)，递归的深度最大可能是整个网格的大小，因此最大可能使用 O(m×n) 的栈空间；
                    同时广度优先搜索里，栈最大可能存储m×n块陆地，故二者也一致。
        """
        m, n = len(grid), len(grid[0])
        max_area = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    # max_area = max(max_area, self.max_area_islands_dfs(grid, i, j))  # 调用深度优先搜索方法
                    max_area = max(max_area, self.max_area_islands_bfs(grid, i, j))  # 调用广度优先搜索方法
        return max_area


graph1 = [
  ["1", "1", "1", "1", "0"],
  ["1", "1", "0", "1", "0"],
  ["1", "1", "0", "0", "0"],
  ["0", "0", "0", "0", "0"]
]  # 输出1
graph2 = [[0, 1, 0, 0],
          [1, 1, 1, 0],
          [0, 1, 0, 0],
          [1, 1, 0, 0]]  # 输出16

graph3 = [[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
          [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
          [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]]  # 输出6

# for r in graph3:
#     print(f"[{', '.join(map(str, r))}],")

print(IslandProblem().nums_islands(graph1))
# print(IslandProblem().perimeter_islands(graph2))
# print(IslandProblem().max_area_islands(graph3))


