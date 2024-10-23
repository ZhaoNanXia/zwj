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

    def nums_islands_dfs(self, grid, i, j):
        """ 岛屿数量问题：深度优先搜索方法 """
        if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] == '0':
            return

        grid[i][j] = '0'
        self.res.append([r[:] for r in grid])
        self.nums_islands_dfs(grid, i + 1, j)
        self.nums_islands_dfs(grid, i - 1, j)
        self.nums_islands_dfs(grid, i, j - 1)
        self.nums_islands_dfs(grid, i, j + 1)

    def nums_islands_bfs(self, grid, i, j):
        """ 岛屿数量问题：广度优先搜索方法 """
        if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] == '0':
            return
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        grid[i][j] = '0'
        self.res.append([r[:] for r in grid])  # 这句代码只为保存结果，验证使用
        for di, dj in directions:
            self.nums_islands_dfs(grid, i + di, j + dj)

    def nums_islands(self, grid):
        """ 200.岛屿数量"""
        m, n = len(grid), len(grid[0])
        count = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    # self.nums_islands_dfs(grid, i, j)  # 调用深度优先搜索方法
                    self.nums_islands_bfs(grid, i, j)  # 调用广度优先搜索方法
                    count += 1
        # 打印每一步遍历的输出变化情况
        for g in self.res:
            for r in g:
                print(' '.join(r))
            print()
        return count

    @staticmethod
    def perimeter_islands_iteration(grid, i, j):
        """ 岛屿的周长问题：迭代法，只需要判断每块陆地周围是否有水域或是否是边界"""
        if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] == 0:
            return 1
        else:
            return 0

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
        """ 463.岛屿的周长 """
        m, n = len(grid), len(grid[0])
        perimeter = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    # perimeter += self.perimeter_islands_iteration(grid, i + 1, j)
                    # perimeter += self.perimeter_islands_iteration(grid, i - 1, j)
                    # perimeter += self.perimeter_islands_iteration(grid, i, j - 1)
                    # perimeter += self.perimeter_islands_iteration(grid, i, j + 1)  # 调用迭代法

                    perimeter += self.perimeter_islands_dfs(grid, i, j)  # 调用深度优先搜索方法

                    for g in self.res:
                        for r in g:
                            print(r)
                        print()

                    return perimeter

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
        if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] == 0:
            return 0
        else:
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            area = 1
            grid[i][j] = 0
            for di, dj in directions:
                area += self.max_area_islands_bfs(grid, i + di, j + dj)
        return area

    def max_area_islands(self, grid):
        """ 695.岛屿的最大面积 """
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
]

graph2 = [[0, 1, 0, 0],
          [1, 1, 1, 0],
          [0, 1, 0, 0],
          [1, 1, 0, 0]]

graph3 = [[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
          [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
          [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]]

# for r in graph3:
#     print(f"[{', '.join(map(str, r))}],")

# print(IslandProblem().nums_islands(graph1))
# print(IslandProblem().perimeter_islands(graph2))
print(IslandProblem().max_area_islands(graph3))
