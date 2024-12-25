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
import collections


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

# print(IslandProblem().nums_islands(graph1))
# print(IslandProblem().perimeter_islands(graph2))
# print(IslandProblem().max_area_islands(graph3))


class AdjacencyListToGraph:
    """ 使用邻接表构造图 """
    def __init__(self, vertices):
        self.v = vertices  # 顶点数
        self.graph = {i: [] for i in range(vertices)}  # 初始化邻接表

    def add_edge(self, u, v, weight=1):
        self.graph[u].append((v, weight))  # 添加边u->v,权重为weight

    def remove_edge(self, u, v):
        self.graph[u] = [x for x in self.graph[u] if x[0] != v]  # 删除边u->v

    def display(self):
        for vertex, edges in self.graph.items():
            print(f'{vertex}: {edges}')


# ALG = AdjacencyListToGraph(3)
# ALG.add_edge(0, 1, 5)
# ALG.add_edge(0, 2, 10)
# ALG.add_edge(1, 3, 2)
# ALG.display()


def can_finish(num_courses, prerequisites):
    """ 207.课程表 """
    graph = collections.defaultdict(list)
    for u, v in prerequisites:
        graph[u].append(v)
    print('每个课程的先修课程统计->graph', graph)
    visited = [0] * num_courses  # 记录每个节点的访问情况,0-未访问,1-正在访问,2-已访问
    print('初始化所有节点的状态->visited:', visited)

    def dfs(i):
        # 如果当前课程正在访问则表明遇到了环,如课程1的先修课程是课程0,但课程0的先修课程是课程1
        if visited[i] == 1:
            print(f'此时课程 {i} 正在被访问,说明遇到了环,任务无法完成!')
            return False
        # 如果当前课程已访问,则表明当前课程已经是其它课程的先修课程了
        if visited[i] == 2:
            print(f'此时课程 {i} 已被访问过,说明遇到了另一条已完成搜索的路径,不必继续往下搜索了!')
            return True
        visited[i] = 1  # 将当前节点状态置为正在访问
        # 对当前节点的邻接节点开始深搜
        for j in graph[i]:
            print(f'开始对课程 {j} 的深搜, 搜索前的课程访问状态->visited: {visited}')
            if not dfs(j):
                return False
        visited[i] = 2  # 将当前节点的状态置为已访问
        return True

    for i in range(num_courses):
        # 对当前节点进行递归深搜,看能否顺利先修完当前课程的前序课程
        print(f'开始对课程 {i} 的深搜, 搜索前的课程访问状态->visited: {visited}')
        if not dfs(i):
            return False
    return True


# numCourses = 4
# prerequisites = [[0, 1], [1, 2], [0, 3], [3, 1]]
# print(can_finish(numCourses, prerequisites))


def find_order(num_courses, prerequisites):
    """ 210.课程表Ⅱ """
    graph = collections.defaultdict(list)
    in_degree = [0] * num_courses  # 记录每个课程节点的入度

    for i, j in prerequisites:
        graph[j].append(i)  # 有向边是从先修课程指向后修课程
        in_degree[i] += 1  # 入度+1,即后修课程节点入度增加

    print('每个课程的先修课程统计->graph', graph)
    print('初始时所有课程节点的入度->in_degree:', in_degree)

    # 先将入度为0的课程节点加入队列当中,即优先处理没有先修课程的课程节点
    queue = [i for i in range(num_courses) if in_degree[i] == 0]
    print('初始状态入度为0的课程节点: ', queue)
    res = []

    while queue:
        node = queue.pop(0)
        res.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
        print(f'当前处理节点: {node}, 当前结果: {res}, 当前各节点的入度: {in_degree}, 当前队列中节点: {queue}')

    if len(res) == num_courses:
        return res
    else:
        return []


# numCourses = 4
# prerequisite = [[0, 1], [1, 2], [0, 3], [3, 1]]
# print(find_order(numCourses, prerequisite))


class UnionFind:
    """ 并查集 """
    def __init__(self, n):
        self.parent = list(range(n))  # 初始化父节点数组，parent[i]表示i的父节点，初始时为其本身
        self.rank = [1] * n  # rank[i]表示以i为根的树的高度，初始为1，称之为元素的秩

    def find(self, x):
        """ 查找操作 """
        if self.parent[x] != x:  # 如果x的父节点不是自己，递归查找
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """ 合并操作 """
        # 首先寻找两个元素的父节点
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:  # x,y 不属于同一集合时才合并
            # 通过比较两个元素的秩来将秩较小的树合并到秩较大的树上面
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            elif self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1

    def find_circle_num(self, is_connected):
        """ 547.省份数量 """
        n = len(is_connected)
        for i in range(n):
            for j in range(i + 1, n):
                if is_connected[i][j] == 1:
                    self.union(i, j)  # 进行合并操作

        print(self.parent)
        provinces = sum(self.parent[i] == i for i in range(n))  # 统计有多少个根节点，就说明有多少个省份
        return provinces


# is_connect = [[1, 1, 0], [1, 1, 0], [0, 0, 1]]
# print(UnionFind(len(is_connect)).find_circle_num(is_connect))

def find_redundant_connection(edges):
    """ 684.冗余连接 """
    n = len(edges)
    parent = list(range(n + 1))
    rank = [1] * (n + 1)

    def find(t):
        if parent[t] == t:
            return t
        else:
            parent[t] = find(parent[t])
        return parent[t]

    def union(i, j):
        parent[find(i)] = find(j)

    for x, y in edges:
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            if rank[root_x] < rank[root_y]:
                union(root_y, root_x)
            elif rank[root_x] > rank[root_y]:
                union(root_x, root_y)
            else:
                union(root_x, root_y)
                rank[root_x] += 1
            print(x, y, parent, rank)
        else:
            return [x, y]


# edge = [[1, 2], [1, 3], [2, 3]]
# print(find_redundant_connection(edge))


def find_redundant_directed_connection(edges):
    """ 685.冗余连接Ⅱ """
    n = len(edges)
    uf = UnionFind(n + 1)


# edge = [[1, 2], [2, 3], [3, 4], [4, 1], [1, 5]]
# print(find_redundant_directed_connection(edge))


""" A*算法 """
import heapq

# 方向向量（上、下、左、右）
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def heuristic(a, b):
    # 曼哈顿距离作为启发式函数
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star_search(grid, start, goal):
    # grid: 二维列表，表示网格，1 表示可通行，0 表示障碍物
    # start: 起点坐标 (x, y)
    # goal: 目标点坐标 (x, y)

    # 使用堆实现优先队列，存储 (f, g, (x, y))，f = g + h
    open_list = []
    heapq.heappush(open_list, (0 + heuristic(start, goal), 0, start))

    # 用来记录路径的字典，记录每个点的父节点
    came_from = {}

    # g 值字典，记录每个点的从起点到当前点的代价
    g_costs = {start: 0}

    while open_list:
        # 获取 f 值最小的节点
        f, g, current = heapq.heappop(open_list)

        # 如果到达目标，返回路径
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()  # 从起点到目标点的路径
            return path

        # 检查相邻的节点
        for direction in DIRECTIONS:
            neighbor = (current[0] + direction[0], current[1] + direction[1])

            # 确保邻居在网格内且是可通行的
            if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]) and grid[neighbor[0]][neighbor[1]] == 1:
                # 计算到达邻居的代价
                tentative_g = g_costs[current] + 1

                # 如果邻居还没有被访问过，或者发现更短的路径
                if neighbor not in g_costs or tentative_g < g_costs[neighbor]:
                    g_costs[neighbor] = tentative_g
                    f_cost = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_cost, tentative_g, neighbor))
                    came_from[neighbor] = current

    return None  # 如果没有找到路径，返回 None


# 测试

# 1 代表可通行，0 代表障碍物
grid = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
    [1, 0, 0, 1, 1],
    [1, 1, 1, 1, 1]
]

start = (0, 0)  # 起点
goal = (4, 4)  # 目标点

path = a_star_search(grid, start, goal)

if path:
    print("路径：", path)
else:
    print("没有找到路径")
