---
title: Algorithm-part6-图
author: Rayest
date: 2023-12-16 21:42:00 +0800
categories: [interview]
tags: [fuck-code]
math: true
---

## 所有可能的路径

解法：图的遍历。记录遍历的路径，当到达目标位置时，保存到结果中。

```python
class Solution(object):
    def allPathsSourceTarget(self, graph):
        """
        :type graph: List[List[int]]
        :rtype: List[List[int]]
        """

        self.res = []
        n = len(graph)
        path = []
        def traverse(graph, i):

            
            path.append(i)

            if i == n-1:
                self.res.append(list(path))
                path.pop()
                return None

            for j in graph[i]:
                traverse(graph, j)

            path.pop()

            return None

        traverse(graph, 0)
        return self.res
```

## 搜索名人

名人：其他人都认识他，他不认识其他人。

解法：假设第一个是名人，遍历其他人，如果发现有人认识这个人，或者这个人认识其他人。就把其他人当成名人。因为名人只有一个，最后只剩下一个人，再判断这个人是不是认识其他人和其他人都认不认识他。

```python
class Solution(object):
    def findCelebrity(self, n):
        """
        :type n: int
        :rtype: int
        """

        
        only_one = 0

        for other in range(1, n):
            if knows(only_one, other) or not knows(other, only_one):
                only_one = other
            else:
                pass

        for other in range(n):
            if only_one == other:
                continue

            else:
                if knows(only_one, other) or not knows(other, only_one):
                    return -1

        return only_one
```

## 课程表

你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses - 1 。
在选修某些课程之前需要一些先修课程。 先修课程按数组 prerequisites 给出，其中 prerequisites[i] = [ai, bi] ，表示如果要学习课程 ai 则 必须 先学习课程  bi 。
例如，先修课程对 [0, 1] 表示：想要学习课程 0 ，你需要先完成课程 1 。
请你判断是否可能完成所有课程的学习？如果可以，返回 true ；否则，返回 false

解法：

1. DFS:判断图是否有环。先构建图，然后遍历图，如果遍历的路径上存在重复节点，则有环。

2. BFS:计算每个节点的入度。先遍历入度为0的节点，遍历到某个节点就把该节点入度-1，并把新的入度放入队列。最后遍历过的节点就是全部节点说明没有环。（假设图有环，那么必然环内的所有节点入度都不为0）

```python
# DFS
class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """

        if not prerequisites:
            return True

        def build_graph(numCourses, prerequisites):
            graph = [[] for i in range(numCourses)]

            for pre in prerequisites:
                a1, b1 = pre[0], pre[1]
                graph[a1].append(b1)

            return graph

        graph = build_graph(numCourses, prerequisites)

        self.visited = []
        self.path = []
        self.flag = True

        def traverse(graph, i):
      
            if i in self.path:
                self.flag = False

            if i in self.visited or not self.flag:        
                return None

            self.visited.append(i)
            self.path.append(i)

            for pre in graph[i]:
                traverse(graph, pre)

            self.path.pop()
            return None

        for i in range(0, len(graph)):
            if self.flag:
                traverse(graph, i)

        return self.flag
    
    
# BFS：
class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """

        graph = [[] for i in range(numCourses)]
        indegree = [0 for _ in range(numCourses)]

        for pre in prerequisites:
            graph[pre[0]].append(pre[1])

            indegree[pre[1]] += 1


        quene = []

        for i in range(numCourses):
            if indegree[i] == 0:
                quene.append(i)

        count = 0
        while quene:
            node = quene.pop(0)
            count += 1

            for i in graph[node]:
                indegree[i] -= 1

                if indegree[i] == 0:
                    quene.append(i)

        if count == numCourses:
            return True

        else:
            return False
```

## 课程表 II

现在你总共有 numCourses 门课需要选，记为 0 到 numCourses - 1。给你一个数组 prerequisites ，其中 prerequisites[i] = [ai, bi] ，表示在选修课程 ai 前 必须 先选修 bi 。
例如，想要学习课程 0 ，你需要先完成课程 1 ，我们用一个匹配来表示：[0,1] 。
返回你为了学完所有课程所安排的学习顺序。可能会有多个正确的顺序，你只要返回 任意一种 就可以了。如果不可能完成所有课程，返回 一个空数组 。

解法：图遍历判断是否有环。同时在后序位置保留输出，对比多叉树，先学完所有子节点，才能学自己。

```python
class Solution(object):
    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """

        graph = [[] for _ in range(numCourses)]

        for pre in prerequisites:
            graph[pre[0]].append(pre[1])

        
        self.visited = []
        self.path = []
        self.postorder = []
        self.flag = True

        def traverse(graph, i):
            if i in self.path:
                self.flag = False

            if i in self.visited or not self.flag:
                return None

            self.visited.append(i)
            self.path.append(i)

            for pre in graph[i]:
                traverse(graph, pre)

            self.postorder.append(self.path.pop())

            return None

        for i in range(numCourses):
            if self.flag:
                traverse(graph, i)

        if self.flag:
            return self.postorder

        return []
    
    
# BFS

class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """

        graph = [[] for i in range(numCourses)]
        indegree = [0 for _ in range(numCourses)]

        for pre in prerequisites:
            graph[pre[1]].append(pre[0])

            indegree[pre[0]] += 1


        quene = []

        for i in range(numCourses):
            if indegree[i] == 0:
                quene.append(i)

        count = 0
        while quene:
            node = quene.pop(0)
            count += 1

            for i in graph[node]:
                indegree[i] -= 1

                if indegree[i] == 0:
                    quene.append(i)

        if count == numCourses:
            return True

        else:
            return False


            
```

## 判断二分图

解法：即二色图，任意两个节点要用不同的颜色。遍历图，要让所有相邻节点和自己颜色不同，如果做不到就不是二分图。

```python
class Solution(object):
    def isBipartite(self, graph):
        """
        :type graph: List[List[int]]
        :rtype: bool
        """

        self.is_bipartitie = True
        self.color = [True for _ in range(len(graph))]
        self.visited = []
        def traverse(graph, i):
            if i in self.visited:
                return

            self.visited.append(i)
            for other in graph[i]:
                
                if other not in self.visited:
                    self.color[other] = not self.color[i]

                else:
                    if self.color[other] == self.color[i]:
                        self.is_bipartitie = False
                traverse(graph, other)

        for i in range(len(graph)):
            if not self.is_bipartitie:
                return self.is_bipartitie
            if i not in self.visited:
                traverse(graph, i)

            
        return self.is_bipartitie
```

## 可能的二分法

给定一组 n 人（编号为 1, 2, ..., n）， 我们想把每个人分进任意大小的两组。每个人都可能不喜欢其他人，那么他们不应该属于同一组。
给定整数 n 和数组 dislikes ，其中 dislikes[i] = [ai, bi] ，表示不允许将编号为 ai 和  bi的人归入同一组。当可以用这种方法将所有人分进两组时，返回 true；否则返回 false。

解法：判断是否为二分图。将dislikes构建成图(注意是无向图)，然后进行着色判断，遍历图，要让所有相邻节点和自己颜色不同，如果做不到就不是二分图。

```python
class Solution(object):
    def possibleBipartition(self, n, dislikes):
        """
        :type n: int
        :type dislikes: List[List[int]]
        :rtype: bool
        """

        self.color = [True for _ in range(n)]
        self.visited = []
        self.is_possible = True

        graph = [[] for i in range(n)]
        for dis in dislikes:
            graph[dis[0]-1].append(dis[1]-1)
            graph[dis[1]-1].append(dis[0]-1)


        def traverse(graph, i):
            if i in self.visited:
                return

            self.visited.append(i)
            for other in graph[i]:
                if other not in self.visited:
                    self.color[other] = not self.color[i]

                else:
                    if self.color[other] == self.color[i]:
                        self.is_possible = False
                        break
                traverse(graph, other)

        for i in range(len(graph)):
            if not self.is_possible:
                return self.is_possible

            if  i not in self.visited:
                traverse(graph, i)

        return self.is_possible

```

## 无向图中连通分量的数目

你有一个包含 n 个节点的图。给定一个整数 n 和一个数组 edges ，其中 edges[i] = [ai, bi] 表示图中 ai 和 bi 之间有一条边。

返回 图中已连接分量的数目 。

解法：构建并查集(union-find)

```python
class Solution(object):
    def countComponents(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: int
        """

        class UF:

            def __init__(self, n):
                self.count = n
                self.parent = [i for i in range(n)]

            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])

                return self.parent[x]

            def union(self, p, q):
                p_parent = self.find(p)
                q_parent = self.find(q)

                if p_parent == q_parent:
                    return

                self.parent[p_parent] = q_parent
                self.count -= 1

            # def connected(self, p, q):
            #     p_parent = self.find(p)
            #     q_parent = self.find(q)

            #     return p_parent == q_parent

        uf = UF(n)
        for edge in edges:
            uf.union(edge[0], edge[1])

        return uf.count
```

## 以图判树

给定编号从 0 到 n - 1 的 n 个结点。给定一个整数 n 和一个 edges 列表，其中 edges[i] = [ai, bi] 表示图中节点 ai 和 bi 之间存在一条无向边。
如果这些边能够形成一个合法有效的树结构，则返回 true ，否则返回 false 。

解法：并查集。判断是否有环或有多个连通变量存在。有环的情况就是两个节点已经是同一个连通变量了，继续连接就会出现环。

```python
class Solution(object):
    def validTree(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: bool
        """

        class UF:

            def __init__(self, n):
                self.count = n
                self.parent = [i for i in range(n)]

            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])

                return self.parent[x]

            def union(self, p, q):
                root_p = self.find(p)
                root_q = self.find(q)

                if root_p == root_q:
                    return
                self.count -= 1
                self.parent[root_p] = root_q

            def connected(self, p, q):
                root_p = self.find(p)
                root_q = self.find(q)

                return root_p == root_q


        uf = UF(n)
        for edge in edges:
            p, q = edge[0], edge[1]
            if uf.connected(p, q):
                return False

            uf.union(p, q)

        if uf.count > 1:
            return False
        return True
```

## 最低成本联通所有城市

想象一下你是个城市基建规划者，地图上有 n 座城市，它们按以 1 到 n 的次序编号。
给你整数 n 和一个数组 conections，其中 connections[i] = [xi, yi, costi] 表示将城市 xi 和城市 yi 连接所要的costi（连接是双向的）。
返回连接所有城市的最低成本，每对城市之间至少有一条路径。如果无法连接所有 n 个城市，返回 -1
该 最小成本 应该是所用全部连接成本的总和。

解法：

1. kruskal最小树算法。利用并查集和贪心算法，优先连接权值最小的节点。同时判断两个节点是否联通(就不必添加新节点)，已经最后联通变量不能超过1(是否所有节点都被连接)。
2. prim算法。以一个节点开始遍历，把这个节点相关的边放入堆队列中，然后弹出边权值最小的下一个节点。继续把下个节点相关的边放入堆队列中，再弹出权值最小的下个节点。直到队列为空为止。记得构建visited数组，防止重复节点结算。

```python
class Solution(object):
    def minimumCost(self, n, connections):
        """
        :type n: int
        :type connections: List[List[int]]
        :rtype: int
        """

        class UF:

            def __init__(self, n):
                self.count = n
                self.parent = [i for i in range(n)]

            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])

                return self.parent[x]

            def union(self, p, q):
                root_p = self.find(p)
                root_q = self.find(q)

                if root_p == root_q:
                    return

                self.count -= 1
                self.parent[root_p] = root_q

            def connected(self, p, q):
                root_p = self.find(p)
                root_q = self.find(q)

                return root_p == root_q

        connections.sort(key =lambda x:x[-1])

        uf = UF(n+1)

        all_weight = 0

        for connection in connections:
            x1 = connection[0]
            y1 = connection[1]
            weight = connection[2]

            if uf.connected(x1, y1):
                continue

            uf.union(x1, y1)
            all_weight += weight

        if uf.count > 2:
            return -1

        return all_weight
    
    
# prim算法

class Solution(object):
    def minimumCost(self, n, connections):
        """
        :type n: int
        :type connections: List[List[int]]
        :rtype: int
        """

        # prim算法

        import heapq

        self.graph = [[] for _ in range(n)]

        for connection in connections:
            self.graph[connection[0]-1].append([connection[2], connection[1]-1])
            self.graph[connection[1]-1].append([connection[2], connection[0]-1])

        self.visited = []
        self.pq = [(0, 0)]
        self.weight = 0

        while self.pq:
            edge = heapq.heappop(self.pq)

            if edge[1] not in self.visited:
                self.visited.append(edge[1])
                self.weight += edge[0]

                for other_edge in self.graph[edge[1]]:
                    if other_edge[1] not in self.visited:
                        heapq.heappush(self.pq, other_edge)

        if len(self.visited) == n:
            return self.weight
        return -1
```

## 连接所有点的最小费用

给你一个points 数组，表示 2D 平面上的一些点，其中 points[i] = [xi, yi] 。
连接点 [xi, yi] 和点 [xj, yj] 的费用为它们之间的 曼哈顿距离 ：|xi - xj| + |yi - yj| ，其中 |val| 表示 val 的绝对值。
请你返回将所有点连接的最小总费用。只有任意两点之间 有且仅有 一条简单路径时，才认为所有点都已连接。

解法：

1. kruskal最小生成树算法。构建并查集，然后贪心算法从距离最小的两个点进行构建图。
2. prim算法。构建邻接表。从第一个节点开始，把到其他点的边放入到堆队列中。先弹出最小权值的节点，再把这个节点相关的边放入到堆队列中。重复上面操作。

```python
class Solution(object):
    def minCostConnectPoints(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """

        class UF:

            def __init__(self, n):
                self.count = n
                self.parent = [i for i in range(n)]

            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])

                return self.parent[x]

            def union(self, p, q):
                root_p = self.find(p)
                root_q = self.find(q)

                if root_p == root_q:
                    return

                self.count -= 1
                self.parent[root_p] = root_q
                return

            def connected(self, p, q):
                return self.find(p) == self.find(q)

        n = len(points)
        uf = UF(n)

        graph = []

        for i in range(n):
            for j in range(i+1, n):
                weight = abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])
                graph.append([i, j, weight])

        graph.sort(key = lambda x:x[-1])
        
        costs = 0
        for edge in graph:
            p = edge[0]
            q = edge[1]
            weight = edge[2]

            if uf.connected(p, q):
                continue

            uf.union(p, q)
            costs += weight

        return costs
    
# prim
class Solution(object):
    def minCostConnectPoints(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """
        import heapq

        n = len(points)
        self.graph = [[] for i in range(n)]

        for index in range(n):
            point = points[index]
            for jndex in range(index+1, n):
                point2 = points[jndex]
                distance = abs(point[1] - point2[1]) + abs(point[0] - point2[0])
                self.graph[index].append([distance, jndex])
                self.graph[jndex].append([distance, index])

        self.pq = [(0, 0)]
        self.cost = 0
        self.visited = []

        while self.pq:
            node = heapq.heappop(self.pq)
            if node[1] not in self.visited:
                self.visited.append(node[1])
                self.cost += node[0]

                for other_node in self.graph[node[1]]:
                    if other_node[1] not in self.visited:
                        heapq.heappush(self.pq, other_node)

                if len(self.visited) == n:
                    return self.cost 
```

## 网络延迟时间

有 n 个网络节点，标记为 1 到 n。
给你一个列表 times，表示信号经过 有向 边的传递时间。 times[i] = (ui, vi, wi)，其中 ui 是源节点，vi 是目标节点， wi 是一个信号从源节点传递到目标节点的时间。
现在，从某个节点 K 发出一个信号。需要多久才能使所有节点都收到信号？如果不能使所有节点收到信号，返回 -1 。

解法：标准的dijkstra找最小路径算法。抽象成找出发点到每个节点的最段时间，然后最长的时间就是我们要的结果。dijkstra算法流程，维护一个数组代表的是出发点到每个节点的最短距离。维护一个优先级队列，按照到出发点的距离进行排序，优先弹出离出发点的节点。算法开始时，将出发点放入队列中，去遍历相邻节点，计算相邻节点到出发点的距离(cur_start_dist + cur_other_dist)，如果比维护的结果要低，就去更新结果，然后把该相邻节点放入到队列中(相当于找到了新的最短路径)。直到队列为空。

```python
class Solution(object):
    def networkDelayTime(self, times, n, k):
        """
        :type times: List[List[int]]
        :type n: int
        :type k: int
        :rtype: int
        """

        from heapq import heappush, heappop

        graph = [[] for _ in range(n)]

        for time in times:
            graph[time[0]-1].append([time[1]-1, time[2]])

        cost_times = [float('inf') for _ in range(n)]
        cost_times[k-1] = 0
        pq = [(0, k-1)]

        while pq:
            node = heappop(pq)
            dist_start = node[0]
            cur_node = node[1]

            if dist_start > cost_times[cur_node]:
                continue

            for other in graph[cur_node]:
                other_node = other[0]
                dist_cur_other = other[1]

                dist_start_other = dist_cur_other + cost_times[cur_node]

                if dist_start_other < cost_times[other_node]:
                    cost_times[other_node] = dist_start_other
                    heappush(pq, [cost_times[other_node], other_node])

        res = max(cost_times)
        if res == float('inf'):
            return -1

        return res
```

## 最小体力消耗路径

你准备参加一场远足活动。给你一个二维 rows x columns 的地图 heights ，其中 heights(row)(col) 表示格子 (row, col) 的高度。一开始你在最左上角的格子 (0, 0) ，且你希望去最右下角的格子 (rows-1, columns-1) （注意下标从 0 开始编号）。你每次可以往 上，下，左，右 四个方向之一移动，你想要找到耗费 体力 最小的一条路径。
一条路径耗费的 体力值 是路径上相邻格子之间 高度差绝对值 的 最大值 决定的。
请你返回从左上角走到右下角的最小 体力消耗值 。

解法：dijkstra最短路径算法。注意构建图的过程，相邻节点是上下左右。然后注意计算路径权值，是按照相邻节点差值的最大值来作为整个路径权值的。要修改下距离更新代码。

```python
class Solution(object):
    def minimumEffortPath(self, heights):
        """
        :type heights: List[List[int]]
        :rtype: int
        """

        from heapq import heappush, heappop

        row = len(heights)
        col = len(heights[0])
        n = row * col

        graph = [[] for i in range(n)]

        for i in range(row):
            for j in range(col):

                if j - 1 >= 0:
                    graph[i*col+j].append([i*col+j-1, abs(heights[i][j] - heights[i][j-1])])
                if j + 1 < col:
                    graph[i*col+j].append([i*col+j+1, abs(heights[i][j] - heights[i][j+1])])
                if i - 1 >= 0:
                    graph[i*col+j].append([i*col+j-col, abs(heights[i][j] - heights[i-1][j])])
                if i + 1 < row:
                    graph[i*col+j].append([i*col+j+col, abs(heights[i][j] - heights[i+1][j])])

        pq = [(0, 0)]
        costs_min = [float('inf') for _ in range(n)]
        costs_min[0] = 0

        while pq:
            node = heappop(pq)
            cur_node = node[1]
            cur_start_cost = node[0]

            if cur_node == n - 1:
                return cur_start_cost

            if cur_start_cost > costs_min[cur_node]:
                continue

             
            for other in graph[cur_node]:
                other_node = other[0]
                other_cur_cost = other[1]

                other_start_cost = max(other_cur_cost, costs_min[cur_node])

                if other_start_cost < costs_min[other_node]:
                    costs_min[other_node] = other_start_cost
                    heappush(pq, (costs_min[other_node], other_node))

        return costs_min[n-1]
        
```

## 概率最大的路径

给你一个由 n 个节点（下标从 0 开始）组成的无向加权图，该图由一个描述边的列表组成，其中 edges[i] = [a, b] 表示连接节点 a 和 b 的一条无向边，且该边遍历成功的概率为 succProb[i] 。
指定两个节点分别作为起点 start 和终点 end ，请你找出从起点到终点成功概率最大的路径，并返回其成功概率。

解法：dijkstra找”最大路径“算法。注意是无向图(双向图)。注意最值问题，找最大值的话，要注意维护的堆队列排序，还有就是更新值之前的比大小问题。

```python
class Solution(object):
    def maxProbability(self, n, edges, succProb, start_node, end_node):
        """
        :type n: int
        :type edges: List[List[int]]
        :type succProb: List[float]
        :type start_node: int
        :type end_node: int
        :rtype: float
        """

        from heapq import heappush, heappop

        graph = [[] for _ in range(n)]

        for index in range(len(edges)):
            graph[edges[index][0]].append([edges[index][1], succProb[index]])
            graph[edges[index][1]].append([edges[index][0], succProb[index]])

        pq = [(-1, start_node)]
        max_prob = [-1 * float('inf') for _ in range(n)]
        max_prob[start_node] = 1

        while pq:
            node = heappop(pq)
            cur_node = node[1]
            cur_start_prob = -1 * node[0]

            if cur_node == end_node:
                return cur_start_prob

            if max_prob[cur_node] > cur_start_prob:
                continue

            for other in graph[cur_node]:
                other_node = other[0]
                other_cur_prob = other[1]

                other_start_prob = other_cur_prob * max_prob[cur_node]

                if other_start_prob > max_prob[other_node]:
                    max_prob[other_node] = other_start_prob
                    heappush(pq, [-1 * max_prob[other_node], other_node])

        return 0
```
