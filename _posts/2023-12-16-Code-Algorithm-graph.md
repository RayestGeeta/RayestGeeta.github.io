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

