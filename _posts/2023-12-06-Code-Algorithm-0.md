---
title: Algorithm-part0-梳理框架
author: Rayest
date: 2023-12-06 19:04:00 +0800
categories: [interview]
tags: [fuck-code]
math: true
---


## 刷题的框架思维

### 数据结构的存储方式

数据结构的存储方式只有两种：__数组（顺序存储）和链表（链式存储）。__

- 数据: 紧凑连续存储，可以随机访问，可快速索引，相对节约存储。但因为连续存储，内存要一次性分配够，后续扩容分配更大空间，需要复制数据，时间复杂度O(N)。进行插入和删除，复杂度也是O(N)。
- 链表: 元素不连续，靠指针指向下一个元素位置，不存在扩容问题，不能随机访问。过前驱和后驱进行删除和插入，复杂度为O(1)。但需要额外存储指针，相对消耗更多一些空间。

### 数据结构的基础操作

对于任何数据结构，其基本操作无非遍历 + 访问，再具体一点就是：增删查改。

- 遍历：
  
  - 数组(线性):
  
    ```python
    def traverse(array):
        for arr in array:
            print(arr)
            
    traverse([1,4,2,3])
    ```

  - 链表:
  
    ```python
    class ListNode:
        def __init__(self, val):
            self.val = val
            self.next = None
            
    def buildListNodes():
        node0 = ListNode(1)
        node1 = ListNode(4)
        node0.next = node1
        node2 = ListNode(2)
        node1.next = node2
        node3 = ListNode(3)
        node2.next = node3
        return node0
            
    def traverse1(listNode):
        while listNode is not None:
            print(listNode.val)
            listNode = listNode.next
            
    def traverse2(listNode):
        if listNode is not None:
            print(listNode.val)
            traverse2(listNode.next)
        
    node0 = buildListNodes()
    traverse1(node0)

    print()
    print()

    node0 = buildListNodes()
    traverse2(node0) 
    ```

- 二叉树(非线性):

```python
class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
def buildTreeNodes():
    node5 = TreeNode(5)
    node6 = TreeNode(6)    
    node3 = TreeNode(3)
    node4 = TreeNode(4)
    node2 = TreeNode(2, node5, node6)
    node1 = TreeNode(1, node3, node4)
    node0 = TreeNode(0, node1, node2)
    return node0

def traverse(treeNode):
    
    if treeNode.left is not None:
        traverse(treeNode.left)
        print(treeNode.left.val)
    if treeNode.right is not None:
        traverse(treeNode.right)
        print(treeNode.right.val)

        
traverse(buildTreeNodes())
```
