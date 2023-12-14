---
title: Algorithm-part5-二叉树
author: Rayest
date: 2023-12-09 22:57:00 +0800
categories: [interview]
tags: [fuck-code]
math: true
---

## 二叉树的最大深度

解法1：前序位置depth+1，后序位置depth-1。当到最大深度时，更新结果。

解法2：分别计算左子树和右子树深度，然后后序位置返回结果depth + 1

```python
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        if not root:
            return 0

        max_left = self.maxDepth(root.left)
        max_right = self.maxDepth(root.right)

        
        return max(max_left, max_right) + 1

```

## 二叉树的前序遍历

解法：emm，递归

```python
class Solution(object):
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        
        res = []

        if not root:
            return []

        res.append(root.val)
        res.extend(self.preorderTraversal(root.left))
        res.extend(self.preorderTraversal(root.right))

        return res
```

## 二叉树的直径

解法：每个节点的最大直径就是左右子树深度之和。在后序位置进行更新结果，且递归返回值应该是max(左子树深度，右子树深度) + 1

```python
class Solution(object):

    
    def diameterOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        self.dia = 0

        def depth(root):

            if not root:
                return 0
            
            left = depth(root.left)
            right = depth(root.right)
            self.dia = max(left + right, self.dia)
        
            return 1 + max(left, right)

        depth(root)
        return self.dia
```

## 翻转二叉树

解法：遍历二叉树。在前序位置，直接调换左右节点即可。

```python
class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        
        def traverse(root):

            if not root:
                return

            root.left, root.right = root.right, root.left

            traverse(root.left)
            traverse(root.right)

            return

        traverse(root)

        return root
```

## 填充每个节点的下一个右侧节点指针

解法：抽象成三叉树。遍历改成 f(node1.left, node1.right)、f(node1.right, node2.left)、f(node2.left, node2.right)

```python
class Solution(object):
    def connect(self, root):
        """
        :type root: Node
        :rtype: Node
        """

        if not root:
            return
            
        def traverse(node1, node2):
            if not node1 and not node2:
                return 

            node1.next = node2
            traverse(node1.left, node1.right)
            traverse(node1.right, node2.left)
            traverse(node2.left, node2.right)
            return 

        traverse(root.left, root.right)

        return root
```

## 二叉树展开为链表

解法：遍历二叉树。在后序位置，将left指针赋值None，right指针指向左节点，然后再将右节点拼接到左节点后面(注意，不能直接拼，要遍历到末尾结点）

```python
class Solution(object):
    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: None Do not return anything, modify root in-place instead.
        """

        def traverse(root):

            if not root:
                return

            traverse(root.left)
            traverse(root.right)

            left = root.left
            right = root.right
            
            root.left = None
            root.right = left

            tmp = root
            while tmp.right:
                tmp = tmp.right

            tmp.right = right
        
            return

        if not root:
            return 

        traverse(root)
        return root
```

## 最大二叉树

解法：先找到最大值，构建root结点。将num切分为两块，左边进入递归作为左子树，右边进入递归作为右子树。

```python
class Solution(object):
    def constructMaximumBinaryTree(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        
        if len(nums) == 0:
            return

        max_num = max(nums)
        index = nums.index(max_num)

        root = TreeNode(val = max_num)
        root.left = self.constructMaximumBinaryTree(nums[:index])
        root.right = self.constructMaximumBinaryTree(nums[index+1:])

        return root
```

## 从前序与中序遍历序列构造二叉树

解法：前序[0]、后序.index(前序[0])为root结点。后序[:后序.index(前序[0])]、前序[1: 后序.index(前序[0])+1] 是左子树。后序[后序.index(前序[0]) + 1:]、前序[后序.index(前序[0])+1:] 是右子树。

```python
class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """

        if not preorder or not inorder:
            return None

        root_val = preorder[0]
        root = TreeNode(root_val)

        index_in = inorder.index(root_val)

        root.left = self.buildTree(preorder[1:index_in+1] , inorder[:index_in])
        root.right = self.buildTree(preorder[index_in+1:], inorder[index_in+1:])

        return root
```

## 从中序与后序遍历序列构造二叉树

解法：

1. root：postorder[-1]、inorder[inorder.index(rootval)]
2. left: inorder[:in_index]、postorder[:in_index]
3. right: inorder[in_index+1:]、postorder[in_index:-1]

```python
class Solution(object):
    def buildTree(self, inorder, postorder):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: TreeNode
        """
        
        if not inorder and not postorder:
            return None

        rootVal = postorder[-1]
        root = TreeNode(rootVal)

        in_index = inorder.index(rootVal)
        root.left = self.buildTree(inorder[:in_index], postorder[:in_index])
        root.right = self.buildTree(inorder[in_index+1:], postorder[in_index:-1])

        return root
```

## 前序后序构建二叉树

解法：

1. root:preorder[0]、postorder[-1]
2. right:postorder[-2]是rightNode，通过索引preorder，可以找到右子树的开始位置right_index。preorder[right_index:]，postorder[right_index-1:-1]。前提是postorder[-2]是rightNode，有可能右子树是空。
3. left:preorder[1: right_index]、postorder[:right_index-1]

```python
class Solution(object):
    def constructFromPrePost(self, preorder, postorder):
        """
        :type preorder: List[int]
        :type postorder: List[int]
        :rtype: TreeNode
        """
        
        if not preorder or not postorder:
            return None

        
        rootVal = preorder[0]
        root = TreeNode(rootVal)

        if len(postorder) < 2:
            return root
            
        rightNode = postorder[-2]
        rightIndex = preorder.index(rightNode)
        leftNums = rightIndex - 1

        root.left = self.constructFromPrePost(preorder[1:leftNums+1], postorder[:leftNums])
        root.right = self.constructFromPrePost(preorder[rightIndex:], postorder[leftNums:-1])

        return root
```

## 寻找重复的子树

给你一棵二叉树的根节点 root ，返回所有 重复的子树 。
对于同一类的重复子树，你只需要返回其中任意 一棵 的根结点即可。
如果两棵树具有 相同的结构 和 相同的结点值 ，则认为二者是 重复 的。

解法：遍历二叉树。在后序位置，使用后序遍历序列化树进行标识。外部使用字典存储标识，如果出现多次保存到结果中。

```python
class Solution(object):
    def findDuplicateSubtrees(self, root):
        """
        :type root: TreeNode
        :rtype: List[TreeNode]
        """
        
        self.dicts = {}
        self.res = []
        def serialize(root):

            if not root:
                return "#"

            left = serialize(root.left)
            right = serialize(root.right)

            serialize_res = left + ',' + right + ',' + str(root.val)
            
            if serialize_res in self.dicts:
                if self.dicts[serialize_res] == 1:
                    self.res.append(root)
                self.dicts[serialize_res] += 1
            
            else:
                self.dicts[serialize_res] = 1

            return serialize_res

        serialize(root)

        return self.res
```

## 二叉树的序列化与反序列化

解法：后序遍历。

1. serialize：后序构建带#的序列，left+right+root
2. deserialize：倒序遍历。最后一个节点就是root节点，递归倒序遍历，先构建右子树，再构建左子树。

```python
class Codec:

    def __init__(self):
        self.seq = ','
        self.null = '#'

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        
        if not root:
            return self.null + self.seq

        left = self.serialize(root.left)
        right = self.serialize(root.right)

        return left + right + str(root.val) + self.seq
        

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        datas = data.split(',')[:-1]


        def build(datas):

            if not datas:
                return None

            last = datas.pop()

            if last == '#':
                return None

            root = TreeNode(int(last))

            root.right = build(datas)
            root.left = build(datas)

            return root

        return build(datas)
```

## 归并排序

解法：后序遍历。递归每次排序sort(l, mid)、sort(mid, r)。然后用双指针的方法做两个有序数组的合并。

```python
class Solution(object):
    def sortArray(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """

        def merge_sort(nums, l, r):

            if l == r:
                return nums[l]

            mid = l + (r - l)/2

            merge_sort(nums, l, mid)
            merge_sort(nums, mid+1, r)

            i, j = l, mid + 1
            tmp = []
            while i <= mid or j <= r:
                if i > mid or (j <= r and nums[i] > nums[j]):
                    tmp.append(nums[j])
                    j += 1
                else:
                    tmp.append(nums[i])
                    i += 1

            nums[l:r+1] = tmp

        merge_sort(nums, 0, len(nums) - 1)

        return nums
```

## 三道hard 日后再做

<https://leetcode.cn/problems/count-of-smaller-numbers-after-self/description/>
<https://leetcode.cn/problems/reverse-pairs/>
<https://leetcode.cn/problems/count-of-range-sum/>
