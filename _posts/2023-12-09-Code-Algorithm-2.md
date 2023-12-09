---
title: Algorithm-part2-数组算法
author: Rayest
date: 2023-12-09 22:57:00 +0800
categories: [interview]
tags: [fuck-code]
math: true
---

## 删除有序数组中的重复项

给你一个 升序排列 的数组 nums ，请你 原地 删除重复出现的元素，使每个元素 只出现一次 ，返回删除后数组的新长度。元素的 相对顺序 应该保持 一致 。然后返回 nums 中唯一元素的个数。

解法：快慢指针。fast走在前面，每找到一个新的元素，就赋值给slow并让slow往前走一步。这样就能保证在[0:slow]之前的值都是不重复的。

```python
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        slow = 0
        fast = 1

        while fast < len(nums):
            if nums[slow] != nums[fast]:
                nums[slow+1] = nums[fast]
                slow += 1

            fast += 1

        return slow+1
    
# 链表
class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head:
            return None

        slow = head
        fast = head

        while fast:
            if slow.val != fast.val:
                slow.next = fast
                slow = slow.next
            fast = fast.next

        slow.next = None
        return head
```

## 移除元素

给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。

解法：快慢指针。快指针去寻找非val值，然后复制给慢指针并让慢指针往前走。这样[0:slow]之间的值都是非val的。

```python
class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """

        slow = 0
        fast = 0

        while fast < len(nums):
            if nums[fast] != val:
                nums[slow] = nums[fast]
                slow += 1

            fast += 1

        return slow
```

## 移动0

给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。

请注意 ，必须在不复制数组的情况下原地对数组进行操作。

解法：快慢指针。快指针找非0值给慢指针。最后把剩下的元素都改成0。

```python
class Solution(object):
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """

        slow, fast = 0, 0

        while fast < len(nums):
            if nums[fast] != 0:
                nums[slow] = nums[fast]
                slow += 1

            fast += 1

        for i in range(slow, len(nums)):
            nums[i] = 0

        return nums
```

## 两数之和

解法：滑动窗口。通过调节 left 和 right 就可以调整 sum 的大小。

```python
class Solution(object):
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """

        left = 0
        right = len(numbers) - 1

        while left < right:
            res = numbers[left] + numbers[right]
            if res == target:
                return [left+1 , right+1]

            elif res > target:
                right -= 1

            else:
                left += 1

        return [-1, -1]
```

## 反转字符串

解法：双指针。相向而行，交换值。

```python
class Solution(object):
    def reverseString(self, s):
        """
        :type s: List[str]
        :rtype: None Do not return anything, modify s in-place instead.
        """

        left, right = 0, len(s) - 1

        while left < right:
            tmp = s[left]

            s[left] = s[right]
            s[right] = tmp

            left += 1
            right -= 1

            
```

## 最长回文子串

解法：中心扩展。左右指针来做。先实现一个函数，从当前位置作为中心，左右扩展找最长字串。要注意考虑奇数偶数情况，然后再循环去找每个位置作为中心的最大字串。

```python
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        if len(s) <= 1:
            return s

        def getIndexLongest(s, l, r):

            while l>=0 and r<len(s) and s[l] == s[r]:
                l -= 1
                r += 1

            return s[l+1:r]

        res = ""
        for i in range(0, len(s)-1):

            s1 = getIndexLongest(s, i, i)
            s2 = getIndexLongest(s, i, i+1)

            res = res if len(res) > len(s1) else s1
            res = res if len(res) > len(s2) else s2

        return res


```

## 区域和检索 - 数组不可变
给定一个整数数组  nums，处理以下类型的多个查询:
计算索引 left 和 right （包含 left 和 right）之间的 nums 元素的 和 ，其中 left <= right

解法：提前计算出一个preSum来记录 preSum[i] = sum(nums[0:i-1])，然后后面再要计算某个索引区间下的值，直接使用这个preSum相减就可以了。

```python
class NumArray(object):

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.preNums = [0] * (len(nums)+1)
        for i in range(1, len(nums)+1):
            self.preNums[i] = self.preNums[i-1] + nums[i-1]


    def sumRange(self, left, right):
        """
        :type left: int
        :type right: int
        :rtype: int
        """
        return self.preNums[right+1] - self.preNums[left]


```

## 二维区域和检索 - 矩阵不可变

给定一个二维矩阵 matrix，以下类型的多个请求：
计算其子矩形范围内元素的总和，该子矩阵的 左上角 为 (row1, col1) ，右下角 为 (row2, col2) 。

解法：前缀做法

```python
class NumMatrix(object):

    def __init__(self, matrix):
        """
        :type matrix: List[List[int]]
        """
        m, n = len(matrix), len(matrix[0])
        self.preMatrix = [[0 for _ in range(n)] for _ in range(m)]

        self.preMatrix[0][0] = matrix[0][0]
        for i in range(1, m):
            self.preMatrix[i][0] = matrix[i][0] + self.preMatrix[i-1][0]

        for j in range(1, n):
            self.preMatrix[0][j] = matrix[0][j] + self.preMatrix[0][j-1]

        for i in range(1, m):
            for j in range(1, n):
                self.preMatrix[i][j] = matrix[i][j] + self.preMatrix[i-1][j] + self.preMatrix[i][j-1] - self.preMatrix[i-1][j-1]


    def sumRegion(self, row1, col1, row2, col2):
        """
        :type row1: int
        :type col1: int
        :type row2: int
        :type col2: int
        :rtype: int
        """
        if row1 > 0 and col1 > 0:
            return self.preMatrix[row2][col2] - self.preMatrix[row2][col1-1] - self.preMatrix[row1-1][col2] + self.preMatrix[row1-1][col1-1]

        elif row1 == 0 and col1 > 0:
            return self.preMatrix[row2][col2] - self.preMatrix[row2][col1-1]
        elif row1 > 0 and col1 == 0:
            return self.preMatrix[row2][col2] - self.preMatrix[row1-1][col2]
        else:
            return self.preMatrix[row2][col2]

```
