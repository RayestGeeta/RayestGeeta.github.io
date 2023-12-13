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

## 二维区域和检索 - 矩阵不可变（重新优化）

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

## 区间加法

假设你有一个长度为 n 的数组，初始情况下所有的数字均为 0，你将会被给出 k个更新的操作。

解法：构建差分数组。res[i] = res[i-1] + diff[i]。对于[i,j]区间+val的话，直接就是diff[i] + val，diff[j+1] - val即可。

```python
class Solution(object):
    def getModifiedArray(self, length, updates):
        """
        :type length: int
        :type updates: List[List[int]]
        :rtype: List[int]
        """

        res = [0 for _ in range(length)]
        diff_list = [0 for _ in range(length)]

        for startIndex, endIndex, inc in updates:
            diff_list[startIndex] += inc

            if endIndex + 1 < length:
                diff_list[endIndex + 1] -= inc

        res[0] = diff_list[0]
        for i in range(1, length):
            res[i] = res[i-1] + diff_list[i]
        return res
```

## 航班预定统计

这里有 n 个航班，它们分别从 1 到 n 进行编号。
有一份航班预订表 bookings ，表中第 i 条预订记录 bookings[i] = [firsti, lasti, seatsi] 意味着在从 firsti 到 lasti （包含 firsti 和 lasti ）的 每个航班 上预订了 seatsi 个座位。
请你返回一个长度为 n 的数组 answer，里面的元素是每个航班预定的座位总数。

解法：构建差分数组。res[i] = res[i-1] + diff[i]。对于[i,j]区间+val的话，直接就是diff[i] + val，diff[j+1] - val即可。

```python
class Solution(object):
    def corpFlightBookings(self, bookings, n):
        """
        :type bookings: List[List[int]]
        :type n: int
        :rtype: List[int]
        """

        res = [0 for _ in range(n)]
        diff = [0 for _ in range(n)]

        for first, last, seats in bookings:
            first -= 1
            last -= 1
            
            diff[first] += seats

            if last + 1 < n:
                diff[last + 1] -= seats

        
        res[0] = diff[0]
        for i in range(1, n):
            res[i] = res[i-1] + diff[i]

        return res
```

## 拼车

车上最初有 capacity 个空座位。车 只能 向一个方向行驶（也就是说，不允许掉头或改变方向）
给定整数 capacity 和一个数组 trips ,  trip[i] = [numPassengersi, fromi, toi] 表示第 i 次旅行有 numPassengersi 乘客，接他们和放他们的位置分别是 fromi 和 toi 。这些位置是从汽车的初始位置向东的公里数。
当且仅当你可以在所有给定的行程中接送所有乘客时，返回 true，否则请返回 false。

解法：差分数组

```python
class Solution(object):
    def carPooling(self, trips, capacity):
        """
        :type trips: List[List[int]]
        :type capacity: int
        :rtype: bool
        """

        maxLength = 1001
        tripNumPassengers = [0 for _ in range(maxLength)]
        diff = [0 for _ in range(maxLength)]

        for numPassengers, from1, to in trips:
            to -= 1
            diff[from1] += numPassengers

            if to + 1 < maxLength:
                diff[to+1] -= numPassengers

        tripNumPassengers[0] = diff[0]
        if tripNumPassengers[0] > capacity:
            return False
            
        for i in range(1, maxLength):
            now_capacity = diff[i] + tripNumPassengers[i-1]
            if now_capacity > capacity:
                return False

            tripNumPassengers[i] = now_capacity

        return True
```

## 反转字符串中的单词

给你一个字符串 s ，请你反转字符串中 单词 的顺序。
单词 是由非空格字符组成的字符串。s 中使用至少一个空格将字符串中的 单词 分隔开。
返回 单词 顺序颠倒且 单词 之间用单个空格连接的结果字符串。
注意：输入字符串 s中可能会存在前导空格、尾随空格或者单词间的多个空格。返回的结果字符串中，单词间应当仅用单个空格分隔，且不包含任何额外的空格。

解法：先反转整个字符串，再反转每个单词。

```python


class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        s_l = s.strip().split()
        s = []

        for i in s_l:
            for j in i:
                s.append(j)
            s.append(' ')

        s = s[:-1]

        def reverse(s):
            left = 0
            right = len(s)-1

            while left < right:
                s[left], s[right] = s[right], s[left]

                left += 1
                right -= 1

            return s

        s = reverse(s)
        s.append(' ')
        left, right = 0, 0
        while right < len(s):
            if s[right] == ' ':
                s[left:right] = reverse(s[left:right])

                right += 1
                left = right

            right += 1

        return ''.join(s[:-1])


```

## 旋转图像

给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。

解法：先按照左上到右下的对角线进行镜像反转。然后每行来反转。

```python
class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)

        for row in range(n):
            for col in range(row, n):
                matrix[row][col], matrix[col][row] = matrix[col][row], matrix[row][col]

        for row in range(n):
            for col in range(n/2):
                matrix[row][col], matrix[row][n-col-1] = matrix[row][n-col-1], matrix[row][col]

# 将二维矩阵原地逆时针旋转 90 度
def rotate2(matrix):
    n = len(matrix)
    # 沿左下到右上的对角线镜像对称二维矩阵
    for i in range(n):
        for j in range(n - i):
            # swap(matrix[i][j], matrix[n-j-1][n-i-1])
            temp = matrix[i][j]
            matrix[i][j] = matrix[n - j - 1][n - i - 1]
            matrix[n - j - 1][n - i - 1] = temp
    # 然后反转二维矩阵的每一行
    for row in matrix:
        reverse(row)

def reverse(arr):
    i, j = 0, len(arr)-1
    while j > i:
        arr[i], arr[j] = arr[j], arr[i]
        i+=1
        j-=1

```

## 螺旋矩阵

给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。

解法：依次按照从左到右、从上到下，从右到左，从下到上顺序遍历

```python
class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        
        rows, cols = len(matrix), len(matrix[0])
        upper, left = 0, 0
        lower, right = rows-1, cols-1

        res = []
        while len(res) < rows*cols:
            if upper <= lower:
                for i in range(left, right+1):
                    res.append(matrix[upper][i])
                upper += 1

            if left <= right:
                for j in range(upper, lower+1):
                    res.append(matrix[j][right])
                right -= 1

            if upper <= lower:
                for i in range(right, left-1, -1):
                    res.append(matrix[lower][i])
                lower -= 1

            if left <= right:
                for j in range(lower, upper-1, -1):
                    res.append(matrix[j][left])
                left += 1

        return res            
```

## 螺旋矩阵 II

给你一个正整数 n ，生成一个包含 1 到 n2 所有元素，且元素按顺时针顺序螺旋排列的 n x n 正方形矩阵 matrix 。

解法：解法：依次按照从左到右、从上到下，从右到左，从下到上顺序遍历

```python
class Solution(object):
    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """

        upper, left = 0, 0
        lower, right = n-1, n-1

        matrix = [[0 for _ in range(n)] for _ in range(n)]
        count = 1

        while count <= n*n:

            if upper <= lower:
                for i in range(left, right+1):
                    matrix[upper][i] = count
                    count += 1

                upper += 1

            if left <= right:
                for j in range(upper, lower+1):
                    matrix[j][right] = count
                    count += 1

                right -= 1

            if upper <= lower:
                for i in range(right, left-1, -1):
                    matrix[lower][i] = count
                    count += 1

                lower -= 1

            if left <= lower:
                for j in range(lower, upper-1, -1):
                    matrix[j][left] = count
                    count += 1

                left += 1

        return matrix
```

## O(1) 时间插入、删除和获取随机元素

解法：用字典存储每个值所在位置。然后每次新增的值都放数组的末尾。做删除时，利用字典存储的位置，将需要删除的值放入数组末尾进行删除。

```python
import random
class RandomizedSet(object):

    def __init__(self):
        self.nums = []
        self.dicts = {}


    def insert(self, val):
        """
        :type val: int
        :rtype: bool
        """

        if val in self.nums:
            return False

        self.dicts[val] = len(self.nums)
        self.nums.append(val)
        

        return True


    def remove(self, val):
        """
        :type val: int
        :rtype: bool
        """

        if val not in self.nums:
            return False

        index = self.dicts[val]
        self.dicts[self.nums[-1]] = index

        self.nums[-1], self.nums[index] = self.nums[index], self.nums[-1]

        self.nums.pop()
        del self.dicts[val]

        return True


    def getRandom(self):
        """
        :rtype: int
        """
        return random.choice(self.nums)



# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()
```

## 黑名单的随机数 以后再做

<https://leetcode.cn/problems/random-pick-with-blacklist/>
