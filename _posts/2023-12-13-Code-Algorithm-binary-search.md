---
title: Algorithm-part4-二分搜索
author: Rayest
date: 2023-12-13 00:31:00 +0800
categories: [interview]
tags: [fuck-code]
math: true
---

## 二分查找

给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target  ，写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。

解法：二分查找。每次和最中间的值比大小，然后调整左右边界。

```python
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """

        left = 0
        right = len(nums) - 1

        while left <= right:
            mid = left + (right - left) // 2

            if nums[mid] == target:
                return mid
            
            elif nums[mid] < target:
                left = mid + 1

            else:
                right = mid - 1

        return -1
```

## 二分查找左右边界

给你一个按照非递减顺序排列的整数数组 nums，和一个目标值 target。请你找出给定目标值在数组中的开始位置和结束位置。如果数组中不存在目标值 target，返回 [-1, -1]。

解法：二分查找。对于左边界，当每次找到target值时，继续缩减右边界。对于右边界，当每次找到target值时，继续缩减左边界。然后最后能找到的left， right即左右边界。

```python
class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """

        res_left, res_right = 0, 0

        # 左边界

        left, right = 0, len(nums) - 1

        while left <= right:
            mid = left + (right - left) // 2

            if nums[mid] == target:
                right = mid - 1

            elif nums[mid] < target:
                left = mid + 1

            else:
                right = mid - 1

        if left < 0 or left >= len(nums) or nums[left] != target:
            res_left = -1
        else:
            res_left = left

        
        # 右边界
        left, right = 0, len(nums) - 1

        while left <= right:
            mid = left + (right - left) // 2

            if nums[mid] == target:
                left = mid + 1

            elif nums[mid] < target:
                left = mid + 1

            else:
                right = mid - 1

        if right < 0 or right >= len(nums) or nums[right] != target:
            res_right = -1
        else:
            res_right = right

        
        return [res_left, res_right]

```

## 爱吃香蕉的珂珂

珂珂喜欢吃香蕉。这里有 n 堆香蕉，第 i 堆中有 piles[i] 根香蕉。警卫已经离开了，将在 h 小时后回来。
珂珂可以决定她吃香蕉的速度 k （单位：根/小时）。每个小时，她将会选择一堆香蕉，从中吃掉 k 根。如果这堆香蕉少于 k 根，她将吃掉这堆的所有香蕉，然后这一小时内不会再吃更多的香蕉。  
珂珂喜欢慢慢吃，但仍然想在警卫回来前吃掉所有的香蕉。
返回她可以在 h 小时内吃掉所有香蕉的最小速度 k（k 为整数）。

解法：二分搜索。搜索的值就是速度，初始上下范围时1到max(piles)，寻找左边界。

```python
class Solution(object):
    def minEatingSpeed(self, piles, h):
        """
        :type piles: List[int]
        :type h: int
        :rtype: int
        """

        left, right = 1, max(piles)


        def cost_time(piles, speed):
            time = 0
            for pile in piles:
                time += pile//speed

                if pile % speed > 0:
                    time += 1
            return time

        while left <= right:

            mid = left + (right - left) // 2
            time = cost_time(piles, mid)

            if time == h:
                right = mid - 1

            elif time > h:
                left = mid + 1

            else:
                right = mid - 1

        return left
```

## 在 D 天内送达包裹的能力

传送带上的包裹必须在 days 天内从一个港口运送到另一个港口。
传送带上的第 i 个包裹的重量为 weights[i]。每一天，我们都会按给出重量（weights）的顺序往传送带上装载包裹。我们装载的重量不会超过船的最大运载重量。
返回能在 days 天内将传送带上的所有包裹送达的船的最低运载能力。

解法：二分查找。搜索值就是最低运载能力，搜索范围是[max(weights), sum(weights)]。注意在计算某运载能力下所需时间的函数。依然是找左边界。

```python
class Solution(object):
    def shipWithinDays(self, weights, days):
        """
        :type weights: List[int]
        :type days: int
        :rtype: int
        """

        left, right = max(weights), sum(weights)


        def cost_days(weights, can_weight):
            cost_day = 0

            tmp = 0
            for i in weights:
                if tmp + i <= can_weight:
                    tmp += i
                else:
                    tmp = i
                    cost_day += 1

            if tmp > 0: cost_day += 1
            return cost_day

        while left <= right:

            mid = left + (right - left) // 2

            cost_day = cost_days(weights, mid)
            if cost_day == days:
                right = mid - 1

            elif cost_day > days:
                left = mid + 1

            else:
                right = mid - 1

        return left
```

## 分割数组的最大值

给定一个非负整数数组 nums 和一个整数 k ，你需要将这个数组分成 k 个非空的连续子数组。
设计一个算法使得这 k 个子数组各自和的最大值最小。

解法：二分搜索。该问题换个问法，当最大值是多少时，恰好能让数组分成k份。搜索区间是[max(nums), sum(nums)]，注意计算当子数组最大值不超过n时，子数组的数量的计算。

```python
class Solution(object):
    def splitArray(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """

        left, right = max(nums), sum(nums)

        def cut_groups(nums, sum_value):

            groups = 0

            tmp = 0
            for num in nums:
                if tmp + num <= sum_value:
                    tmp += num
                else:
                    tmp = num
                    groups += 1

            if tmp > 0: groups += 1

            return groups

        while left <= right:


            mid = left + (right - left) // 2

            cut_group = cut_groups(nums, mid)

            if cut_group == k:
                right = mid - 1

            elif cut_group > k:
                left = mid + 1

            else:
                right = mid - 1

        return left
```
