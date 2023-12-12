---
title: Algorithm-part3-滑动窗口
author: Rayest
date: 2023-12-11 22:20:00 +0800
categories: [interview]
tags: [fuck-code]
math: true
---

## 最小覆盖子串

给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。

解法：滑动窗口。先向右滑动，找到可以满足的字串。然后再缩小左边界，直到不再满足，继续增加右边界。再缩小边界的时候，不断更新结果。

```python
class Solution(object):
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """

        vaild = 0
        need_t = {}
        windows_t = {}
        for i in t:
            if i not in need_t:
                need_t[i] = 1
                windows_t[i] = 0
            else:
                need_t[i] += 1

        left, right = 0, 0
        res_start, res_length = 0, float('inf')

        while right < len(s):

            strs = s[right]
            right += 1

            if strs in need_t:
                windows_t[strs] += 1

                if windows_t[strs] == need_t[strs]:
                    vaild += 1
        
            while vaild == len(need_t):
                
                if right - left < res_length:
                    res_start = left
                    res_length = right - left

                pop_strs = s[left]
                left += 1

                if pop_strs in need_t:

                    if windows_t[pop_strs] == need_t[pop_strs]:
                        vaild -= 1
                    windows_t[pop_strs] -= 1

        return "" if res_length == float('inf') else s[res_start:res_start+res_length]
```

## 字符串的排列

给你两个字符串 s1 和 s2 ，写一个函数来判断 s2 是否包含 s1 的排列。如果是，返回 true ；否则，返回 false 。

换句话说，s1 的排列之一是 s2 的 子串 。

解法：滑动窗口。先增加右边界，当窗口长度大于s1长度时，就要开始收缩到s1长度。

```python
class Solution(object):
    def checkInclusion(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """


        from collections import defaultdict
        need, window = defaultdict(int), defaultdict(int)
        for c in s1:
            need[c] += 1

        left, right = 0, 0
        valid = 0
        while right < len(s2):
            c = s2[right]
            right += 1

            if c in need:
                window[c] += 1
                if window[c] == need[c]:
                    valid += 1


            while right - left >= len(s1):

                if valid == len(need):
                    return True
                d = s2[left]
                left += 1

                if d in need:
                    if window[d] == need[d]:
                        valid -= 1
                    window[d] -= 1


        return False

        
```

## 找到字符串中所有字母异位词

给定两个字符串 s 和 p，找到 s 中所有 p 的 异位词 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。

异位词 指由相同字母重排列形成的字符串（包括相同的字符串）。

解法：滑动窗口。先增加右边界，当窗口长度大于p长度时，缩减边界。当满足条件时，保存index。

```python
class Solution(object):
    def findAnagrams(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: List[int]
        """

        vaild = 0
        need = {}
        window = {}
        for i in p:
            if i not in need:
                need[i] = 1
                window[i] = 0

            else:
                need[i] += 1


        indexs = []
        left, right = 0, 0

        while right < len(s):

            strs = s[right]
            right += 1

            if strs in need:
                window[strs] += 1
                if window[strs] == need[strs]:
                    vaild += 1

            while right - left >= len(p):
                if vaild == len(need):
                    indexs.append(left)

                pop_strs = s[left]
                left += 1

                if pop_strs in need:
                    if window[pop_strs] == need[pop_strs]:
                        vaild -= 1

                    window[pop_strs] -= 1

        return indexs

```

## 无重复字符的最长子串

给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。

解法：滑动窗口。扩大右边界，每次扩大都更新结果。当发现有重复字符时，可以缩减左边界，直到无重复字符。

```python
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """

        window_s = []

        res_max = 0
        left, right = 0, 0

        while right < len(s):

            strs = s[right]
            right += 1

            if strs not in window_s:
                window_s.append(strs)

                if right - left > res_max:
                    res_max = right - left

            else:

                while s[left] != strs:
                    pop_strs = s[left]
                    left += 1

                    window_s.remove(pop_strs)
                left += 1

        return res_max
                
```

## 优势洗牌

给定两个长度相等的数组 nums1 和 nums2，nums1 相对于 nums2 的优势可以用满足 nums1[i] > nums2[i] 的索引 i 的数目来描述。
返回 nums1 的任意排列，使其相对于 nums2 的优势最大化。

```python
class Solution(object):
    def advantageCount(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """

        # 和对面最大的值比，比得过就打，比不过就拿最小值顶

        index1, index2 = [i for i in range(len(nums1))], [i for i in range(len(nums2))]

        index1.sort(key = lambda x:nums1[x], reverse=True)
        index2.sort(key = lambda x:nums2[x], reverse=True)


        res = [0] * len(nums1)
        left, right = 0, len(nums1) - 1
        for i in range(len(nums1)):

            if nums2[index2[i]] < nums1[index1[left]]:

                res[index2[i]] = nums1[index1[left]]
                left += 1


            else:
                res[index2[i]] = nums1[index1[right]]
                right -= 1
                
        return res
```
