---
title: Algorithm-part7-动态规划
author: Rayest
date: 2023-12-19 21:42:00 +0800
categories: [interview]
tags: [fuck-code]
math: true
---

## 零钱兑换

给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。
计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。
你可以认为每种硬币的数量是无限的。

解法：动态规划。抽象子问题，变成1+ dp(amount - 1)。状态转移公式dp[i] = min(dp[i], 1+dp[i-coin])

```python
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """

        dp = [amount+1 for i in range(amount+1)]

        dp[0] = 0
        for i in range(1, len(dp)):
            for coin in coins:

                if i < coin:
                    continue

                dp[i] = min(dp[i], 1 + dp[i - coin])

        return dp[-1] if dp[-1] != amount+1 else -1
```

## 最长递增子序列

给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
子序列 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。

解法：动态规划。dp[i]表示的到第i的元素为结尾的最长递增子序列长度。dp[i] = max(dp[i], 1+dp[j]) nums[j] < num[i]

```python
class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        n = len(nums)
        dp = [1 for _ in range(n)]

        for i in range(len(dp)):
            for j in range(0, i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], 1 + dp[j])

        return max(dp)
```

## 俄罗斯套娃信封问题

给你一个二维整数数组 envelopes ，其中 envelopes[i] = [wi, hi] ，表示第 i 个信封的宽度和高度。
当另一个信封的宽度和高度都比这个信封大的时候，这个信封就可以放进另一个信封里，如同俄罗斯套娃一样。
请计算 最多能有多少个 信封能组成一组“俄罗斯套娃”信封（即可以把一个信封放到另一个信封里面）。

解法:动态规划.对w继续升序,对h降序(因为只有w,h都比前一个才能套娃).

```python
class Solution(object):
    def maxEnvelopes(self, envelopes):
        """
        :type envelopes: List[List[int]]
        :rtype: int
        """

        envelopes.sort(key = lambda x:(x[0], -x[1]))

        heights = [i[1] for i in envelopes]
        dp = [1 for i in range(len(heights))]

        for i in range(len(dp)):
            for j in range(0, i):
                if heights[i] > heights[j]:
                    dp[i] = max(dp[i], 1 + dp[j])
        
        return max(dp)
```

## 下降路径最小和

给你一个 n x n 的 方形 整数数组 matrix ，请你找出并返回通过 matrix 的下降路径 的 最小和 。

`解法:二维动态规划.dp[i][j]表示到matrix[i][j]时的最小和.dp[i][j] = matrix[i][j] + min([dp[i-1][j-1], dp[i-1][j], dp[i-1][j+1]])`

```python
class Solution(object):
    def minFallingPathSum(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: int
        """

        n = len(matrix)
        dp = [[float('inf') for _ in range(n)] for _ in range(n)]

        dp[0] = matrix[0]

        for i in range(1, n):
            for j in range(n):


                left = float('inf')
                if j-1 >= 0:
                    left = dp[i-1][j-1]

                right = float('inf')
                if j+1 < n:
                    right = dp[i-1][j+1]


                dp[i][j] = matrix[i][j] + min([left, dp[i-1][j], right])

        return min(dp[-1])
```

## 不同的子序列

解法:日后再看

```python
class Solution(object):
    def numDistinct(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: int
        """
        m, n = len(s), len(t)
        dp = [1]+[0]*n
        for i in range(m):
            for j in range(n-1, -1, -1):
                if s[i] == t[j]:
                    dp[j+1] += dp[j]
        return dp[-1] % (10**9+7)
```

## 单词拆分

给你一个字符串 s 和一个字符串列表 wordDict 作为字典。请你判断是否可以利用字典中出现的单词拼接出 s 。
注意：不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。

解法:动态规划.dp[i]代表截至到s[0:i]可以拼出.

```python
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """

        n = len(s)
        dp = [False for _ in range(n)]


        for i in range(n):
            dp[i] = s[:i+1] in wordDict

            for j in range(0, i):
                if dp[i]:
                    continue
                dp[i] = dp[j] and s[j+1:i+1] in wordDict

        return dp[-1]
```

## 单词拆分II

日后再说

```python
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: List[str]
        """

        res = []
        #
        wordDict = set(wordDict)

        def dfs(wordDict,temp,pos):
            #
            if pos == len(s):
                res.append(" ".join(temp))
                return
            for i in range(pos,len(s)+1):
                if s[pos:i] in wordDict:
                    temp.append(s[pos:i])
                    dfs(wordDict,temp,i)
                    temp.pop() 
            #
                       
            
        dfs(wordDict,[],0)
        return res

```
