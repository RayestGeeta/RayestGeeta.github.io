---
title: Algorithm-part1-链表算法
author: Rayest
date: 2023-12-07 20:29:00 +0800
categories: [interview]
tags: [fuck-code]
math: true
---


## 合并两个有序链表

将两个升序链表合并为一个新的 升序 链表并返回。

解法：循环遍历两个链表，比较当前节点的大小。谁小谁拼在新链表上，直到其中一个链表为空。然后将另外一个链表剩下结点拼在新链表上就结束了。

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
def mergeLists(list1, list2):
    
    result = ListNode(-1)
    
    tmp = result
    p1 = list1
    p2 = list2
    
    while p1 and p2:
        if p1.val < p2.val:
            tmp.next = p1
            p1 = p1.next
            
        else:
            tmp.next = p2
            p2 = p2.next
            
        tmp = tmp.next
        
    if p1: tmp.next = p1
    if p2: tmp.next = p2
    
    return result.next

def buildLists(vals):
    head = ListNode(vals[0])
    tmp = head
    
    for val in vals[1:]:
        tmp.next = ListNode(val)
        tmp = tmp.next
        
    return head

list1 = buildLists([1,2,3])
list2 = buildLists([2,3,4])

res = mergeLists(list1, list2)

while res:
    print(res.val)
    res = res.next
```

## 单链表的分解

给你一个链表的头节点 head 和一个特定值 x ，请你对链表进行分隔，使得所有 小于 x 的节点都出现在 大于或等于 x 的节点之前。你应当 保留 两个分区中每个节点的初始相对位置。

解法：和上面的双链表合并相反。可以先拆成两个链表，一个放比x小的结点。一个放大于等于x的结点。最后两个链表拼起来。注意就是在链表向后走的时候记得断开指针。

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
def splitList(listNodes, x):
    
    list1 = ListNode(-1)
    list2 = ListNode(-1)
    
    p1, p2 = list1, list2
    head = listNodes
    
    while head:
        if head.val < x:
            p1.next = head
            p1 = p1.next
            
        else:
            p2.next = head
            p2 = p2.next
            
        tmp = head.next
        head.next = None
        head = tmp
        
    p1.next = list2.next
    
    return list1.next

def buildListNodes(vals):
    res = ListNode(vals[0])
    head = res
    
    for val in vals[1:]:
        head.next = ListNode(val)
        head = head.next
    
    return res

listNodes = buildListNodes([1,4,3,2,5,2])

res = splitList(listNodes, 3)

while res:
    print(res.val)
    res = res.next
```

## 单链表的倒数第K个节点

解法：双指针。第一个指针先走k步，然后第二个指针从头开始。两个指针一块走，直到指针1走到结尾。指针2的位置就是结果。

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
def countBack(listNodes, k):
    
    p1 = listNodes
    for i in range(k):
        p1 = p1.next
        
    p2 = listNodes
    
    while p1:
        p1 = p1.next
        p2 = p2.next
        
    return p2

def buildListNodes(vals):
    head = ListNode(vals[0])
    
    tmp = head
    for val in vals[1:]:
        tmp.next = ListNode(val)
        tmp = tmp.next
        
    return head

listNodes = buildListNodes([6,5,4,3,2,1,0])

res = countBack(listNodes, 4)
print(res.val)

```

## 删除链表的第N个节点

给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。

解法：还是双指针。找到倒数第n+1个节点。然后短路第n个节点就可以。注意就是要用虚拟节点。可能会有空指针的情况。比如删除头节点。

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
def removeBack(head, n):
    
    tmpNode = ListNode(-1, head)
    
    p1 = head
    for i in range(n):
        p1 = p1.next
        
    p2 = tmpNode
    while p1:
        p1 = p1.next
        p2 = p2.next
        
    p2.next = p2.next.next
    
    return tmpNode.next

def buildListNodes(vals):
    head = ListNode(vals[0])
    
    tmp = head
    for val in vals[1:]:
        tmp.next = ListNode(val)
        tmp = tmp.next
        
    return head

head = buildListNodes([1,2,3,4,5,6])

res = removeBack(head, 3)

while res:
    print(res.val)
    res = res.next
```

## 链表的中点

解法：快慢指针。快指针走两步，满指针走一步。快指针走到底的时候，满指针就在中间。

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
def middleLists(head):
    
    slow = head
    fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
    return slow

def buildListNodes(vals):
    head = ListNode(vals[0])
    
    tmp = head
    for val in vals[1:]:
        tmp.next = ListNode(val)
        tmp = tmp.next
        
    return head

head = buildListNodes([1,2,3,4,5,6])

middle = middleLists(head)

print(middle.val)
```

## 判断链表是否包含环

解法：还是快慢指针。如果快慢指针会相遇，那么就有环。

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
def hasCycle(head):
    
    slow = head
    fast = head
    
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next
        
        if fast == slow:
            return True
        
    return False

def buildHaveCycleNodes():
    head = ListNode(0)
    
    cycle0 = ListNode(1)
    cycle1 = ListNode(2)
    
    cycle2 = ListNode(3)
    
    end = ListNode(4)
    
    head.next = cycle0
    cycle0.next = cycle1
    cycle1.next = cycle2
    cycle2.next = end
    end.next = cycle0
    
    return head

head = buildHaveCycleNodes()

print(hasCycle(head))
```

## 找到链表中环的位置

解法：还是快慢指针。当快慢指针相遇时，假设慢指针走k步，快指针走了2k步，慢指针距离环头节点距离m。

1. 快指针比慢指针多走的k步，都是在圈里兜圈子。也就是k的环长度的n倍。
2. 慢指针肯定没有兜圈。那么就知道环头节点距离头节点 k-m步。
3. 快指针再走k-m步也能到环头节点。
4. 根据2，3。把慢指针扔回头节点，和快指针一起走k-m步。再次相遇时就是环头节点。

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
def getCyclePos(head):
    
    slow = head
    fast = head
    
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next
        
        if fast == slow:
            slow = head
            while slow != fast:
                slow = slow.next
                fast = fast.next
                
            return slow
        
    return False 

def buildHaveCycleNodes():
    head = ListNode(0)
    
    cycle0 = ListNode(1)
    cycle1 = ListNode(2)
    
    cycle2 = ListNode(3)
    
    end = ListNode(4)
    
    head.next = cycle0
    cycle0.next = cycle1
    cycle1.next = cycle2
    cycle2.next = end
    end.next = cycle0
    
    return head

head = buildHaveCycleNodes()

print(getCyclePos(head).val)
```

## 两个链表是否相交

解法：两个链表肯定是后续才相交的。问题是在于长度不一样。所以先算出两个链表的长度。然后依次同时遍历，比对节点是否一致。如果存在一致就是有相交，反之则没有。

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
        
def buildTowNodesInterSect():
    list1 = ListNode(4)
    list2 = ListNode(5)
    
    p1 = list1
    p2 = list2
    
    shareNode = ListNode(8)
    tmpNode = shareNode
    for val in [4, 5]:
        tmpNode.next = ListNode(val)
        tmpNode = tmpNode.next
        
    p1.next = ListNode(1)
    p1.next.next = shareNode
    
    p2.next = ListNode(6)
    p2.next.next = ListNode(1)
    p2.next.next.next = shareNode
    
    return list1, list2

def getInterSectNode(list1, list2):
    
    p1 = list1
    p2 = list2
    
    len1, len2 = 0, 0
    
    while p1:
        p1 = p1.next
        len1 += 1
        
    while p2:
        p2 = p2.next
        len2 += 1 
        
    p1 = list1
    p2 = list2
    
    if len1 > len2:
        for i in range(len1 - len2):
            p1 = p1.next
    else:
        for i in range(len2 - len1):
            p2 = p2.next
            
    while p1:
        if p1 == p2:
            return p1
        
        p1 = p1.next
        p2 = p2.next
        
    return False

list1, list2 = buildTowNodesInterSect()
print(getInterSectNode(list1, list2).val)
```

## 合并K个有序链表

## 等到二叉堆数据结构做到了，再回头来写

## 反转链表

解法：递归。假设只有两个结点，进行反转。再往上递归到第三个、四个结点反转。

```python
def reverse(head):
    
    if head or head.next:
        return head
    
    last = reverse(head.next)
    
    head.next.next = head
    head.next = None
    
    return last
```

## 反转前N个结点 链表

解法：递归。和反转链表就一个区别，反转后head不是最后结点，还要连上未反转的部分。

```python
raw = None

def reverseN(head, n):
    global raw
    
    if n == 1:
        
        raw = head.next
        return head
    
    last = reverseN(head.next, n - 1)
    
    head.next.next = head
    head.next = raw
    
    return last
    
```

## 反转部分链表

给一个索引区间 [m, n]（索引从 1 开始），仅仅反转区间中的链表元素。
解法：先反转前N个链表，然后用递归从第m个结点开始反转。

```python
def reverseBetween(head, left, right):
    
    # if left == 1:
    
    raw = None
    
    def reverseN(head, right):
        
        if right == 1:
            raw = head.next
            return head
        
        last = reverseN(head.next, right - 1)
        
        head.next.next = head
        head.next = raw
        
        return last
        
    if left == 1:
        return reverseN(head, right)
    
    head.next = reverseBetween(head.next, left-1, right-1)
    
    return head
        

```

## 链表中 k个一组进行反转

解法：纯递归

```python
def reverseKGroup(head, k):
    """
    :type head: ListNode
    :type k: int
    :rtype: ListNode
    """

    def reverseBetween(head, left, right):

        raw = None

        def reverseN(head, right):
            global raw

            if right == 1:
                raw = head.next
                return head

            last = reverseN(head.next, right - 1)

            head.next.next = head
            head.next = raw

            return last

        if left == 1:
            return reverseN(head, right)

        head.next = reverseBetween(head.next, left-1, right-1)

        return head

    tmp = head
    left = 1
    flag = True
    while tmp:
        for i in range(k):
            if tmp is None:
                flag = False
                break
            tmp = tmp.next

        if flag:
            head = reverseBetween(head, left, left+k-1)
            left += k

    return head

```
