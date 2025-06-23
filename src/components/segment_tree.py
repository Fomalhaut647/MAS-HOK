import operator


# 直接来自OpenAI Baseline实现 (https://github.com/openai/baselines)
class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """构建线段树数据结构。
        https://en.wikipedia.org/wiki/Segment_tree
        可以像常规数组一样使用，但有两个重要差异：
            a) 设置项目值稍慢。时间复杂度为O(lg capacity)而不是O(1)。
            b) 用户可以访问高效的（O(log segment size)）
               `reduce`操作，该操作在数组的连续子序列上执行`operation`。
        参数
        ---------
        capacity: int
            数组的总大小 - 必须是2的幂。
        operation: lambda obj, obj -> obj
            用于组合元素的操作（例如sum、max）
            必须与数组元素的可能值集合形成数学群（即满足结合律）
        neutral_element: obj
            上述操作的中性元素。例如max的float('-inf')和sum的0。
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """返回对数组连续子序列应用`self.operation`的结果。
            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
        参数
        ----------
        start: int
            子序列的开始位置
        end: int
            子序列的结束位置
        返回
        -------
        reduced: obj
            在指定数组元素范围上应用self.operation的结果。
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # 叶子节点的索引
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """找到数组中满足条件的最高索引`i`，使得
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        如果数组值是概率，这个函数
        允许根据离散概率高效地采样索引。
        参数
        ----------
        perfixsum: float
            数组前缀和的上界
        返回
        -------
        idx: int
            满足前缀和约束的最高索引
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # 当非叶子节点时
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """返回 min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)