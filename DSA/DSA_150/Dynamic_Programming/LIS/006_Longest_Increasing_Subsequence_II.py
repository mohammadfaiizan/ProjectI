"""
Problem: Longest Increasing Subsequence II
URL: https://leetcode.com/problems/longest-increasing-subsequence-ii/

Problem Statement:
You are given an integer array nums and an integer k.
Find the longest subsequence of nums that meets the following requirements:
- The subsequence is strictly increasing.
- The difference between adjacent elements in the subsequence is at most k.
Return the length of the longest subsequence that meets the requirements.

Sample Input/Output:
Input: nums = [4,2,1,4,3,4,5,8,15], k = 3
Output: 5
Explanation: The longest subsequence that meets the requirements is [1,3,4,5,8].

Input: nums = [7,4,5,1,8,12,4,7], k = 5
Output: 4
Explanation: The longest subsequence that meets the requirements is [4,5,8,12].
"""

from typing import List

class Solution:
    def Length_Of_LIS_Brute_Force(self, nums: List[int], k: int) -> int:
        """
        Brute Force - Try all possible subsequences
        Time Complexity: O(2^n * n)
        Space Complexity: O(n)
        """
        def Is_Valid_Subsequence(subseq: List[int]) -> bool:
            if len(subseq) <= 1:
                return True
            
            for i in range(1, len(subseq)):
                if subseq[i] <= subseq[i-1] or subseq[i] - subseq[i-1] > k:
                    return False
            
            return True
        
        def Generate_Subsequences(index: int, current: List[int], all_subseq: List[List[int]]) -> None:
            if index >= len(nums):
                if current:
                    all_subseq.append(current[:])
                return
            
            Generate_Subsequences(index + 1, current, all_subseq)
            
            current.append(nums[index])
            Generate_Subsequences(index + 1, current, all_subseq)
            current.pop()
        
        all_subsequences = []
        Generate_Subsequences(0, [], all_subsequences)
        
        max_length = 0
        for subseq in all_subsequences:
            if Is_Valid_Subsequence(subseq):
                max_length = max(max_length, len(subseq))
        
        return max_length
    
    def Length_Of_LIS_DP_Quadratic(self, nums: List[int], k: int) -> int:
        """
        DP Quadratic - O(n²) DP approach
        Time Complexity: O(n²)
        Space Complexity: O(n)
        """
        if not nums:
            return 0
        
        n = len(nums)
        dp = [1] * n
        
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i] and nums[i] - nums[j] <= k:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)
    
    def Length_Of_LIS_Segment_Tree_Optimal(self, nums: List[int], k: int) -> int:
        """
        Segment Tree Optimal - Use segment tree for range maximum
        Time Complexity: O(n log(max_val))
        Space Complexity: O(max_val)
        """
        if not nums:
            return 0
        
        max_val = max(nums)
        
        class SegmentTree:
            def __init__(self, size: int):
                self.size = size
                self.tree = [0] * (4 * size)
            
            def Update(self, node: int, start: int, end: int, idx: int, val: int) -> None:
                if start == end:
                    self.tree[node] = max(self.tree[node], val)
                else:
                    mid = (start + end) // 2
                    if idx <= mid:
                        self.Update(2 * node, start, mid, idx, val)
                    else:
                        self.Update(2 * node + 1, mid + 1, end, idx, val)
                    
                    self.tree[node] = max(self.tree[2 * node], self.tree[2 * node + 1])
            
            def Query(self, node: int, start: int, end: int, l: int, r: int) -> int:
                if r < start or end < l or l > r:
                    return 0
                if l <= start and end <= r:
                    return self.tree[node]
                
                mid = (start + end) // 2
                return max(self.Query(2 * node, start, mid, l, r),
                          self.Query(2 * node + 1, mid + 1, end, l, r))
        
        seg_tree = SegmentTree(max_val + 1)
        
        for num in nums:
            left_bound = max(1, num - k)
            right_bound = num - 1
            
            max_prev = seg_tree.Query(1, 0, max_val, left_bound, right_bound)
            seg_tree.Update(1, 0, max_val, num, max_prev + 1)
        
        return seg_tree.Query(1, 0, max_val, 1, max_val)
    
    def Length_Of_LIS_Coordinate_Compression(self, nums: List[int], k: int) -> int:
        """
        Coordinate Compression - Compress values for smaller range
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        if not nums:
            return 0
        
        sorted_unique = sorted(set(nums))
        coord_map = {v: i for i, v in enumerate(sorted_unique)}
        reverse_map = {i: v for i, v in enumerate(sorted_unique)}
        
        compressed_nums = [coord_map[num] for num in nums]
        n = len(sorted_unique)
        
        class SegmentTree:
            def __init__(self, size: int):
                self.size = size
                self.tree = [0] * (4 * size)
            
            def Update(self, node: int, start: int, end: int, idx: int, val: int) -> None:
                if start == end:
                    self.tree[node] = max(self.tree[node], val)
                else:
                    mid = (start + end) // 2
                    if idx <= mid:
                        self.Update(2 * node, start, mid, idx, val)
                    else:
                        self.Update(2 * node + 1, mid + 1, end, idx, val)
                    
                    self.tree[node] = max(self.tree[2 * node], self.tree[2 * node + 1])
            
            def Query(self, node: int, start: int, end: int, l: int, r: int) -> int:
                if r < start or end < l or l > r:
                    return 0
                if l <= start and end <= r:
                    return self.tree[node]
                
                mid = (start + end) // 2
                return max(self.Query(2 * node, start, mid, l, r),
                          self.Query(2 * node + 1, mid + 1, end, l, r))
        
        seg_tree = SegmentTree(n)
        
        for compressed_num in compressed_nums:
            actual_val = reverse_map[compressed_num]
            
            left_bound = actual_val - k
            right_bound = actual_val - 1
            
            left_idx = 0
            right_idx = compressed_num - 1
            
            for i, val in enumerate(sorted_unique):
                if val >= left_bound:
                    left_idx = i
                    break
            
            max_prev = seg_tree.Query(1, 0, n - 1, left_idx, right_idx) if right_idx >= left_idx else 0
            seg_tree.Update(1, 0, n - 1, compressed_num, max_prev + 1)
        
        return seg_tree.Query(1, 0, n - 1, 0, n - 1)
    
    def Length_Of_LIS_Binary_Indexed_Tree(self, nums: List[int], k: int) -> int:
        """
        Binary Indexed Tree - Use BIT for efficient updates
        Time Complexity: O(n log(max_val))
        Space Complexity: O(max_val)
        """
        if not nums:
            return 0
        
        offset = 1
        max_val = max(nums) + offset
        
        class BIT:
            def __init__(self, size: int):
                self.size = size
                self.tree = [0] * (size + 1)
            
            def Update(self, idx: int, val: int) -> None:
                idx += 1
                while idx <= self.size:
                    self.tree[idx] = max(self.tree[idx], val)
                    idx += idx & (-idx)
            
            def Query(self, idx: int) -> int:
                idx += 1
                result = 0
                while idx > 0:
                    result = max(result, self.tree[idx])
                    idx -= idx & (-idx)
                return result
            
            def Range_Query(self, left: int, right: int) -> int:
                if left > right:
                    return 0
                if left == 0:
                    return self.Query(right)
                return max(self.Query(right) - self.Query(left - 1), 0)
        
        bit = BIT(max_val)
        
        for num in nums:
            left_bound = max(offset, num - k)
            right_bound = num - 1
            
            max_prev = 0
            for i in range(left_bound, min(right_bound + 1, num)):
                max_prev = max(max_prev, bit.Query(i - 1))
            
            bit.Update(num, max_prev + 1)
        
        result = 0
        for i in range(max_val):
            result = max(result, bit.Query(i))
        
        return result
    
    def Length_Of_LIS_With_Sequence(self, nums: List[int], k: int) -> tuple:
        """
        With Sequence - Return length and actual LIS
        Time Complexity: O(n²)
        Space Complexity: O(n)
        """
        if not nums:
            return 0, []
        
        n = len(nums)
        dp = [1] * n
        parent = [-1] * n
        
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i] and nums[i] - nums[j] <= k:
                    if dp[j] + 1 > dp[i]:
                        dp[i] = dp[j] + 1
                        parent[i] = j
        
        max_length = max(dp)
        max_index = dp.index(max_length)
        
        sequence = []
        current = max_index
        
        while current != -1:
            sequence.append(nums[current])
            current = parent[current]
        
        return max_length, sequence[::-1]

def Test_Length_Of_LIS():
    solution = Solution()
    
    test_cases = [
        ([4,2,1,4,3,4,5,8,15], 3, 5),
        ([7,4,5,1,8,12,4,7], 5, 4),
        ([1,3,6,7,9,4,10,5,6], 3, 6),
        ([1,2,3], 1, 3),
        ([1,5,2], 3, 2)
    ]
    
    methods = [
        ("DP Quadratic", solution.Length_Of_LIS_DP_Quadratic),
        ("Segment Tree Optimal", solution.Length_Of_LIS_Segment_Tree_Optimal),
        ("Coordinate Compression", solution.Length_Of_LIS_Coordinate_Compression)
    ]
    
    for nums, k, expected in test_cases:
        print(f"Array: {nums}, k: {k}")
        print(f"Expected: {expected}")
        
        if len(nums) <= 8:
            result_bf = solution.Length_Of_LIS_Brute_Force(nums.copy(), k)
            print(f"Brute Force: {result_bf}")
        
        for method_name, method in methods:
            try:
                result = method(nums.copy(), k)
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        length, sequence = solution.Length_Of_LIS_With_Sequence(nums.copy(), k)
        print(f"With Sequence: Length={length}, Sequence={sequence}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Length_Of_LIS()
