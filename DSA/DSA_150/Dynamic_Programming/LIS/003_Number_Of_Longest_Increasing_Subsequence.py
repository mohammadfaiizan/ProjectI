"""
Problem: Number of Longest Increasing Subsequence
URL: https://leetcode.com/problems/number-of-longest-increasing-subsequence/

Problem Statement:
Given an integer array nums, return the number of longest increasing subsequences.
Notice that the sequence has to be strictly increasing.

Sample Input/Output:
Input: nums = [1,3,5,4,7]
Output: 2
Explanation: The two longest increasing subsequences are [1, 3, 4, 7] and [1, 3, 5, 7].

Input: nums = [2,2,2,2,2]
Output: 5
Explanation: The length of longest continuous increasing subsequence is 1, and there are 5 subsequences' length is 1, so output 5.
"""

from typing import List

class Solution:
    def Find_Number_Of_LIS_Brute_Force(self, nums: List[int]) -> int:
        """
        Brute Force - Generate all subsequences and count LIS
        Time Complexity: O(2^n * n)
        Space Complexity: O(n)
        """
        def Generate_All_Subsequences(index: int, current: List[int], all_subseq: List[List[int]]) -> None:
            if index >= len(nums):
                if current:
                    all_subseq.append(current[:])
                return
            
            Generate_All_Subsequences(index + 1, current, all_subseq)
            
            if not current or nums[index] > current[-1]:
                current.append(nums[index])
                Generate_All_Subsequences(index + 1, current, all_subseq)
                current.pop()
        
        all_subsequences = []
        Generate_All_Subsequences(0, [], all_subsequences)
        
        max_length = max(len(subseq) for subseq in all_subsequences) if all_subsequences else 0
        return sum(1 for subseq in all_subsequences if len(subseq) == max_length)
    
    def Find_Number_Of_LIS_DP_Optimal(self, nums: List[int]) -> int:
        """
        DP Optimal - Track length and count simultaneously
        Time Complexity: O(n²)
        Space Complexity: O(n)
        """
        if not nums:
            return 0
        
        n = len(nums)
        lengths = [1] * n
        counts = [1] * n
        
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    if lengths[j] + 1 > lengths[i]:
                        lengths[i] = lengths[j] + 1
                        counts[i] = counts[j]
                    elif lengths[j] + 1 == lengths[i]:
                        counts[i] += counts[j]
        
        max_length = max(lengths)
        return sum(counts[i] for i in range(n) if lengths[i] == max_length)
    
    def Find_Number_Of_LIS_Segment_Tree(self, nums: List[int]) -> int:
        """
        Segment Tree - Coordinate compression with segment tree
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        def Coordinate_Compress(arr: List[int]) -> List[int]:
            sorted_unique = sorted(set(arr))
            coord_map = {v: i for i, v in enumerate(sorted_unique)}
            return [coord_map[x] for x in arr]
        
        class SegmentTree:
            def __init__(self, size: int):
                self.size = size
                self.tree = [(0, 0)] * (4 * size)
            
            def Combine(self, left: tuple, right: tuple) -> tuple:
                length1, count1 = left
                length2, count2 = right
                
                if length1 > length2:
                    return (length1, count1)
                elif length2 > length1:
                    return (length2, count2)
                else:
                    return (length1, count1 + count2)
            
            def Update(self, node: int, start: int, end: int, idx: int, val: tuple) -> None:
                if start == end:
                    self.tree[node] = self.Combine(self.tree[node], val)
                else:
                    mid = (start + end) // 2
                    if idx <= mid:
                        self.Update(2 * node, start, mid, idx, val)
                    else:
                        self.Update(2 * node + 1, mid + 1, end, idx, val)
                    
                    self.tree[node] = self.Combine(self.tree[2 * node], self.tree[2 * node + 1])
            
            def Query(self, node: int, start: int, end: int, l: int, r: int) -> tuple:
                if r < start or end < l:
                    return (0, 1)
                if l <= start and end <= r:
                    return self.tree[node]
                
                mid = (start + end) // 2
                left_result = self.Query(2 * node, start, mid, l, r)
                right_result = self.Query(2 * node + 1, mid + 1, end, l, r)
                
                return self.Combine(left_result, right_result)
        
        compressed = Coordinate_Compress(nums)
        seg_tree = SegmentTree(len(set(compressed)))
        
        for compressed_val in compressed:
            length, count = seg_tree.Query(1, 0, len(set(compressed)) - 1, 0, compressed_val - 1)
            seg_tree.Update(1, 0, len(set(compressed)) - 1, compressed_val, (length + 1, count))
        
        final_length, final_count = seg_tree.Query(1, 0, len(set(compressed)) - 1, 0, len(set(compressed)) - 1)
        return final_count
    
    def Find_Number_Of_LIS_Binary_Indexed_Tree(self, nums: List[int]) -> int:
        """
        Binary Indexed Tree - Use BIT for range queries
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        class BIT:
            def __init__(self, size: int):
                self.size = size
                self.tree = [(0, 0)] * (size + 1)
            
            def Update(self, idx: int, val: tuple) -> None:
                idx += 1
                while idx <= self.size:
                    if self.tree[idx][0] < val[0]:
                        self.tree[idx] = val
                    elif self.tree[idx][0] == val[0]:
                        self.tree[idx] = (val[0], self.tree[idx][1] + val[1])
                    idx += idx & (-idx)
            
            def Query(self, idx: int) -> tuple:
                idx += 1
                result = (0, 1)
                while idx > 0:
                    if result[0] < self.tree[idx][0]:
                        result = self.tree[idx]
                    elif result[0] == self.tree[idx][0]:
                        result = (result[0], result[1] + self.tree[idx][1])
                    idx -= idx & (-idx)
                return result
        
        sorted_nums = sorted(set(nums))
        coord_map = {v: i for i, v in enumerate(sorted_nums)}
        
        bit = BIT(len(sorted_nums))
        
        for num in nums:
            compressed = coord_map[num]
            length, count = bit.Query(compressed - 1) if compressed > 0 else (0, 1)
            bit.Update(compressed, (length + 1, count))
        
        return bit.Query(len(sorted_nums) - 1)[1]
    
    def Find_Number_Of_LIS_Memoized(self, nums: List[int]) -> int:
        """
        Memoized - Use memoization with recursion
        Time Complexity: O(n²)
        Space Complexity: O(n²)
        """
        memo = {}
        
        def LIS_From_Index(index: int, prev_val: int) -> tuple:
            if index >= len(nums):
                return (0, 1)
            
            if (index, prev_val) in memo:
                return memo[(index, prev_val)]
            
            exclude_length, exclude_count = LIS_From_Index(index + 1, prev_val)
            include_length, include_count = 0, 0
            
            if nums[index] > prev_val:
                sub_length, sub_count = LIS_From_Index(index + 1, nums[index])
                include_length = 1 + sub_length
                include_count = sub_count
            
            if include_length > exclude_length:
                result = (include_length, include_count)
            elif exclude_length > include_length:
                result = (exclude_length, exclude_count)
            else:
                result = (exclude_length, exclude_count + include_count)
            
            memo[(index, prev_val)] = result
            return result
        
        length, count = LIS_From_Index(0, float('-inf'))
        return count
    
    def Find_Number_Of_LIS_With_Sequences(self, nums: List[int]) -> tuple:
        """
        With Sequences - Return count and all LIS sequences
        Time Complexity: O(n² * result_count)
        Space Complexity: O(n * result_count)
        """
        if not nums:
            return 0, []
        
        n = len(nums)
        lengths = [1] * n
        counts = [1] * n
        
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    if lengths[j] + 1 > lengths[i]:
                        lengths[i] = lengths[j] + 1
                        counts[i] = counts[j]
                    elif lengths[j] + 1 == lengths[i]:
                        counts[i] += counts[j]
        
        max_length = max(lengths)
        total_count = sum(counts[i] for i in range(n) if lengths[i] == max_length)
        
        def Generate_LIS(index: int, current_length: int, current_seq: List[int], all_lis: List[List[int]]) -> None:
            if current_length == max_length:
                all_lis.append(current_seq[:])
                return
            
            if index >= n:
                return
            
            for i in range(index, n):
                if (not current_seq or nums[i] > current_seq[-1]) and lengths[i] == max_length - current_length:
                    current_seq.append(nums[i])
                    Generate_LIS(i + 1, current_length + 1, current_seq, all_lis)
                    current_seq.pop()
        
        all_lis = []
        if len(nums) <= 10 and max_length <= 6:
            Generate_LIS(0, 0, [], all_lis)
        
        return total_count, all_lis

def Test_Find_Number_Of_LIS():
    solution = Solution()
    
    test_cases = [
        ([1,3,5,4,7], 2),
        ([2,2,2,2,2], 5),
        ([1,2,4,3,5,4,7,2], 3),
        ([1], 1),
        ([1,3,2], 2)
    ]
    
    methods = [
        ("DP Optimal", solution.Find_Number_Of_LIS_DP_Optimal),
        ("Memoized", solution.Find_Number_Of_LIS_Memoized)
    ]
    
    for nums, expected in test_cases:
        print(f"Array: {nums}")
        print(f"Expected: {expected}")
        
        if len(nums) <= 8:
            result_bf = solution.Find_Number_Of_LIS_Brute_Force(nums.copy())
            print(f"Brute Force: {result_bf}")
        
        for method_name, method in methods:
            try:
                result = method(nums.copy())
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        if len(nums) <= 8:
            count, sequences = solution.Find_Number_Of_LIS_With_Sequences(nums.copy())
            print(f"With Sequences: Count={count}")
            for seq in sequences:
                print(f"  {seq}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Find_Number_Of_LIS()
