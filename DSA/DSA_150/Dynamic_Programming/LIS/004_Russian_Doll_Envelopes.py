"""
Problem: Russian Doll Envelopes
URL: https://leetcode.com/problems/russian-doll-envelopes/

Problem Statement:
You are given a 2D array of integers envelopes where envelopes[i] = [wi, hi] represents the width and the height of an envelope.
One envelope can fit into another if and only if both the width and height of one envelope are greater than the other envelope's width and height.
What is the maximum number of envelopes you can Russian doll? (put one inside other)

Sample Input/Output:
Input: envelopes = [[5,4],[6,4],[6,7],[2,3]]
Output: 3
Explanation: The maximum number of envelopes you can Russian doll is 3 ([2,3] => [5,4] => [6,7]).

Input: envelopes = [[1,1],[1,1],[1,1]]
Output: 1
"""

from typing import List
import bisect

class Solution:
    def Max_Envelopes_Brute_Force(self, envelopes: List[List[int]]) -> int:
        """
        Brute Force - Try all possible arrangements
        Time Complexity: O(2^n * n)
        Space Complexity: O(n)
        """
        def Can_Fit(env1: List[int], env2: List[int]) -> bool:
            return env1[0] < env2[0] and env1[1] < env2[1]
        
        def Max_Dolls(index: int, current_stack: List[List[int]]) -> int:
            if index >= len(envelopes):
                return len(current_stack)
            
            exclude = Max_Dolls(index + 1, current_stack)
            include = 0
            
            if not current_stack or Can_Fit(current_stack[-1], envelopes[index]):
                current_stack.append(envelopes[index])
                include = Max_Dolls(index + 1, current_stack)
                current_stack.pop()
            
            return max(include, exclude)
        
        return Max_Dolls(0, [])
    
    def Max_Envelopes_DP_2D(self, envelopes: List[List[int]]) -> int:
        """
        2D DP - Sort and apply LIS logic
        Time Complexity: O(n²)
        Space Complexity: O(n)
        """
        if not envelopes:
            return 0
        
        envelopes.sort(key=lambda x: (x[0], x[1]))
        n = len(envelopes)
        dp = [1] * n
        
        for i in range(1, n):
            for j in range(i):
                if (envelopes[j][0] < envelopes[i][0] and 
                    envelopes[j][1] < envelopes[i][1]):
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)
    
    def Max_Envelopes_Binary_Search_Optimal(self, envelopes: List[List[int]]) -> int:
        """
        Binary Search Optimal - Sort by width, LIS on height
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        if not envelopes:
            return 0
        
        envelopes.sort(key=lambda x: (x[0], -x[1]))
        
        tails = []
        
        for _, height in envelopes:
            pos = bisect.bisect_left(tails, height)
            
            if pos == len(tails):
                tails.append(height)
            else:
                tails[pos] = height
        
        return len(tails)
    
    def Max_Envelopes_Segment_Tree(self, envelopes: List[List[int]]) -> int:
        """
        Segment Tree - Use segment tree for range maximum queries
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        if not envelopes:
            return 0
        
        envelopes.sort()
        
        heights = sorted(set(env[1] for env in envelopes))
        height_map = {h: i for i, h in enumerate(heights)}
        
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
                if r < start or end < l:
                    return 0
                if l <= start and end <= r:
                    return self.tree[node]
                
                mid = (start + end) // 2
                return max(self.Query(2 * node, start, mid, l, r),
                          self.Query(2 * node + 1, mid + 1, end, l, r))
        
        seg_tree = SegmentTree(len(heights))
        max_dolls = 0
        
        for width, height in envelopes:
            height_idx = height_map[height]
            
            if height_idx > 0:
                max_prev = seg_tree.Query(1, 0, len(heights) - 1, 0, height_idx - 1)
            else:
                max_prev = 0
            
            current_max = max_prev + 1
            seg_tree.Update(1, 0, len(heights) - 1, height_idx, current_max)
            max_dolls = max(max_dolls, current_max)
        
        return max_dolls
    
    def Max_Envelopes_Coordinate_Compression(self, envelopes: List[List[int]]) -> int:
        """
        Coordinate Compression - Compress coordinates and use DP
        Time Complexity: O(n²)
        Space Complexity: O(n)
        """
        if not envelopes:
            return 0
        
        widths = sorted(set(env[0] for env in envelopes))
        heights = sorted(set(env[1] for env in envelopes))
        
        width_map = {w: i for i, w in enumerate(widths)}
        height_map = {h: i for i, h in enumerate(heights)}
        
        compressed = [(width_map[w], height_map[h]) for w, h in envelopes]
        compressed.sort()
        
        n = len(compressed)
        dp = [1] * n
        
        for i in range(1, n):
            for j in range(i):
                if (compressed[j][0] < compressed[i][0] and 
                    compressed[j][1] < compressed[i][1]):
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp) if dp else 0
    
    def Max_Envelopes_With_Sequence(self, envelopes: List[List[int]]) -> tuple:
        """
        With Sequence - Return max count and actual sequence
        Time Complexity: O(n²)
        Space Complexity: O(n)
        """
        if not envelopes:
            return 0, []
        
        envelopes.sort(key=lambda x: (x[0], x[1]))
        n = len(envelopes)
        dp = [1] * n
        parent = [-1] * n
        
        for i in range(1, n):
            for j in range(i):
                if (envelopes[j][0] < envelopes[i][0] and 
                    envelopes[j][1] < envelopes[i][1]):
                    if dp[j] + 1 > dp[i]:
                        dp[i] = dp[j] + 1
                        parent[i] = j
        
        max_count = max(dp)
        max_index = dp.index(max_count)
        
        sequence = []
        current = max_index
        
        while current != -1:
            sequence.append(envelopes[current])
            current = parent[current]
        
        return max_count, sequence[::-1]

def Test_Max_Envelopes():
    solution = Solution()
    
    test_cases = [
        ([[5,4],[6,4],[6,7],[2,3]], 3),
        ([[1,1],[1,1],[1,1]], 1),
        ([[4,5],[4,6],[6,7],[2,3],[1,1]], 4),
        ([[1,3],[3,5],[6,7],[6,8],[8,4],[9,5]], 3),
        ([[2,100],[3,200],[4,300],[5,500],[5,400],[5,250],[6,370],[6,360],[7,380]], 5)
    ]
    
    methods = [
        ("2D DP", solution.Max_Envelopes_DP_2D),
        ("Binary Search Optimal", solution.Max_Envelopes_Binary_Search_Optimal),
        ("Segment Tree", solution.Max_Envelopes_Segment_Tree),
        ("Coordinate Compression", solution.Max_Envelopes_Coordinate_Compression)
    ]
    
    for envelopes, expected in test_cases:
        print(f"Envelopes: {envelopes}")
        print(f"Expected: {expected}")
        
        if len(envelopes) <= 8:
            result_bf = solution.Max_Envelopes_Brute_Force(envelopes.copy())
            print(f"Brute Force: {result_bf}")
        
        for method_name, method in methods:
            try:
                result = method([env.copy() for env in envelopes])
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        max_count, sequence = solution.Max_Envelopes_With_Sequence([env.copy() for env in envelopes])
        print(f"With Sequence: Count={max_count}, Sequence={sequence}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Max_Envelopes()
