"""
Specialized Sorting Techniques
=============================

Topics: Hybrid sorts, topological sort, special sorts
Companies: Google, Facebook, Amazon, Microsoft
Difficulty: Medium to Hard
"""

from typing import List, Dict
from collections import defaultdict, deque

class SpecializedSortingTechniques:
    
    def tim_sort(self, arr: List[int]) -> List[int]:
        """Tim Sort - Python's built-in algorithm
        Time: O(n log n), Space: O(n)
        """
        MIN_MERGE = 32
        
        def insertion_sort(arr, left, right):
            for i in range(left + 1, right + 1):
                key = arr[i]
                j = i - 1
                while j >= left and arr[j] > key:
                    arr[j + 1] = arr[j]
                    j -= 1
                arr[j + 1] = key
        
        def merge(arr, left, mid, right):
            left_part = arr[left:mid + 1]
            right_part = arr[mid + 1:right + 1]
            
            i = j = k = 0
            k = left
            
            while i < len(left_part) and j < len(right_part):
                if left_part[i] <= right_part[j]:
                    arr[k] = left_part[i]
                    i += 1
                else:
                    arr[k] = right_part[j]
                    j += 1
                k += 1
            
            while i < len(left_part):
                arr[k] = left_part[i]
                i += 1
                k += 1
            
            while j < len(right_part):
                arr[k] = right_part[j]
                j += 1
                k += 1
        
        arr = arr.copy()
        n = len(arr)
        
        # Sort small chunks with insertion sort
        for start in range(0, n, MIN_MERGE):
            end = min(start + MIN_MERGE - 1, n - 1)
            insertion_sort(arr, start, end)
        
        # Merge chunks
        size = MIN_MERGE
        while size < n:
            for start in range(0, n, size * 2):
                mid = min(start + size - 1, n - 1)
                end = min(start + size * 2 - 1, n - 1)
                
                if mid < end:
                    merge(arr, start, mid, end)
            
            size *= 2
        
        return arr
    
    def topological_sort_dfs(self, graph: Dict[int, List[int]]) -> List[int]:
        """Topological sort using DFS
        Time: O(V + E), Space: O(V)
        """
        visited = set()
        stack = []
        
        def dfs(node):
            visited.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor)
            stack.append(node)
        
        for node in graph:
            if node not in visited:
                dfs(node)
        
        return stack[::-1]
    
    def topological_sort_kahn(self, graph: Dict[int, List[int]]) -> List[int]:
        """Topological sort using Kahn's algorithm
        Time: O(V + E), Space: O(V)
        """
        in_degree = defaultdict(int)
        all_nodes = set(graph.keys())
        
        for node in graph:
            for neighbor in graph[node]:
                in_degree[neighbor] += 1
                all_nodes.add(neighbor)
        
        queue = deque([node for node in all_nodes if in_degree[node] == 0])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            for neighbor in graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result if len(result) == len(all_nodes) else []
    
    def pancake_sort(self, arr: List[int]) -> List[int]:
        """LC 969: Pancake Sorting
        Time: O(n²), Space: O(1)
        """
        def flip(arr, k):
            arr[:k] = arr[:k][::-1]
        
        def find_max_index(arr, n):
            max_idx = 0
            for i in range(1, n):
                if arr[i] > arr[max_idx]:
                    max_idx = i
            return max_idx
        
        arr = arr.copy()
        n = len(arr)
        
        for curr_size in range(n, 1, -1):
            max_idx = find_max_index(arr, curr_size)
            
            if max_idx != curr_size - 1:
                if max_idx != 0:
                    flip(arr, max_idx + 1)
                flip(arr, curr_size)
        
        return arr
    
    def wiggle_sort(self, nums: List[int]) -> List[int]:
        """LC 280: Wiggle Sort
        Time: O(n), Space: O(1)
        """
        nums = nums.copy()
        
        for i in range(len(nums) - 1):
            if (i % 2 == 0 and nums[i] > nums[i + 1]) or \
               (i % 2 == 1 and nums[i] < nums[i + 1]):
                nums[i], nums[i + 1] = nums[i + 1], nums[i]
        
        return nums
    
    def wiggle_sort_ii(self, nums: List[int]) -> List[int]:
        """LC 324: Wiggle Sort II
        Time: O(n log n), Space: O(n)
        """
        nums_sorted = sorted(nums)
        n = len(nums)
        mid = (n + 1) // 2
        
        result = [0] * n
        
        # Fill odd positions with smaller half (reverse)
        j = mid - 1
        for i in range(1, n, 2):
            result[i] = nums_sorted[j]
            j -= 1
        
        # Fill even positions with larger half (reverse)
        j = n - 1
        for i in range(0, n, 2):
            result[i] = nums_sorted[j]
            j -= 1
        
        return result

# Test Examples
def run_examples():
    sst = SpecializedSortingTechniques()
    
    print("=== SPECIALIZED SORTING TECHNIQUES ===\n")
    
    # Tim Sort
    test_arr = [64, 34, 25, 12, 22, 11, 90, 5]
    print("Tim Sort:", sst.tim_sort(test_arr))
    
    # Topological Sort
    graph = {0: [1, 2], 1: [3], 2: [3], 3: []}
    print("Topological Sort (DFS):", sst.topological_sort_dfs(graph))
    print("Topological Sort (Kahn):", sst.topological_sort_kahn(graph))
    
    # Special sorts
    pancake_arr = [3, 2, 4, 1]
    print("Pancake Sort:", sst.pancake_sort(pancake_arr))
    
    wiggle_arr = [1, 5, 1, 1, 6, 4]
    print("Wiggle Sort:", sst.wiggle_sort(wiggle_arr))

if __name__ == "__main__":
    run_examples() 