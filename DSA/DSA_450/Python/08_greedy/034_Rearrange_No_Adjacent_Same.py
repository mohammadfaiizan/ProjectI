"""
Problem: Rearrange Characters So No Two Adjacent Are Same
URL: https://www.geeksforgeeks.org/rearrange-characters-string-no-two-adjacent/

Problem Statement:
Rearrange characters in a string so no two adjacent characters are same. Return if possible and the rearranged string.

Sample Input/Output:
Input: "aabb"
Output: "abab"
Explanation: Max-heap frequency based approach or count check approach.
"""

import heapq
from collections import Counter


class Solution:
    def Rearrange_Max_Heap(self, s):
        """
        Max-heap frequency based approach
        Time Complexity: O(n log k) where k is distinct chars
        Space Complexity: O(n)
        """
        freq = Counter(s)
        
        max_heap = [(-freq[char], char) for char in freq]
        heapq.heapify(max_heap)
        
        result = ""
        prev = (-1, '#')
        
        while max_heap or prev[0] < 0:
            if prev[0] < 0 and not max_heap:
                return ""
            
            curr = heapq.heappop(max_heap)
            
            result += curr[1]
            curr = (curr[0] + 1, curr[1])
            
            if prev[0] < 0:
                heapq.heappush(max_heap, prev)
            
            prev = curr
        
        return result
    
    def Rearrange_Count_Check(self, s):
        """
        Count check approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        freq = Counter(s)
        max_freq = max(freq.values())
        
        if max_freq > (len(s) + 1) // 2:
            return ""
        
        return self.Rearrange_Max_Heap(s)


def Test_Rearrange_No_Adjacent_Same():
    solution = Solution()
    
    print(f"Test 1 (Max-Heap): {solution.Rearrange_Max_Heap('aabb')}")
    print(f"Test 1 (Count Check): {solution.Rearrange_Count_Check('aabb')}")
    
    print(f"Test 2 (Max-Heap): {solution.Rearrange_Max_Heap('aaabc')}")
    print(f"Test 2 (Count Check): {solution.Rearrange_Count_Check('aaabc')}")
    
    print(f"Test 3 (Max-Heap): {solution.Rearrange_Max_Heap('aaa')}")
    print(f"Test 3 (Count Check): {solution.Rearrange_Count_Check('aaa')}")


if __name__ == "__main__":
    Test_Rearrange_No_Adjacent_Same()
