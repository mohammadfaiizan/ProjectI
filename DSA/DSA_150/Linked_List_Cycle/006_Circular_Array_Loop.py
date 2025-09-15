"""
Problem: Circular Array Loop
URL: https://leetcode.com/problems/circular-array-loop/

Problem Statement:
You are playing a game involving a circular array of non-zero integers nums. Each nums[i] denotes the number of indices forward/backward you must move if you are located at index i:
- If nums[i] is positive, move nums[i] steps forward, and
- If nums[i] is negative, move nums[i] steps backward.
Since the array is circular, you may assume that moving forward from the last element puts you on the first element, and moving backward from the first element puts you on the last element.
A cycle in the array consists of a sequence of indices seq of length k where:
- Following the movement rules above results in the same sequence seq.
- Every nums[seq[j]] is either all positive or all negative.
- k > 1
Return true if there is a cycle in nums, or false otherwise.

Sample Input/Output:
Input: nums = [2,-1,1,2,2]
Output: true
Explanation: The indices 0 -> 2 -> 3 -> 0 form a cycle.

Input: nums = [-1,-2,-3,-4,-5,6]
Output: false
Explanation: The movement from index 3 -> 3 is not a cycle because the cycle length is 1.
"""

from typing import List

class Solution:
    def Circular_Array_Loop_Floyd(self, nums: List[int]) -> bool:
        """
        Floyd's Cycle Detection - Apply to circular array
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(nums)
        
        def Get_Next_Index(index: int) -> int:
            return (index + nums[index]) % n
        
        for i in range(n):
            if nums[i] == 0:
                continue
            
            slow = fast = i
            
            while (nums[fast] * nums[Get_Next_Index(fast)] > 0 and 
                   nums[Get_Next_Index(fast)] * nums[Get_Next_Index(Get_Next_Index(fast))] > 0):
                
                slow = Get_Next_Index(slow)
                fast = Get_Next_Index(Get_Next_Index(fast))
                
                if slow == fast:
                    if slow == Get_Next_Index(slow):
                        break
                    return True
            
            slow = i
            val = nums[i]
            while nums[slow] * val > 0:
                next_slow = Get_Next_Index(slow)
                nums[slow] = 0
                slow = next_slow
        
        return False
    
    def Circular_Array_Loop_DFS(self, nums: List[int]) -> bool:
        """
        DFS Approach - Use DFS to detect cycles
        Time Complexity: O(n²)
        Space Complexity: O(n)
        """
        n = len(nums)
        visited = [False] * n
        
        def Get_Next_Index(index: int) -> int:
            return (index + nums[index]) % n
        
        def DFS(start: int, current: int, direction: int) -> bool:
            if visited[current]:
                return current == start
            
            if nums[current] * direction <= 0:
                return False
            
            next_index = Get_Next_Index(current)
            
            if next_index == current:
                return False
            
            visited[current] = True
            result = DFS(start, next_index, direction)
            visited[current] = False
            
            return result
        
        for i in range(n):
            if DFS(i, i, nums[i]):
                return True
        
        return False
    
    def Circular_Array_Loop_Marking(self, nums: List[int]) -> bool:
        """
        Marking Approach - Mark visited indices
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(nums)
        
        def Get_Next_Index(index: int) -> int:
            return (index + nums[index]) % n
        
        for i in range(n):
            if nums[i] == 0:
                continue
            
            slow = fast = i
            direction = nums[i] > 0
            
            while True:
                slow = Get_Next_Index(slow)
                
                if nums[slow] == 0 or (nums[slow] > 0) != direction:
                    break
                
                fast = Get_Next_Index(fast)
                if nums[fast] == 0 or (nums[fast] > 0) != direction:
                    break
                
                fast = Get_Next_Index(fast)
                if nums[fast] == 0 or (nums[fast] > 0) != direction:
                    break
                
                if slow == fast:
                    if slow == Get_Next_Index(slow):
                        break
                    return True
            
            slow = i
            val = nums[i]
            while nums[slow] * val > 0:
                next_slow = Get_Next_Index(slow)
                nums[slow] = 0
                slow = next_slow
        
        return False
    
    def Circular_Array_Loop_Path_Compression(self, nums: List[int]) -> bool:
        """
        Path Compression - Optimize with path compression
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(nums)
        
        def Get_Next_Index(index: int) -> int:
            return (index + nums[index]) % n
        
        def Same_Direction(i: int, j: int) -> bool:
            return (nums[i] > 0) == (nums[j] > 0)
        
        for i in range(n):
            if nums[i] == 0:
                continue
            
            slow = fast = i
            
            while (Same_Direction(fast, Get_Next_Index(fast)) and 
                   Same_Direction(Get_Next_Index(fast), Get_Next_Index(Get_Next_Index(fast)))):
                
                slow = Get_Next_Index(slow)
                fast = Get_Next_Index(Get_Next_Index(fast))
                
                if slow == fast:
                    if slow == Get_Next_Index(slow):
                        break
                    return True
            
            slow = i
            while Same_Direction(slow, Get_Next_Index(slow)):
                next_slow = Get_Next_Index(slow)
                nums[slow] = 0
                slow = next_slow
        
        return False
    
    def Circular_Array_Loop_Simulation(self, nums: List[int]) -> bool:
        """
        Simulation - Simulate movement and detect cycles
        Time Complexity: O(n²)
        Space Complexity: O(n)
        """
        n = len(nums)
        
        def Get_Next_Index(index: int) -> int:
            return (index + nums[index]) % n
        
        for start in range(n):
            if nums[start] == 0:
                continue
            
            visited = set()
            current = start
            direction = nums[start] > 0
            
            while current not in visited:
                if nums[current] == 0 or (nums[current] > 0) != direction:
                    break
                
                visited.add(current)
                next_index = Get_Next_Index(current)
                
                if next_index == current:
                    break
                
                current = next_index
            
            if current in visited and len(visited) > 1:
                return True
        
        return False

def Test_Circular_Array_Loop():
    solution = Solution()
    
    test_cases = [
        ([2,-1,1,2,2], True),
        ([-1,-2,-3,-4,-5,6], False),
        ([1,-1,5,1,4], True),
        ([-2,1,-1,-2,-2], False),
        ([2,2,2,2,2], True)
    ]
    
    methods = [
        ("Floyd", solution.Circular_Array_Loop_Floyd),
        ("DFS", solution.Circular_Array_Loop_DFS),
        ("Marking", solution.Circular_Array_Loop_Marking),
        ("Path Compression", solution.Circular_Array_Loop_Path_Compression),
        ("Simulation", solution.Circular_Array_Loop_Simulation)
    ]
    
    for nums, expected in test_cases:
        print(f"Array: {nums}")
        print(f"Expected: {expected}")
        
        for method_name, method in methods:
            try:
                result = method(nums.copy())
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Circular_Array_Loop()
