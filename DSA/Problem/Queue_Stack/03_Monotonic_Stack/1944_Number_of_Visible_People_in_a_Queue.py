"""
1944. Number of Visible People in a Queue - Multiple Approaches
Difficulty: Hard

There are n people standing in a queue, and an array heights of length n represents the heights of the people from front to back.

A person can see another person in front of them if all the people in between are shorter than both the person and the person they want to see.

Return an array answer of length n where answer[i] is the number of people the ith person can see to their right in the queue.
"""

from typing import List

class NumberOfVisiblePeopleInQueue:
    """Multiple approaches to count visible people in queue"""
    
    def canSeePersonsCount_stack_approach(self, heights: List[int]) -> List[int]:
        """
        Approach 1: Monotonic Stack (Optimal)
        
        Use decreasing stack to track visible people.
        
        Time: O(n), Space: O(n)
        """
        n = len(heights)
        result = [0] * n
        stack = []  # Store indices in decreasing order of heights
        
        # Process from right to left
        for i in range(n - 1, -1, -1):
            count = 0
            
            # Pop people shorter than current person
            while stack and heights[stack[-1]] < heights[i]:
                stack.pop()
                count += 1
            
            # If stack is not empty, current person can see the top person
            if stack:
                count += 1
            
            result[i] = count
            stack.append(i)
        
        return result
    
    def canSeePersonsCount_brute_force(self, heights: List[int]) -> List[int]:
        """
        Approach 2: Brute Force
        
        For each person, scan to the right and count visible people.
        
        Time: O(n²), Space: O(1)
        """
        n = len(heights)
        result = [0] * n
        
        for i in range(n):
            max_height_seen = 0
            
            for j in range(i + 1, n):
                # Can see person j if they're taller than all people in between
                if heights[j] > max_height_seen:
                    result[i] += 1
                    max_height_seen = heights[j]
                    
                    # If person j is taller than person i, can't see beyond
                    if heights[j] >= heights[i]:
                        break
        
        return result


def test_number_of_visible_people():
    """Test number of visible people algorithms"""
    solver = NumberOfVisiblePeopleInQueue()
    
    test_cases = [
        ([10,6,8,5,11,9], [3,1,2,1,1,0], "Example 1"),
        ([5,1,2,3,10], [4,1,1,1,0], "Example 2"),
        ([1,2,3,4,5], [1,1,1,1,0], "Increasing heights"),
        ([5,4,3,2,1], [4,3,2,1,0], "Decreasing heights"),
        ([1], [0], "Single person"),
        ([5,5,5], [1,1,0], "All same heights"),
    ]
    
    algorithms = [
        ("Stack Approach", solver.canSeePersonsCount_stack_approach),
        ("Brute Force", solver.canSeePersonsCount_brute_force),
    ]
    
    print("=== Testing Number of Visible People in Queue ===")
    
    for heights, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Heights: {heights}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(heights)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


if __name__ == "__main__":
    test_number_of_visible_people()

"""
Number of Visible People in a Queue demonstrates advanced monotonic stack
applications for visibility problems with geometric line-of-sight calculations.
"""
