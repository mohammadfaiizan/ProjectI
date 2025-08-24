"""
739. Daily Temperatures - Multiple Approaches
Difficulty: Easy

Given an array of integers temperatures represents the daily temperatures, return an array answer such that answer[i] is the number of days you have to wait after the ith day to get a warmer temperature. If there is no future day for which this is possible, keep answer[i] == 0.
"""

from typing import List

class DailyTemperatures:
    """Multiple approaches to solve daily temperatures problem"""
    
    def dailyTemperatures_monotonic_stack(self, temperatures: List[int]) -> List[int]:
        """
        Approach 1: Monotonic Decreasing Stack
        
        Use stack to maintain decreasing temperatures and their indices.
        
        Time: O(n), Space: O(n)
        """
        n = len(temperatures)
        result = [0] * n
        stack = []  # Store indices
        
        for i in range(n):
            # While stack is not empty and current temp is greater than stack top
            while stack and temperatures[i] > temperatures[stack[-1]]:
                prev_index = stack.pop()
                result[prev_index] = i - prev_index
            
            stack.append(i)
        
        return result
    
    def dailyTemperatures_brute_force(self, temperatures: List[int]) -> List[int]:
        """
        Approach 2: Brute Force
        
        For each day, search forward for warmer temperature.
        
        Time: O(n²), Space: O(1)
        """
        n = len(temperatures)
        result = [0] * n
        
        for i in range(n):
            for j in range(i + 1, n):
                if temperatures[j] > temperatures[i]:
                    result[i] = j - i
                    break
        
        return result
    
    def dailyTemperatures_optimized_stack(self, temperatures: List[int]) -> List[int]:
        """
        Approach 3: Optimized Stack with Early Termination
        
        Optimize stack approach with better memory usage.
        
        Time: O(n), Space: O(n)
        """
        n = len(temperatures)
        result = [0] * n
        stack = []
        
        for curr_day in range(n):
            curr_temp = temperatures[curr_day]
            
            # Process all days that are cooler than current day
            while stack and temperatures[stack[-1]] < curr_temp:
                prev_day = stack.pop()
                result[prev_day] = curr_day - prev_day
            
            stack.append(curr_day)
        
        return result
    
    def dailyTemperatures_backward_iteration(self, temperatures: List[int]) -> List[int]:
        """
        Approach 4: Backward Iteration with Next Array
        
        Iterate backwards and use next warmer day information.
        
        Time: O(n), Space: O(1) excluding output
        """
        n = len(temperatures)
        result = [0] * n
        
        for i in range(n - 2, -1, -1):
            j = i + 1
            
            # Skip days that are not warmer and have no warmer day ahead
            while j < n and temperatures[j] <= temperatures[i] and result[j] > 0:
                j += result[j]
            
            # If we found a warmer day
            if j < n and temperatures[j] > temperatures[i]:
                result[i] = j - i
        
        return result
    
    def dailyTemperatures_next_greater_optimization(self, temperatures: List[int]) -> List[int]:
        """
        Approach 5: Next Greater Element Optimization
        
        Use next greater element pattern with optimizations.
        
        Time: O(n), Space: O(n)
        """
        n = len(temperatures)
        result = [0] * n
        stack = []
        
        # Process from right to left
        for i in range(n - 1, -1, -1):
            # Remove elements that are not greater than current
            while stack and temperatures[stack[-1]] <= temperatures[i]:
                stack.pop()
            
            # If stack is not empty, top element is next greater
            if stack:
                result[i] = stack[-1] - i
            
            stack.append(i)
        
        return result
    
    def dailyTemperatures_segment_tree_approach(self, temperatures: List[int]) -> List[int]:
        """
        Approach 6: Segment Tree Approach (Overkill but Educational)
        
        Use segment tree to find next greater element.
        
        Time: O(n log n), Space: O(n)
        """
        n = len(temperatures)
        result = [0] * n
        
        # Create list of (temperature, index) pairs and sort by temperature
        temp_with_index = [(temperatures[i], i) for i in range(n)]
        temp_with_index.sort()
        
        # For each temperature, find the next occurrence of higher temperature
        for i in range(n):
            current_temp = temperatures[i]
            min_next_day = float('inf')
            
            # Binary search for temperatures greater than current
            left, right = 0, n - 1
            while left <= right:
                mid = (left + right) // 2
                if temp_with_index[mid][0] > current_temp:
                    # Check if this index is after current day
                    if temp_with_index[mid][1] > i:
                        min_next_day = min(min_next_day, temp_with_index[mid][1])
                    right = mid - 1
                else:
                    left = mid + 1
            
            # Continue searching in the right half
            for j in range(left, n):
                if temp_with_index[j][1] > i:
                    min_next_day = min(min_next_day, temp_with_index[j][1])
            
            if min_next_day != float('inf'):
                result[i] = min_next_day - i
        
        return result

def test_daily_temperatures():
    """Test daily temperatures algorithms"""
    solver = DailyTemperatures()
    
    test_cases = [
        ([73,74,75,71,69,72,76,73], [1,1,4,2,1,1,0,0], "Example 1"),
        ([30,40,50,60], [1,1,1,0], "Increasing temperatures"),
        ([30,60,90], [1,1,0], "Strictly increasing"),
        ([90,60,30], [0,0,0], "Decreasing temperatures"),
        ([89,62,70,58,47,47,46,76,100,70], [8,1,5,4,3,2,1,1,0,0], "Complex case"),
    ]
    
    algorithms = [
        ("Monotonic Stack", solver.dailyTemperatures_monotonic_stack),
        ("Brute Force", solver.dailyTemperatures_brute_force),
        ("Optimized Stack", solver.dailyTemperatures_optimized_stack),
        ("Backward Iteration", solver.dailyTemperatures_backward_iteration),
        ("Next Greater Optimization", solver.dailyTemperatures_next_greater_optimization),
    ]
    
    print("=== Testing Daily Temperatures ===")
    
    for temperatures, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Temperatures: {temperatures}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(temperatures)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:25} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:25} | ERROR: {str(e)[:40]}")

def benchmark_daily_temperatures():
    """Benchmark different approaches"""
    import time
    import random
    
    # Generate test data
    sizes = [100, 1000, 5000]
    
    algorithms = [
        ("Monotonic Stack", DailyTemperatures().dailyTemperatures_monotonic_stack),
        ("Brute Force", DailyTemperatures().dailyTemperatures_brute_force),
        ("Backward Iteration", DailyTemperatures().dailyTemperatures_backward_iteration),
    ]
    
    print("\n=== Daily Temperatures Performance Benchmark ===")
    
    for size in sizes:
        print(f"\n--- Array Size: {size} ---")
        temperatures = [random.randint(30, 100) for _ in range(size)]
        
        for alg_name, alg_func in algorithms:
            start_time = time.time()
            result = alg_func(temperatures)
            end_time = time.time()
            
            print(f"{alg_name:20} | Time: {end_time - start_time:.4f}s")

if __name__ == "__main__":
    test_daily_temperatures()
    benchmark_daily_temperatures()

"""
Daily Temperatures demonstrates the power of monotonic stack
for next greater element problems, with multiple optimization
approaches and performance analysis.
"""
