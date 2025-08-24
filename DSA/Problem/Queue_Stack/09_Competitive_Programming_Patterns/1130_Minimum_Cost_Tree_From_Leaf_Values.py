"""
1130. Minimum Cost Tree From Leaf Values - Multiple Approaches
Difficulty: Medium

Given an array arr of positive integers, consider all binary trees such that:
- Each node has either 0 or 2 children;
- The values of arr correspond to the values of each leaf in an in-order traversal of the tree.
- The value of each non-leaf node is equal to the product of the largest leaf value in its left and right subtree respectively.

Among all possible binary trees considered, return the smallest possible sum of values of all non-leaf nodes. It is guaranteed this sum fits in a 32-bit integer.
"""

from typing import List

class MinimumCostTreeFromLeafValues:
    """Multiple approaches to find minimum cost tree from leaf values"""
    
    def mctFromLeafValues_stack_monotonic(self, arr: List[int]) -> int:
        """
        Approach 1: Monotonic Stack (Optimal)
        
        Use monotonic decreasing stack to find optimal merging order.
        
        Time: O(n), Space: O(n)
        """
        stack = []
        result = 0
        
        for num in arr:
            # While stack is not empty and top is <= current number
            while stack and stack[-1] <= num:
                # Remove smaller element and add cost
                mid = stack.pop()
                
                # Cost is mid * min(left_neighbor, right_neighbor)
                if stack:
                    # Has left neighbor
                    result += mid * min(stack[-1], num)
                else:
                    # No left neighbor, only right
                    result += mid * num
            
            stack.append(num)
        
        # Process remaining elements in stack
        while len(stack) > 1:
            result += stack.pop() * stack[-1]
        
        return result
    
    def mctFromLeafValues_dp_interval(self, arr: List[int]) -> int:
        """
        Approach 2: Dynamic Programming (Interval DP)
        
        Use interval DP to find optimal cost for each subarray.
        
        Time: O(n³), Space: O(n²)
        """
        n = len(arr)
        
        # dp[i][j] = minimum cost for subarray arr[i:j+1]
        dp = [[0] * n for _ in range(n)]
        
        # max_val[i][j] = maximum value in subarray arr[i:j+1]
        max_val = [[0] * n for _ in range(n)]
        
        # Initialize max values
        for i in range(n):
            max_val[i][i] = arr[i]
            for j in range(i + 1, n):
                max_val[i][j] = max(max_val[i][j-1], arr[j])
        
        # Fill DP table
        for length in range(2, n + 1):  # length of subarray
            for i in range(n - length + 1):
                j = i + length - 1
                dp[i][j] = float('inf')
                
                # Try all possible split points
                for k in range(i, j):
                    cost = dp[i][k] + dp[k+1][j] + max_val[i][k] * max_val[k+1][j]
                    dp[i][j] = min(dp[i][j], cost)
        
        return dp[0][n-1]
    
    def mctFromLeafValues_recursive_memo(self, arr: List[int]) -> int:
        """
        Approach 3: Recursive with Memoization
        
        Use recursion with memoization to find optimal cost.
        
        Time: O(n³), Space: O(n²)
        """
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def dp(i: int, j: int) -> int:
            if i == j:
                return 0
            
            result = float('inf')
            max_left = max_right = 0
            
            for k in range(i, j):
                # Calculate max values for left and right parts
                max_left = max(arr[i:k+1])
                max_right = max(arr[k+1:j+1])
                
                cost = dp(i, k) + dp(k+1, j) + max_left * max_right
                result = min(result, cost)
            
            return result
        
        return dp(0, len(arr) - 1)
    
    def mctFromLeafValues_greedy(self, arr: List[int]) -> int:
        """
        Approach 4: Greedy Approach
        
        Greedily remove the smallest element that has a smaller neighbor.
        
        Time: O(n²), Space: O(1)
        """
        result = 0
        
        while len(arr) > 1:
            # Find the smallest element with a smaller neighbor
            min_idx = -1
            min_val = float('inf')
            
            for i in range(len(arr)):
                if ((i > 0 and arr[i-1] < arr[i]) or 
                    (i < len(arr) - 1 and arr[i+1] < arr[i])):
                    if arr[i] < min_val:
                        min_val = arr[i]
                        min_idx = i
            
            # If no such element found, remove the smallest overall
            if min_idx == -1:
                min_idx = arr.index(min(arr))
            
            # Calculate cost and remove element
            left = arr[min_idx - 1] if min_idx > 0 else float('inf')
            right = arr[min_idx + 1] if min_idx < len(arr) - 1 else float('inf')
            
            result += arr[min_idx] * min(left, right)
            arr.pop(min_idx)
        
        return result
    
    def mctFromLeafValues_divide_conquer(self, arr: List[int]) -> int:
        """
        Approach 5: Divide and Conquer
        
        Use divide and conquer to split array optimally.
        
        Time: O(n³), Space: O(n) due to recursion
        """
        def solve(start: int, end: int) -> tuple:
            """Returns (min_cost, max_value) for subarray arr[start:end+1]"""
            if start == end:
                return 0, arr[start]
            
            min_cost = float('inf')
            max_val = max(arr[start:end+1])
            
            for k in range(start, end):
                left_cost, left_max = solve(start, k)
                right_cost, right_max = solve(k + 1, end)
                
                total_cost = left_cost + right_cost + left_max * right_max
                min_cost = min(min_cost, total_cost)
            
            return min_cost, max_val
        
        cost, _ = solve(0, len(arr) - 1)
        return cost


def test_minimum_cost_tree():
    """Test minimum cost tree algorithms"""
    solver = MinimumCostTreeFromLeafValues()
    
    test_cases = [
        ([6,2,4], 32, "Example 1"),
        ([4,11], 44, "Example 2"),
        ([1,2,3,4], 20, "Increasing sequence"),
        ([4,3,2,1], 20, "Decreasing sequence"),
        ([1], 0, "Single element"),
        ([2,3], 6, "Two elements"),
        ([1,2,1], 4, "Valley pattern"),
        ([3,1,2], 5, "Mixed pattern"),
        ([6,2,4,1], 40, "Complex case"),
        ([15,13,5,3,15], 500, "Larger example"),
        ([1,3,2,4], 14, "Another mixed"),
        ([2,1,4,3], 16, "Alternating"),
    ]
    
    algorithms = [
        ("Monotonic Stack", solver.mctFromLeafValues_stack_monotonic),
        ("Interval DP", solver.mctFromLeafValues_dp_interval),
        ("Recursive Memo", solver.mctFromLeafValues_recursive_memo),
        ("Greedy", solver.mctFromLeafValues_greedy),
        ("Divide Conquer", solver.mctFromLeafValues_divide_conquer),
    ]
    
    print("=== Testing Minimum Cost Tree From Leaf Values ===")
    
    for arr, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Input: {arr}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(arr[:])  # Copy to avoid modification
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_monotonic_stack_approach():
    """Demonstrate monotonic stack approach step by step"""
    print("\n=== Monotonic Stack Approach Step-by-Step Demo ===")
    
    arr = [6, 2, 4]
    
    print(f"Input: {arr}")
    print("Strategy: Use monotonic decreasing stack to find optimal merging")
    print("Key insight: Smaller elements should be merged first with their smaller neighbors")
    
    stack = []
    result = 0
    
    print(f"\nStep-by-step processing:")
    
    for i, num in enumerate(arr):
        print(f"\nStep {i+1}: Processing {num}")
        print(f"  Current stack: {stack}")
        print(f"  Current result: {result}")
        
        # Process elements in stack that are <= current number
        while stack and stack[-1] <= num:
            mid = stack.pop()
            print(f"    Removing {mid} from stack")
            
            if stack:
                cost = mid * min(stack[-1], num)
                print(f"    Cost: {mid} * min({stack[-1]}, {num}) = {cost}")
            else:
                cost = mid * num
                print(f"    Cost: {mid} * {num} = {cost}")
            
            result += cost
            print(f"    New result: {result}")
        
        stack.append(num)
        print(f"  Stack after adding {num}: {stack}")
    
    # Process remaining elements
    print(f"\nProcessing remaining elements in stack: {stack}")
    while len(stack) > 1:
        cost = stack.pop() * stack[-1]
        print(f"  Cost: {cost}")
        result += cost
    
    print(f"\nFinal result: {result}")


def visualize_tree_construction():
    """Visualize tree construction process"""
    print("\n=== Tree Construction Visualization ===")
    
    arr = [6, 2, 4]
    
    print(f"Input array: {arr}")
    print("Constructing optimal binary tree:")
    
    print(f"\nStep 1: Merge 2 and 4 (smallest adjacent pair)")
    print("  Cost: 2 * 4 = 8")
    print("  Tree: (2, 4) with internal node value 8")
    print("  Remaining: [6, 8]")
    
    print(f"\nStep 2: Merge 6 and 8")
    print("  Cost: 6 * 8 = 48")
    print("  Tree: (6, (2, 4)) with root value 48")
    
    print(f"\nTotal cost: 8 + 48 = 56")
    print("Wait, this doesn't match optimal...")
    
    print(f"\nOptimal construction:")
    print("Step 1: Merge 6 and 2")
    print("  Cost: 6 * 2 = 12")
    print("  Remaining: [12, 4]")
    
    print("Step 2: Merge 12 and 4")
    print("  Cost: 12 * 4 = 48")
    print("  Total: 12 + 48 = 60")
    
    print("Actually, let's check the monotonic stack result...")


def demonstrate_competitive_programming_patterns():
    """Demonstrate competitive programming patterns"""
    print("\n=== Competitive Programming Patterns ===")
    
    solver = MinimumCostTreeFromLeafValues()
    
    # Pattern 1: Monotonic stack for optimization
    print("1. Monotonic Stack Optimization:")
    print("   Use monotonic stack to find optimal merging order")
    print("   Key insight: merge smaller elements with their smaller neighbors first")
    
    example1 = [6, 2, 4]
    result1 = solver.mctFromLeafValues_stack_monotonic(example1)
    print(f"   {example1} -> {result1}")
    
    # Pattern 2: Interval DP
    print(f"\n2. Interval Dynamic Programming:")
    print("   dp[i][j] = minimum cost for subarray arr[i:j+1]")
    print("   Try all possible split points for each interval")
    
    # Pattern 3: Greedy approach
    print(f"\n3. Greedy Strategy:")
    print("   Always remove the smallest element with a smaller neighbor")
    print("   Minimizes the cost at each step")
    
    # Pattern 4: Problem transformation
    print(f"\n4. Problem Transformation:")
    print("   Transform tree construction to optimal parenthesization")
    print("   Similar to matrix chain multiplication")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Monotonic Stack", "O(n)", "O(n)", "Each element pushed/popped once"),
        ("Interval DP", "O(n³)", "O(n²)", "3 nested loops, 2D DP table"),
        ("Recursive Memo", "O(n³)", "O(n²)", "Memoized recursion"),
        ("Greedy", "O(n²)", "O(1)", "Remove elements one by one"),
        ("Divide Conquer", "O(n³)", "O(n)", "Recursive splitting"),
    ]
    
    print(f"{'Approach':<20} | {'Time':<8} | {'Space':<8} | {'Notes'}")
    print("-" * 65)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<20} | {time_comp:<8} | {space_comp:<8} | {notes}")
    
    print(f"\nMonotonic Stack is optimal for competitive programming")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = MinimumCostTreeFromLeafValues()
    
    edge_cases = [
        ([1], 0, "Single element"),
        ([1, 2], 2, "Two elements"),
        ([2, 1], 2, "Two elements reverse"),
        ([1, 1, 1], 2, "All same elements"),
        ([1, 2, 3], 4, "Increasing sequence"),
        ([3, 2, 1], 4, "Decreasing sequence"),
        ([1, 3, 1], 6, "Peak in middle"),
        ([3, 1, 3], 6, "Valley in middle"),
        ([1, 2, 1, 2], 8, "Alternating pattern"),
        ([5, 4, 3, 2, 1], 40, "Long decreasing"),
        ([1, 2, 3, 4, 5], 40, "Long increasing"),
    ]
    
    for arr, expected, description in edge_cases:
        try:
            result = solver.mctFromLeafValues_stack_monotonic(arr[:])
            status = "✓" if result == expected else "✗"
            print(f"{description:20} | {status} | {arr} -> {result}")
        except Exception as e:
            print(f"{description:20} | ERROR: {str(e)[:30]}")


def demonstrate_dp_approach():
    """Demonstrate DP approach"""
    print("\n=== Dynamic Programming Approach Demo ===")
    
    arr = [6, 2, 4]
    n = len(arr)
    
    print(f"Input: {arr}")
    print("DP approach: dp[i][j] = minimum cost for subarray arr[i:j+1]")
    
    # Initialize DP table
    dp = [[0] * n for _ in range(n)]
    max_val = [[0] * n for _ in range(n)]
    
    # Fill max values
    for i in range(n):
        max_val[i][i] = arr[i]
        for j in range(i + 1, n):
            max_val[i][j] = max(max_val[i][j-1], arr[j])
    
    print(f"\nMax value table:")
    for i in range(n):
        print(f"  {max_val[i]}")
    
    # Fill DP table
    print(f"\nFilling DP table:")
    for length in range(2, n + 1):
        print(f"\nLength {length}:")
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            
            print(f"  dp[{i}][{j}] (subarray {arr[i:j+1]}):")
            
            for k in range(i, j):
                left_cost = dp[i][k]
                right_cost = dp[k+1][j]
                merge_cost = max_val[i][k] * max_val[k+1][j]
                total_cost = left_cost + right_cost + merge_cost
                
                print(f"    Split at {k}: {left_cost} + {right_cost} + {merge_cost} = {total_cost}")
                dp[i][j] = min(dp[i][j], total_cost)
            
            print(f"    Result: {dp[i][j]}")
    
    print(f"\nFinal DP table:")
    for i in range(n):
        print(f"  {dp[i]}")
    
    print(f"\nAnswer: dp[0][{n-1}] = {dp[0][n-1]}")


if __name__ == "__main__":
    test_minimum_cost_tree()
    demonstrate_monotonic_stack_approach()
    visualize_tree_construction()
    demonstrate_competitive_programming_patterns()
    analyze_time_complexity()
    test_edge_cases()
    demonstrate_dp_approach()

"""
Minimum Cost Tree From Leaf Values demonstrates competitive programming
patterns with monotonic stack optimization, interval DP, and tree
construction algorithms for optimal binary tree formation.
"""
