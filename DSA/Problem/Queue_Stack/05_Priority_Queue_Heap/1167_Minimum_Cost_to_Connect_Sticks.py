"""
1167. Minimum Cost to Connect Sticks - Multiple Approaches
Difficulty: Medium

You have some number of sticks with positive integer lengths. These lengths are given as an array sticks, where sticks[i] is the length of the ith stick.

You can connect any two sticks of lengths x and y into one stick by paying a cost of x + y. You must connect all the sticks until there is only one stick remaining.

Return the minimum cost of connecting all the sticks.
"""

from typing import List
import heapq

class MinimumCostToConnectSticks:
    """Multiple approaches to find minimum cost to connect sticks"""
    
    def connectSticks_min_heap(self, sticks: List[int]) -> int:
        """
        Approach 1: Min Heap (Optimal)
        
        Always connect two smallest sticks to minimize cost.
        
        Time: O(n log n), Space: O(n)
        """
        if len(sticks) <= 1:
            return 0
        
        # Build min heap
        heapq.heapify(sticks)
        
        total_cost = 0
        
        while len(sticks) > 1:
            # Get two smallest sticks
            first = heapq.heappop(sticks)
            second = heapq.heappop(sticks)
            
            # Cost to connect them
            cost = first + second
            total_cost += cost
            
            # Add the combined stick back
            heapq.heappush(sticks, cost)
        
        return total_cost
    
    def connectSticks_sorting_greedy(self, sticks: List[int]) -> int:
        """
        Approach 2: Sorting with Greedy (Suboptimal)
        
        Sort and always connect smallest available sticks.
        
        Time: O(n² log n), Space: O(1)
        """
        if len(sticks) <= 1:
            return 0
        
        sticks = sticks[:]  # Make copy
        total_cost = 0
        
        while len(sticks) > 1:
            sticks.sort()
            
            # Connect two smallest
            first = sticks.pop(0)
            second = sticks.pop(0)
            
            cost = first + second
            total_cost += cost
            
            # Add combined stick back
            sticks.append(cost)
        
        return total_cost
    
    def connectSticks_priority_queue_simulation(self, sticks: List[int]) -> int:
        """
        Approach 3: Priority Queue Simulation
        
        Simulate priority queue with list operations.
        
        Time: O(n² log n), Space: O(n)
        """
        if len(sticks) <= 1:
            return 0
        
        sticks = sticks[:]
        total_cost = 0
        
        while len(sticks) > 1:
            # Find two minimum elements
            sticks.sort()
            
            first = sticks[0]
            second = sticks[1]
            
            # Remove them
            sticks = sticks[2:]
            
            # Calculate cost and add combined stick
            cost = first + second
            total_cost += cost
            
            # Insert combined stick in sorted order
            inserted = False
            for i in range(len(sticks)):
                if cost <= sticks[i]:
                    sticks.insert(i, cost)
                    inserted = True
                    break
            
            if not inserted:
                sticks.append(cost)
        
        return total_cost
    
    def connectSticks_recursive(self, sticks: List[int]) -> int:
        """
        Approach 4: Recursive Approach
        
        Use recursion with memoization.
        
        Time: O(n² log n), Space: O(n)
        """
        def connect_recursive(stick_list: List[int]) -> int:
            if len(stick_list) <= 1:
                return 0
            
            stick_list.sort()
            
            # Connect two smallest
            first = stick_list[0]
            second = stick_list[1]
            cost = first + second
            
            # Create new list with combined stick
            new_sticks = [cost] + stick_list[2:]
            
            return cost + connect_recursive(new_sticks)
        
        return connect_recursive(sticks[:])
    
    def connectSticks_dynamic_programming(self, sticks: List[int]) -> int:
        """
        Approach 5: Dynamic Programming (Suboptimal for this problem)
        
        Use DP to explore different connection orders.
        
        Time: O(2^n), Space: O(2^n)
        """
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def dp(stick_tuple):
            stick_list = list(stick_tuple)
            
            if len(stick_list) <= 1:
                return 0
            
            if len(stick_list) == 2:
                return stick_list[0] + stick_list[1]
            
            min_cost = float('inf')
            
            # Try connecting each pair
            for i in range(len(stick_list)):
                for j in range(i + 1, len(stick_list)):
                    # Connect sticks i and j
                    cost = stick_list[i] + stick_list[j]
                    
                    # Create new list without i and j, plus combined stick
                    new_sticks = []
                    for k in range(len(stick_list)):
                        if k != i and k != j:
                            new_sticks.append(stick_list[k])
                    new_sticks.append(cost)
                    
                    total_cost = cost + dp(tuple(sorted(new_sticks)))
                    min_cost = min(min_cost, total_cost)
            
            return min_cost
        
        return dp(tuple(sorted(sticks)))


def test_minimum_cost_to_connect_sticks():
    """Test minimum cost to connect sticks algorithms"""
    solver = MinimumCostToConnectSticks()
    
    test_cases = [
        ([2,4,3], 14, "Example 1"),
        ([1,8,3,5], 30, "Example 2"),
        ([5], 0, "Single stick"),
        ([1,2], 3, "Two sticks"),
        ([1,1,1,1], 8, "All same length"),
        ([10,20,30], 90, "Increasing lengths"),
        ([30,20,10], 90, "Decreasing lengths"),
        ([1,2,3,4,5], 33, "Sequential lengths"),
    ]
    
    algorithms = [
        ("Min Heap", solver.connectSticks_min_heap),
        ("Sorting Greedy", solver.connectSticks_sorting_greedy),
        ("Priority Queue Sim", solver.connectSticks_priority_queue_simulation),
        ("Recursive", solver.connectSticks_recursive),
        # ("Dynamic Programming", solver.connectSticks_dynamic_programming),  # Too slow for large inputs
    ]
    
    print("=== Testing Minimum Cost to Connect Sticks ===")
    
    for sticks, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Sticks: {sticks}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(sticks[:])  # Pass copy
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_min_heap_approach():
    """Demonstrate min heap approach step by step"""
    print("\n=== Min Heap Approach Step-by-Step Demo ===")
    
    sticks = [2, 4, 3]
    print(f"Initial sticks: {sticks}")
    
    # Build min heap
    heap = sticks[:]
    heapq.heapify(heap)
    
    print(f"Min heap: {heap}")
    
    total_cost = 0
    step = 1
    
    while len(heap) > 1:
        print(f"\nStep {step}:")
        print(f"  Current heap: {heap}")
        
        # Get two smallest
        first = heapq.heappop(heap)
        second = heapq.heappop(heap)
        
        print(f"  Connect sticks of length {first} and {second}")
        
        # Calculate cost
        cost = first + second
        total_cost += cost
        
        print(f"  Cost: {first} + {second} = {cost}")
        print(f"  Total cost so far: {total_cost}")
        
        # Add combined stick back
        heapq.heappush(heap, cost)
        
        print(f"  New stick of length {cost} added to heap")
        
        step += 1
    
    print(f"\nFinal total cost: {total_cost}")


def demonstrate_greedy_strategy():
    """Demonstrate why greedy strategy works"""
    print("\n=== Greedy Strategy Demonstration ===")
    
    sticks = [1, 2, 3, 4]
    print(f"Sticks: {sticks}")
    
    print(f"\nWhy always connect two smallest sticks?")
    print("- Each stick contributes to the cost multiple times")
    print("- Smaller sticks should be combined first to minimize their contribution")
    
    print(f"\nOptimal strategy (always connect two smallest):")
    
    heap = sticks[:]
    heapq.heapify(heap)
    total_cost = 0
    
    while len(heap) > 1:
        first = heapq.heappop(heap)
        second = heapq.heappop(heap)
        cost = first + second
        total_cost += cost
        
        print(f"  Connect {first} + {second} = {cost}, total = {total_cost}")
        heapq.heappush(heap, cost)
    
    print(f"Optimal total cost: {total_cost}")
    
    print(f"\nSuboptimal strategy (connect largest first):")
    
    sticks_copy = sticks[:]
    suboptimal_cost = 0
    
    # Connect largest first (just for demonstration)
    sticks_copy.sort(reverse=True)
    
    while len(sticks_copy) > 1:
        first = sticks_copy.pop(0)
        second = sticks_copy.pop(0)
        cost = first + second
        suboptimal_cost += cost
        
        print(f"  Connect {first} + {second} = {cost}, total = {suboptimal_cost}")
        
        # Insert back in sorted order
        inserted = False
        for i in range(len(sticks_copy)):
            if cost >= sticks_copy[i]:
                sticks_copy.insert(i, cost)
                inserted = True
                break
        if not inserted:
            sticks_copy.append(cost)
    
    print(f"Suboptimal total cost: {suboptimal_cost}")
    print(f"Difference: {suboptimal_cost - total_cost}")


def visualize_connection_process():
    """Visualize the connection process"""
    print("\n=== Connection Process Visualization ===")
    
    sticks = [1, 3, 2, 4]
    print(f"Initial sticks: {sticks}")
    
    heap = sticks[:]
    heapq.heapify(heap)
    
    total_cost = 0
    step = 1
    
    print(f"\nVisualization:")
    print("Sticks: " + " ".join(f"[{s}]" for s in sorted(sticks)))
    
    while len(heap) > 1:
        print(f"\nStep {step}:")
        
        # Show current sticks
        current_sticks = sorted(heap)
        print("Current: " + " ".join(f"[{s}]" for s in current_sticks))
        
        # Connect two smallest
        first = heapq.heappop(heap)
        second = heapq.heappop(heap)
        cost = first + second
        total_cost += cost
        
        print(f"Connect: [{first}] + [{second}] -> [{cost}] (cost: {cost})")
        
        heapq.heappush(heap, cost)
        
        step += 1
    
    print(f"\nFinal stick: [{heap[0]}]")
    print(f"Total cost: {total_cost}")


if __name__ == "__main__":
    test_minimum_cost_to_connect_sticks()
    demonstrate_min_heap_approach()
    demonstrate_greedy_strategy()
    visualize_connection_process()

"""
Minimum Cost to Connect Sticks demonstrates heap applications for
greedy optimization problems, including cost minimization strategies
and multiple approaches for stick connection algorithms.
"""
