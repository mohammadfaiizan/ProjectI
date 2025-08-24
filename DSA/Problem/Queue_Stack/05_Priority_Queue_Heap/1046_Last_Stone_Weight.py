"""
1046. Last Stone Weight - Multiple Approaches
Difficulty: Easy

You are given an array of integers stones where stones[i] is the weight of the ith stone.

We are playing a game with the stones. On each turn, we choose the heaviest two stones and smash them together. Suppose the heaviest two stones have weights x and y with x <= y. The result of this smash is:

- If x == y, both stones are destroyed.
- If x != y, the stone of weight x is destroyed, and the stone of weight y has new weight y - x.

At the end of the game, there is at most one stone left.

Return the weight of the last stone. If there are no stones left, return 0.
"""

from typing import List
import heapq

class LastStoneWeight:
    """Multiple approaches to solve last stone weight problem"""
    
    def lastStoneWeight_max_heap(self, stones: List[int]) -> int:
        """
        Approach 1: Max Heap (Optimal)
        
        Use max heap to always get the two heaviest stones.
        
        Time: O(n log n), Space: O(n)
        """
        if not stones:
            return 0
        
        # Python heapq is min heap, so negate values for max heap
        heap = [-stone for stone in stones]
        heapq.heapify(heap)
        
        while len(heap) > 1:
            # Get two heaviest stones
            first = -heapq.heappop(heap)   # Heaviest
            second = -heapq.heappop(heap)  # Second heaviest
            
            # If they're different, put the difference back
            if first != second:
                heapq.heappush(heap, -(first - second))
        
        # Return last stone weight or 0 if no stones left
        return -heap[0] if heap else 0
    
    def lastStoneWeight_sorting(self, stones: List[int]) -> int:
        """
        Approach 2: Sorting
        
        Sort array and process from the end.
        
        Time: O(n² log n), Space: O(1)
        """
        stones = stones[:]  # Make a copy
        
        while len(stones) > 1:
            stones.sort()
            
            # Get two heaviest stones
            first = stones.pop()   # Heaviest
            second = stones.pop()  # Second heaviest
            
            # If they're different, add the difference back
            if first != second:
                stones.append(first - second)
        
        return stones[0] if stones else 0
    
    def lastStoneWeight_priority_queue_simulation(self, stones: List[int]) -> int:
        """
        Approach 3: Priority Queue Simulation
        
        Simulate priority queue with list operations.
        
        Time: O(n² log n), Space: O(n)
        """
        stones = stones[:]
        
        while len(stones) > 1:
            # Find two maximum elements
            stones.sort(reverse=True)
            
            first = stones[0]
            second = stones[1]
            
            # Remove the two heaviest stones
            stones = stones[2:]
            
            # Add difference if stones are different
            if first != second:
                # Insert in sorted order
                diff = first - second
                inserted = False
                
                for i in range(len(stones)):
                    if diff > stones[i]:
                        stones.insert(i, diff)
                        inserted = True
                        break
                
                if not inserted:
                    stones.append(diff)
        
        return stones[0] if stones else 0
    
    def lastStoneWeight_recursive(self, stones: List[int]) -> int:
        """
        Approach 4: Recursive Approach
        
        Use recursion to process stones.
        
        Time: O(n² log n), Space: O(n)
        """
        def solve(stones_list: List[int]) -> int:
            if len(stones_list) <= 1:
                return stones_list[0] if stones_list else 0
            
            # Sort to get heaviest stones
            stones_list.sort(reverse=True)
            
            first = stones_list[0]
            second = stones_list[1]
            remaining = stones_list[2:]
            
            if first == second:
                # Both stones destroyed
                return solve(remaining)
            else:
                # Add difference back and recurse
                remaining.append(first - second)
                return solve(remaining)
        
        return solve(stones[:])
    
    def lastStoneWeight_bucket_sort(self, stones: List[int]) -> int:
        """
        Approach 5: Bucket Sort Optimization
        
        Use bucket sort for better performance when weights are bounded.
        
        Time: O(n + W), Space: O(W) where W is max weight
        """
        if not stones:
            return 0
        
        max_weight = max(stones)
        
        # Count frequency of each weight
        buckets = [0] * (max_weight + 1)
        for stone in stones:
            buckets[stone] += 1
        
        # Process from heaviest to lightest
        while True:
            # Find two heaviest stones
            first = second = -1
            
            for weight in range(max_weight, -1, -1):
                if buckets[weight] > 0:
                    if first == -1:
                        first = weight
                        buckets[weight] -= 1
                        if buckets[weight] > 0:
                            second = weight
                            buckets[weight] -= 1
                            break
                    else:
                        second = weight
                        buckets[weight] -= 1
                        break
            
            # If we couldn't find two stones, we're done
            if second == -1:
                return first if first != -1 else 0
            
            # Process the collision
            if first != second:
                diff = first - second
                buckets[diff] += 1
    
    def lastStoneWeight_multiset_simulation(self, stones: List[int]) -> int:
        """
        Approach 6: Multiset Simulation
        
        Simulate multiset with sorted list.
        
        Time: O(n² log n), Space: O(n)
        """
        from bisect import bisect_left, insort
        
        stones_sorted = sorted(stones, reverse=True)
        
        while len(stones_sorted) > 1:
            # Get two heaviest
            first = stones_sorted.pop(0)
            second = stones_sorted.pop(0)
            
            # Add difference if needed
            if first != second:
                diff = first - second
                insort(stones_sorted, diff)
                stones_sorted.reverse()  # Keep descending order
                stones_sorted.reverse()  # Correct the order
                
                # Manual insertion in descending order
                stones_sorted = sorted(stones_sorted, reverse=True)
        
        return stones_sorted[0] if stones_sorted else 0
    
    def lastStoneWeight_optimized_heap(self, stones: List[int]) -> int:
        """
        Approach 7: Optimized Heap with Early Termination
        
        Optimize heap approach with early termination.
        
        Time: O(n log n), Space: O(n)
        """
        if not stones:
            return 0
        
        if len(stones) == 1:
            return stones[0]
        
        # Use max heap (negate values)
        heap = [-stone for stone in stones]
        heapq.heapify(heap)
        
        while len(heap) > 1:
            first = -heapq.heappop(heap)
            second = -heapq.heappop(heap)
            
            if first == second:
                # Both destroyed, continue
                continue
            else:
                # Add difference back
                diff = first - second
                heapq.heappush(heap, -diff)
                
                # Early termination: if only one stone left after this operation
                if len(heap) == 1:
                    break
        
        return -heap[0] if heap else 0


def test_last_stone_weight():
    """Test last stone weight algorithms"""
    solver = LastStoneWeight()
    
    test_cases = [
        ([2,7,4,1,8,1], 1, "Example 1"),
        ([1], 1, "Single stone"),
        ([2,2], 0, "Two equal stones"),
        ([3,7,2], 2, "Three stones"),
        ([1,3], 2, "Two different stones"),
        ([10,4,2,10], 2, "Four stones with duplicates"),
        ([1,1,1,1], 0, "All equal stones"),
        ([5,4,3,2,1], 1, "Descending order"),
        ([1,2,3,4,5], 1, "Ascending order"),
        ([20,10,5,15], 0, "Complex case"),
    ]
    
    algorithms = [
        ("Max Heap", solver.lastStoneWeight_max_heap),
        ("Sorting", solver.lastStoneWeight_sorting),
        ("Priority Queue Sim", solver.lastStoneWeight_priority_queue_simulation),
        ("Recursive", solver.lastStoneWeight_recursive),
        ("Bucket Sort", solver.lastStoneWeight_bucket_sort),
        ("Multiset Sim", solver.lastStoneWeight_multiset_simulation),
        ("Optimized Heap", solver.lastStoneWeight_optimized_heap),
    ]
    
    print("=== Testing Last Stone Weight ===")
    
    for stones, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Stones: {stones}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(stones)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_max_heap_approach():
    """Demonstrate max heap approach step by step"""
    print("\n=== Max Heap Approach Step-by-Step Demo ===")
    
    stones = [2, 7, 4, 1, 8, 1]
    print(f"Initial stones: {stones}")
    
    # Create max heap (negate values for min heap)
    heap = [-stone for stone in stones]
    heapq.heapify(heap)
    
    print(f"Max heap (negated): {heap}")
    print(f"Actual values: {[-x for x in heap]}")
    
    step = 1
    while len(heap) > 1:
        print(f"\nStep {step}:")
        
        # Get two heaviest stones
        first = -heapq.heappop(heap)
        second = -heapq.heappop(heap)
        
        print(f"  Heaviest stones: {first} and {second}")
        
        if first == second:
            print(f"  Both stones destroyed (equal weight)")
        else:
            diff = first - second
            heapq.heappush(heap, -diff)
            print(f"  Difference {diff} added back to heap")
        
        remaining = [-x for x in heap]
        print(f"  Remaining stones: {remaining}")
        
        step += 1
    
    result = -heap[0] if heap else 0
    print(f"\nFinal result: {result}")


def visualize_stone_collisions():
    """Visualize stone collisions"""
    print("\n=== Stone Collisions Visualization ===")
    
    stones = [8, 7, 6, 2, 1]
    print(f"Initial stones: {stones}")
    
    stones_copy = stones[:]
    collision = 1
    
    while len(stones_copy) > 1:
        stones_copy.sort(reverse=True)
        
        first = stones_copy[0]
        second = stones_copy[1]
        
        print(f"\nCollision {collision}:")
        print(f"  Before: {stones_copy}")
        print(f"  Heaviest stones collide: {first} vs {second}")
        
        # Remove the two stones
        stones_copy = stones_copy[2:]
        
        if first == second:
            print(f"  Both stones destroyed (equal weight)")
        else:
            diff = first - second
            stones_copy.append(diff)
            print(f"  Stone of weight {diff} remains")
        
        print(f"  After: {sorted(stones_copy, reverse=True) if stones_copy else []}")
        
        collision += 1
    
    result = stones_copy[0] if stones_copy else 0
    print(f"\nFinal stone weight: {result}")


def demonstrate_heap_operations():
    """Demonstrate heap operations in detail"""
    print("\n=== Heap Operations Demonstration ===")
    
    print("Max Heap using Min Heap (negated values):")
    print("- Python heapq implements min heap")
    print("- To simulate max heap, negate all values")
    print("- Smallest negated value = largest original value")
    
    stones = [3, 5, 1, 4]
    print(f"\nOriginal stones: {stones}")
    
    # Create heap step by step
    heap = []
    
    for stone in stones:
        print(f"\nAdding stone {stone}:")
        heapq.heappush(heap, -stone)
        print(f"  Heap (negated): {heap}")
        print(f"  Actual values: {[-x for x in heap]}")
        print(f"  Max stone available: {-heap[0]}")
    
    print(f"\nExtracting stones in order:")
    while heap:
        max_stone = -heapq.heappop(heap)
        print(f"  Extracted: {max_stone}")
        print(f"  Remaining heap: {[-x for x in heap] if heap else []}")


def benchmark_last_stone_weight():
    """Benchmark different approaches"""
    import time
    import random
    
    algorithms = [
        ("Max Heap", LastStoneWeight().lastStoneWeight_max_heap),
        ("Sorting", LastStoneWeight().lastStoneWeight_sorting),
        ("Bucket Sort", LastStoneWeight().lastStoneWeight_bucket_sort),
        ("Optimized Heap", LastStoneWeight().lastStoneWeight_optimized_heap),
    ]
    
    # Test with different array sizes
    test_sizes = [100, 1000, 5000]
    
    print("\n=== Last Stone Weight Performance Benchmark ===")
    
    for size in test_sizes:
        print(f"\n--- Array Size: {size} ---")
        
        # Generate random stones (weights 1-100)
        stones = [random.randint(1, 100) for _ in range(size)]
        
        for alg_name, alg_func in algorithms:
            start_time = time.time()
            
            try:
                result = alg_func(stones[:])  # Pass copy
                end_time = time.time()
                print(f"{alg_name:20} | Time: {end_time - start_time:.4f}s | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:30]}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = LastStoneWeight()
    
    edge_cases = [
        ([], 0, "Empty array"),
        ([1], 1, "Single stone"),
        ([1, 1], 0, "Two equal stones"),
        ([2, 1], 1, "Two different stones"),
        ([1, 1, 1], 1, "Three equal stones"),
        ([1, 1, 1, 1], 0, "Four equal stones"),
        ([100], 100, "Single large stone"),
        ([50, 50], 0, "Two large equal stones"),
        ([1, 2, 3, 4, 5], 1, "Sequential stones"),
        ([5, 4, 3, 2, 1], 1, "Reverse sequential"),
    ]
    
    for stones, expected, description in edge_cases:
        try:
            result = solver.lastStoneWeight_max_heap(stones)
            status = "✓" if result == expected else "✗"
            print(f"{description:25} | {status} | stones: {stones} -> {result}")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")


def compare_approaches():
    """Compare different approaches"""
    print("\n=== Approach Comparison ===")
    
    test_cases = [
        [2, 7, 4, 1, 8, 1],
        [1],
        [2, 2],
        [10, 4, 2, 10],
    ]
    
    solver = LastStoneWeight()
    
    approaches = [
        ("Max Heap", solver.lastStoneWeight_max_heap),
        ("Sorting", solver.lastStoneWeight_sorting),
        ("Recursive", solver.lastStoneWeight_recursive),
        ("Bucket Sort", solver.lastStoneWeight_bucket_sort),
        ("Optimized Heap", solver.lastStoneWeight_optimized_heap),
    ]
    
    for i, stones in enumerate(test_cases):
        print(f"\nTest case {i+1}: {stones}")
        
        results = {}
        
        for name, func in approaches:
            try:
                result = func(stones[:])  # Pass copy
                results[name] = result
                print(f"{name:15} | Result: {result}")
            except Exception as e:
                print(f"{name:15} | ERROR: {str(e)[:40]}")
        
        # Check consistency
        if results:
            first_result = list(results.values())[0]
            all_same = all(result == first_result for result in results.values())
            print(f"All approaches agree: {'✓' if all_same else '✗'}")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Max Heap", "O(n log n)", "O(n)", "Heap operations for n stones"),
        ("Sorting", "O(n² log n)", "O(1)", "Sort after each collision"),
        ("Priority Queue Sim", "O(n² log n)", "O(n)", "Repeated sorting and insertion"),
        ("Recursive", "O(n² log n)", "O(n)", "Recursive calls with sorting"),
        ("Bucket Sort", "O(n + W)", "O(W)", "W = max weight, bounded input"),
        ("Multiset Sim", "O(n² log n)", "O(n)", "Sorted list operations"),
        ("Optimized Heap", "O(n log n)", "O(n)", "Heap with early termination"),
    ]
    
    print(f"{'Approach':<20} | {'Time':<15} | {'Space':<8} | {'Notes'}")
    print("-" * 70)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<20} | {time_comp:<15} | {space_comp:<8} | {notes}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Resource merging in games
    print("1. Game Resource Merging - Combine materials:")
    materials = [10, 15, 8, 12, 20]  # Material quantities
    
    solver = LastStoneWeight()
    remaining = solver.lastStoneWeight_max_heap(materials[:])
    
    print(f"  Initial materials: {materials}")
    print(f"  After optimal merging: {remaining} units remain")
    
    # Application 2: Load balancing
    print("\n2. Server Load Balancing - Redistribute loads:")
    server_loads = [100, 80, 120, 90, 110]  # Server loads
    
    remaining_load = solver.lastStoneWeight_max_heap(server_loads[:])
    
    print(f"  Initial server loads: {server_loads}")
    print(f"  After load redistribution: {remaining_load} imbalance remains")
    
    # Application 3: Chemical reaction simulation
    print("\n3. Chemical Reactions - Reactant consumption:")
    reactants = [25, 30, 15, 20, 35]  # Reactant amounts (grams)
    
    final_amount = solver.lastStoneWeight_max_heap(reactants[:])
    
    print(f"  Initial reactants (g): {reactants}")
    print(f"  Final product amount: {final_amount}g")


def demonstrate_game_simulation():
    """Demonstrate the stone smashing game"""
    print("\n=== Stone Smashing Game Simulation ===")
    
    stones = [8, 4, 6, 2, 3]
    print(f"Starting stones: {stones}")
    print("Game rules:")
    print("- Take two heaviest stones")
    print("- If equal: both destroyed")
    print("- If different: replace with difference")
    
    heap = [-stone for stone in stones]
    heapq.heapify(heap)
    
    round_num = 1
    
    while len(heap) > 1:
        print(f"\nRound {round_num}:")
        
        # Show current state
        current_stones = sorted([-x for x in heap], reverse=True)
        print(f"  Current stones: {current_stones}")
        
        # Get two heaviest
        first = -heapq.heappop(heap)
        second = -heapq.heappop(heap)
        
        print(f"  Smashing: {first} vs {second}")
        
        if first == second:
            print(f"  Result: Both stones destroyed!")
        else:
            diff = first - second
            heapq.heappush(heap, -diff)
            print(f"  Result: Stone of weight {diff} remains")
        
        round_num += 1
    
    final_result = -heap[0] if heap else 0
    print(f"\nGame Over! Final stone weight: {final_result}")


def demonstrate_heap_vs_sorting():
    """Demonstrate heap vs sorting performance characteristics"""
    print("\n=== Heap vs Sorting Performance Characteristics ===")
    
    stones = [10, 8, 6, 4, 2, 9, 7, 5, 3, 1]
    
    print("Heap Approach:")
    print("- Build heap once: O(n)")
    print("- Each collision: O(log n) to extract + O(log n) to insert")
    print("- Total: O(n log n)")
    
    print("\nSorting Approach:")
    print("- Each collision: O(n log n) to sort entire array")
    print("- n collisions in worst case")
    print("- Total: O(n² log n)")
    
    print(f"\nExample with {len(stones)} stones:")
    
    # Simulate operations count
    heap_ops = len(stones) + 2 * (len(stones) - 1)  # heapify + 2 ops per collision
    sort_ops = (len(stones) - 1) * len(stones)  # sort per collision
    
    print(f"  Heap operations: ~{heap_ops}")
    print(f"  Sorting operations: ~{sort_ops}")
    print(f"  Heap is ~{sort_ops // heap_ops}x more efficient")


if __name__ == "__main__":
    test_last_stone_weight()
    demonstrate_max_heap_approach()
    visualize_stone_collisions()
    demonstrate_heap_operations()
    demonstrate_game_simulation()
    demonstrate_heap_vs_sorting()
    demonstrate_real_world_applications()
    test_edge_cases()
    compare_approaches()
    analyze_time_complexity()
    benchmark_last_stone_weight()

"""
Last Stone Weight demonstrates priority queue applications for simulation
problems, including max heap implementation and multiple optimization
strategies for collision-based game mechanics.
"""
