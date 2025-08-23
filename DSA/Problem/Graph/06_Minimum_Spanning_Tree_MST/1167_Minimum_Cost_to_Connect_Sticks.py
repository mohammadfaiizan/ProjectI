"""
1167. Minimum Cost to Connect Sticks
Difficulty: Medium

Problem:
You have some number of sticks with positive integer lengths. These lengths are given as an array sticks,
where sticks[i] is the length of the ith stick.

You can connect any two sticks of lengths x and y into one stick by paying a cost of x + y. 
You must connect all the sticks until there is only one stick remaining.

Return the minimum cost of connecting all the sticks.

Examples:
Input: sticks = [2,4,3]
Output: 14
Explanation: You start with sticks = [2,4,3].
1. Combine sticks 2 and 3 for a cost of 2 + 3 = 5. Now you have sticks = [5,4].
2. Combine sticks 5 and 4 for a cost of 5 + 4 = 9. Now you have sticks = [9].
There is only one stick left, so you are done. The total cost is 5 + 9 = 14.

Input: sticks = [1,8,3,5]
Output: 30
Explanation: You start with sticks = [1,8,3,5].
1. Combine sticks 1 and 3 for a cost of 1 + 3 = 4. Now you have sticks = [4,8,5].
2. Combine sticks 4 and 5 for a cost of 4 + 5 = 9. Now you have sticks = [9,8].
3. Combine sticks 9 and 8 for a cost of 9 + 8 = 17. Now you have sticks = [17].
The total cost is 4 + 9 + 17 = 30.

Input: sticks = [5]
Output: 0

Constraints:
- 1 <= sticks.length <= 10^4
- 1 <= sticks[i] <= 10^4
"""

from typing import List
import heapq

class Solution:
    def connectSticks_approach1_min_heap(self, sticks: List[int]) -> int:
        """
        Approach 1: Min Heap (Optimal)
        
        Always combine the two smallest sticks to minimize total cost.
        This is equivalent to Huffman coding problem.
        
        Time: O(N log N)
        Space: O(N)
        """
        if len(sticks) <= 1:
            return 0
        
        # Create min heap
        heapq.heapify(sticks)
        total_cost = 0
        
        # Combine sticks until only one remains
        while len(sticks) > 1:
            # Get two smallest sticks
            first = heapq.heappop(sticks)
            second = heapq.heappop(sticks)
            
            # Calculate cost and add to total
            cost = first + second
            total_cost += cost
            
            # Put combined stick back
            heapq.heappush(sticks, cost)
        
        return total_cost
    
    def connectSticks_approach2_priority_queue_simulation(self, sticks: List[int]) -> int:
        """
        Approach 2: Priority Queue with Step-by-step Simulation
        
        Detailed simulation showing each combination step.
        
        Time: O(N log N)
        Space: O(N)
        """
        if len(sticks) <= 1:
            return 0
        
        # Convert to min heap
        min_heap = sticks[:]
        heapq.heapify(min_heap)
        
        total_cost = 0
        combination_steps = []
        
        while len(min_heap) > 1:
            # Extract two minimum elements
            stick1 = heapq.heappop(min_heap)
            stick2 = heapq.heappop(min_heap)
            
            # Calculate combination cost
            combined_length = stick1 + stick2
            total_cost += combined_length
            
            # Record step for analysis
            combination_steps.append({
                'stick1': stick1,
                'stick2': stick2,
                'cost': combined_length,
                'total_cost': total_cost
            })
            
            # Add combined stick back to heap
            heapq.heappush(min_heap, combined_length)
        
        return total_cost
    
    def connectSticks_approach3_huffman_tree_building(self, sticks: List[int]) -> int:
        """
        Approach 3: Huffman Tree Construction
        
        Build Huffman tree and calculate total internal node weights.
        
        Time: O(N log N)
        Space: O(N)
        """
        if len(sticks) <= 1:
            return 0
        
        class TreeNode:
            def __init__(self, weight, left=None, right=None):
                self.weight = weight
                self.left = left
                self.right = right
            
            def __lt__(self, other):
                return self.weight < other.weight
        
        # Create leaf nodes for each stick
        heap = [TreeNode(weight) for weight in sticks]
        heapq.heapify(heap)
        
        total_cost = 0
        
        # Build Huffman tree
        while len(heap) > 1:
            # Get two nodes with minimum weight
            node1 = heapq.heappop(heap)
            node2 = heapq.heappop(heap)
            
            # Create internal node
            combined_weight = node1.weight + node2.weight
            internal_node = TreeNode(combined_weight, node1, node2)
            
            # Add cost (weight of internal node)
            total_cost += combined_weight
            
            # Add internal node back to heap
            heapq.heappush(heap, internal_node)
        
        return total_cost
    
    def connectSticks_approach4_dynamic_programming_memoization(self, sticks: List[int]) -> int:
        """
        Approach 4: Dynamic Programming with Memoization (Alternative approach)
        
        Use memoization to explore different combination orders.
        Note: This has exponential complexity and is not optimal.
        
        Time: O(2^N) - Exponential
        Space: O(2^N)
        """
        if len(sticks) <= 1:
            return 0
        
        memo = {}
        
        def min_cost(remaining_sticks):
            """Find minimum cost to combine remaining sticks"""
            if len(remaining_sticks) == 1:
                return 0
            
            key = tuple(sorted(remaining_sticks))
            if key in memo:
                return memo[key]
            
            min_total = float('inf')
            
            # Try all possible pairs
            for i in range(len(remaining_sticks)):
                for j in range(i + 1, len(remaining_sticks)):
                    # Combine sticks i and j
                    stick1, stick2 = remaining_sticks[i], remaining_sticks[j]
                    combination_cost = stick1 + stick2
                    
                    # Create new list with combined stick
                    new_sticks = []
                    for k in range(len(remaining_sticks)):
                        if k != i and k != j:
                            new_sticks.append(remaining_sticks[k])
                    new_sticks.append(combination_cost)
                    
                    # Recursive call
                    total_cost = combination_cost + min_cost(new_sticks)
                    min_total = min(min_total, total_cost)
            
            memo[key] = min_total
            return min_total
        
        return min_cost(sticks)
    
    def connectSticks_approach5_greedy_validation(self, sticks: List[int]) -> int:
        """
        Approach 5: Greedy Strategy Validation
        
        Verify that greedy approach (always combine smallest) is optimal.
        
        Time: O(N log N)
        Space: O(N)
        """
        if len(sticks) <= 1:
            return 0
        
        # Sort sticks to analyze greedy choice
        sorted_sticks = sorted(sticks)
        
        # Use min heap for greedy combinations
        heap = sorted_sticks[:]
        total_cost = 0
        combination_order = []
        
        while len(heap) > 1:
            # Always pick two smallest (greedy choice)
            first = heap.pop(0)  # Smallest
            second = heap.pop(0)  # Second smallest
            
            # Combine and calculate cost
            combined = first + second
            total_cost += combined
            
            # Record combination
            combination_order.append((first, second, combined))
            
            # Insert combined stick in sorted order
            # Find correct position to maintain sorted order
            insert_pos = 0
            while insert_pos < len(heap) and heap[insert_pos] < combined:
                insert_pos += 1
            heap.insert(insert_pos, combined)
        
        return total_cost

def test_minimum_cost_connect_sticks():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (sticks, expected)
        ([2,4,3], 14),
        ([1,8,3,5], 30),
        ([5], 0),
        ([1,2], 3),
        ([1,2,3,4,5], 33),
        ([20,4,8,2], 54),
        ([1], 0),
        ([10,10,10,10], 60),
    ]
    
    approaches = [
        ("Min Heap", solution.connectSticks_approach1_min_heap),
        ("Priority Queue Simulation", solution.connectSticks_approach2_priority_queue_simulation),
        ("Huffman Tree", solution.connectSticks_approach3_huffman_tree_building),
        ("Greedy Validation", solution.connectSticks_approach5_greedy_validation),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (sticks, expected) in enumerate(test_cases):
            result = func(sticks[:])  # Deep copy
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} sticks={sticks}, expected={expected}, got={result}")

def demonstrate_huffman_connection():
    """Demonstrate connection to Huffman coding"""
    print("\n=== Huffman Coding Connection Demo ===")
    
    sticks = [2, 4, 3]
    print(f"Sticks: {sticks}")
    
    print(f"\nHuffman Tree Construction:")
    print(f"1. Start with leaves: 2, 3, 4")
    print(f"2. Combine smallest (2, 3) → cost = 5, new node weight = 5")
    print(f"3. Combine remaining (4, 5) → cost = 9, new node weight = 9")
    print(f"4. Total cost = 5 + 9 = 14")
    
    print(f"\nTree structure:")
    print(f"        9")
    print(f"       / \\")
    print(f"      4   5")
    print(f"         / \\")
    print(f"        2   3")
    
    print(f"\nKey insight: Each internal node represents a combination")
    print(f"The weight of internal nodes is the cost of combination")
    print(f"Total cost = sum of all internal node weights")

def demonstrate_greedy_optimality():
    """Demonstrate why greedy approach is optimal"""
    print("\n=== Greedy Optimality Proof ===")
    
    print("Why always combining smallest sticks is optimal:")
    
    print("\n1. **Exchange Argument:**")
    print("   • Suppose optimal solution doesn't combine two smallest first")
    print("   • We can exchange to get better or equal solution")
    print("   • This proves greedy choice is safe")
    
    print("\n2. **Cost Analysis:**")
    print("   • Each stick contributes to cost based on its depth in tree")
    print("   • Smaller sticks should be deeper to minimize total cost")
    print("   • Greedy approach ensures this property")
    
    print("\n3. **Example Comparison:**")
    sticks = [1, 3, 5, 8]
    
    print(f"   Sticks: {sticks}")
    print(f"   Greedy: (1,3)→4, (4,5)→9, (8,9)→17, cost = 4+9+17 = 30")
    print(f"   Alternative: (1,8)→9, (3,5)→8, (8,9)→17, cost = 9+8+17 = 34")
    print(f"   Greedy is better!")
    
    print("\n4. **Mathematical Proof Sketch:**")
    print("   • Let a₁ ≤ a₂ ≤ ... ≤ aₙ be sorted sticks")
    print("   • Any optimal tree has a₁ and a₂ as siblings (can be proven)")
    print("   • Combining them first reduces problem size optimally")

def analyze_problem_variations():
    """Analyze variations of the stick connection problem"""
    print("\n=== Problem Variations Analysis ===")
    
    print("Related Problems and Variations:")
    
    print("\n1. **Huffman Coding:**")
    print("   • Character frequencies → stick lengths")
    print("   • Minimize expected code length")
    print("   • Same greedy algorithm")
    
    print("\n2. **File Merging:**")
    print("   • Merge sorted files optimally")
    print("   • Cost = sum of file sizes being merged")
    print("   • Minimize total I/O operations")
    
    print("\n3. **Matrix Chain Multiplication:**")
    print("   • Different problem but similar tree structure")
    print("   • Optimal parenthesization")
    print("   • Dynamic programming solution")
    
    print("\n4. **Optimal Binary Search Tree:**")
    print("   • Weight-based tree construction")
    print("   • Minimize expected search cost")
    print("   • Similar to Huffman but more complex")
    
    print("\n5. **Rope/String Concatenation:**")
    print("   • Minimize character copy operations")
    print("   • Each merge costs sum of string lengths")
    print("   • Same greedy strategy applies")

def demonstrate_step_by_step_solution():
    """Demonstrate detailed step-by-step solution"""
    print("\n=== Step-by-Step Solution Demo ===")
    
    sticks = [1, 8, 3, 5]
    print(f"Initial sticks: {sticks}")
    
    # Simulate min heap operations
    import heapq
    heap = sticks[:]
    heapq.heapify(heap)
    
    total_cost = 0
    step = 0
    
    print(f"\nHeap after initialization: {heap}")
    
    while len(heap) > 1:
        step += 1
        # Get two smallest
        first = heapq.heappop(heap)
        second = heapq.heappop(heap)
        
        # Calculate cost
        cost = first + second
        total_cost += cost
        
        print(f"\nStep {step}:")
        print(f"  Combine: {first} + {second} = {cost}")
        print(f"  Step cost: {cost}")
        print(f"  Total cost so far: {total_cost}")
        
        # Add back to heap
        heapq.heappush(heap, cost)
        print(f"  Heap after combination: {heap}")
    
    print(f"\nFinal total cost: {total_cost}")
    
    # Verify with direct calculation
    solution = Solution()
    result = solution.connectSticks_approach1_min_heap(sticks)
    print(f"Verification: {result}")

def analyze_complexity_and_optimization():
    """Analyze complexity and potential optimizations"""
    print("\n=== Complexity Analysis ===")
    
    print("Time Complexity Analysis:")
    
    print("\n1. **Min Heap Approach:**")
    print("   • Heap initialization: O(N)")
    print("   • N-1 combinations, each O(log N)")
    print("   • Total: O(N log N)")
    print("   • Optimal for this problem")
    
    print("\n2. **Sorting + Linear Scan:**")
    print("   • Not applicable due to dynamic insertions")
    print("   • Heap maintains order efficiently")
    
    print("\n3. **Space Complexity:**")
    print("   • O(N) for heap storage")
    print("   • O(1) additional space")
    
    print("\nOptimization Considerations:")
    
    print("\n1. **Input Size:**")
    print("   • For small N (<100), simple sorting might be faster")
    print("   • For large N, heap approach is essential")
    
    print("\n2. **Implementation Details:**")
    print("   • Python heapq is efficient")
    print("   • In-place operations when possible")
    print("   • Avoid unnecessary copying")
    
    print("\n3. **Memory Usage:**")
    print("   • Heap uses original array space")
    print("   • No additional data structures needed")
    print("   • Cache-friendly access patterns")

if __name__ == "__main__":
    test_minimum_cost_connect_sticks()
    demonstrate_huffman_connection()
    demonstrate_greedy_optimality()
    analyze_problem_variations()
    demonstrate_step_by_step_solution()
    analyze_complexity_and_optimization()

"""
Minimum Cost to Connect Sticks and Huffman Coding Concepts:
1. Greedy Algorithm for Optimal Tree Construction
2. Min Heap for Efficient Minimum Selection
3. Huffman Coding Tree Building Process
4. Cost Optimization through Strategic Combination
5. Priority Queue Applications in Graph Algorithms

Key Problem Insights:
- Always combine two smallest elements for optimal cost
- Equivalent to Huffman coding tree construction
- Greedy approach guaranteed to be optimal
- Min heap provides efficient implementation

Algorithm Strategy:
1. Use min heap to always access smallest elements
2. Combine two smallest, add cost to total
3. Insert combined result back to heap
4. Repeat until only one element remains

Greedy Optimality:
- Smaller elements should be deeper in combination tree
- Each element's contribution = depth × weight
- Greedy ensures minimal weighted depth sum
- Exchange argument proves optimality

Applications:
- File merging optimization
- Huffman coding for data compression
- Resource allocation and scheduling
- Network design and optimization
- Manufacturing process optimization

This problem demonstrates fundamental greedy algorithms
and their connection to tree construction and optimization.
"""
