"""
1337. The K Weakest Rows in a Matrix
Difficulty: Easy

Problem:
You are given an m x n binary matrix mat of 1's (representing soldiers) and 0's 
(representing civilians). The soldiers are positioned in front of the civilians in 
each row (i.e., all the 1's will appear to the left of all the 0's in each row).

A row i is weaker than a row j if one of the following is true:
- The number of soldiers in row i is less than the number of soldiers in row j.
- Both rows have the same number of soldiers and i < j.

Return the indices of the k weakest rows in the matrix ordered from weakest to strongest.

Examples:
Input: mat = [[1,1,0,0,0],[1,1,1,1,0],[1,0,0,0,0],[1,1,0,0,0],[1,1,1,1,1]], k = 3
Output: [2,0,3]

Input: mat = [[1,0,0,0],[1,1,1,1],[1,0,0,0],[1,0,0,0]], k = 2
Output: [0,2]

Constraints:
- m == mat.length
- n == mat[i].length
- 2 <= n, m <= 100
- 1 <= k <= m
- mat[i][j] is either 0 or 1
- mat[i] is sorted in a non-increasing order
"""

from typing import List
import heapq

class Solution:
    def kWeakestRows_approach1_binary_search_heap(self, mat: List[List[int]], k: int) -> List[int]:
        """
        Approach 1: Binary Search + Min Heap (Optimal)
        
        Use binary search to count soldiers efficiently, then use heap for k smallest.
        
        Time: O(m * log n + k log m)
        Space: O(m)
        """
        def count_soldiers(row):
            """Binary search to count soldiers (1s) in sorted row"""
            left, right = 0, len(row)
            
            while left < right:
                mid = (left + right) // 2
                if row[mid] == 1:
                    left = mid + 1
                else:
                    right = mid
            
            return left
        
        # Create list of (soldier_count, row_index) pairs
        row_strengths = []
        for i, row in enumerate(mat):
            soldier_count = count_soldiers(row)
            row_strengths.append((soldier_count, i))
        
        # Sort by soldier count, then by row index
        row_strengths.sort()
        
        return [row_idx for _, row_idx in row_strengths[:k]]
    
    def kWeakestRows_approach2_linear_scan_heap(self, mat: List[List[int]], k: int) -> List[int]:
        """
        Approach 2: Linear Scan + Min Heap
        
        Count soldiers with linear scan, use heap for k smallest.
        
        Time: O(m * n + k log m)
        Space: O(m)
        """
        # Count soldiers in each row
        row_strengths = []
        for i, row in enumerate(mat):
            soldier_count = sum(row)
            row_strengths.append((soldier_count, i))
        
        # Sort and return first k
        row_strengths.sort()
        return [row_idx for _, row_idx in row_strengths[:k]]
    
    def kWeakestRows_approach3_max_heap_size_k(self, mat: List[List[int]], k: int) -> List[int]:
        """
        Approach 3: Max Heap of Size K
        
        Maintain max heap of size k to find k weakest rows.
        
        Time: O(m * log n + m log k)
        Space: O(k)
        """
        def count_soldiers_binary_search(row):
            """Binary search for soldier count"""
            left, right = 0, len(row)
            while left < right:
                mid = (left + right) // 2
                if row[mid] == 1:
                    left = mid + 1
                else:
                    right = mid
            return left
        
        # Max heap to maintain k weakest rows
        max_heap = []
        
        for i, row in enumerate(mat):
            soldier_count = count_soldiers_binary_search(row)
            
            if len(max_heap) < k:
                # Negative values for max heap simulation
                heapq.heappush(max_heap, (-soldier_count, -i))
            else:
                # Compare with strongest in current k weakest
                if (-soldier_count, -i) > max_heap[0]:
                    heapq.heapreplace(max_heap, (-soldier_count, -i))
        
        # Extract and sort results
        result = []
        while max_heap:
            neg_soldiers, neg_idx = heapq.heappop(max_heap)
            result.append((-neg_soldiers, -neg_idx))
        
        # Sort by strength (soldiers, then index)
        result.sort()
        return [idx for _, idx in result]
    
    def kWeakestRows_approach4_two_pass_scanning(self, mat: List[List[int]], k: int) -> List[int]:
        """
        Approach 4: Two-Pass Scanning
        
        First pass to count, second pass to find k weakest.
        
        Time: O(m * n)
        Space: O(m)
        """
        m, n = len(mat), len(mat[0])
        
        # First pass: count soldiers
        strengths = []
        for i in range(m):
            count = 0
            for j in range(n):
                if mat[i][j] == 1:
                    count += 1
                else:
                    break  # Since row is sorted
            strengths.append((count, i))
        
        # Sort by strength, then by index
        strengths.sort()
        
        return [idx for _, idx in strengths[:k]]
    
    def kWeakestRows_approach5_vertical_scanning(self, mat: List[List[int]], k: int) -> List[int]:
        """
        Approach 5: Vertical Column Scanning
        
        Scan columns from left to right, mark rows as they become weak.
        
        Time: O(m * n)
        Space: O(m)
        """
        m, n = len(mat), len(mat[0])
        
        # Track soldier counts
        soldier_counts = [0] * m
        
        # Scan each column from left to right
        for j in range(n):
            for i in range(m):
                if mat[i][j] == 1:
                    soldier_counts[i] += 1
        
        # Create (count, index) pairs and sort
        indexed_counts = [(soldier_counts[i], i) for i in range(m)]
        indexed_counts.sort()
        
        return [idx for _, idx in indexed_counts[:k]]

def test_k_weakest_rows():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (mat, k, expected)
        ([[1,1,0,0,0],[1,1,1,1,0],[1,0,0,0,0],[1,1,0,0,0],[1,1,1,1,1]], 3, [2,0,3]),
        ([[1,0,0,0],[1,1,1,1],[1,0,0,0],[1,0,0,0]], 2, [0,2]),
        ([[1,0],[0,0],[1,0]], 2, [1,0]),
        ([[1,1,1],[1,1,1],[0,0,0]], 1, [2]),
        ([[0,0,0],[1,1,1]], 1, [0]),
    ]
    
    approaches = [
        ("Binary Search + Heap", solution.kWeakestRows_approach1_binary_search_heap),
        ("Linear Scan + Heap", solution.kWeakestRows_approach2_linear_scan_heap),
        ("Max Heap Size K", solution.kWeakestRows_approach3_max_heap_size_k),
        ("Two-Pass Scanning", solution.kWeakestRows_approach4_two_pass_scanning),
        ("Vertical Scanning", solution.kWeakestRows_approach5_vertical_scanning),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (mat, k, expected) in enumerate(test_cases):
            result = func([row[:] for row in mat], k)  # Deep copy
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} k={k}, expected={expected}, got={result}")

def demonstrate_binary_search_optimization():
    """Demonstrate binary search optimization for counting soldiers"""
    print("\n=== Binary Search Optimization Demo ===")
    
    row = [1, 1, 1, 0, 0, 0, 0]
    print(f"Row: {row}")
    print(f"Goal: Count soldiers (1s) efficiently")
    
    # Linear scan approach
    linear_count = 0
    for val in row:
        if val == 1:
            linear_count += 1
        else:
            break
    
    print(f"\nLinear scan:")
    print(f"  Scan each element until first 0")
    print(f"  Operations: {linear_count + 1}")
    print(f"  Result: {linear_count} soldiers")
    
    # Binary search approach
    print(f"\nBinary search:")
    left, right = 0, len(row)
    operations = 0
    
    while left < right:
        operations += 1
        mid = (left + right) // 2
        print(f"  Step {operations}: left={left}, right={right}, mid={mid}, row[{mid}]={row[mid]}")
        
        if row[mid] == 1:
            left = mid + 1
            print(f"    Found 1, search right half")
        else:
            right = mid
            print(f"    Found 0, search left half")
    
    print(f"  Operations: {operations}")
    print(f"  Result: {left} soldiers")
    print(f"  Efficiency gain: {(linear_count + 1) / operations:.1f}x faster")

def analyze_heap_approaches():
    """Analyze different heap-based approaches"""
    print("\n=== Heap Approaches Analysis ===")
    
    print("Problem: Find k smallest elements with custom comparison")
    print("Elements: (soldier_count, row_index)")
    print("Comparison: First by soldier_count, then by row_index")
    
    print("\n1. **Sort All + Take First K:**")
    print("   • Time: O(m log m)")
    print("   • Space: O(m)")
    print("   • Simple but processes all elements")
    
    print("\n2. **Min Heap (All Elements):**")
    print("   • Time: O(m + k log m)")
    print("   • Space: O(m)")
    print("   • Build heap, then extract k elements")
    
    print("\n3. **Max Heap (Size K):**")
    print("   • Time: O(m log k)")
    print("   • Space: O(k)")
    print("   • Maintain only k elements at any time")
    print("   • Better space complexity for small k")
    
    print("\n4. **Quick Select (if needed):**")
    print("   • Time: O(m) average, O(m²) worst")
    print("   • Space: O(1)")
    print("   • Find k-th element, then partition")
    
    print("\nOptimal Choice:")
    print("• **Small k:** Max heap of size k")
    print("• **Large k:** Sort all elements")
    print("• **k ≈ m/2:** Either approach works well")

def demonstrate_weakness_comparison():
    """Demonstrate how row weakness is determined"""
    print("\n=== Row Weakness Comparison Demo ===")
    
    mat = [[1,1,0,0,0],[1,1,1,1,0],[1,0,0,0,0],[1,1,0,0,0],[1,1,1,1,1]]
    
    print(f"Matrix:")
    for i, row in enumerate(mat):
        print(f"  Row {i}: {row}")
    
    print(f"\nSoldier count analysis:")
    
    strengths = []
    for i, row in enumerate(mat):
        soldier_count = sum(row)
        strengths.append((soldier_count, i))
        print(f"  Row {i}: {soldier_count} soldiers")
    
    print(f"\nSorting by weakness (fewer soldiers = weaker):")
    strengths.sort()
    
    for rank, (soldiers, row_idx) in enumerate(strengths):
        print(f"  Rank {rank + 1}: Row {row_idx} ({soldiers} soldiers)")
    
    print(f"\nTie-breaking rules:")
    print(f"• Primary: Fewer soldiers = weaker")
    print(f"• Secondary: Lower row index = weaker (if same soldier count)")
    
    # Demonstrate tie-breaking
    tie_example = [[1,0,0], [1,0,0], [1,1,0]]
    print(f"\nTie-breaking example: {tie_example}")
    
    tie_strengths = []
    for i, row in enumerate(tie_example):
        soldier_count = sum(row)
        tie_strengths.append((soldier_count, i))
        print(f"  Row {i}: {soldier_count} soldiers")
    
    tie_strengths.sort()
    print(f"\nAfter sorting with tie-breaking:")
    for rank, (soldiers, row_idx) in enumerate(tie_strengths):
        print(f"  Rank {rank + 1}: Row {row_idx} ({soldiers} soldiers)")

def compare_scanning_strategies():
    """Compare different scanning strategies"""
    print("\n=== Scanning Strategies Comparison ===")
    
    print("1. **Linear Row Scanning:**")
    print("   • Scan each row left to right until first 0")
    print("   • Time: O(m * average_soldiers)")
    print("   • Best for rows with few soldiers")
    
    print("\n2. **Binary Search per Row:**")
    print("   • Binary search for first 0 in each row")
    print("   • Time: O(m * log n)")
    print("   • Best for wide matrices (large n)")
    
    print("\n3. **Full Linear Scan:**")
    print("   • Count all 1s in each row")
    print("   • Time: O(m * n)")
    print("   • Simplest implementation")
    
    print("\n4. **Vertical Column Scanning:**")
    print("   • Scan columns left to right")
    print("   • Time: O(m * n)")
    print("   • Can terminate early for specific k")
    
    print("\n5. **Smart Early Termination:**")
    print("   • Combine approaches based on matrix characteristics")
    print("   • Adaptive strategy selection")
    
    print("\nChoice Guidelines:")
    print("• **Wide matrices (n >> m):** Binary search per row")
    print("• **Tall matrices (m >> n):** Linear scanning")
    print("• **Small k:** Can optimize with early termination")
    print("• **General case:** Binary search + sorting")

def analyze_problem_variants():
    """Analyze variants of the k weakest rows problem"""
    print("\n=== Problem Variants Analysis ===")
    
    print("Core Problem: K Weakest Rows")
    print("• Find k rows with fewest soldiers")
    print("• Tie-breaking by row index")
    
    print("\nRelated Problems:")
    
    print("\n1. **K Strongest Rows:**")
    print("   • Find k rows with most soldiers")
    print("   • Reverse sorting criteria")
    
    print("\n2. **Top K with Custom Criteria:**")
    print("   • Different strength definitions")
    print("   • Multiple tie-breaking rules")
    
    print("\n3. **Dynamic K Selection:**")
    print("   • K changes based on matrix properties")
    print("   • Adaptive threshold selection")
    
    print("\n4. **Range Queries:**")
    print("   • Find rows in strength range [min, max]")
    print("   • Not just top k")
    
    print("\n5. **Online Updates:**")
    print("   • Matrix changes, maintain k weakest")
    print("   • Incremental updates")
    
    print("\nReal-world Applications:")
    print("• **Military:** Unit strength analysis")
    print("• **Sports:** Team ranking systems")
    print("• **Business:** Performance evaluation")
    print("• **Games:** Difficulty balancing")
    print("• **Resource allocation:** Priority assignment")

if __name__ == "__main__":
    test_k_weakest_rows()
    demonstrate_binary_search_optimization()
    analyze_heap_approaches()
    demonstrate_weakness_comparison()
    compare_scanning_strategies()
    analyze_problem_variants()

"""
Shortest Path Concepts:
1. Efficient Counting with Binary Search
2. Priority-based Selection with Heaps
3. Sorting and Ranking Algorithms
4. Matrix Analysis and Processing
5. Top-K Selection Strategies

Key Problem Insights:
- Count soldiers efficiently in sorted rows
- Find k weakest based on soldier count and index
- Tie-breaking rules: fewer soldiers, then lower index
- Binary search optimization for sorted rows

Algorithm Strategy:
1. Count soldiers in each row (binary search or linear)
2. Create (soldier_count, row_index) pairs
3. Sort by soldier count, then by row index
4. Return first k row indices

Optimization Techniques:
- Binary search for O(log n) counting per row
- Max heap of size k for O(m log k) selection
- Early termination strategies
- Adaptive approach selection

Matrix Processing Patterns:
- Row-wise analysis with efficient counting
- Sorting with custom comparison functions
- Top-k selection with heap data structures
- Binary search in sorted sequences

Real-world Applications:
- Military unit strength analysis
- Resource allocation and prioritization
- Performance ranking systems
- Game difficulty balancing
- Data analysis and reporting

This problem demonstrates efficient matrix analysis
and top-k selection algorithms with optimization.
"""
