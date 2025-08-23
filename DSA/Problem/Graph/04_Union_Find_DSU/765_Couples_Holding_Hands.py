"""
765. Couples Holding Hands
Difficulty: Hard

Problem:
There are n couples sitting in 2n seats arranged in a row and want to hold hands.

The people and seats are represented by an integer array row where row[i] is the ID of 
the person sitting in the ith seat. The couples are numbered in order, the first couple 
being (0, 1), the second couple being (2, 3), and so on with the last couple being (2n-2, 2n-1).

Return the minimum number of swaps so that every couple is sitting side by side. A swap 
consists of choosing any two people, then they stand up and switch seats.

Examples:
Input: row = [0,2,1,3]
Output: 1

Input: row = [3,2,0,1]
Output: 0

Constraints:
- 2n == row.length
- 2 <= 2n <= 60
- 0 <= row[i] < 2n
- All the values of row are unique
- row[i] is even if i is even, and row[i] is odd if i is odd
"""

from typing import List

class UnionFind:
    """Union-Find for tracking couple groups"""
    
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n
    
    def find(self, x):
        """Find with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """Union by rank"""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        self.components -= 1
        return True
    
    def get_components_count(self):
        """Get number of connected components"""
        return self.components

class Solution:
    def minSwapsCouples_approach1_union_find(self, row: List[int]) -> int:
        """
        Approach 1: Union-Find (Optimal)
        
        Model as graph where each couple is a node. Connect couples that 
        need to be rearranged. Swaps needed = total couples - components.
        
        Time: O(N * α(N)) ≈ O(N)
        Space: O(N)
        """
        n = len(row) // 2  # Number of couples
        uf = UnionFind(n)
        
        # Process each adjacent pair
        for i in range(0, len(row), 2):
            person1 = row[i]
            person2 = row[i + 1]
            
            couple1 = person1 // 2  # Which couple person1 belongs to
            couple2 = person2 // 2  # Which couple person2 belongs to
            
            # If different couples are sitting together, union them
            if couple1 != couple2:
                uf.union(couple1, couple2)
        
        # Number of swaps = total couples - number of components
        return n - uf.get_components_count()
    
    def minSwapsCouples_approach2_greedy_swapping(self, row: List[int]) -> int:
        """
        Approach 2: Greedy Swapping
        
        For each position, if couple is not together, find partner and swap.
        
        Time: O(N²) - each swap might require linear search
        Space: O(N) - position tracking
        """
        def get_partner(person):
            """Get partner of a person"""
            return person + 1 if person % 2 == 0 else person - 1
        
        # Create position mapping for quick lookup
        pos = {}
        for i, person in enumerate(row):
            pos[person] = i
        
        swaps = 0
        
        # Check each couple position (0,1), (2,3), (4,5), ...
        for i in range(0, len(row), 2):
            person1 = row[i]
            person2 = row[i + 1]
            
            # Check if they are partners
            if get_partner(person1) == person2:
                continue  # Already a couple
            
            # Find where person1's partner is sitting
            partner = get_partner(person1)
            partner_pos = pos[partner]
            
            # Swap person2 with person1's partner
            row[i + 1], row[partner_pos] = row[partner_pos], row[i + 1]
            
            # Update position mapping
            pos[person2] = partner_pos
            pos[partner] = i + 1
            
            swaps += 1
        
        return swaps
    
    def minSwapsCouples_approach3_cycle_detection(self, row: List[int]) -> int:
        """
        Approach 3: Cycle Detection in Permutation
        
        Find cycles in the couple arrangement and count swaps needed.
        
        Time: O(N)
        Space: O(N)
        """
        n = len(row) // 2
        couple_positions = [0] * n
        
        # Map each couple to their seating positions
        for i in range(0, len(row), 2):
            person1 = row[i]
            person2 = row[i + 1]
            
            couple1 = person1 // 2
            couple2 = person2 // 2
            
            couple_positions[couple1] = couple2
            couple_positions[couple2] = couple1
        
        visited = [False] * n
        swaps = 0
        
        # Find cycles in the permutation
        for i in range(n):
            if not visited[i]:
                # Count cycle length
                cycle_length = 0
                current = i
                
                while not visited[current]:
                    visited[current] = True
                    current = couple_positions[current]
                    cycle_length += 1
                
                # Swaps needed for cycle = cycle_length - 1
                if cycle_length > 1:
                    swaps += cycle_length - 1
        
        return swaps
    
    def minSwapsCouples_approach4_graph_components(self, row: List[int]) -> int:
        """
        Approach 4: Graph Components Analysis
        
        Build graph of "wrong" couple connections and analyze components.
        
        Time: O(N)
        Space: O(N)
        """
        from collections import defaultdict
        
        n = len(row) // 2
        graph = defaultdict(list)
        
        # Build graph of couples that need rearrangement
        for i in range(0, len(row), 2):
            person1 = row[i]
            person2 = row[i + 1]
            
            couple1 = person1 // 2
            couple2 = person2 // 2
            
            if couple1 != couple2:
                graph[couple1].append(couple2)
                graph[couple2].append(couple1)
        
        visited = set()
        components = 0
        component_sizes = []
        
        def dfs(node):
            """DFS to find component size"""
            if node in visited:
                return 0
            
            visited.add(node)
            size = 1
            
            for neighbor in graph[node]:
                size += dfs(neighbor)
            
            return size
        
        # Find all components
        for couple in range(n):
            if couple not in visited and couple in graph:
                size = dfs(couple)
                components += 1
                component_sizes.append(size)
        
        # Calculate total swaps needed
        total_swaps = 0
        for size in component_sizes:
            total_swaps += size - 1
        
        return total_swaps

def test_couples_holding_hands():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (row, expected)
        ([0,2,1,3], 1),
        ([3,2,0,1], 0),
        ([0,2,4,6,7,1,3,5], 3),
        ([5,4,2,6,3,1,0,7], 2),
        ([0,1], 0),
        ([1,0], 0),
        ([0,2,1,3,4,5], 1),
    ]
    
    approaches = [
        ("Union-Find", solution.minSwapsCouples_approach1_union_find),
        ("Greedy Swapping", solution.minSwapsCouples_approach2_greedy_swapping),
        ("Cycle Detection", solution.minSwapsCouples_approach3_cycle_detection),
        ("Graph Components", solution.minSwapsCouples_approach4_graph_components),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (row, expected) in enumerate(test_cases):
            result = func(row[:])  # Copy to avoid modification
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} Row: {row}, Expected: {expected}, Got: {result}")

def demonstrate_union_find_approach():
    """Demonstrate Union-Find approach for couples problem"""
    print("\n=== Union-Find Approach Demo ===")
    
    row = [0,2,1,3]
    print(f"Row: {row}")
    print(f"Couples: (0,1), (2,3)")
    print(f"Current seating: (0,2), (1,3)")
    
    n = len(row) // 2
    print(f"Number of couples: {n}")
    
    uf = UnionFind(n)
    print(f"Initial components: {uf.get_components_count()}")
    
    print(f"\nAnalyzing seating pairs:")
    
    for i in range(0, len(row), 2):
        person1 = row[i]
        person2 = row[i + 1]
        
        couple1 = person1 // 2
        couple2 = person2 // 2
        
        print(f"  Pair ({person1}, {person2}): Couple {couple1} with Couple {couple2}")
        
        if couple1 != couple2:
            print(f"    Different couples! Union({couple1}, {couple2})")
            uf.union(couple1, couple2)
        else:
            print(f"    Same couple - already correct")
        
        print(f"    Components after: {uf.get_components_count()}")
    
    swaps_needed = n - uf.get_components_count()
    print(f"\nSwaps needed: {n} couples - {uf.get_components_count()} components = {swaps_needed}")

def demonstrate_greedy_approach():
    """Demonstrate greedy swapping approach"""
    print("\n=== Greedy Swapping Demo ===")
    
    row = [0,2,1,3]
    print(f"Initial row: {row}")
    
    def get_partner(person):
        return person + 1 if person % 2 == 0 else person - 1
    
    # Create position mapping
    pos = {}
    for i, person in enumerate(row):
        pos[person] = i
    
    print(f"Position mapping: {pos}")
    
    swaps = 0
    
    for i in range(0, len(row), 2):
        person1 = row[i]
        person2 = row[i + 1]
        
        print(f"\nChecking position {i//2}: ({person1}, {person2})")
        
        if get_partner(person1) == person2:
            print(f"  ✅ Already a couple!")
            continue
        
        # Find person1's partner
        partner = get_partner(person1)
        partner_pos = pos[partner]
        
        print(f"  ❌ Not a couple. Person {person1}'s partner is {partner} at position {partner_pos}")
        print(f"  Swapping position {i+1} ({person2}) with position {partner_pos} ({partner})")
        
        # Perform swap
        row[i + 1], row[partner_pos] = row[partner_pos], row[i + 1]
        
        # Update positions
        pos[person2] = partner_pos
        pos[partner] = i + 1
        
        swaps += 1
        
        print(f"  After swap: {row}")
        print(f"  Updated positions: {pos}")
    
    print(f"\nTotal swaps: {swaps}")

def analyze_couples_problem():
    """Analyze the couples holding hands problem"""
    print("\n=== Couples Problem Analysis ===")
    
    print("Problem Structure:")
    print("• N couples sitting in 2N seats")
    print("• Couples numbered (0,1), (2,3), (4,5), ...")
    print("• Goal: Minimum swaps for all couples to sit together")
    print("• Each swap exchanges two people's positions")
    
    print("\nKey Insights:")
    print("1. **Couple Identification:** person i belongs to couple i//2")
    print("2. **Partner Function:** partner of even person x is x+1, odd person x is x-1")
    print("3. **Wrong Pairs:** When couple A person sits with couple B person")
    print("4. **Graph Modeling:** Each couple is a node, wrong pairs create edges")
    
    print("\nUnion-Find Approach Logic:")
    print("• Create component for each couple initially")
    print("• Union couples that have members sitting together")
    print("• Each component represents a 'tangle' of couples")
    print("• Swaps needed = total couples - number of components")
    
    print("\nWhy This Formula Works:")
    print("• In each component of size k, we need k-1 swaps")
    print("• Total swaps = Σ(component_size - 1)")
    print("• This equals: total_couples - number_of_components")
    print("• Union-Find efficiently counts components")
    
    print("\nExample Component Analysis:")
    print("Row: [0,2,1,3] → Couples (0,1) and (2,3)")
    print("• Position 0-1: persons 0,2 → couples 0,1 → Union(0,1)")
    print("• Position 2-3: persons 1,3 → couples 0,1 → already unioned")
    print("• Result: 1 component of size 2")
    print("• Swaps: 2 couples - 1 component = 1 swap")

def compare_solution_approaches():
    """Compare different solution approaches"""
    print("\n=== Solution Approaches Comparison ===")
    
    print("1. **Union-Find Approach:**")
    print("   ✅ Optimal O(N) time complexity")
    print("   ✅ Clean mathematical formulation")
    print("   ✅ Handles complex couple tangles elegantly")
    print("   ✅ Direct component counting")
    print("   ❌ Requires Union-Find knowledge")
    
    print("\n2. **Greedy Swapping:**")
    print("   ✅ Intuitive step-by-step process")
    print("   ✅ Directly simulates actual swaps")
    print("   ✅ Easy to understand and implement")
    print("   ❌ O(N²) worst case time complexity")
    print("   ❌ More complex position tracking")
    
    print("\n3. **Cycle Detection:**")
    print("   ✅ Mathematical insight into permutation cycles")
    print("   ✅ O(N) time complexity")
    print("   ✅ Educational value for permutation analysis")
    print("   ❌ Less intuitive problem modeling")
    
    print("\n4. **Graph Components:**")
    print("   ✅ Clear graph theory modeling")
    print("   ✅ DFS/BFS component finding")
    print("   ✅ Explicit graph construction")
    print("   ❌ More complex than Union-Find")
    print("   ❌ Extra space for adjacency lists")
    
    print("\nReal-world Applications:")
    print("• **Seating Arrangement:** Wedding, theater, conference")
    print("• **Task Assignment:** Pairing complementary skills")
    print("• **Network Optimization:** Pairing nodes for efficiency")
    print("• **Scheduling:** Coordinating dependent activities")
    print("• **Resource Allocation:** Pairing resources optimally")
    
    print("\nKey Algorithmic Insights:")
    print("• Union-Find excellent for connectivity problems")
    print("• Component counting yields swap count directly")
    print("• Greedy swapping gives constructive solution")
    print("• Problem reducible to graph connectivity")
    print("• Mathematical formula: swaps = n - components")

if __name__ == "__main__":
    test_couples_holding_hands()
    demonstrate_union_find_approach()
    demonstrate_greedy_approach()
    analyze_couples_problem()
    compare_solution_approaches()

"""
Union-Find Concepts:
1. Connected Components in Constraint Problems
2. Graph Modeling of Arrangement Problems
3. Component Counting for Optimization
4. Swapping Operations as Graph Edges

Key Problem Insights:
- Model couples as graph nodes
- Wrong seating creates edges between couples
- Connected components represent "tangles"
- Swaps needed = couples - components

Algorithm Strategy:
1. Identify couples from person IDs (id//2)
2. Union couples that share seating pairs
3. Count connected components
4. Apply formula: swaps = total_couples - components

Union-Find Advantages:
- O(N) time with path compression
- Natural fit for connectivity analysis
- Direct component counting
- Elegant mathematical solution

Mathematical Insight:
- Each component of size k needs k-1 swaps
- Total swaps = Σ(component_size - 1)
- Simplifies to: total_couples - components
- Union-Find efficiently computes this

Real-world Applications:
- Seating arrangement optimization
- Task pairing and assignment
- Resource allocation problems
- Network topology design
- Scheduling and coordination

This problem demonstrates Union-Find for
optimization in arrangement and pairing problems.
"""
