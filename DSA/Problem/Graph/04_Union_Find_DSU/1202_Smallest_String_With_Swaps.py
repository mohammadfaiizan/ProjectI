"""
1202. Smallest String With Swaps
Difficulty: Medium

Problem:
You are given a string s, and an array of pairs of indices in the string pairs where 
pairs[i] = [a, b] indicates 2 indices(0-indexed) of the string.

You can swap the characters at any pair of indices in the given pairs any number of times.

Return the lexicographically smallest string that s can be transformed to after using 
the swaps.

Examples:
Input: s = "dcab", pairs = [[0,3],[1,2]]
Output: "bacd"

Input: s = "dcab", pairs = [[0,3],[1,2],[0,2]]
Output: "abcd"

Input: s = "cba", pairs = [[0,1],[1,2]]
Output: "abc"

Constraints:
- 1 <= s.length <= 10^5
- 0 <= pairs.length <= 10^5
- 0 <= pairs[i][0], pairs[i][1] < s.length
- s contains only lowercase English letters
"""

from typing import List
from collections import defaultdict

class UnionFind:
    """Union-Find for grouping swappable positions"""
    
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
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
        
        return True

class Solution:
    def smallestStringWithSwaps_approach1_union_find(self, s: str, pairs: List[List[int]]) -> str:
        """
        Approach 1: Union-Find (Optimal)
        
        Use Union-Find to group swappable positions, then sort characters 
        within each group to get lexicographically smallest arrangement.
        
        Time: O(N * α(N) + N log N) - Union-Find + sorting
        Space: O(N)
        """
        n = len(s)
        uf = UnionFind(n)
        
        # Union all swappable positions
        for a, b in pairs:
            uf.union(a, b)
        
        # Group positions by their root
        groups = defaultdict(list)
        for i in range(n):
            root = uf.find(i)
            groups[root].append(i)
        
        # Sort characters within each group
        result = list(s)
        
        for positions in groups.values():
            if len(positions) > 1:  # Only sort if multiple positions
                # Get characters at these positions
                chars = [s[pos] for pos in positions]
                
                # Sort characters and positions
                chars.sort()
                positions.sort()
                
                # Place sorted characters at sorted positions
                for i, pos in enumerate(positions):
                    result[pos] = chars[i]
        
        return ''.join(result)
    
    def smallestStringWithSwaps_approach2_dfs(self, s: str, pairs: List[List[int]]) -> str:
        """
        Approach 2: DFS on Adjacency List
        
        Build graph of swappable positions and use DFS to find components.
        
        Time: O(N + E + N log N)
        Space: O(N + E)
        """
        n = len(s)
        
        # Build adjacency list
        graph = defaultdict(list)
        for a, b in pairs:
            graph[a].append(b)
            graph[b].append(a)
        
        visited = [False] * n
        result = list(s)
        
        def dfs(node, component):
            """DFS to collect all positions in connected component"""
            if visited[node]:
                return
            
            visited[node] = True
            component.append(node)
            
            for neighbor in graph[node]:
                dfs(neighbor, component)
        
        # Find all connected components and sort within each
        for i in range(n):
            if not visited[i]:
                component = []
                dfs(i, component)
                
                if len(component) > 1:
                    # Get characters and sort them
                    chars = [s[pos] for pos in component]
                    chars.sort()
                    component.sort()
                    
                    # Place sorted characters
                    for j, pos in enumerate(component):
                        result[pos] = chars[j]
        
        return ''.join(result)
    
    def smallestStringWithSwaps_approach3_bfs(self, s: str, pairs: List[List[int]]) -> str:
        """
        Approach 3: BFS on Adjacency List
        
        Use BFS instead of DFS to find connected components.
        
        Time: O(N + E + N log N)
        Space: O(N + E)
        """
        from collections import deque
        
        n = len(s)
        
        # Build adjacency list
        graph = defaultdict(list)
        for a, b in pairs:
            graph[a].append(b)
            graph[b].append(a)
        
        visited = [False] * n
        result = list(s)
        
        def bfs(start):
            """BFS to find all positions in connected component"""
            component = []
            queue = deque([start])
            visited[start] = True
            
            while queue:
                node = queue.popleft()
                component.append(node)
                
                for neighbor in graph[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            
            return component
        
        # Find all components using BFS
        for i in range(n):
            if not visited[i]:
                component = bfs(i)
                
                if len(component) > 1:
                    # Sort characters within component
                    chars = [s[pos] for pos in component]
                    chars.sort()
                    component.sort()
                    
                    for j, pos in enumerate(component):
                        result[pos] = chars[j]
        
        return ''.join(result)
    
    def smallestStringWithSwaps_approach4_optimized_union_find(self, s: str, pairs: List[List[int]]) -> str:
        """
        Approach 4: Optimized Union-Find with Early Character Sorting
        
        Optimize by reducing memory allocations and using efficient sorting.
        
        Time: O(N * α(N) + N log N)
        Space: O(N)
        """
        n = len(s)
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            root_x, root_y = find(x), find(y)
            if root_x != root_y:
                parent[root_x] = root_y
        
        # Union all pairs
        for a, b in pairs:
            union(a, b)
        
        # Group by root and sort
        groups = defaultdict(lambda: {'positions': [], 'chars': []})
        
        for i in range(n):
            root = find(i)
            groups[root]['positions'].append(i)
            groups[root]['chars'].append(s[i])
        
        result = [''] * n
        
        for group in groups.values():
            positions = group['positions']
            chars = group['chars']
            
            if len(positions) > 1:
                chars.sort()
                positions.sort()
            
            for pos, char in zip(positions, chars):
                result[pos] = char
        
        return ''.join(result)

def test_smallest_string_with_swaps():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (s, pairs, expected)
        ("dcab", [[0,3],[1,2]], "bacd"),
        ("dcab", [[0,3],[1,2],[0,2]], "abcd"),
        ("cba", [[0,1],[1,2]], "abc"),
        ("abcd", [], "abcd"),  # No swaps
        ("abcd", [[0,1]], "abcd"),  # Already optimal
        ("dcba", [[0,3],[1,2]], "bacd"),
        ("edcba", [[1,4],[0,3]], "ebcda"),
    ]
    
    approaches = [
        ("Union-Find", solution.smallestStringWithSwaps_approach1_union_find),
        ("DFS", solution.smallestStringWithSwaps_approach2_dfs),
        ("BFS", solution.smallestStringWithSwaps_approach3_bfs),
        ("Optimized UF", solution.smallestStringWithSwaps_approach4_optimized_union_find),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (s, pairs, expected) in enumerate(test_cases):
            result = func(s, pairs[:])  # Copy pairs to avoid modification
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} s='{s}', pairs={pairs}, expected='{expected}', got='{result}'")

def demonstrate_union_find_process():
    """Demonstrate Union-Find process for string swapping"""
    print("\n=== Union-Find String Swapping Demo ===")
    
    s = "dcab"
    pairs = [[0,3],[1,2],[0,2]]
    print(f"String: '{s}'")
    print(f"Pairs: {pairs}")
    print(f"Characters by position: {[(i, s[i]) for i in range(len(s))]}")
    
    # Union-Find process
    n = len(s)
    uf = UnionFind(n)
    
    print(f"\nUnion-Find operations:")
    
    for i, (a, b) in enumerate(pairs):
        print(f"\nStep {i+1}: Union({a}, {b}) - can swap '{s[a]}' and '{s[b]}'")
        uf.union(a, b)
        
        # Show current components
        components = defaultdict(list)
        for pos in range(n):
            root = uf.find(pos)
            components[root].append(pos)
        
        comp_list = list(components.values())
        print(f"  Current components: {comp_list}")
        
        # Show characters in each component
        for component in comp_list:
            chars = [s[pos] for pos in component]
            print(f"    Positions {component}: characters {chars}")
    
    # Final optimization
    print(f"\nFinal optimization:")
    
    groups = defaultdict(list)
    for i in range(n):
        root = uf.find(i)
        groups[root].append(i)
    
    result = list(s)
    
    for positions in groups.values():
        if len(positions) > 1:
            chars = [s[pos] for pos in positions]
            chars.sort()
            positions.sort()
            
            print(f"  Component {positions}:")
            print(f"    Original chars: {[s[pos] for pos in sorted(positions)]}")
            print(f"    Sorted chars: {chars}")
            print(f"    Assignment: {list(zip(sorted(positions), chars))}")
            
            for i, pos in enumerate(sorted(positions)):
                result[pos] = chars[i]
    
    final_result = ''.join(result)
    print(f"\nFinal result: '{final_result}'")

def analyze_string_swapping():
    """Analyze the string swapping problem"""
    print("\n=== String Swapping Analysis ===")
    
    print("Problem Characteristics:")
    print("• Can swap characters at specified index pairs")
    print("• Can perform swaps any number of times")
    print("• Transitivity: if (a,b) and (b,c) pairs exist, can effectively swap (a,c)")
    print("• Goal: Lexicographically smallest possible string")
    
    print("\nKey Insights:")
    print("1. **Transitivity:** Swapping creates equivalence classes of positions")
    print("2. **Union-Find Natural Fit:** Groups positions that can be rearranged")
    print("3. **Optimal Strategy:** Sort characters within each group")
    print("4. **Lexicographic Order:** Smaller characters should come first")
    
    print("\nAlgorithm Strategy:")
    print("1. Use Union-Find to group all positions that can exchange characters")
    print("2. For each group, collect the characters at those positions")
    print("3. Sort characters in ascending order (for lexicographic minimum)")
    print("4. Sort positions in ascending order")
    print("5. Assign sorted characters to sorted positions")
    
    print("\nWhy This Works:")
    print("• Characters within a group can be arranged in any order")
    print("• To minimize lexicographically, put smallest chars in leftmost positions")
    print("• Sorting achieves this optimal arrangement")
    print("• Union-Find efficiently finds all swappable groups")
    
    print("\nComplexity Analysis:")
    print("• Union-Find operations: O(N * α(N)) ≈ O(N)")
    print("• Sorting within groups: O(N log N) in worst case")
    print("• Total time: O(N log N)")
    print("• Space: O(N) for Union-Find structure")

def demonstrate_edge_cases():
    """Demonstrate handling of edge cases"""
    print("\n=== Edge Cases Demo ===")
    
    edge_cases = [
        {
            "name": "No Swaps",
            "s": "hello",
            "pairs": [],
            "description": "No pairs means no swaps possible"
        },
        {
            "name": "Self Loop", 
            "s": "abc",
            "pairs": [[0,0]],
            "description": "Swapping position with itself"
        },
        {
            "name": "All Connected",
            "s": "dcba", 
            "pairs": [[0,1],[1,2],[2,3]],
            "description": "All positions can be rearranged"
        },
        {
            "name": "Duplicate Pairs",
            "s": "cab",
            "pairs": [[0,2],[2,0],[0,2]],
            "description": "Duplicate pairs should not affect result"
        },
        {
            "name": "Already Optimal",
            "s": "abc",
            "pairs": [[0,1],[1,2]],
            "description": "String already in optimal order"
        }
    ]
    
    solution = Solution()
    
    for case in edge_cases:
        print(f"\n{case['name']}:")
        print(f"  Description: {case['description']}")
        print(f"  Input: s='{case['s']}', pairs={case['pairs']}")
        
        result = solution.smallestStringWithSwaps_approach1_union_find(case['s'], case['pairs'])
        print(f"  Output: '{result}'")
        
        # Show if any change occurred
        if result == case['s']:
            print(f"  No change needed")
        else:
            print(f"  Optimized: '{case['s']}' → '{result}'")

def compare_approaches():
    """Compare different approaches to string swapping"""
    print("\n=== Approach Comparison ===")
    
    print("1. **Union-Find Approach:**")
    print("   ✅ Optimal O(N log N) time complexity")
    print("   ✅ Efficient grouping of swappable positions")
    print("   ✅ Clean separation of concerns")
    print("   ✅ Handles complex transitivity naturally")
    print("   ❌ Requires Union-Find data structure knowledge")
    
    print("\n2. **DFS Graph Approach:**")
    print("   ✅ Intuitive graph representation")
    print("   ✅ Standard connected components algorithm")
    print("   ✅ Easy to understand and debug")
    print("   ❌ Extra space for adjacency lists")
    print("   ❌ Potential recursion depth issues")
    
    print("\n3. **BFS Graph Approach:**")
    print("   ✅ Iterative, no recursion concerns")
    print("   ✅ Level-by-level component discovery")
    print("   ✅ Queue-based implementation")
    print("   ❌ Additional queue space overhead")
    
    print("\nPerformance Comparison:")
    print("• **Time Complexity:** All approaches O(N log N)")
    print("• **Space Complexity:**")
    print("  - Union-Find: O(N)")
    print("  - DFS/BFS: O(N + E) where E is number of pairs")
    print("• **Practical Performance:** Union-Find typically fastest")
    
    print("\nReal-world Applications:")
    print("• **Text Processing:** Anagram generation, word games")
    print("• **Data Sorting:** Constrained rearrangement problems")
    print("• **Puzzle Games:** Character swapping mechanics")
    print("• **Optimization:** Arrangement under constraints")
    print("• **Cryptography:** Permutation-based transformations")

if __name__ == "__main__":
    test_smallest_string_with_swaps()
    demonstrate_union_find_process()
    analyze_string_swapping()
    demonstrate_edge_cases()
    compare_approaches()

"""
Union-Find Concepts:
1. Equivalence Classes in Constrained Optimization
2. Transitive Relationships Through Swapping
3. Lexicographic Optimization Within Groups
4. Character Position Mapping

Key Problem Insights:
- Swapping creates equivalence classes of positions
- Characters within same class can be rearranged freely
- Optimal strategy: sort characters within each class
- Union-Find naturally models transitive swapping

Algorithm Strategy:
1. Use Union-Find to group swappable positions
2. Collect characters for each position group
3. Sort characters in ascending order
4. Sort positions in ascending order
5. Assign sorted characters to sorted positions

String Optimization Pattern:
- Identify which positions can exchange characters
- Within each exchangeable group, arrange optimally
- Lexicographic minimum = smallest chars leftmost
- Union-Find efficiently finds all groups

Real-world Applications:
- Text processing and anagram generation
- Constrained sorting and arrangement
- Game mechanics with limited moves
- Optimization under swap constraints
- Permutation-based problem solving

This problem demonstrates Union-Find for
constrained optimization in string manipulation.
"""
