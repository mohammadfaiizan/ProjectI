"""
1061. Lexicographically Smallest Equivalent String
Difficulty: Medium

Problem:
You are given two strings of the same length s1 and s2 and a string baseStr.

We say s1[i] and s2[i] are equivalent characters.

- For example, if s1 = "abc" and s2 = "cde", then we have 'a' == 'c', 'b' == 'd', 'c' == 'e'.

Equivalent characters follow the usual rules of equivalence relations:

- Reflexivity: 'a' == 'a'
- Symmetry: 'a' == 'b' implies 'b' == 'a'
- Transitivity: 'a' == 'b' and 'b' == 'c' implies 'a' == 'c'

For example, given the equivalency information from s1 = "abc" and s2 = "cde", "acd" and "aab" 
are equivalent strings of baseStr = "eed", and "aab" is the lexicographically smallest equivalent string of baseStr.

Return the lexicographically smallest equivalent string of baseStr.

Examples:
Input: s1 = "parker", s2 = "morris", baseStr = "parser"
Output: "makkek"

Input: s1 = "hello", s2 = "world", baseStr = "hold"
Output: "hdld"

Input: s1 = "leetcode", s2 = "programs", baseStr = "sourcecode"
Output: "aauaaaaada"

Constraints:
- 1 <= s1.length, s2.length, baseStr.length <= 1000
- s1.length == s2.length
- s1, s2, and baseStr consist of lowercase English letters
"""

from typing import List

class UnionFind:
    """Union-Find for character equivalence classes"""
    
    def __init__(self):
        # Use character ASCII values as indices
        self.parent = list(range(26))  # a-z
        self.rank = [0] * 26
    
    def find(self, x):
        """Find with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """Union by rank with lexicographic preference"""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return
        
        # Always make the lexicographically smaller character the root
        if root_x > root_y:
            root_x, root_y = root_y, root_x
        
        # Union by rank, but prefer lexicographically smaller root
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
            # Update to maintain lexicographic order
            if root_y > root_x:
                self.parent[root_y] = root_x
                self.parent[root_x] = root_x
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
    
    def get_smallest_equivalent(self, char):
        """Get lexicographically smallest equivalent character"""
        return chr(ord('a') + self.find(ord(char) - ord('a')))

class Solution:
    def smallestEquivalentString_approach1_union_find_lexicographic(self, s1: str, s2: str, baseStr: str) -> str:
        """
        Approach 1: Union-Find with Lexicographic Root Selection (Optimal)
        
        Use Union-Find where root is always the lexicographically smallest character.
        
        Time: O(N * α(26)) ≈ O(N) where N = total characters processed
        Space: O(26) = O(1)
        """
        uf = UnionFind()
        
        # Process equivalence pairs
        for c1, c2 in zip(s1, s2):
            uf.union(ord(c1) - ord('a'), ord(c2) - ord('a'))
        
        # Build result string
        result = []
        for char in baseStr:
            smallest_equiv = uf.get_smallest_equivalent(char)
            result.append(smallest_equiv)
        
        return ''.join(result)
    
    def smallestEquivalentString_approach2_simple_union_find(self, s1: str, s2: str, baseStr: str) -> str:
        """
        Approach 2: Simple Union-Find with Post-Processing
        
        Use standard Union-Find, then find smallest in each component.
        
        Time: O(N)
        Space: O(1)
        """
        parent = list(range(26))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            root_x, root_y = find(x), find(y)
            if root_x != root_y:
                # Always attach higher to lower for lexicographic order
                if root_x > root_y:
                    parent[root_x] = root_y
                else:
                    parent[root_y] = root_x
        
        # Process equivalences
        for c1, c2 in zip(s1, s2):
            union(ord(c1) - ord('a'), ord(c2) - ord('a'))
        
        # Build result
        result = []
        for char in baseStr:
            char_idx = ord(char) - ord('a')
            root = find(char_idx)
            smallest_char = chr(ord('a') + root)
            result.append(smallest_char)
        
        return ''.join(result)
    
    def smallestEquivalentString_approach3_dfs_components(self, s1: str, s2: str, baseStr: str) -> str:
        """
        Approach 3: DFS to Find Connected Components
        
        Build adjacency list and use DFS to find components.
        
        Time: O(N + 26)
        Space: O(26)
        """
        from collections import defaultdict
        
        # Build adjacency list
        graph = defaultdict(set)
        
        for c1, c2 in zip(s1, s2):
            graph[c1].add(c2)
            graph[c2].add(c1)
        
        # Find smallest character in each component
        visited = set()
        char_to_smallest = {}
        
        def dfs(char, component):
            """DFS to collect all characters in component"""
            if char in visited:
                return
            
            visited.add(char)
            component.append(char)
            
            for neighbor in graph[char]:
                dfs(neighbor, component)
        
        # Find all components
        for char in 'abcdefghijklmnopqrstuvwxyz':
            if char not in visited:
                component = []
                dfs(char, component)
                
                # Find smallest in component
                smallest = min(component)
                for c in component:
                    char_to_smallest[c] = smallest
        
        # Build result
        result = []
        for char in baseStr:
            result.append(char_to_smallest.get(char, char))
        
        return ''.join(result)
    
    def smallestEquivalentString_approach4_iterative_reduction(self, s1: str, s2: str, baseStr: str) -> str:
        """
        Approach 4: Iterative Character Reduction
        
        Repeatedly merge character equivalences until stable.
        
        Time: O(N * 26)
        Space: O(1)
        """
        # Initialize each character mapping to itself
        char_map = {}
        for c in 'abcdefghijklmnopqrstuvwxyz':
            char_map[c] = c
        
        # Process equivalences with iterative improvement
        for c1, c2 in zip(s1, s2):
            # Find current mappings
            mapped_c1 = char_map[c1]
            mapped_c2 = char_map[c2]
            
            # Use lexicographically smaller mapping
            smaller = min(mapped_c1, mapped_c2)
            
            # Update all characters that map to either value
            for char in char_map:
                if char_map[char] in [mapped_c1, mapped_c2]:
                    char_map[char] = smaller
        
        # Iteratively reduce until stable
        changed = True
        while changed:
            changed = False
            for char in char_map:
                # Follow the mapping chain
                current = char_map[char]
                if char_map[current] < current:
                    char_map[char] = char_map[current]
                    changed = True
        
        # Build result
        result = []
        for char in baseStr:
            result.append(char_map[char])
        
        return ''.join(result)

def test_lexicographically_smallest_equivalent():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (s1, s2, baseStr, expected)
        ("parker", "morris", "parser", "makkek"),
        ("hello", "world", "hold", "hdld"),
        ("leetcode", "programs", "sourcecode", "aauaaaaada"),
        ("abc", "cde", "eed", "aab"),
        ("aa", "bb", "ab", "aa"),
        ("a", "z", "az", "aa"),
    ]
    
    approaches = [
        ("Union-Find Lexicographic", solution.smallestEquivalentString_approach1_union_find_lexicographic),
        ("Simple Union-Find", solution.smallestEquivalentString_approach2_simple_union_find),
        ("DFS Components", solution.smallestEquivalentString_approach3_dfs_components),
        ("Iterative Reduction", solution.smallestEquivalentString_approach4_iterative_reduction),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (s1, s2, baseStr, expected) in enumerate(test_cases):
            result = func(s1, s2, baseStr)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} s1='{s1}', s2='{s2}', baseStr='{baseStr}', expected='{expected}', got='{result}'")

def demonstrate_equivalence_process():
    """Demonstrate character equivalence process"""
    print("\n=== Character Equivalence Demo ===")
    
    s1 = "abc"
    s2 = "cde"
    baseStr = "eed"
    
    print(f"s1: '{s1}'")
    print(f"s2: '{s2}'")
    print(f"baseStr: '{baseStr}'")
    
    print(f"\nEquivalence pairs:")
    for i, (c1, c2) in enumerate(zip(s1, s2)):
        print(f"  Position {i}: '{c1}' ≡ '{c2}'")
    
    # Union-Find simulation
    print(f"\nUnion-Find simulation:")
    parent = list(range(26))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        root_x, root_y = find(x), find(y)
        if root_x != root_y:
            if root_x > root_y:
                parent[root_x] = root_y
            else:
                parent[root_y] = root_x
    
    def char_to_idx(c):
        return ord(c) - ord('a')
    
    def idx_to_char(i):
        return chr(ord('a') + i)
    
    for c1, c2 in zip(s1, s2):
        idx1, idx2 = char_to_idx(c1), char_to_idx(c2)
        print(f"\nUnion('{c1}', '{c2}'):")
        print(f"  Before: '{c1}' -> {idx_to_char(find(idx1))}, '{c2}' -> {idx_to_char(find(idx2))}")
        union(idx1, idx2)
        print(f"  After:  '{c1}' -> {idx_to_char(find(idx1))}, '{c2}' -> {idx_to_char(find(idx2))}")
    
    # Show equivalence classes
    print(f"\nEquivalence classes:")
    classes = {}
    for i in range(26):
        char = idx_to_char(i)
        root = idx_to_char(find(i))
        if root not in classes:
            classes[root] = []
        classes[root].append(char)
    
    for root, chars in sorted(classes.items()):
        if len(chars) > 1 or root in s1 + s2:
            print(f"  {root}: {chars}")
    
    # Transform baseStr
    print(f"\nTransforming baseStr:")
    result = []
    for char in baseStr:
        equiv = idx_to_char(find(char_to_idx(char)))
        result.append(equiv)
        print(f"  '{char}' -> '{equiv}'")
    
    final_result = ''.join(result)
    print(f"\nResult: '{baseStr}' -> '{final_result}'")

def analyze_equivalence_relations():
    """Analyze equivalence relation properties"""
    print("\n=== Equivalence Relations Analysis ===")
    
    print("Equivalence Relation Properties:")
    
    print("\n1. **Reflexivity:** a ≡ a")
    print("   • Every character is equivalent to itself")
    print("   • Automatically satisfied in Union-Find")
    print("   • parent[i] starts as i")
    
    print("\n2. **Symmetry:** a ≡ b ⟹ b ≡ a")
    print("   • If a equivalent to b, then b equivalent to a")
    print("   • Union operation is symmetric")
    print("   • union(a,b) same as union(b,a)")
    
    print("\n3. **Transitivity:** a ≡ b and b ≡ c ⟹ a ≡ c")
    print("   • Equivalence chains propagate")
    print("   • Union-Find handles transitivity naturally")
    print("   • Path compression optimizes transitive queries")
    
    print("\nLexicographic Optimization:")
    print("• Goal: Replace each character with smallest equivalent")
    print("• Union-Find root = smallest character in equivalence class")
    print("• Maintain lexicographic order during union operations")
    print("• Final mapping: char → find(char) gives smallest equivalent")
    
    print("\nExample Transitivity:")
    print("  Given: a ≡ c, c ≡ e")
    print("  Then: a ≡ e (transitively)")
    print("  Equivalence class: {a, c, e}")
    print("  Smallest representative: a")
    print("  All map to: a")

def demonstrate_edge_cases():
    """Demonstrate handling of edge cases"""
    print("\n=== Edge Cases Demo ===")
    
    edge_cases = [
        {
            "name": "No Equivalences",
            "s1": "a",
            "s2": "a", 
            "baseStr": "hello",
            "description": "Character equivalent to itself"
        },
        {
            "name": "All Same Character",
            "s1": "aaaa",
            "s2": "aaaa",
            "baseStr": "test",
            "description": "Redundant equivalences"
        },
        {
            "name": "Complete Chain",
            "s1": "abcd",
            "s2": "bcde",
            "baseStr": "abcde",
            "description": "Transitive chain a→b→c→d→e"
        },
        {
            "name": "Disjoint Groups",
            "s1": "ac",
            "s2": "bd",
            "baseStr": "abcd",
            "description": "Separate equivalence classes"
        },
        {
            "name": "Single Character Base",
            "s1": "ab",
            "s2": "ba",
            "baseStr": "a",
            "description": "Short base string"
        }
    ]
    
    solution = Solution()
    
    for case in edge_cases:
        print(f"\n{case['name']}:")
        print(f"  Description: {case['description']}")
        print(f"  Input: s1='{case['s1']}', s2='{case['s2']}', baseStr='{case['baseStr']}'")
        
        result = solution.smallestEquivalentString_approach1_union_find_lexicographic(
            case['s1'], case['s2'], case['baseStr'])
        print(f"  Output: '{result}'")
        
        # Show character mappings
        uf = UnionFind()
        for c1, c2 in zip(case['s1'], case['s2']):
            uf.union(ord(c1) - ord('a'), ord(c2) - ord('a'))
        
        unique_chars = set(case['s1'] + case['s2'] + case['baseStr'])
        mappings = []
        for char in sorted(unique_chars):
            equiv = uf.get_smallest_equivalent(char)
            if char != equiv:
                mappings.append(f"{char}→{equiv}")
        
        if mappings:
            print(f"  Mappings: {', '.join(mappings)}")
        else:
            print(f"  No character mappings needed")

def compare_implementation_approaches():
    """Compare different implementation approaches"""
    print("\n=== Implementation Approaches Comparison ===")
    
    print("1. **Union-Find with Lexicographic Root:**")
    print("   ✅ Optimal O(N) time complexity")
    print("   ✅ Automatic lexicographic ordering")
    print("   ✅ Natural equivalence relation modeling")
    print("   ✅ Path compression optimization")
    print("   ❌ Requires careful root selection logic")
    
    print("\n2. **Simple Union-Find:**")
    print("   ✅ Straightforward implementation")
    print("   ✅ Standard Union-Find operations")
    print("   ✅ Clear separation of concerns")
    print("   ❌ Manual lexicographic root management")
    
    print("\n3. **DFS Connected Components:**")
    print("   ✅ Intuitive graph-based approach")
    print("   ✅ Explicit component identification")
    print("   ✅ Easy to understand and debug")
    print("   ❌ Higher space complexity for adjacency lists")
    print("   ❌ Potential recursion depth issues")
    
    print("\n4. **Iterative Character Reduction:**")
    print("   ✅ No complex data structures needed")
    print("   ✅ Direct character mapping approach")
    print("   ❌ O(N * 26) time complexity")
    print("   ❌ Multiple passes needed for convergence")
    
    print("\nPerformance Analysis:")
    print("• **Time Complexity:** Union-Find O(N), others O(N) to O(N²)")
    print("• **Space Complexity:** All O(1) for 26 characters")
    print("• **Practical Performance:** Union-Find fastest")
    print("• **Implementation Complexity:** DFS simplest conceptually")
    
    print("\nReal-world Applications:")
    print("• **Text Processing:** Character normalization")
    print("• **String Matching:** Fuzzy string comparison")
    print("• **Data Cleaning:** Character standardization")
    print("• **Linguistics:** Phonetic equivalence modeling")
    print("• **Cryptography:** Character substitution analysis")
    
    print("\nKey Design Insights:")
    print("• Union-Find perfect for equivalence relations")
    print("• Lexicographic ordering during union operations")
    print("• Path compression essential for performance")
    print("• Character indexing simplifies implementation")
    print("• Transitivity handling is automatic")

if __name__ == "__main__":
    test_lexicographically_smallest_equivalent()
    demonstrate_equivalence_process()
    analyze_equivalence_relations()
    demonstrate_edge_cases()
    compare_implementation_approaches()

"""
Union-Find Concepts:
1. Equivalence Relations in Character Mapping
2. Lexicographic Optimization with Union-Find
3. Character Equivalence Class Management
4. Transitivity in String Transformations

Key Problem Insights:
- Character equivalences form equivalence relations
- Union-Find naturally models transitivity
- Lexicographic optimization requires careful root selection
- Character indexing simplifies implementation

Algorithm Strategy:
1. Process character equivalence pairs with Union-Find
2. Maintain lexicographically smallest root for each class
3. Transform base string using equivalence mappings
4. Return lexicographically optimal result

Equivalence Relation Properties:
- Reflexive: each character equivalent to itself
- Symmetric: bidirectional equivalence
- Transitive: equivalence chains propagate
- Union-Find handles all properties automatically

Lexicographic Optimization:
- Root selection determines final character mapping
- Always choose smallest character as root
- Union operations preserve lexicographic order
- Path compression maintains efficiency

Real-world Applications:
- Text normalization and standardization
- Character encoding transformations
- Linguistic analysis and phonetics
- Data cleaning and preprocessing
- String similarity and matching

This problem demonstrates Union-Find for
character equivalence and lexicographic optimization.
"""
