"""
990. Satisfiability of Equality Equations
Difficulty: Medium

Problem:
You are given an array of strings equations that represent relationships between variables 
where each string equations[i] is of length 4 and takes one of two different forms: 
"xi==xj" or "xi!=xj". Here, xi and xj are lowercase letters (not necessarily different) 
that represent one-letter variable names.

Return true if it is possible to assign integers to variable names so as to satisfy all 
the given equations, or false otherwise.

Examples:
Input: equations = ["a==b","b!=c","c==a"]
Output: false

Input: equations = ["b==a","a==b"]
Output: true

Input: equations = ["a==b","b==c","a==c"]
Output: true

Constraints:
- 1 <= equations.length <= 500
- equations[i].length == 4
- equations[i][0] is a lowercase letter
- equations[i][1] is either '=' or '!'
- equations[i][2] is '='
- equations[i][3] is a lowercase letter
"""

from typing import List

class UnionFind:
    """Union-Find data structure with path compression and union by rank"""
    
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size
        self.components = size
    
    def find(self, x):
        """Find root with path compression"""
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
            root_x, root_y = root_y, root_x
        
        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        
        self.components -= 1
        return True
    
    def connected(self, x, y):
        """Check if two elements are in same component"""
        return self.find(x) == self.find(y)

class Solution:
    def equationsPossible_approach1_union_find(self, equations: List[str]) -> bool:
        """
        Approach 1: Union-Find (Optimal)
        
        Use Union-Find to group equal variables, then check inequalities.
        
        Time: O(N * α(26)) ≈ O(N) where α is inverse Ackermann
        Space: O(26) = O(1)
        """
        # Initialize Union-Find for 26 letters
        uf = UnionFind(26)
        
        # Process equality equations first
        for eq in equations:
            if eq[1] == '=':  # "x==y"
                x = ord(eq[0]) - ord('a')
                y = ord(eq[3]) - ord('a')
                uf.union(x, y)
        
        # Check inequality equations
        for eq in equations:
            if eq[1] == '!':  # "x!=y"
                x = ord(eq[0]) - ord('a')
                y = ord(eq[3]) - ord('a')
                
                # If they're in same component, contradiction
                if uf.connected(x, y):
                    return False
        
        return True
    
    def equationsPossible_approach2_two_pass(self, equations: List[str]) -> bool:
        """
        Approach 2: Two-Pass Union-Find
        
        Separate processing for clarity.
        
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
                parent[root_y] = root_x
        
        # First pass: process equalities
        equalities = []
        inequalities = []
        
        for eq in equations:
            if eq[1] == '=':
                equalities.append(eq)
            else:
                inequalities.append(eq)
        
        # Union equal variables
        for eq in equalities:
            x = ord(eq[0]) - ord('a')
            y = ord(eq[3]) - ord('a')
            union(x, y)
        
        # Check inequalities
        for eq in inequalities:
            x = ord(eq[0]) - ord('a')
            y = ord(eq[3]) - ord('a')
            
            if find(x) == find(y):
                return False
        
        return True
    
    def equationsPossible_approach3_graph_coloring(self, equations: List[str]) -> bool:
        """
        Approach 3: Graph Coloring Alternative
        
        Build equality graph and check inequality constraints.
        
        Time: O(N + 26)
        Space: O(26)
        """
        from collections import defaultdict
        
        # Build equality graph
        graph = defaultdict(list)
        inequalities = []
        
        for eq in equations:
            x, y = eq[0], eq[3]
            
            if eq[1] == '=':
                graph[x].append(y)
                graph[y].append(x)
            else:
                inequalities.append((x, y))
        
        # Find connected components using DFS
        visited = set()
        components = {}
        component_id = 0
        
        def dfs(node, comp_id):
            if node in visited:
                return
            
            visited.add(node)
            components[node] = comp_id
            
            for neighbor in graph[node]:
                dfs(neighbor, comp_id)
        
        # Assign component IDs
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            if letter not in visited:
                dfs(letter, component_id)
                component_id += 1
        
        # Check inequality constraints
        for x, y in inequalities:
            if components.get(x, -1) == components.get(y, -2):
                return False
        
        return True
    
    def equationsPossible_approach4_optimized_uf(self, equations: List[str]) -> bool:
        """
        Approach 4: Optimized Union-Find with Early Termination
        
        Add optimizations for better average case.
        
        Time: O(N)
        Space: O(1)
        """
        parent = list(range(26))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        # Quick check: if any variable equals and not-equals itself
        for eq in equations:
            if eq[0] == eq[3]:
                if eq[1] == '!':  # "x!=x" is impossible
                    return False
                # "x==x" is always true, skip
                continue
        
        # Process equalities
        for eq in equations:
            if eq[1] == '=' and eq[0] != eq[3]:
                x = ord(eq[0]) - ord('a')
                y = ord(eq[3]) - ord('a')
                
                root_x, root_y = find(x), find(y)
                if root_x != root_y:
                    parent[root_y] = root_x
        
        # Check inequalities
        for eq in equations:
            if eq[1] == '!' and eq[0] != eq[3]:
                x = ord(eq[0]) - ord('a')
                y = ord(eq[3]) - ord('a')
                
                if find(x) == find(y):
                    return False
        
        return True

def test_satisfiability_equations():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (equations, expected)
        (["a==b","b!=c","c==a"], False),
        (["b==a","a==b"], True),
        (["a==b","b==c","a==c"], True),
        (["c==c","b==d","x!=z"], True),
        (["a==b","b!=a"], False),
        (["a!=a"], False),
        (["a==a"], True),
        (["a==b","b==c","c==d","d!=a"], False),
        (["a!=b","b!=c","c!=a"], True),
    ]
    
    approaches = [
        ("Union-Find", solution.equationsPossible_approach1_union_find),
        ("Two-Pass UF", solution.equationsPossible_approach2_two_pass),
        ("Graph Coloring", solution.equationsPossible_approach3_graph_coloring),
        ("Optimized UF", solution.equationsPossible_approach4_optimized_uf),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (equations, expected) in enumerate(test_cases):
            result = func(equations)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} {equations} -> Expected: {expected}, Got: {result}")

def demonstrate_union_find_process():
    """Demonstrate Union-Find process for equation solving"""
    print("\n=== Union-Find Process Demo ===")
    
    equations = ["a==b", "b==c", "c!=d", "d==e", "a!=e"]
    print(f"Equations: {equations}")
    
    # Initialize Union-Find
    uf = UnionFind(26)
    
    print(f"\nProcessing equalities:")
    
    # Process equalities
    for eq in equations:
        if eq[1] == '=':
            x = ord(eq[0]) - ord('a')
            y = ord(eq[3]) - ord('a')
            
            print(f"  {eq}: Union({eq[0]}, {eq[3]})")
            uf.union(x, y)
            
            # Show current components
            components = {}
            for letter in [eq[0], eq[3]]:
                idx = ord(letter) - ord('a')
                root = uf.find(idx)
                if root not in components:
                    components[root] = []
                components[root].append(letter)
            
            print(f"    Components: {list(components.values())}")
    
    print(f"\nChecking inequalities:")
    
    # Check inequalities
    for eq in equations:
        if eq[1] == '!':
            x = ord(eq[0]) - ord('a')
            y = ord(eq[3]) - ord('a')
            
            connected = uf.connected(x, y)
            print(f"  {eq}: Connected({eq[0]}, {eq[3]}) = {connected}")
            
            if connected:
                print(f"    ❌ Contradiction! {eq[0]} and {eq[3]} must be equal and not equal")
                return False
            else:
                print(f"    ✓ Valid: {eq[0]} and {eq[3]} are in different components")
    
    print(f"\n✅ All equations are satisfiable!")
    return True

def analyze_union_find_optimizations():
    """Analyze Union-Find optimizations"""
    print("\n=== Union-Find Optimizations Analysis ===")
    
    print("1. **Path Compression:**")
    print("   • Make all nodes point directly to root")
    print("   • Flattens tree structure during find")
    print("   • Reduces future find operations to O(1)")
    print("   • Achieves nearly O(1) amortized time")
    
    print("\n2. **Union by Rank:**")
    print("   • Always attach smaller tree under larger tree")
    print("   • Keeps tree height logarithmic")
    print("   • Prevents degenerate linear chains")
    print("   • Complements path compression")
    
    print("\n3. **Combined Effect:**")
    print("   • Time complexity: O(α(n)) per operation")
    print("   • α(n) is inverse Ackermann function")
    print("   • For practical purposes: α(n) ≤ 5")
    print("   • Nearly constant time operations")
    
    print("\nEquation Problem Specifics:")
    print("• Only 26 possible variables (a-z)")
    print("• Union-Find size is constant: O(26)")
    print("• Total time: O(N) for N equations")
    print("• Space: O(1) constant space")
    
    print("\nTwo-Phase Strategy:")
    print("1. **Phase 1:** Process all equality constraints")
    print("   • Build connected components")
    print("   • Variables in same component must be equal")
    
    print("2. **Phase 2:** Check inequality constraints")
    print("   • Verify no contradictions")
    print("   • Variables in same component cannot be unequal")

def compare_approaches():
    """Compare different approaches to equation satisfiability"""
    print("\n=== Approach Comparison ===")
    
    print("1. **Union-Find Approach:**")
    print("   ✅ Optimal time complexity O(N)")
    print("   ✅ Constant space O(1)")
    print("   ✅ Direct modeling of equivalence relations")
    print("   ✅ Handles transitive equality naturally")
    print("   ❌ Requires understanding of Union-Find")
    
    print("\n2. **Graph Coloring Approach:**")
    print("   ✅ Intuitive graph-based thinking")
    print("   ✅ Clear separation of concerns")
    print("   ✅ Easy to understand and debug")
    print("   ❌ Slightly more complex implementation")
    print("   ❌ Additional space for graph structure")
    
    print("\n3. **Two-Pass Approach:**")
    print("   ✅ Clear separation of equality/inequality")
    print("   ✅ Easy to follow logic")
    print("   ✅ Good for educational purposes")
    print("   ❌ Redundant passes over equations")
    
    print("\nReal-world Applications:")
    print("• **Constraint Satisfaction:** Variable equality constraints")
    print("• **Type Inference:** Programming language type systems")
    print("• **Database Joins:** Equality constraints in SQL")
    print("• **Circuit Design:** Equivalent signal analysis")
    print("• **Mathematical Proof:** Equation system validation")
    
    print("\nKey Insights:")
    print("• Equality is transitive: if a=b and b=c, then a=c")
    print("• Union-Find naturally models transitivity")
    print("• Inequality breaks transitivity chains")
    print("• Two-phase processing separates concerns")
    print("• Constant alphabet size makes problem tractable")

if __name__ == "__main__":
    test_satisfiability_equations()
    demonstrate_union_find_process()
    analyze_union_find_optimizations()
    compare_approaches()

"""
Union-Find Concepts:
1. Disjoint Set Union Data Structure
2. Path Compression Optimization
3. Union by Rank Strategy
4. Equivalence Relation Modeling

Key Problem Insights:
- Equality constraints create equivalence classes
- Union-Find naturally models transitivity
- Inequality constraints must respect equivalence
- Two-phase processing: union then check

Algorithm Strategy:
1. Process equality equations with Union-Find
2. Check inequality equations for contradictions
3. Use path compression for efficiency
4. Leverage constant alphabet size (26 letters)

Union-Find Optimizations:
- Path compression: O(α(n)) amortized time
- Union by rank: keeps trees balanced
- Combined: nearly constant time operations
- For 26 variables: effectively O(1) per operation

Real-world Applications:
- Constraint satisfaction problems
- Type inference systems
- Database query optimization
- Circuit equivalence analysis
- Mathematical equation validation

This problem demonstrates Union-Find for
equivalence relation constraint satisfaction.
"""
