"""
1791. Find Center of Star Graph - Multiple Approaches
Difficulty: Easy

There is an undirected star graph consisting of n nodes labeled from 1 to n. 
A star graph is a graph where there is one center node and exactly n - 1 edges 
that connect the center node with every other node.

You are given a 2D integer array edges where each edges[i] = [ui, vi] indicates 
that there is an edge between the nodes ui and vi. Return the center of the given star graph.
"""

from typing import List, Dict, Set
from collections import defaultdict, Counter

class FindCenterStarGraph:
    """Multiple approaches to find center of star graph"""
    
    def findCenter_degree_counting(self, edges: List[List[int]]) -> int:
        """
        Approach 1: Degree Counting
        
        In a star graph, center has degree n-1, others have degree 1.
        
        Time: O(E), Space: O(V)
        """
        degree = defaultdict(int)
        
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        
        # Find node with maximum degree
        return max(degree.keys(), key=lambda x: degree[x])
    
    def findCenter_first_two_edges(self, edges: List[List[int]]) -> int:
        """
        Approach 2: Check First Two Edges (Optimal)
        
        Center must appear in both first and second edge.
        
        Time: O(1), Space: O(1)
        """
        # The center must be in both first and second edge
        first_edge = edges[0]
        second_edge = edges[1]
        
        # Find common node
        if first_edge[0] in second_edge:
            return first_edge[0]
        else:
            return first_edge[1]
    
    def findCenter_set_intersection(self, edges: List[List[int]]) -> int:
        """
        Approach 3: Set Intersection
        
        Use set intersection of first two edges.
        
        Time: O(1), Space: O(1)
        """
        return list(set(edges[0]) & set(edges[1]))[0]
    
    def findCenter_frequency_count(self, edges: List[List[int]]) -> int:
        """
        Approach 4: Frequency Counting
        
        Count frequency and find most frequent node.
        
        Time: O(E), Space: O(V)
        """
        counter = Counter()
        
        for u, v in edges:
            counter[u] += 1
            counter[v] += 1
        
        return counter.most_common(1)[0][0]
    
    def findCenter_early_termination(self, edges: List[List[int]]) -> int:
        """
        Approach 5: Early Termination
        
        Stop as soon as we find a node with degree > 1.
        
        Time: O(E) worst case, O(1) average, Space: O(V)
        """
        degree = defaultdict(int)
        
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
            
            # Early termination: if degree > 1, it's the center
            if degree[u] > 1:
                return u
            if degree[v] > 1:
                return v
        
        # Should not reach here for valid star graph
        return -1
    
    def findCenter_mathematical(self, edges: List[List[int]]) -> int:
        """
        Approach 6: Mathematical Property
        
        Use mathematical properties of star graph.
        
        Time: O(1), Space: O(1)
        """
        # In a star graph with n nodes, there are n-1 edges
        # Center appears in all edges, so appears 2*(n-1) times in edge list
        # Other nodes appear exactly once
        
        # Since we know it's a star graph, just check first two edges
        a, b = edges[0]
        c, d = edges[1]
        
        # The center is the common node
        return a if a == c or a == d else b

def test_find_center():
    """Test find center algorithms"""
    solver = FindCenterStarGraph()
    
    test_cases = [
        ([[1,2],[2,3],[4,2]], 2, "Star with center 2"),
        ([[1,2],[5,1],[1,3],[1,4]], 1, "Star with center 1"),
        ([[1,2]], 1, "Two nodes (either can be center)"),  # Note: both 1 and 2 are valid
        ([[3,1],[2,3],[3,4],[3,5]], 3, "Star with center 3"),
    ]
    
    algorithms = [
        ("Degree Counting", solver.findCenter_degree_counting),
        ("First Two Edges", solver.findCenter_first_two_edges),
        ("Set Intersection", solver.findCenter_set_intersection),
        ("Frequency Count", solver.findCenter_frequency_count),
        ("Early Termination", solver.findCenter_early_termination),
        ("Mathematical", solver.findCenter_mathematical),
    ]
    
    print("=== Testing Find Center of Star Graph ===")
    
    for edges, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        print(f"Edges: {edges}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(edges)
                # For two-node case, both nodes are valid centers
                if len(edges) == 1:
                    status = "✓" if result in edges[0] else "✗"
                else:
                    status = "✓" if result == expected else "✗"
                print(f"{alg_name:18} | {status} | Center: {result}")
            except Exception as e:
                print(f"{alg_name:18} | ERROR: {str(e)[:30]}")

def demonstrate_star_graph_properties():
    """Demonstrate star graph properties"""
    print("\n=== Star Graph Properties ===")
    
    print("Key Properties:")
    print("• Exactly one center node")
    print("• Center connected to all other nodes")
    print("• Center has degree n-1")
    print("• All other nodes have degree 1")
    print("• Total edges = n-1")
    
    print("\nOptimization Insights:")
    print("• Center must appear in every edge")
    print("• Only need to check first two edges")
    print("• Common node in first two edges is center")
    print("• O(1) solution possible")
    
    print("\nCompetitive Programming Tips:")
    print("• Always look for O(1) solutions")
    print("• Use problem constraints effectively")
    print("• Mathematical properties often lead to optimal solutions")
    print("• Early termination can improve average case")

def analyze_complexity():
    """Analyze complexity of different approaches"""
    print("\n=== Complexity Analysis ===")
    
    print("Algorithm Comparison:")
    
    print("\n1. **First Two Edges (Optimal):**")
    print("   • Time: O(1)")
    print("   • Space: O(1)")
    print("   • Pros: Optimal complexity, simple logic")
    print("   • Cons: Requires understanding of star graph properties")
    
    print("\n2. **Set Intersection:**")
    print("   • Time: O(1)")
    print("   • Space: O(1)")
    print("   • Pros: Clean code, optimal complexity")
    print("   • Cons: Slightly more overhead than direct comparison")
    
    print("\n3. **Degree Counting:**")
    print("   • Time: O(E)")
    print("   • Space: O(V)")
    print("   • Pros: General approach, works for any graph")
    print("   • Cons: Not optimal for star graph")
    
    print("\n4. **Early Termination:**")
    print("   • Time: O(1) average, O(E) worst case")
    print("   • Space: O(V)")
    print("   • Pros: Good average case performance")
    print("   • Cons: Still not optimal")
    
    print("\nRecommendation: Use 'First Two Edges' approach for optimal performance")

if __name__ == "__main__":
    test_find_center()
    demonstrate_star_graph_properties()
    analyze_complexity()

"""
Find Center of Star Graph - Key Insights:

1. **Problem Understanding:**
   - Star graph has exactly one center node
   - Center connects to all other nodes
   - Center has degree n-1, others have degree 1
   - Given edges form a valid star graph

2. **Optimization Opportunities:**
   - Center must appear in every edge
   - Only need first two edges to find center
   - Common node in first two edges is center
   - O(1) solution possible

3. **Algorithm Categories:**
   - Brute force: Count all degrees
   - Optimized: Use first two edges
   - Mathematical: Leverage star graph properties
   - Early termination: Stop when center found

4. **Competitive Programming Lessons:**
   - Always analyze problem constraints
   - Look for mathematical properties
   - Optimal solutions often use problem structure
   - Simple observations lead to elegant solutions

5. **Implementation Considerations:**
   - Handle edge cases (two nodes)
   - Choose appropriate data structures
   - Consider space-time tradeoffs
   - Validate assumptions about input

This problem demonstrates how understanding problem structure
can lead to dramatically more efficient solutions.
"""
