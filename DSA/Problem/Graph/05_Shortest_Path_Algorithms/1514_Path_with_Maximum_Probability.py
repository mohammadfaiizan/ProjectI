"""
1514. Path with Maximum Probability
Difficulty: Medium

Problem:
You are given an undirected weighted graph of n nodes (0-indexed), represented by an 
edge list where edges[i] = [a, b] is an undirected edge connecting the nodes a and b 
with a probability of success of traversing that edge succProb[i].

Given two nodes start and end, find the path with the maximum probability of success 
to go from start to end and return its success probability.

If there is no path from start to end, return 0. Your answer will be accepted if it 
differs from the correct answer by at most 1e-5.

Examples:
Input: n = 3, edges = [[0,1],[1,2],[0,2]], succProb = [0.5,0.5,0.2], start = 0, end = 2
Output: 0.25000

Input: n = 3, edges = [[0,1],[1,2],[0,2]], succProb = [0.5,0.5,0.3], start = 0, end = 2
Output: 0.30000

Input: n = 3, edges = [[0,1]], succProb = [0.5], start = 0, end = 2
Output: 0.00000

Constraints:
- 2 <= n <= 10^4
- 0 <= start, end < n
- start != end
- 0 <= edges.length <= 2 * 10^4
- edges[i].length == 2
- 0 <= a, b < n
- a != b
- 0 <= succProb[i] <= 1
- There is at most one edge between every pair of nodes
"""

from typing import List
import heapq
from collections import defaultdict, deque

class Solution:
    def maxProbability_approach1_modified_dijkstra(self, n: int, edges: List[List[int]], succProb: List[float], start: int, end: int) -> float:
        """
        Approach 1: Modified Dijkstra's Algorithm (Optimal)
        
        Use Dijkstra's algorithm but maximize probability instead of minimizing distance.
        Use negative log probabilities to convert to shortest path problem.
        
        Time: O(E log V)
        Space: O(V + E)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            prob = succProb[i]
            graph[u].append((v, prob))
            graph[v].append((u, prob))
        
        # Modified Dijkstra: maximize probability
        max_prob = [0.0] * n
        max_prob[start] = 1.0
        
        # Use negative probability for max-heap simulation
        pq = [(-1.0, start)]
        
        while pq:
            neg_prob, node = heapq.heappop(pq)
            prob = -neg_prob
            
            if node == end:
                return prob
            
            if prob < max_prob[node]:
                continue  # Already found better path to this node
            
            for neighbor, edge_prob in graph[node]:
                new_prob = prob * edge_prob
                
                if new_prob > max_prob[neighbor]:
                    max_prob[neighbor] = new_prob
                    heapq.heappush(pq, (-new_prob, neighbor))
        
        return max_prob[end]
    
    def maxProbability_approach2_bellman_ford_variant(self, n: int, edges: List[List[int]], succProb: List[float], start: int, end: int) -> float:
        """
        Approach 2: Bellman-Ford Variant for Maximum Probability
        
        Use Bellman-Ford style relaxation to find maximum probability paths.
        
        Time: O(V * E)
        Space: O(V + E)
        """
        # Initialize probabilities
        max_prob = [0.0] * n
        max_prob[start] = 1.0
        
        # Relax edges up to n-1 times
        for _ in range(n - 1):
            updated = False
            
            for i, (u, v) in enumerate(edges):
                prob = succProb[i]
                
                # Check both directions (undirected graph)
                if max_prob[u] * prob > max_prob[v]:
                    max_prob[v] = max_prob[u] * prob
                    updated = True
                
                if max_prob[v] * prob > max_prob[u]:
                    max_prob[u] = max_prob[v] * prob
                    updated = True
            
            if not updated:
                break  # Early termination
        
        return max_prob[end]
    
    def maxProbability_approach3_spfa_maximum_variant(self, n: int, edges: List[List[int]], succProb: List[float], start: int, end: int) -> float:
        """
        Approach 3: SPFA Variant for Maximum Probability
        
        Use SPFA (queue-optimized Bellman-Ford) for maximum probability.
        
        Time: O(V * E) worst case, often better
        Space: O(V + E)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            prob = succProb[i]
            graph[u].append((v, prob))
            graph[v].append((u, prob))
        
        # SPFA for maximum probability
        max_prob = [0.0] * n
        max_prob[start] = 1.0
        
        queue = deque([start])
        in_queue = [False] * n
        in_queue[start] = True
        
        while queue:
            u = queue.popleft()
            in_queue[u] = False
            
            for v, prob in graph[u]:
                new_prob = max_prob[u] * prob
                
                if new_prob > max_prob[v]:
                    max_prob[v] = new_prob
                    
                    if not in_queue[v]:
                        queue.append(v)
                        in_queue[v] = True
        
        return max_prob[end]
    
    def maxProbability_approach4_log_transformation(self, n: int, edges: List[List[int]], succProb: List[float], start: int, end: int) -> float:
        """
        Approach 4: Log Transformation to Shortest Path
        
        Convert to shortest path by taking negative logarithms.
        
        Time: O(E log V)
        Space: O(V + E)
        """
        import math
        
        # Build adjacency list with negative log weights
        graph = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            prob = succProb[i]
            if prob > 0:  # Avoid log(0)
                weight = -math.log(prob)
                graph[u].append((v, weight))
                graph[v].append((u, weight))
        
        # Standard Dijkstra on transformed graph
        distances = [float('inf')] * n
        distances[start] = 0.0
        
        pq = [(0.0, start)]
        
        while pq:
            dist, node = heapq.heappop(pq)
            
            if node == end:
                return math.exp(-dist)
            
            if dist > distances[node]:
                continue
            
            for neighbor, weight in graph[node]:
                new_dist = dist + weight
                
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))
        
        return 0.0 if distances[end] == float('inf') else math.exp(-distances[end])
    
    def maxProbability_approach5_dfs_memoization(self, n: int, edges: List[List[int]], succProb: List[float], start: int, end: int) -> float:
        """
        Approach 5: DFS with Memoization
        
        Use DFS to explore all paths with memoization for efficiency.
        
        Time: O(V + E) with memoization
        Space: O(V + E)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            prob = succProb[i]
            graph[u].append((v, prob))
            graph[v].append((u, prob))
        
        # Memoization for maximum probabilities
        memo = {}
        
        def dfs(node, target, visited):
            """DFS to find maximum probability from node to target"""
            if node == target:
                return 1.0
            
            if node in visited:
                return 0.0  # Avoid cycles
            
            if node in memo:
                return memo[node]
            
            visited.add(node)
            max_prob = 0.0
            
            for neighbor, prob in graph[node]:
                if neighbor not in visited:
                    max_prob = max(max_prob, prob * dfs(neighbor, target, visited))
            
            visited.remove(node)
            memo[node] = max_prob
            return max_prob
        
        return dfs(start, end, set())

def test_max_probability():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (n, edges, succProb, start, end, expected)
        (3, [[0,1],[1,2],[0,2]], [0.5,0.5,0.2], 0, 2, 0.25),
        (3, [[0,1],[1,2],[0,2]], [0.5,0.5,0.3], 0, 2, 0.3),
        (3, [[0,1]], [0.5], 0, 2, 0.0),
        (4, [[0,1],[1,2],[2,3],[0,3]], [0.5,0.5,0.3,0.4], 0, 3, 0.4),
        (2, [[0,1]], [1.0], 0, 1, 1.0),
    ]
    
    approaches = [
        ("Modified Dijkstra", solution.maxProbability_approach1_modified_dijkstra),
        ("Bellman-Ford Variant", solution.maxProbability_approach2_bellman_ford_variant),
        ("SPFA Maximum", solution.maxProbability_approach3_spfa_maximum_variant),
        ("Log Transformation", solution.maxProbability_approach4_log_transformation),
        ("DFS Memoization", solution.maxProbability_approach5_dfs_memoization),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (n, edges, succProb, start, end, expected) in enumerate(test_cases):
            result = func(n, edges[:], succProb[:], start, end)
            status = "✓" if abs(result - expected) < 1e-5 else "✗"
            print(f"Test {i+1}: {status} Expected: {expected:.5f}, Got: {result:.5f}")

def demonstrate_probability_maximization():
    """Demonstrate probability maximization process"""
    print("\n=== Probability Maximization Demo ===")
    
    n = 3
    edges = [[0,1],[1,2],[0,2]]
    succProb = [0.5,0.5,0.2]
    start = 0
    end = 2
    
    print(f"Graph: {n} nodes")
    print(f"Edges: {edges}")
    print(f"Success probabilities: {succProb}")
    print(f"Find path from {start} to {end} with maximum probability")
    
    # Build adjacency list
    graph = defaultdict(list)
    for i, (u, v) in enumerate(edges):
        prob = succProb[i]
        graph[u].append((v, prob))
        graph[v].append((u, prob))
    
    print(f"\nAdjacency list: {dict(graph)}")
    
    print(f"\nPossible paths from {start} to {end}:")
    print(f"1. Direct path: 0 -> 2, probability = {succProb[2]}")
    print(f"2. Via node 1: 0 -> 1 -> 2, probability = {succProb[0]} * {succProb[1]} = {succProb[0] * succProb[1]}")
    
    max_prob = max(succProb[2], succProb[0] * succProb[1])
    print(f"\nMaximum probability: {max_prob}")
    
    # Modified Dijkstra demonstration
    print(f"\nModified Dijkstra's algorithm:")
    
    max_prob_array = [0.0] * n
    max_prob_array[start] = 1.0
    pq = [(-1.0, start)]
    
    print(f"Initial: max_prob = {max_prob_array}, pq = {pq}")
    
    step = 0
    while pq:
        step += 1
        neg_prob, node = heapq.heappop(pq)
        prob = -neg_prob
        
        print(f"\nStep {step}: Processing node {node} with probability {prob:.3f}")
        
        if node == end:
            print(f"  Reached target! Maximum probability = {prob:.3f}")
            break
        
        if prob < max_prob_array[node]:
            print(f"  Skipping (already have better path)")
            continue
        
        for neighbor, edge_prob in graph[node]:
            new_prob = prob * edge_prob
            print(f"  Edge to {neighbor}: {prob:.3f} * {edge_prob} = {new_prob:.3f}")
            
            if new_prob > max_prob_array[neighbor]:
                print(f"    Updating max_prob[{neighbor}] = {new_prob:.3f}")
                max_prob_array[neighbor] = new_prob
                heapq.heappush(pq, (-new_prob, neighbor))
        
        print(f"  Current max_prob: {[f'{p:.3f}' for p in max_prob_array]}")
        print(f"  Priority queue: {[f'({-p:.3f}, {n})' for p, n in sorted(pq)]}")

def analyze_probability_vs_distance():
    """Analyze the relationship between probability and distance optimization"""
    print("\n=== Probability vs Distance Optimization Analysis ===")
    
    print("Key Differences:")
    
    print("\n1. **Objective Function:**")
    print("   • Distance: Minimize sum of edge weights")
    print("   • Probability: Maximize product of edge probabilities")
    
    print("\n2. **Mathematical Transformation:**")
    print("   • Distance: d(path) = d₁ + d₂ + ... + dₙ")
    print("   • Probability: p(path) = p₁ × p₂ × ... × pₙ")
    print("   • Log transform: log(p) = log(p₁) + log(p₂) + ... + log(pₙ)")
    
    print("\n3. **Algorithm Adaptation:**")
    print("   • Max-heap instead of min-heap")
    print("   • Multiplication instead of addition")
    print("   • Initialize with 1.0 instead of 0")
    print("   • Update condition: new_prob > old_prob")
    
    print("\n4. **Edge Cases:**")
    print("   • Zero probability edges (disconnected)")
    print("   • Floating point precision")
    print("   • Probability = 1.0 (perfect reliability)")
    
    print("\nLog Transformation Approach:")
    print("• Convert: max(p₁ × p₂ × ... × pₙ)")
    print("• To: min(-log(p₁) - log(p₂) - ... - log(pₙ))")
    print("• Advantages:")
    print("  - Use standard shortest path algorithms")
    print("  - Avoid floating point multiplication errors")
    print("  - Numerical stability for small probabilities")
    
    print("\nNumerical Considerations:")
    print("• **Underflow:** Very small probabilities may underflow")
    print("• **Precision:** Floating point comparison with epsilon")
    print("• **Log domain:** More stable for very small probabilities")
    print("• **Zero handling:** log(0) = -∞, need special handling")

def demonstrate_log_transformation():
    """Demonstrate log transformation technique"""
    print("\n=== Log Transformation Demo ===")
    
    import math
    
    probabilities = [0.5, 0.8, 0.3, 0.9]
    print(f"Original probabilities: {probabilities}")
    
    # Convert to negative log distances
    log_distances = [-math.log(p) for p in probabilities]
    print(f"Negative log distances: {[f'{d:.3f}' for d in log_distances]}")
    
    print(f"\nPath probability calculation:")
    path_probs = [0.5, 0.8]
    path_prob = 1.0
    for p in path_probs:
        path_prob *= p
    print(f"Direct: {path_probs[0]} × {path_probs[1]} = {path_prob}")
    
    path_log_dist = sum(-math.log(p) for p in path_probs)
    converted_prob = math.exp(-path_log_dist)
    print(f"Log transform: exp(-({-math.log(path_probs[0]):.3f} + {-math.log(path_probs[1]):.3f})) = {converted_prob:.3f}")
    
    print(f"\nAdvantages of log transformation:")
    print(f"• Numerical stability for small probabilities")
    print(f"• Can use standard shortest path algorithms")
    print(f"• Avoids underflow in probability multiplication")
    print(f"• Natural handling of zero probabilities")

def compare_algorithmic_approaches():
    """Compare different algorithmic approaches for maximum probability"""
    print("\n=== Algorithmic Approaches Comparison ===")
    
    print("1. **Modified Dijkstra (Recommended):**")
    print("   ✅ Optimal O(E log V) time complexity")
    print("   ✅ Handles probabilistic weights naturally")
    print("   ✅ Early termination when target reached")
    print("   ❌ Requires max-heap simulation or priority queue modification")
    
    print("\n2. **Bellman-Ford Variant:**")
    print("   ✅ Simple modification of standard algorithm")
    print("   ✅ Easy to understand and implement")
    print("   ✅ Handles general case (could handle negative log weights)")
    print("   ❌ O(V × E) time complexity - slower for dense graphs")
    
    print("\n3. **SPFA Variant:**")
    print("   ✅ Often faster than Bellman-Ford in practice")
    print("   ✅ Queue-based optimization")
    print("   ✅ Good average case performance")
    print("   ❌ Still O(V × E) worst case")
    
    print("\n4. **Log Transformation + Standard Dijkstra:**")
    print("   ✅ Can use existing shortest path implementations")
    print("   ✅ Numerically stable")
    print("   ✅ Handles edge cases (zero probability) well")
    print("   ❌ Additional math operations (log/exp)")
    
    print("\n5. **DFS with Memoization:**")
    print("   ✅ Intuitive recursive approach")
    print("   ✅ Natural handling of all paths")
    print("   ❌ May have exponential worst case without good memoization")
    print("   ❌ Stack overflow risk for deep graphs")
    
    print("\nWhen to Use Each:")
    print("• **Modified Dijkstra:** Most cases, especially dense graphs")
    print("• **Bellman-Ford:** When you need algorithm simplicity")
    print("• **SPFA:** Sparse graphs with good average case")
    print("• **Log Transformation:** When numerical stability is crucial")
    print("• **DFS Memoization:** Educational purposes or very sparse graphs")
    
    print("\nReal-world Applications:")
    print("• **Network reliability:** Communication network success probability")
    print("• **Route planning:** Most reliable path in unreliable networks")
    print("• **Investment analysis:** Maximum probability of profitable trades")
    print("• **Game AI:** Highest success probability strategies")
    print("• **Biological networks:** Most likely protein interaction paths")

if __name__ == "__main__":
    test_max_probability()
    demonstrate_probability_maximization()
    analyze_probability_vs_distance()
    demonstrate_log_transformation()
    compare_algorithmic_approaches()

"""
Shortest Path Concepts:
1. Maximum Probability Path (Dual of Shortest Path)
2. Modified Dijkstra's Algorithm for Optimization
3. Log Transformation for Mathematical Stability
4. Probabilistic Graph Analysis
5. Alternative Optimization Objectives

Key Problem Insights:
- Maximize probability product instead of minimizing distance sum
- Use modified Dijkstra with max-heap behavior
- Log transformation converts multiplication to addition
- Floating point precision requires careful handling

Algorithm Strategy:
1. Model network as undirected weighted graph
2. Use modified Dijkstra to maximize probability
3. Track maximum probability to each node
4. Return probability to target node

Modified Dijkstra Adaptations:
- Use negative probabilities in min-heap for max behavior
- Initialize source probability to 1.0
- Update condition: new_prob > current_max_prob
- Multiply probabilities instead of adding distances

Mathematical Foundation:
- Path probability = product of edge probabilities
- Log transformation: max(∏pᵢ) = max(exp(∑log(pᵢ)))
- Equivalent to: min(-∑log(pᵢ)) in distance space
- Numerical stability through log domain computation

Real-world Applications:
- Network reliability analysis
- Communication system optimization
- Financial risk assessment
- Game theory and strategy
- Biological pathway analysis

This problem demonstrates adaptation of shortest path
algorithms for alternative optimization objectives.
"""
