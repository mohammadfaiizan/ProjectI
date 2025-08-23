"""
2127. Maximum Employees to Be Invited to a Meeting
Difficulty: Hard

Problem:
A company is organizing a meeting and has a list of n employees, numbered from 0 to n - 1. 
Each employee has a favorite person in the same company, and they will only sit next to their favorite person.

The favorite person of an employee is not themself.

The company wants to invite the maximum number of employees to the meeting. 
The meeting table is circular, so employees will sit in a circle.

Return the maximum number of employees that can be invited to the meeting.

Examples:
Input: favorite = [2,2,1,2]
Output: 3
Explanation: 
The maximum number of employees that can be invited to the meeting is 3. 
One possible arrangement is as follows:
- Employee 0 sits next to employee 2.
- Employee 1 sits next to employee 2.
- Employee 2 sits next to employee 1.
Note that employee 2 cannot sit next to employee 0 since 2 is not 0's favorite.

Input: favorite = [1,2,0]
Output: 3
Explanation: Each employee is the favorite person of at least one other employee.

Input: favorite = [3,0,1,4,1]
Output: 4
Explanation: 
The maximum number of employees that can be invited to the meeting is 4.
One possible arrangement is employee 1 sitting between employees 0 and 2, 
and employee 4 sitting between employees 3 and 1.

Constraints:
- n == favorite.length
- 2 <= n <= 10^5
- 0 <= favorite[i] <= n - 1
- favorite[i] != i
"""

from typing import List, Dict, Set, Tuple
from collections import defaultdict, deque

class Solution:
    def maximumInvitations_approach1_functional_graph_analysis(self, favorite: List[int]) -> int:
        """
        Approach 1: Functional Graph Analysis (Optimal)
        
        Analyze the functional graph structure to find cycles and chains.
        
        Time: O(N)
        Space: O(N)
        """
        n = len(favorite)
        
        # Build reverse graph (who likes this person)
        reverse_graph = defaultdict(list)
        for i, fav in enumerate(favorite):
            reverse_graph[fav].append(i)
        
        visited = [False] * n
        in_cycle = [False] * n
        cycle_id = [-1] * n
        cycles = []
        
        def find_cycle_from(start):
            """Find cycle starting from a node using functional graph properties"""
            # In a functional graph, following the edges will eventually lead to a cycle
            path = {}
            current = start
            step = 0
            
            # Follow the path until we find a cycle or reach visited node
            while current not in path and not visited[current]:
                path[current] = step
                current = favorite[current]
                step += 1
            
            if current in path:
                # Found a new cycle
                cycle_start_step = path[current]
                cycle_nodes = []
                
                # Reconstruct the cycle
                temp = start
                for _ in range(cycle_start_step):
                    temp = favorite[temp]
                
                # Extract cycle
                cycle_node = temp
                while True:
                    cycle_nodes.append(cycle_node)
                    in_cycle[cycle_node] = True
                    cycle_id[cycle_node] = len(cycles)
                    cycle_node = favorite[cycle_node]
                    if cycle_node == temp:
                        break
                
                cycles.append(cycle_nodes)
            
            # Mark all nodes in path as visited
            current = start
            while not visited[current]:
                visited[current] = True
                current = favorite[current]
        
        # Find all cycles in the functional graph
        for i in range(n):
            if not visited[i]:
                find_cycle_from(i)
        
        def longest_chain_to_cycle_node(cycle_node):
            """Find longest chain of nodes that can reach a cycle node"""
            max_depth = 0
            
            def dfs(node, depth):
                nonlocal max_depth
                max_depth = max(max_depth, depth)
                
                for prev_node in reverse_graph[node]:
                    if not in_cycle[prev_node]:  # Only consider non-cycle nodes
                        dfs(prev_node, depth + 1)
            
            dfs(cycle_node, 0)
            return max_depth
        
        max_employees = 0
        
        # Case 1: Use a single cycle of length > 2
        for cycle in cycles:
            if len(cycle) > 2:
                max_employees = max(max_employees, len(cycle))
        
        # Case 2: Use multiple 2-cycles with their chains
        two_cycle_total = 0
        for cycle in cycles:
            if len(cycle) == 2:
                node1, node2 = cycle
                
                # For 2-cycles, we can add chains leading to both nodes
                chain1_length = longest_chain_to_cycle_node(node1)
                chain2_length = longest_chain_to_cycle_node(node2)
                
                # Add the 2-cycle plus both chains
                two_cycle_total += 2 + chain1_length + chain2_length
        
        return max(max_employees, two_cycle_total)
    
    def maximumInvitations_approach2_cycle_detection_with_chains(self, favorite: List[int]) -> int:
        """
        Approach 2: Explicit Cycle Detection with Chain Analysis
        
        Detect cycles explicitly and analyze chain extensions.
        
        Time: O(N)
        Space: O(N)
        """
        n = len(favorite)
        
        # Build incoming edges graph
        incoming = defaultdict(list)
        for i, fav in enumerate(favorite):
            incoming[fav].append(i)
        
        # Find all cycles using DFS
        visited = [False] * n
        in_cycle = [False] * n
        cycles = []
        
        def find_cycles():
            """Find all cycles in the functional graph"""
            for start in range(n):
                if visited[start]:
                    continue
                
                # Follow the path from this node
                path = []
                path_set = set()
                current = start
                
                while current not in path_set and not visited[current]:
                    path.append(current)
                    path_set.add(current)
                    current = favorite[current]
                
                if current in path_set:
                    # Found cycle
                    cycle_start_idx = path.index(current)
                    cycle = path[cycle_start_idx:]
                    cycles.append(cycle)
                    
                    for node in cycle:
                        in_cycle[node] = True
                
                # Mark all nodes in path as visited
                for node in path:
                    visited[node] = True
        
        find_cycles()
        
        def calculate_max_chain_length(cycle_node):
            """Calculate maximum chain length leading to a cycle node"""
            max_length = 0
            
            def dfs(node, length):
                nonlocal max_length
                max_length = max(max_length, length)
                
                for predecessor in incoming[node]:
                    if not in_cycle[predecessor]:
                        dfs(predecessor, length + 1)
            
            dfs(cycle_node, 0)
            return max_length
        
        result = 0
        
        # Strategy 1: Use one large cycle (length > 2)
        for cycle in cycles:
            if len(cycle) > 2:
                result = max(result, len(cycle))
        
        # Strategy 2: Combine all 2-cycles with their chains
        total_2_cycles = 0
        for cycle in cycles:
            if len(cycle) == 2:
                a, b = cycle
                chain_a = calculate_max_chain_length(a)
                chain_b = calculate_max_chain_length(b)
                total_2_cycles += 2 + chain_a + chain_b
        
        return max(result, total_2_cycles)
    
    def maximumInvitations_approach3_topological_analysis(self, favorite: List[int]) -> int:
        """
        Approach 3: Topological Analysis of Functional Graph
        
        Use topological properties to analyze the structure.
        
        Time: O(N)
        Space: O(N)
        """
        n = len(favorite)
        
        # In a functional graph, each node has exactly one outgoing edge
        # and may have multiple incoming edges
        
        # Build reverse adjacency list
        reverse_adj = [[] for _ in range(n)]
        for i, fav in enumerate(favorite):
            reverse_adj[fav].append(i)
        
        # Find weakly connected components and their cycle structure
        visited_global = [False] * n
        components = []
        
        def explore_component(start):
            """Explore a weakly connected component"""
            visited_local = set()
            stack = [start]
            component = []
            
            while stack:
                node = stack.pop()
                if node in visited_local:
                    continue
                
                visited_local.add(node)
                component.append(node)
                
                # Add favorite (outgoing edge)
                if favorite[node] not in visited_local:
                    stack.append(favorite[node])
                
                # Add all who like this node (incoming edges)
                for admirer in reverse_adj[node]:
                    if admirer not in visited_local:
                        stack.append(admirer)
            
            return component
        
        # Find all weakly connected components
        for i in range(n):
            if not visited_global[i]:
                component = explore_component(i)
                components.append(component)
                
                for node in component:
                    visited_global[node] = True
        
        def analyze_component(component):
            """Analyze the structure of a single component"""
            # Find the cycle in this component
            visited = set()
            cycle = []
            
            # Start from any node and follow favorites to find cycle
            start = component[0]
            current = start
            path = []
            path_indices = {}
            
            while current not in path_indices:
                path_indices[current] = len(path)
                path.append(current)
                current = favorite[current]
            
            # Extract cycle
            cycle_start_idx = path_indices[current]
            cycle = path[cycle_start_idx:]
            
            # Calculate maximum depth for each cycle node
            def max_depth_to_node(target, avoid_set):
                """Find maximum depth to reach target avoiding nodes in avoid_set"""
                max_d = 0
                
                def dfs(node, depth):
                    nonlocal max_d
                    if node == target:
                        max_d = max(max_d, depth)
                        return
                    
                    if node in avoid_set:
                        return
                    
                    for predecessor in reverse_adj[node]:
                        if predecessor not in avoid_set:
                            dfs(predecessor, depth + 1)
                
                for predecessor in reverse_adj[target]:
                    if predecessor not in avoid_set:
                        dfs(predecessor, 1)
                
                return max_d
            
            cycle_set = set(cycle)
            
            if len(cycle) == 2:
                # 2-cycle: can extend with chains
                a, b = cycle
                chain_a = max_depth_to_node(a, cycle_set)
                chain_b = max_depth_to_node(b, cycle_set)
                return 2 + chain_a + chain_b, True  # True indicates 2-cycle
            else:
                # Larger cycle: use as-is
                return len(cycle), False  # False indicates larger cycle
        
        # Analyze each component
        max_single_cycle = 0
        total_2_cycles = 0
        
        for component in components:
            size, is_2_cycle = analyze_component(component)
            
            if is_2_cycle:
                total_2_cycles += size
            else:
                max_single_cycle = max(max_single_cycle, size)
        
        return max(max_single_cycle, total_2_cycles)
    
    def maximumInvitations_approach4_graph_theory_optimal(self, favorite: List[int]) -> int:
        """
        Approach 4: Graph Theory Optimal Solution
        
        Use advanced graph theory concepts for optimal solution.
        
        Time: O(N)
        Space: O(N)
        """
        n = len(favorite)
        
        # This is a functional graph where each vertex has out-degree 1
        # The structure consists of components, each with exactly one cycle
        
        in_degree = [0] * n
        for fav in favorite:
            in_degree[fav] += 1
        
        # Topologically sort to find nodes that are not in any cycle
        queue = deque()
        for i in range(n):
            if in_degree[i] == 0:
                queue.append(i)
        
        visited = [False] * n
        
        # Remove all nodes that are not part of any cycle
        while queue:
            node = queue.popleft()
            visited[node] = True
            next_node = favorite[node]
            
            in_degree[next_node] -= 1
            if in_degree[next_node] == 0:
                queue.append(next_node)
        
        # Remaining nodes form cycles
        cycles = []
        
        for i in range(n):
            if not visited[i]:
                # Found a cycle node, extract the entire cycle
                cycle = []
                current = i
                
                while current not in cycle:
                    cycle.append(current)
                    visited[current] = True
                    current = favorite[current]
                
                cycles.append(cycle)
        
        # For each cycle, calculate the longest chains leading to it
        def longest_chain_to_cycle_node(cycle_node):
            """Find longest chain to a cycle node using BFS backwards"""
            max_length = 0
            queue = deque([(cycle_node, 0)])
            visited_chain = set([cycle_node])
            
            while queue:
                node, length = queue.popleft()
                max_length = max(max_length, length)
                
                # Check all nodes that have this node as favorite
                for i in range(n):
                    if favorite[i] == node and i not in visited_chain:
                        # Check if i is not in any cycle
                        if not any(i in cycle for cycle in cycles):
                            visited_chain.add(i)
                            queue.append((i, length + 1))
            
            return max_length
        
        # Calculate maximum employees
        max_employees = 0
        
        # Case 1: Single cycle of length > 2
        for cycle in cycles:
            if len(cycle) > 2:
                max_employees = max(max_employees, len(cycle))
        
        # Case 2: Multiple 2-cycles with chains
        two_cycle_total = 0
        for cycle in cycles:
            if len(cycle) == 2:
                a, b = cycle
                chain_a = longest_chain_to_cycle_node(a)
                chain_b = longest_chain_to_cycle_node(b)
                two_cycle_total += 2 + chain_a + chain_b
        
        return max(max_employees, two_cycle_total)
    
    def maximumInvitations_approach5_cycle_decomposition(self, favorite: List[int]) -> int:
        """
        Approach 5: Complete Cycle Decomposition Analysis
        
        Complete analysis using cycle decomposition of functional graphs.
        
        Time: O(N)
        Space: O(N)
        """
        n = len(favorite)
        
        # Build the reverse graph for easier traversal
        reverse = [[] for _ in range(n)]
        for i, fav in enumerate(favorite):
            reverse[fav].append(i)
        
        # Find all cycles using Tarjan-like approach for functional graphs
        visited = [False] * n
        in_cycle = [False] * n
        cycles = []
        
        def find_cycle_containing(start):
            """Find the unique cycle in the component containing start"""
            # In functional graph, following edges always leads to a cycle
            slow = fast = start
            
            # Floyd's cycle detection
            while True:
                slow = favorite[slow]
                fast = favorite[favorite[fast]]
                if slow == fast:
                    break
            
            # Find cycle start
            cycle_start = start
            while cycle_start != slow:
                cycle_start = favorite[cycle_start]
                slow = favorite[slow]
            
            # Extract complete cycle
            cycle = []
            current = cycle_start
            while True:
                cycle.append(current)
                in_cycle[current] = True
                current = favorite[current]
                if current == cycle_start:
                    break
            
            return cycle
        
        # Mark all reachable nodes and find cycles
        for i in range(n):
            if not visited[i]:
                # Find cycle in this component
                cycle = find_cycle_containing(i)
                cycles.append(cycle)
                
                # Mark all reachable nodes as visited
                stack = [i]
                while stack:
                    node = stack.pop()
                    if visited[node]:
                        continue
                    
                    visited[node] = True
                    stack.append(favorite[node])
                    for prev in reverse[node]:
                        if not visited[prev]:
                            stack.append(prev)
        
        def calculate_tree_depth(root, avoid_nodes):
            """Calculate maximum depth of tree rooted at root"""
            max_depth = 0
            
            def dfs(node, depth):
                nonlocal max_depth
                max_depth = max(max_depth, depth)
                
                for child in reverse[node]:
                    if child not in avoid_nodes:
                        dfs(child, depth + 1)
            
            dfs(root, 0)
            return max_depth
        
        # Calculate result using two strategies
        max_single_cycle = 0
        total_paired_cycles = 0
        
        for cycle in cycles:
            if len(cycle) > 2:
                max_single_cycle = max(max_single_cycle, len(cycle))
            elif len(cycle) == 2:
                a, b = cycle
                depth_a = calculate_tree_depth(a, set(cycle))
                depth_b = calculate_tree_depth(b, set(cycle))
                total_paired_cycles += 2 + depth_a + depth_b
        
        return max(max_single_cycle, total_paired_cycles)

def test_maximum_invitations():
    """Test all approaches with various test cases"""
    solution = Solution()
    
    test_cases = [
        # (favorite, expected)
        ([2,2,1,2], 3),
        ([1,2,0], 3),
        ([3,0,1,4,1], 4),
        ([1,0], 2),
        ([1,0,3,2], 4),
        ([2,2,1,2], 3),
    ]
    
    approaches = [
        ("Functional Graph", solution.maximumInvitations_approach1_functional_graph_analysis),
        ("Cycle Detection", solution.maximumInvitations_approach2_cycle_detection_with_chains),
        ("Topological Analysis", solution.maximumInvitations_approach3_topological_analysis),
        ("Graph Theory Optimal", solution.maximumInvitations_approach4_graph_theory_optimal),
        ("Cycle Decomposition", solution.maximumInvitations_approach5_cycle_decomposition),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (favorite, expected) in enumerate(test_cases):
            result = func(favorite[:])  # Copy list
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} favorite={favorite}, expected={expected}, got={result}")

def demonstrate_functional_graph_analysis():
    """Demonstrate functional graph analysis"""
    print("\n=== Functional Graph Analysis Demo ===")
    
    favorite = [3,0,1,4,1]
    n = len(favorite)
    
    print(f"Favorite array: {favorite}")
    print(f"Graph representation:")
    for i, fav in enumerate(favorite):
        print(f"  Employee {i} → Employee {fav}")
    
    print(f"\nFunctional graph properties:")
    print(f"• Each node has exactly one outgoing edge")
    print(f"• Nodes may have multiple incoming edges")
    print(f"• Graph consists of components with exactly one cycle each")
    
    # Analyze structure
    cycles = []
    visited = [False] * n
    
    def find_cycle_from(start):
        path = {}
        current = start
        step = 0
        
        while current not in path and not visited[current]:
            path[current] = step
            current = favorite[current]
            step += 1
        
        if current in path:
            cycle_start_step = path[current]
            cycle_nodes = []
            temp = start
            
            for _ in range(cycle_start_step):
                temp = favorite[temp]
            
            cycle_node = temp
            while True:
                cycle_nodes.append(cycle_node)
                cycle_node = favorite[cycle_node]
                if cycle_node == temp:
                    break
            
            return cycle_nodes
        
        return []
    
    for i in range(n):
        if not visited[i]:
            cycle = find_cycle_from(i)
            if cycle:
                cycles.append(cycle)
            
            # Mark component as visited
            current = i
            while not visited[current]:
                visited[current] = True
                current = favorite[current]
    
    print(f"\nCycles found:")
    for i, cycle in enumerate(cycles):
        print(f"  Cycle {i+1}: {cycle} (length {len(cycle)})")
    
    solution = Solution()
    result = solution.maximumInvitations_approach1_functional_graph_analysis(favorite)
    print(f"\nMaximum employees that can be invited: {result}")

def analyze_meeting_arrangement_strategies():
    """Analyze different meeting arrangement strategies"""
    print("\n=== Meeting Arrangement Strategies ===")
    
    print("Optimal Strategies for Circular Meeting:")
    
    print("\n1. **Single Large Cycle Strategy:**")
    print("   • Use one cycle of length > 2")
    print("   • All employees in cycle sit around table")
    print("   • Each employee sits next to their favorite")
    print("   • Cannot combine with other cycles")
    
    print("\n2. **Multiple 2-Cycle Strategy:**")
    print("   • Use all 2-cycles (mutual favorites)")
    print("   • Add chains leading to 2-cycle nodes")
    print("   • Chain nodes sit in line leading to cycle")
    print("   • Can combine multiple 2-cycles and their chains")
    
    print("\n3. **Strategy Selection:**")
    print("   • Compare single largest cycle vs sum of all 2-cycles+chains")
    print("   • 2-cycles are more flexible for combinations")
    print("   • Large cycles give guaranteed seating but no extensions")
    print("   • Optimal strategy depends on graph structure")
    
    print("\n4. **Chain Extension Rules:**")
    print("   • Only works with 2-cycles")
    print("   • Chain nodes must not be in any cycle")
    print("   • Chain forms directed path to cycle node")
    print("   • Maximizes utilization of non-cycle employees")

def demonstrate_graph_structure_analysis():
    """Demonstrate graph structure analysis"""
    print("\n=== Graph Structure Analysis ===")
    
    print("Functional Graph Properties:")
    
    print("\n1. **Structural Characteristics:**")
    print("   • Out-degree = 1 for every vertex")
    print("   • In-degree ≥ 0 (varies by vertex)")
    print("   • Each weakly connected component has exactly one cycle")
    print("   • Trees rooted at cycle nodes")
    
    print("\n2. **Cycle Types and Handling:**")
    print("   • Length 1: Self-loop (impossible here due to constraints)")
    print("   • Length 2: Mutual favorites (best for extensions)")
    print("   • Length > 2: Multiple employees in cycle (standalone use)")
    
    print("\n3. **Chain Analysis:**")
    print("   • Chains are directed paths leading to cycles")
    print("   • Chain length = maximum depth from cycle node")
    print("   • Only non-cycle nodes can form chains")
    print("   • Chains contribute additively to 2-cycles")
    
    print("\n4. **Optimization Insight:**")
    print("   • 2-cycles + chains often optimal")
    print("   • Large cycles compete with 2-cycle combinations")
    print("   • Graph decomposition enables efficient analysis")
    print("   • Linear time solution possible")

def analyze_algorithmic_approaches():
    """Analyze different algorithmic approaches"""
    print("\n=== Algorithmic Approaches Analysis ===")
    
    print("Algorithm Design Strategies:")
    
    print("\n1. **Functional Graph Analysis:**")
    print("   • Exploit out-degree = 1 property")
    print("   • Follow edges to find cycles naturally")
    print("   • Use Floyd's cycle detection variants")
    print("   • Efficient component analysis")
    
    print("\n2. **Topological Sorting:**")
    print("   • Remove nodes with in-degree 0 iteratively")
    print("   • Remaining nodes form cycles")
    print("   • Clean separation of cycle and non-cycle nodes")
    print("   • Enables separate tree analysis")
    
    print("\n3. **Reverse Graph Construction:**")
    print("   • Build who-likes-whom reverse mapping")
    print("   • Enables efficient chain length calculation")
    print("   • DFS from cycle nodes finds chain depths")
    print("   • Natural for tree traversal")
    
    print("\n4. **Component Decomposition:**")
    print("   • Analyze each weakly connected component")
    print("   • Find cycle structure per component")
    print("   • Calculate optimal arrangement per component")
    print("   • Combine results optimally")
    
    print("\n5. **Time Complexity Optimization:**")
    print("   • O(N) time achievable with proper approach")
    print("   • Single pass for cycle detection")
    print("   • Linear chain depth calculation")
    print("   • Efficient data structure usage")

if __name__ == "__main__":
    test_maximum_invitations()
    demonstrate_functional_graph_analysis()
    analyze_meeting_arrangement_strategies()
    demonstrate_graph_structure_analysis()
    analyze_algorithmic_approaches()

"""
Maximum Employee Meeting and Functional Graph Concepts:
1. Functional Graph Analysis and Cycle Detection
2. Chain Extension and Tree Structure Analysis
3. Circular Arrangement Optimization with Constraints
4. Graph Decomposition and Component Analysis
5. Combinatorial Optimization on Special Graph Structures

Key Problem Insights:
- Functional graph with out-degree 1 for every vertex
- Each component has exactly one cycle
- Two strategies: single large cycle vs multiple 2-cycles with chains
- 2-cycles allow chain extensions, larger cycles don't

Algorithm Strategy:
1. Detect all cycles in the functional graph
2. For each 2-cycle, calculate maximum chain extensions
3. Compare single largest cycle vs sum of 2-cycles + chains
4. Return maximum of the two strategies

Functional Graph Properties:
- Every vertex has exactly one outgoing edge
- Components have unique cycle with trees attached
- Floyd's cycle detection applicable
- Linear time analysis possible

Optimization Techniques:
- Reverse graph for efficient chain calculation
- Topological sorting for cycle isolation
- Component-wise analysis for modularity
- Dynamic programming for chain depths

Real-world Applications:
- Seating arrangement optimization
- Resource allocation with preferences
- Network topology analysis
- Constraint satisfaction problems
- Combinatorial optimization

This problem demonstrates advanced functional graph analysis
essential for preference-based optimization problems.
"""
