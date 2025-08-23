"""
1376. Time Needed to Inform All Employees - Multiple Approaches
Difficulty: Medium

A company has n employees with a unique ID for each employee from 0 to n - 1. 
The head of the company is the one with headID.

Each employee has one direct manager given in the manager array where manager[i] 
is the direct manager of the i-th employee, manager[headID] = -1. Also, it is 
guaranteed that the subordination relationships have a tree structure.

The head of the company wants to inform all the company employees of an urgent 
piece of news. He will inform his direct subordinates, and they will inform 
their subordinates, and so on until all employees know about the news.

The i-th employee needs informTime[i] minutes to inform all of his direct 
subordinates (i.e., After informTime[i] minutes, all his direct subordinates 
can start spreading the news).

Return the number of minutes needed to inform all the employees about the news.
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque
import heapq

class TimeToInformEmployees:
    """Multiple approaches to solve the employee information propagation problem"""
    
    def numOfMinutes_dfs_recursive(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        """
        Approach 1: DFS Recursive (Top-down)
        
        Build tree from manager relationships and use DFS to find maximum time.
        
        Time: O(n) - visit each employee once
        Space: O(n) - recursion stack and adjacency list
        """
        # Build adjacency list (manager -> subordinates)
        subordinates = defaultdict(list)
        for i in range(n):
            if manager[i] != -1:
                subordinates[manager[i]].append(i)
        
        def dfs(employee_id):
            """DFS to find maximum time to inform all subordinates"""
            if not subordinates[employee_id]:
                return 0  # Leaf employee, no one to inform
            
            max_time = 0
            for subordinate in subordinates[employee_id]:
                max_time = max(max_time, dfs(subordinate))
            
            return informTime[employee_id] + max_time
        
        return dfs(headID)
    
    def numOfMinutes_dfs_iterative(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        """
        Approach 2: DFS Iterative (Stack-based)
        
        Use explicit stack to avoid recursion depth issues.
        
        Time: O(n)
        Space: O(n)
        """
        # Build adjacency list
        subordinates = defaultdict(list)
        for i in range(n):
            if manager[i] != -1:
                subordinates[manager[i]].append(i)
        
        # Stack: (employee_id, time_so_far)
        stack = [(headID, 0)]
        max_time = 0
        
        while stack:
            employee_id, current_time = stack.pop()
            
            if not subordinates[employee_id]:
                # Leaf employee - update max time
                max_time = max(max_time, current_time)
            else:
                # Add subordinates to stack
                new_time = current_time + informTime[employee_id]
                for subordinate in subordinates[employee_id]:
                    stack.append((subordinate, new_time))
        
        return max_time
    
    def numOfMinutes_bfs_level_order(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        """
        Approach 3: BFS Level Order Traversal
        
        Process employees level by level, tracking maximum time at each level.
        
        Time: O(n)
        Space: O(n)
        """
        # Build adjacency list
        subordinates = defaultdict(list)
        for i in range(n):
            if manager[i] != -1:
                subordinates[manager[i]].append(i)
        
        # BFS queue: (employee_id, time_to_reach)
        queue = deque([(headID, 0)])
        max_time = 0
        
        while queue:
            employee_id, current_time = queue.popleft()
            
            if not subordinates[employee_id]:
                # Leaf employee
                max_time = max(max_time, current_time)
            else:
                # Add subordinates with updated time
                inform_time = current_time + informTime[employee_id]
                for subordinate in subordinates[employee_id]:
                    queue.append((subordinate, inform_time))
        
        return max_time
    
    def numOfMinutes_bottom_up(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        """
        Approach 4: Bottom-up Dynamic Programming
        
        Start from leaf employees and propagate time upwards.
        
        Time: O(n)
        Space: O(n)
        """
        # Build adjacency list and find leaf employees
        subordinates = defaultdict(list)
        for i in range(n):
            if manager[i] != -1:
                subordinates[manager[i]].append(i)
        
        # Memoization for time to inform all subordinates
        memo = {}
        
        def get_inform_time(employee_id):
            """Get time to inform all subordinates of this employee"""
            if employee_id in memo:
                return memo[employee_id]
            
            if not subordinates[employee_id]:
                memo[employee_id] = 0
                return 0
            
            max_subordinate_time = 0
            for subordinate in subordinates[employee_id]:
                max_subordinate_time = max(max_subordinate_time, get_inform_time(subordinate))
            
            memo[employee_id] = informTime[employee_id] + max_subordinate_time
            return memo[employee_id]
        
        return get_inform_time(headID)
    
    def numOfMinutes_path_compression(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        """
        Approach 5: Path Compression Optimization
        
        Use path compression to optimize repeated calculations.
        
        Time: O(n) amortized
        Space: O(n)
        """
        # Memoization with path compression
        total_time = {}
        
        def get_total_time(employee_id):
            """Get total time from this employee to inform all below"""
            if employee_id in total_time:
                return total_time[employee_id]
            
            # Build subordinates list for this employee
            subordinates = []
            for i in range(n):
                if manager[i] == employee_id:
                    subordinates.append(i)
            
            if not subordinates:
                total_time[employee_id] = 0
                return 0
            
            max_time = 0
            for subordinate in subordinates:
                max_time = max(max_time, get_total_time(subordinate))
            
            total_time[employee_id] = informTime[employee_id] + max_time
            return total_time[employee_id]
        
        return get_total_time(headID)
    
    def numOfMinutes_parallel_simulation(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        """
        Approach 6: Parallel Time Simulation
        
        Simulate the information spreading process minute by minute.
        
        Time: O(n * max_time) - can be slow for large times
        Space: O(n)
        """
        # Build adjacency list
        subordinates = defaultdict(list)
        for i in range(n):
            if manager[i] != -1:
                subordinates[manager[i]].append(i)
        
        # Track when each employee gets informed
        informed_at = [-1] * n
        informed_at[headID] = 0
        
        # Priority queue: (time_informed, employee_id)
        pq = [(0, headID)]
        max_inform_time = 0
        
        while pq:
            inform_time, employee_id = heapq.heappop(pq)
            
            if inform_time > informed_at[employee_id]:
                continue  # Already processed with earlier time
            
            max_inform_time = max(max_inform_time, inform_time)
            
            # Inform subordinates
            if subordinates[employee_id]:
                new_inform_time = inform_time + informTime[employee_id]
                for subordinate in subordinates[employee_id]:
                    if informed_at[subordinate] == -1 or new_inform_time < informed_at[subordinate]:
                        informed_at[subordinate] = new_inform_time
                        heapq.heappush(pq, (new_inform_time, subordinate))
        
        return max_inform_time
    
    def numOfMinutes_tree_diameter_approach(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        """
        Approach 7: Tree Diameter Inspired Approach
        
        Find the longest weighted path from head to any leaf.
        
        Time: O(n)
        Space: O(n)
        """
        # Build adjacency list with weights
        graph = defaultdict(list)
        for i in range(n):
            if manager[i] != -1:
                graph[manager[i]].append((i, informTime[manager[i]]))
        
        def dfs_max_path(node):
            """Find maximum weighted path from this node to any leaf"""
            if not graph[node]:
                return 0
            
            max_path = 0
            for neighbor, weight in graph[node]:
                max_path = max(max_path, weight + dfs_max_path(neighbor))
            
            return max_path
        
        return dfs_max_path(headID)
    
    def numOfMinutes_optimized_single_pass(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        """
        Approach 8: Optimized Single Pass
        
        Calculate result in single pass without building explicit tree.
        
        Time: O(n)
        Space: O(n)
        """
        # For each employee, calculate time to inform all their subtree
        subtree_time = [0] * n
        
        # Process in reverse topological order (leaves first)
        processed = [False] * n
        
        def process_employee(emp_id):
            """Process employee and return time to inform their subtree"""
            if processed[emp_id]:
                return subtree_time[emp_id]
            
            # Find all subordinates
            max_subordinate_time = 0
            for i in range(n):
                if manager[i] == emp_id:
                    max_subordinate_time = max(max_subordinate_time, process_employee(i))
            
            subtree_time[emp_id] = informTime[emp_id] + max_subordinate_time
            processed[emp_id] = True
            return subtree_time[emp_id]
        
        return process_employee(headID)

def test_time_to_inform():
    """Test all approaches with various test cases"""
    solver = TimeToInformEmployees()
    
    test_cases = [
        # (n, headID, manager, informTime, expected, description)
        (1, 0, [-1], [0], 0, "Single employee"),
        (6, 2, [2, 2, -1, 2, 2, 2], [0, 0, 1, 0, 0, 0], 1, "Star structure"),
        (7, 6, [1, 1, 1, 1, 1, 1, -1], [0, 6, 0, 0, 0, 0, 1], 7, "Deep tree"),
        (15, 0, [-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6], 
         [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 3, "Balanced tree"),
        (4, 0, [-1, 0, 1, 2], [1, 2, 3, 0], 6, "Linear chain"),
        (8, 0, [-1, 0, 0, 1, 2, 3, 4, 5], [1, 2, 3, 1, 1, 1, 1, 0], 8, "Mixed structure"),
    ]
    
    approaches = [
        ("DFS Recursive", solver.numOfMinutes_dfs_recursive),
        ("DFS Iterative", solver.numOfMinutes_dfs_iterative),
        ("BFS Level Order", solver.numOfMinutes_bfs_level_order),
        ("Bottom-up DP", solver.numOfMinutes_bottom_up),
        ("Path Compression", solver.numOfMinutes_path_compression),
        ("Parallel Simulation", solver.numOfMinutes_parallel_simulation),
        ("Tree Diameter", solver.numOfMinutes_tree_diameter_approach),
        ("Single Pass", solver.numOfMinutes_optimized_single_pass),
    ]
    
    print("=== Testing Time to Inform All Employees ===")
    
    for n, headID, manager, informTime, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"n={n}, headID={headID}")
        print(f"manager={manager}")
        print(f"informTime={informTime}")
        print(f"Expected: {expected}")
        
        for approach_name, approach_func in approaches:
            try:
                result = approach_func(n, headID, manager, informTime)
                status = "✓" if result == expected else "✗"
                print(f"{approach_name:18} | {status} | Result: {result}")
            except Exception as e:
                print(f"{approach_name:18} | ERROR: {str(e)[:30]}")

def demonstrate_tree_structure_analysis():
    """Demonstrate tree structure analysis for information flow"""
    print("\n=== Tree Structure Analysis ===")
    
    # Example: Company hierarchy
    n, headID = 7, 6
    manager = [1, 1, 1, 1, 1, 1, -1]
    informTime = [0, 6, 0, 0, 0, 0, 1]
    
    print(f"Company structure:")
    print(f"Head (ID {headID}) needs {informTime[headID]} minutes to inform direct reports")
    
    # Build and display tree structure
    subordinates = defaultdict(list)
    for i in range(n):
        if manager[i] != -1:
            subordinates[manager[i]].append(i)
    
    def print_tree(node, depth=0):
        indent = "  " * depth
        inform_time = informTime[node]
        subs = subordinates[node]
        print(f"{indent}Employee {node} (inform time: {inform_time}, subordinates: {len(subs)})")
        for sub in subs:
            print_tree(sub, depth + 1)
    
    print_tree(headID)
    
    # Calculate and show result
    solver = TimeToInformEmployees()
    result = solver.numOfMinutes_dfs_recursive(n, headID, manager, informTime)
    print(f"\nTotal time to inform all employees: {result} minutes")
    
    print(f"\nInformation flow analysis:")
    print(f"• Head informs 5 direct reports in 1 minute")
    print(f"• Employee 1 then informs employees 0,2,3,4,5 in 6 minutes")
    print(f"• Total time: 1 + 6 = 7 minutes")

def analyze_algorithm_complexity():
    """Analyze time and space complexity of different approaches"""
    print("\n=== Algorithm Complexity Analysis ===")
    
    print("Time Complexity Comparison:")
    print("• DFS Recursive:      O(n) - visit each node once")
    print("• DFS Iterative:      O(n) - explicit stack traversal")
    print("• BFS Level Order:    O(n) - queue-based traversal")
    print("• Bottom-up DP:       O(n) - memoized recursion")
    print("• Path Compression:   O(n) - amortized with memoization")
    print("• Parallel Simulation: O(n log n) - priority queue operations")
    print("• Tree Diameter:      O(n) - single DFS traversal")
    print("• Single Pass:        O(n) - optimized single traversal")
    
    print("\nSpace Complexity Comparison:")
    print("• DFS Recursive:      O(h) - recursion stack height")
    print("• DFS Iterative:      O(n) - explicit stack")
    print("• BFS Level Order:    O(w) - queue width (max level size)")
    print("• Bottom-up DP:       O(n) - memoization table")
    print("• Path Compression:   O(n) - memoization cache")
    print("• Parallel Simulation: O(n) - priority queue")
    print("• Tree Diameter:      O(h) - recursion stack")
    print("• Single Pass:        O(n) - processing array")
    
    print("\nBest Approach Selection:")
    print("• General use: DFS Recursive (simple and efficient)")
    print("• Deep trees: DFS Iterative (avoid stack overflow)")
    print("• Level analysis: BFS Level Order")
    print("• Repeated queries: Bottom-up DP with memoization")
    print("• Memory constrained: Tree Diameter (minimal extra space)")

def demonstrate_real_world_applications():
    """Demonstrate real-world applications of information propagation"""
    print("\n=== Real-World Applications ===")
    
    print("Information Propagation Applications:")
    
    print("\n1. **Corporate Communication:**")
    print("   • Emergency notifications in organizations")
    print("   • Policy updates through management hierarchy")
    print("   • Training program rollout scheduling")
    print("   • Performance review cascading")
    
    print("\n2. **Network Broadcasting:**")
    print("   • Software update distribution")
    print("   • Content delivery network optimization")
    print("   • Peer-to-peer file sharing")
    print("   • Blockchain transaction propagation")
    
    print("\n3. **Social Network Analysis:**")
    print("   • Viral content spread modeling")
    print("   • Influence propagation in social graphs")
    print("   • Rumor spreading analysis")
    print("   • Marketing campaign reach estimation")
    
    print("\n4. **Infrastructure Management:**")
    print("   • Power grid failure cascade analysis")
    print("   • Transportation delay propagation")
    print("   • Supply chain disruption modeling")
    print("   • Emergency response coordination")
    
    print("\n5. **Biological Systems:**")
    print("   • Disease outbreak modeling")
    print("   • Neural signal propagation")
    print("   • Genetic information flow")
    print("   • Ecosystem change propagation")
    
    print("\nKey Optimization Considerations:")
    print("• Minimize maximum propagation time")
    print("• Balance load across intermediate nodes")
    print("• Handle failures and redundancy")
    print("• Optimize for different network topologies")

if __name__ == "__main__":
    test_time_to_inform()
    demonstrate_tree_structure_analysis()
    analyze_algorithm_complexity()
    demonstrate_real_world_applications()

"""
Time to Inform All Employees - Key Insights:

1. **Problem Structure:**
   - Tree-based hierarchy with weighted edges (inform times)
   - Find maximum path from root to any leaf
   - Each node contributes its inform time to all paths through it

2. **Algorithm Categories:**
   - Tree Traversal: DFS/BFS to explore all paths
   - Dynamic Programming: Memoization for repeated subproblems
   - Simulation: Model actual information flow process
   - Optimization: Single-pass and path compression techniques

3. **Key Observations:**
   - Maximum time = longest weighted path from head to leaf
   - Each manager's inform time affects all subordinates
   - Tree structure allows efficient O(n) solutions
   - No cycles due to management hierarchy constraint

4. **Optimization Strategies:**
   - Memoization for repeated subtree calculations
   - Iterative approaches to avoid recursion limits
   - Single-pass algorithms for minimal overhead
   - Priority queue simulation for complex scenarios

5. **Real-World Relevance:**
   - Corporate communication optimization
   - Network broadcasting efficiency
   - Social influence modeling
   - Infrastructure failure analysis

The problem demonstrates fundamental tree algorithms
applied to practical organizational scenarios.
"""
