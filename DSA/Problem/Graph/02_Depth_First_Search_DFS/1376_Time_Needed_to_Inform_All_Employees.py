"""
1376. Time Needed to Inform All Employees - Multiple Approaches
Difficulty: Medium

A company has n employees with a unique ID for each employee from 0 to n - 1. The head of the company is the one with headID.

Each employee has one direct manager given in the manager array where manager[i] is the direct manager of the i-th employee, manager[headID] = -1. Also, it is guaranteed that the subordination relationships have a tree structure.

The head of the company wants to inform all the company employees of an urgent piece of news. He will inform his direct subordinates, and they will inform their subordinates, and so on until all employees know about the news.

The i-th employee needs informTime[i] minutes to inform all of his direct subordinates.

Return the number of minutes needed to inform all the employees about the news.
"""

from typing import List, Dict
from collections import defaultdict, deque

class TimeToInformEmployees:
    """Multiple approaches to calculate information propagation time"""
    
    def numOfMinutes_dfs_recursive(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        """
        Approach 1: Recursive DFS from head
        
        Build tree and recursively calculate max time to inform all subtrees.
        
        Time: O(n), Space: O(n)
        """
        # Build adjacency list of subordinates
        subordinates = defaultdict(list)
        for i, mgr in enumerate(manager):
            if mgr != -1:
                subordinates[mgr].append(i)
        
        def dfs(employee_id: int) -> int:
            """Return max time to inform all employees in this subtree"""
            if not subordinates[employee_id]:  # No subordinates
                return 0
            
            max_time = 0
            for subordinate in subordinates[employee_id]:
                max_time = max(max_time, dfs(subordinate))
            
            return informTime[employee_id] + max_time
        
        return dfs(headID)
    
    def numOfMinutes_dfs_iterative(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        """
        Approach 2: Iterative DFS using stack
        
        Use explicit stack to avoid recursion depth issues.
        
        Time: O(n), Space: O(n)
        """
        # Build adjacency list
        subordinates = defaultdict(list)
        for i, mgr in enumerate(manager):
            if mgr != -1:
                subordinates[mgr].append(i)
        
        # Stack: (employee_id, accumulated_time)
        stack = [(headID, 0)]
        max_time = 0
        
        while stack:
            employee_id, accumulated_time = stack.pop()
            
            # If this is a leaf node (no subordinates)
            if not subordinates[employee_id]:
                max_time = max(max_time, accumulated_time)
            else:
                # Add all subordinates to stack
                new_time = accumulated_time + informTime[employee_id]
                for subordinate in subordinates[employee_id]:
                    stack.append((subordinate, new_time))
        
        return max_time
    
    def numOfMinutes_bfs(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        """
        Approach 3: BFS level-by-level processing
        
        Process employees level by level, tracking maximum time.
        
        Time: O(n), Space: O(n)
        """
        # Build adjacency list
        subordinates = defaultdict(list)
        for i, mgr in enumerate(manager):
            if mgr != -1:
                subordinates[mgr].append(i)
        
        # BFS: (employee_id, time_when_informed)
        queue = deque([(headID, 0)])
        max_time = 0
        
        while queue:
            employee_id, time_informed = queue.popleft()
            
            # Update maximum time
            max_time = max(max_time, time_informed)
            
            # Inform all subordinates
            inform_time = time_informed + informTime[employee_id]
            for subordinate in subordinates[employee_id]:
                queue.append((subordinate, inform_time))
        
        return max_time
    
    def numOfMinutes_bottom_up(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        """
        Approach 4: Bottom-up calculation
        
        Calculate time for each employee to reach head, then find maximum.
        
        Time: O(n), Space: O(n)
        """
        # Memoization for time to inform from each employee to all their subordinates
        memo = {}
        
        def calculate_time(employee_id: int) -> int:
            """Calculate max time to inform all subordinates of this employee"""
            if employee_id in memo:
                return memo[employee_id]
            
            # Build subordinates list for this employee
            subordinates = []
            for i, mgr in enumerate(manager):
                if mgr == employee_id:
                    subordinates.append(i)
            
            if not subordinates:  # Leaf node
                memo[employee_id] = 0
                return 0
            
            max_subordinate_time = 0
            for subordinate in subordinates:
                max_subordinate_time = max(max_subordinate_time, calculate_time(subordinate))
            
            memo[employee_id] = informTime[employee_id] + max_subordinate_time
            return memo[employee_id]
        
        return calculate_time(headID)
    
    def numOfMinutes_path_to_leaves(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        """
        Approach 5: Find all paths to leaves
        
        Calculate time for all root-to-leaf paths and return maximum.
        
        Time: O(n), Space: O(n)
        """
        # Build adjacency list
        subordinates = defaultdict(list)
        for i, mgr in enumerate(manager):
            if mgr != -1:
                subordinates[mgr].append(i)
        
        max_time = 0
        
        def dfs_paths(employee_id: int, current_time: int):
            """DFS to find all paths and track maximum time"""
            nonlocal max_time
            
            if not subordinates[employee_id]:  # Leaf node
                max_time = max(max_time, current_time)
                return
            
            # Explore all subordinates
            new_time = current_time + informTime[employee_id]
            for subordinate in subordinates[employee_id]:
                dfs_paths(subordinate, new_time)
        
        dfs_paths(headID, 0)
        return max_time
    
    def numOfMinutes_optimized_single_pass(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        """
        Approach 6: Optimized single-pass calculation
        
        Calculate result in single DFS pass without building explicit tree.
        
        Time: O(n), Space: O(n)
        """
        # Cache for calculated times
        time_cache = {}
        
        def get_inform_time(employee_id: int) -> int:
            """Get total time to inform all employees under this employee"""
            if employee_id in time_cache:
                return time_cache[employee_id]
            
            # Find all direct subordinates
            max_subordinate_time = 0
            for i, mgr in enumerate(manager):
                if mgr == employee_id:
                    max_subordinate_time = max(max_subordinate_time, get_inform_time(i))
            
            time_cache[employee_id] = informTime[employee_id] + max_subordinate_time
            return time_cache[employee_id]
        
        return get_inform_time(headID)

def test_time_to_inform_employees():
    """Test time to inform employees algorithms"""
    solver = TimeToInformEmployees()
    
    test_cases = [
        (1, 0, [-1], [0], 0, "Single employee"),
        (6, 2, [2,2,-1,2,2,2], [0,0,1,0,0,0], 1, "Head with 5 subordinates"),
        (7, 6, [1,1,3,3,3,3,-1], [0,6,0,1,0,0,1], 7, "Multi-level hierarchy"),
        (15, 0, [-1,0,0,1,1,2,2,3,3,4,4,5,5,6,6], [1,1,1,1,1,1,1,0,0,0,0,0,0,0,0], 3, "Deep tree"),
    ]
    
    algorithms = [
        ("Recursive DFS", solver.numOfMinutes_dfs_recursive),
        ("Iterative DFS", solver.numOfMinutes_dfs_iterative),
        ("BFS", solver.numOfMinutes_bfs),
        ("Bottom-up", solver.numOfMinutes_bottom_up),
        ("Path to Leaves", solver.numOfMinutes_path_to_leaves),
        ("Optimized Single Pass", solver.numOfMinutes_optimized_single_pass),
    ]
    
    print("=== Testing Time to Inform All Employees ===")
    
    for n, headID, manager, informTime, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        print(f"n={n}, headID={headID}, manager={manager}, informTime={informTime}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(n, headID, manager, informTime)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Time: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_time_to_inform_employees()

"""
Time to Inform All Employees demonstrates tree traversal
and path analysis for hierarchical information propagation
with multiple optimization approaches.
"""
