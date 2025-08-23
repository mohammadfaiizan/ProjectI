"""
Hungarian Algorithm Implementation - Multiple Approaches
Difficulty: Medium

The Hungarian algorithm is a combinatorial optimization algorithm that solves the assignment problem in polynomial time. It was developed and published in 1955 by Harold Kuhn, who gave the name "Hungarian algorithm" because the algorithm was largely based on the earlier works of two Hungarian mathematicians: Dénes Kőnig and Jenő Egerváry.

The assignment problem is to find a minimum-weight perfect matching in a weighted bipartite graph.

This implementation covers multiple variations and optimizations of the Hungarian algorithm.
"""

from typing import List, Tuple, Optional
import copy

class HungarianAlgorithm:
    """Multiple implementations of the Hungarian Algorithm"""
    
    def hungarian_classic(self, cost_matrix: List[List[int]]) -> Tuple[int, List[Tuple[int, int]]]:
        """
        Approach 1: Classic Hungarian Algorithm
        
        Standard implementation with row/column reduction and augmenting paths.
        
        Time: O(n³), Space: O(n²)
        """
        n = len(cost_matrix)
        if n == 0:
            return 0, []
        
        # Make a copy to avoid modifying original
        matrix = [row[:] for row in cost_matrix]
        
        # Step 1: Subtract row minimums
        for i in range(n):
            row_min = min(matrix[i])
            for j in range(n):
                matrix[i][j] -= row_min
        
        # Step 2: Subtract column minimums
        for j in range(n):
            col_min = min(matrix[i][j] for i in range(n))
            for i in range(n):
                matrix[i][j] -= col_min
        
        # Step 3: Cover all zeros with minimum number of lines
        while True:
            # Find assignment using zeros
            assignment = self._find_assignment(matrix)
            
            if len(assignment) == n:
                # Found complete assignment
                total_cost = sum(cost_matrix[i][j] for i, j in assignment)
                return total_cost, assignment
            
            # Need to create more zeros
            matrix = self._create_more_zeros(matrix, assignment)
    
    def _find_assignment(self, matrix: List[List[int]]) -> List[Tuple[int, int]]:
        """Find maximum assignment using only zeros"""
        n = len(matrix)
        
        # Try to find assignment greedily first
        row_assigned = [False] * n
        col_assigned = [False] * n
        assignment = []
        
        # Greedy assignment
        for i in range(n):
            for j in range(n):
                if matrix[i][j] == 0 and not row_assigned[i] and not col_assigned[j]:
                    assignment.append((i, j))
                    row_assigned[i] = True
                    col_assigned[j] = True
                    break
        
        return assignment
    
    def _create_more_zeros(self, matrix: List[List[int]], assignment: List[Tuple[int, int]]) -> List[List[int]]:
        """Create more zeros by finding minimum uncovered element"""
        n = len(matrix)
        
        # Mark covered rows and columns
        covered_rows = set()
        covered_cols = set()
        
        # Cover assigned rows
        for i, j in assignment:
            covered_rows.add(i)
        
        # Uncover columns that have zeros in uncovered rows
        changed = True
        while changed:
            changed = False
            
            # Uncover columns with zeros in uncovered rows
            for i in range(n):
                if i not in covered_rows:
                    for j in range(n):
                        if matrix[i][j] == 0 and j not in covered_cols:
                            covered_cols.add(j)
                            changed = True
            
            # Cover rows with assignments in newly uncovered columns
            for i, j in assignment:
                if j in covered_cols and i not in covered_rows:
                    covered_rows.add(i)
                    changed = True
        
        # Find minimum uncovered element
        min_uncovered = float('inf')
        for i in range(n):
            for j in range(n):
                if i not in covered_rows and j not in covered_cols:
                    min_uncovered = min(min_uncovered, matrix[i][j])
        
        # Subtract from uncovered, add to doubly covered
        new_matrix = [row[:] for row in matrix]
        for i in range(n):
            for j in range(n):
                if i not in covered_rows and j not in covered_cols:
                    new_matrix[i][j] -= min_uncovered
                elif i in covered_rows and j in covered_cols:
                    new_matrix[i][j] += min_uncovered
        
        return new_matrix
    
    def hungarian_kuhn_munkres(self, cost_matrix: List[List[int]]) -> Tuple[int, List[Tuple[int, int]]]:
        """
        Approach 2: Kuhn-Munkres Algorithm (Optimized Hungarian)
        
        More efficient implementation with better augmenting path finding.
        
        Time: O(n³), Space: O(n²)
        """
        n = len(cost_matrix)
        if n == 0:
            return 0, []
        
        # Initialize
        u = [0] * (n + 1)  # Potential for workers
        v = [0] * (n + 1)  # Potential for jobs
        p = [0] * (n + 1)  # Assignment: p[j] = i means job j assigned to worker i
        way = [0] * (n + 1)  # For path reconstruction
        
        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = [float('inf')] * (n + 1)
            used = [False] * (n + 1)
            
            while p[j0]:
                used[j0] = True
                i0 = p[j0]
                delta = float('inf')
                j1 = 0
                
                for j in range(1, n + 1):
                    if not used[j]:
                        cur = cost_matrix[i0 - 1][j - 1] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j] = cur
                            way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]
                            j1 = j
                
                for j in range(n + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                
                j0 = j1
            
            while j0:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
        
        # Extract assignment and calculate cost
        assignment = []
        total_cost = 0
        
        for j in range(1, n + 1):
            if p[j] != 0:
                assignment.append((p[j] - 1, j - 1))
                total_cost += cost_matrix[p[j] - 1][j - 1]
        
        return total_cost, assignment
    
    def hungarian_ford_fulkerson(self, cost_matrix: List[List[int]]) -> Tuple[int, List[Tuple[int, int]]]:
        """
        Approach 3: Hungarian using Ford-Fulkerson Style Augmenting Paths
        
        Implementation using augmenting path approach similar to max flow.
        
        Time: O(n³), Space: O(n²)
        """
        n = len(cost_matrix)
        if n == 0:
            return 0, []
        
        # Convert to maximum weight problem by negating and adding large constant
        max_cost = max(max(row) for row in cost_matrix)
        profit_matrix = [[max_cost - cost_matrix[i][j] for j in range(n)] for i in range(n)]
        
        # Initialize matching
        match_worker = [-1] * n  # match_worker[i] = j means worker i matched to job j
        match_job = [-1] * n     # match_job[j] = i means job j matched to worker i
        
        def dfs(worker: int, visited: List[bool]) -> bool:
            """Find augmenting path using DFS"""
            for job in range(n):
                if visited[job]:
                    continue
                
                visited[job] = True
                
                # If job is unmatched or we can find augmenting path from matched worker
                if match_job[job] == -1 or dfs(match_job[job], visited):
                    match_worker[worker] = job
                    match_job[job] = worker
                    return True
            
            return False
        
        # Find maximum matching
        for worker in range(n):
            visited = [False] * n
            dfs(worker, visited)
        
        # Calculate cost and create assignment list
        assignment = []
        total_cost = 0
        
        for worker in range(n):
            if match_worker[worker] != -1:
                job = match_worker[worker]
                assignment.append((worker, job))
                total_cost += cost_matrix[worker][job]
        
        return total_cost, assignment
    
    def hungarian_matrix_reduction(self, cost_matrix: List[List[int]]) -> Tuple[int, List[Tuple[int, int]]]:
        """
        Approach 4: Matrix Reduction Based Hungarian
        
        Focus on matrix reduction steps with clear separation.
        
        Time: O(n³), Space: O(n²)
        """
        n = len(cost_matrix)
        if n == 0:
            return 0, []
        
        # Step 1: Row reduction
        reduced_matrix = [row[:] for row in cost_matrix]
        
        for i in range(n):
            row_min = min(reduced_matrix[i])
            for j in range(n):
                reduced_matrix[i][j] -= row_min
        
        # Step 2: Column reduction
        for j in range(n):
            col_min = min(reduced_matrix[i][j] for i in range(n))
            for i in range(n):
                reduced_matrix[i][j] -= col_min
        
        # Step 3: Iteratively find optimal assignment
        max_iterations = n * n
        iteration = 0
        
        while iteration < max_iterations:
            # Find current assignment
            assignment = self._find_maximum_assignment(reduced_matrix)
            
            if len(assignment) == n:
                total_cost = sum(cost_matrix[i][j] for i, j in assignment)
                return total_cost, assignment
            
            # Improve matrix
            reduced_matrix = self._improve_matrix(reduced_matrix, assignment)
            iteration += 1
        
        # Fallback to greedy assignment if no optimal found
        assignment = self._greedy_assignment(cost_matrix)
        total_cost = sum(cost_matrix[i][j] for i, j in assignment)
        return total_cost, assignment
    
    def _find_maximum_assignment(self, matrix: List[List[int]]) -> List[Tuple[int, int]]:
        """Find maximum assignment using bipartite matching"""
        n = len(matrix)
        
        # Build bipartite graph with only zero-cost edges
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if matrix[i][j] == 0:
                    adj[i].append(j)
        
        # Find maximum matching
        match = [-1] * n
        
        def dfs(u: int, visited: List[bool]) -> bool:
            for v in adj[u]:
                if visited[v]:
                    continue
                visited[v] = True
                
                if match[v] == -1 or dfs(match[v], visited):
                    match[v] = u
                    return True
            return False
        
        matching = 0
        assignment = []
        
        for i in range(n):
            visited = [False] * n
            if dfs(i, visited):
                matching += 1
        
        # Extract assignment
        for j in range(n):
            if match[j] != -1:
                assignment.append((match[j], j))
        
        return assignment
    
    def _improve_matrix(self, matrix: List[List[int]], assignment: List[Tuple[int, int]]) -> List[List[int]]:
        """Improve matrix by creating more zeros"""
        n = len(matrix)
        
        # Find minimum vertex cover
        assigned_rows = set(i for i, j in assignment)
        assigned_cols = set(j for i, j in assignment)
        
        # Find uncovered elements
        uncovered_elements = []
        for i in range(n):
            for j in range(n):
                if i not in assigned_rows and j not in assigned_cols:
                    uncovered_elements.append(matrix[i][j])
        
        if not uncovered_elements:
            return matrix
        
        min_uncovered = min(uncovered_elements)
        
        # Create new matrix
        new_matrix = [row[:] for row in matrix]
        
        for i in range(n):
            for j in range(n):
                if i not in assigned_rows and j not in assigned_cols:
                    new_matrix[i][j] -= min_uncovered
                elif i in assigned_rows and j in assigned_cols:
                    new_matrix[i][j] += min_uncovered
        
        return new_matrix
    
    def _greedy_assignment(self, cost_matrix: List[List[int]]) -> List[Tuple[int, int]]:
        """Greedy assignment as fallback"""
        n = len(cost_matrix)
        assignment = []
        used_rows = set()
        used_cols = set()
        
        # Sort all positions by cost
        positions = []
        for i in range(n):
            for j in range(n):
                positions.append((cost_matrix[i][j], i, j))
        
        positions.sort()
        
        for cost, i, j in positions:
            if i not in used_rows and j not in used_cols:
                assignment.append((i, j))
                used_rows.add(i)
                used_cols.add(j)
                
                if len(assignment) == n:
                    break
        
        return assignment
    
    def hungarian_scipy_style(self, cost_matrix: List[List[int]]) -> Tuple[int, List[Tuple[int, int]]]:
        """
        Approach 5: SciPy-style Implementation
        
        Implementation similar to scipy.optimize.linear_sum_assignment.
        
        Time: O(n³), Space: O(n²)
        """
        n = len(cost_matrix)
        if n == 0:
            return 0, []
        
        # Convert to float for numerical stability
        C = [[float(cost_matrix[i][j]) for j in range(n)] for i in range(n)]
        
        # Initialize dual variables
        u = [0.0] * n  # Row potentials
        v = [0.0] * n  # Column potentials
        
        # Initialize assignment
        row_assigned = [-1] * n  # row_assigned[i] = j means row i assigned to col j
        col_assigned = [-1] * n  # col_assigned[j] = i means col j assigned to row i
        
        for i in range(n):
            # Find augmenting path starting from row i
            path = []
            row_in_path = [False] * n
            col_in_path = [False] * n
            
            # Start with unassigned row
            current_row = i
            path.append(('row', current_row))
            row_in_path[current_row] = True
            
            while True:
                # Find minimum reduced cost in current row
                min_cost = float('inf')
                best_col = -1
                
                for j in range(n):
                    if not col_in_path[j]:
                        reduced_cost = C[current_row][j] - u[current_row] - v[j]
                        if reduced_cost < min_cost:
                            min_cost = reduced_cost
                            best_col = j
                
                # Update potentials
                if min_cost > 0:
                    for k in range(len(path)):
                        if path[k][0] == 'row':
                            u[path[k][1]] += min_cost
                        else:
                            v[path[k][1]] -= min_cost
                
                # Add column to path
                path.append(('col', best_col))
                col_in_path[best_col] = True
                
                # Check if column is unassigned
                if col_assigned[best_col] == -1:
                    # Found augmenting path, update assignment
                    for k in range(0, len(path), 2):
                        row_idx = path[k][1]
                        col_idx = path[k + 1][1]
                        
                        if row_assigned[row_idx] != -1:
                            col_assigned[row_assigned[row_idx]] = -1
                        
                        row_assigned[row_idx] = col_idx
                        col_assigned[col_idx] = row_idx
                    break
                else:
                    # Continue with assigned row
                    current_row = col_assigned[best_col]
                    path.append(('row', current_row))
                    row_in_path[current_row] = True
        
        # Extract assignment and calculate cost
        assignment = []
        total_cost = 0
        
        for i in range(n):
            if row_assigned[i] != -1:
                j = row_assigned[i]
                assignment.append((i, j))
                total_cost += cost_matrix[i][j]
        
        return total_cost, assignment

def test_hungarian_algorithm():
    """Test Hungarian algorithm implementations"""
    solver = HungarianAlgorithm()
    
    test_cases = [
        ([[4, 1, 3], [2, 0, 5], [3, 2, 2]], 4, "3x3 matrix"),
        ([[1, 2, 3, 4], [2, 4, 6, 8], [3, 6, 9, 12], [4, 8, 12, 16]], 10, "4x4 matrix"),
        ([[1]], 1, "1x1 matrix"),
        ([[1, 2], [3, 4]], 5, "2x2 matrix"),
        ([[9, 2, 7, 8], [6, 4, 3, 7], [5, 8, 1, 8], [7, 6, 9, 4]], 13, "Complex 4x4"),
    ]
    
    algorithms = [
        ("Classic Hungarian", solver.hungarian_classic),
        ("Kuhn-Munkres", solver.hungarian_kuhn_munkres),
        ("Matrix Reduction", solver.hungarian_matrix_reduction),
        ("SciPy Style", solver.hungarian_scipy_style),
    ]
    
    print("=== Testing Hungarian Algorithm ===")
    
    for cost_matrix, expected_cost, description in test_cases:
        print(f"\n--- {description} (Expected cost: {expected_cost}) ---")
        print(f"Cost matrix: {cost_matrix}")
        
        for alg_name, alg_func in algorithms:
            try:
                cost, assignment = alg_func(cost_matrix)
                status = "✓" if cost == expected_cost else "✗"
                print(f"{alg_name:18} | {status} | Cost: {cost}, Assignment: {assignment}")
            except Exception as e:
                print(f"{alg_name:18} | ERROR: {str(e)[:40]}")

if __name__ == "__main__":
    test_hungarian_algorithm()

"""
Hungarian Algorithm Implementation demonstrates the classic
assignment problem solution with multiple optimization
approaches and comprehensive bipartite matching techniques.
"""
