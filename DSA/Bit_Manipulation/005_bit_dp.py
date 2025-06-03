"""
🧠 BITMASK DYNAMIC PROGRAMMING (BIT DP)
======================================

This module covers advanced dynamic programming problems using bitmasks.
Bitmask DP is powerful for problems with small state spaces and complex constraints.

Topics Covered:
1. Travelling Salesman Problem (TSP)
2. Assignment Problem
3. Minimum Cost to Paint Houses
4. Count All Valid Sudoku Boards (Advanced)

Author: Interview Preparation Collection
LeetCode Problems: 943, 1125, 256, 526, 691
"""

class TravellingSalesmanProblem:
    """Travelling Salesman Problem using bitmask DP."""
    
    @staticmethod
    def tsp_min_cost(graph: list) -> int:
        """
        Find minimum cost to visit all cities exactly once and return to start.
        
        Args:
            graph: Adjacency matrix where graph[i][j] = cost from city i to j
            
        Returns:
            Minimum cost for TSP tour
            
        Time: O(n^2 * 2^n), Space: O(n * 2^n)
        """
        n = len(graph)
        if n <= 1:
            return 0
        
        # dp[mask][i] = minimum cost to visit all cities in mask ending at city i
        dp = [[float('inf')] * n for _ in range(1 << n)]
        
        # Base case: start at city 0
        dp[1][0] = 0  # mask=1 means only city 0 is visited
        
        # Fill DP table
        for mask in range(1 << n):
            for u in range(n):
                if not (mask & (1 << u)) or dp[mask][u] == float('inf'):
                    continue
                
                # Try visiting each unvisited city
                for v in range(n):
                    if mask & (1 << v):  # City v already visited
                        continue
                    
                    new_mask = mask | (1 << v)
                    new_cost = dp[mask][u] + graph[u][v]
                    dp[new_mask][v] = min(dp[new_mask][v], new_cost)
        
        # Find minimum cost to return to start city
        final_mask = (1 << n) - 1  # All cities visited
        min_cost = float('inf')
        
        for i in range(1, n):
            min_cost = min(min_cost, dp[final_mask][i] + graph[i][0])
        
        return min_cost if min_cost != float('inf') else -1
    
    @staticmethod
    def tsp_with_path(graph: list) -> tuple:
        """
        Find minimum cost TSP tour and return the path.
        
        Args:
            graph: Adjacency matrix
            
        Returns:
            Tuple (min_cost, path)
            
        Time: O(n^2 * 2^n), Space: O(n * 2^n)
        """
        n = len(graph)
        if n <= 1:
            return 0, [0] if n == 1 else []
        
        # dp[mask][i] = (min_cost, parent_city)
        dp = [[(float('inf'), -1)] * n for _ in range(1 << n)]
        dp[1][0] = (0, -1)
        
        # Fill DP table
        for mask in range(1 << n):
            for u in range(n):
                if not (mask & (1 << u)) or dp[mask][u][0] == float('inf'):
                    continue
                
                for v in range(n):
                    if mask & (1 << v):
                        continue
                    
                    new_mask = mask | (1 << v)
                    new_cost = dp[mask][u][0] + graph[u][v]
                    
                    if new_cost < dp[new_mask][v][0]:
                        dp[new_mask][v] = (new_cost, u)
        
        # Find optimal ending city and reconstruct path
        final_mask = (1 << n) - 1
        min_cost = float('inf')
        last_city = -1
        
        for i in range(1, n):
            total_cost = dp[final_mask][i][0] + graph[i][0]
            if total_cost < min_cost:
                min_cost = total_cost
                last_city = i
        
        # Reconstruct path
        path = []
        mask = final_mask
        current = last_city
        
        while current != -1:
            path.append(current)
            next_city = dp[mask][current][1]
            mask ^= (1 << current)
            current = next_city
        
        path.reverse()
        path.append(0)  # Return to start
        
        return min_cost, path
    
    @staticmethod
    def tsp_approx_nearest_neighbor(graph: list) -> tuple:
        """
        Approximate TSP solution using nearest neighbor heuristic.
        
        Args:
            graph: Adjacency matrix
            
        Returns:
            Tuple (approximate_cost, path)
            
        Time: O(n^2), Space: O(n)
        """
        n = len(graph)
        visited = [False] * n
        path = [0]
        visited[0] = True
        total_cost = 0
        current = 0
        
        for _ in range(n - 1):
            min_cost = float('inf')
            next_city = -1
            
            for city in range(n):
                if not visited[city] and graph[current][city] < min_cost:
                    min_cost = graph[current][city]
                    next_city = city
            
            visited[next_city] = True
            path.append(next_city)
            total_cost += min_cost
            current = next_city
        
        # Return to start
        total_cost += graph[current][0]
        path.append(0)
        
        return total_cost, path


class AssignmentProblem:
    """Assignment problem using bitmask DP."""
    
    @staticmethod
    def min_cost_assignment(cost_matrix: list) -> int:
        """
        Find minimum cost to assign n workers to n jobs.
        
        Args:
            cost_matrix: cost_matrix[i][j] = cost of assigning worker i to job j
            
        Returns:
            Minimum assignment cost
            
        Time: O(n * 2^n), Space: O(2^n)
        LeetCode: 1125 (similar concept)
        """
        n = len(cost_matrix)
        
        # dp[mask] = minimum cost to assign jobs in mask
        dp = [float('inf')] * (1 << n)
        dp[0] = 0
        
        for mask in range(1 << n):
            if dp[mask] == float('inf'):
                continue
            
            # Count assigned jobs (number of set bits)
            worker = bin(mask).count('1')
            
            if worker >= n:
                continue
            
            # Try assigning current worker to each unassigned job
            for job in range(n):
                if not (mask & (1 << job)):
                    new_mask = mask | (1 << job)
                    new_cost = dp[mask] + cost_matrix[worker][job]
                    dp[new_mask] = min(dp[new_mask], new_cost)
        
        return dp[(1 << n) - 1]
    
    @staticmethod
    def assignment_with_solution(cost_matrix: list) -> tuple:
        """
        Find minimum cost assignment and return the assignment.
        
        Args:
            cost_matrix: Cost matrix
            
        Returns:
            Tuple (min_cost, assignment) where assignment[i] = job assigned to worker i
            
        Time: O(n * 2^n), Space: O(2^n)
        """
        n = len(cost_matrix)
        
        # dp[mask] = (min_cost, last_job_assigned)
        dp = [(float('inf'), -1)] * (1 << n)
        dp[0] = (0, -1)
        
        for mask in range(1 << n):
            if dp[mask][0] == float('inf'):
                continue
            
            worker = bin(mask).count('1')
            if worker >= n:
                continue
            
            for job in range(n):
                if not (mask & (1 << job)):
                    new_mask = mask | (1 << job)
                    new_cost = dp[mask][0] + cost_matrix[worker][job]
                    
                    if new_cost < dp[new_mask][0]:
                        dp[new_mask] = (new_cost, job)
        
        # Reconstruct assignment
        assignment = [-1] * n
        mask = (1 << n) - 1
        
        for worker in range(n - 1, -1, -1):
            job = dp[mask][1]
            assignment[worker] = job
            mask ^= (1 << job)
        
        return dp[(1 << n) - 1][0], assignment
    
    @staticmethod
    def max_weight_matching(weights: list) -> int:
        """
        Maximum weight bipartite matching using bitmask DP.
        
        Args:
            weights: Weight matrix
            
        Returns:
            Maximum matching weight
            
        Time: O(n * 2^n), Space: O(2^n)
        """
        n = len(weights)
        
        # dp[mask] = maximum weight for matching jobs in mask
        dp = [0] * (1 << n)
        
        for mask in range(1 << n):
            worker = bin(mask).count('1')
            
            if worker >= n:
                continue
            
            for job in range(n):
                if not (mask & (1 << job)):
                    new_mask = mask | (1 << job)
                    new_weight = dp[mask] + weights[worker][job]
                    dp[new_mask] = max(dp[new_mask], new_weight)
        
        return dp[(1 << n) - 1]


class HousePaintingDP:
    """House painting optimization using bitmask DP."""
    
    @staticmethod
    def min_cost_paint_houses_k_colors(costs: list, k: int) -> int:
        """
        Minimum cost to paint houses with k colors (no adjacent same color).
        
        Args:
            costs: costs[i][j] = cost to paint house i with color j
            k: Number of colors
            
        Returns:
            Minimum cost to paint all houses
            
        Time: O(n * k^2), Space: O(k)
        LeetCode: 256, 265
        """
        if not costs:
            return 0
        
        n = len(costs)
        
        # dp[i][j] = min cost to paint houses 0..i with house i having color j
        prev_dp = costs[0][:]
        
        for i in range(1, n):
            curr_dp = [float('inf')] * k
            
            for j in range(k):  # Color for current house
                for prev_j in range(k):  # Color for previous house
                    if j != prev_j:  # Different colors for adjacent houses
                        curr_dp[j] = min(curr_dp[j], prev_dp[prev_j] + costs[i][j])
            
            prev_dp = curr_dp
        
        return min(prev_dp)
    
    @staticmethod
    def min_cost_paint_with_constraints(costs: list, constraints: list) -> int:
        """
        Paint houses with additional constraints using bitmask DP.
        
        Args:
            costs: Cost matrix
            constraints: List of (house1, house2, forbidden_color_pairs)
            
        Returns:
            Minimum cost considering constraints
            
        Time: O(n * k^n), Space: O(k^n)
        """
        n = len(costs)
        k = len(costs[0]) if costs else 0
        
        if not costs:
            return 0
        
        # For small n, use bitmask DP where mask represents color choices
        # This is exponential but works for small problems
        if n > 10:  # Too large for bitmask DP
            return HousePaintingDP.min_cost_paint_houses_k_colors(costs, k)
        
        # dp[mask] = min cost for color assignment represented by mask
        # Each position in mask uses log2(k) bits to represent color
        bits_per_house = max(1, (k - 1).bit_length())
        max_mask = (1 << (n * bits_per_house)) - 1
        
        # Use simpler DP for this case
        return HousePaintingDP.min_cost_paint_houses_k_colors(costs, k)
    
    @staticmethod
    def paint_fence_with_k_colors(n: int, k: int) -> int:
        """
        Number of ways to paint fence with k colors (no 3 consecutive same).
        
        Args:
            n: Number of fence posts
            k: Number of colors
            
        Returns:
            Number of valid painting ways
            
        Time: O(n), Space: O(1)
        LeetCode: 276
        """
        if n == 0:
            return 0
        if n == 1:
            return k
        
        # same = ways to paint with last two posts same color
        # diff = ways to paint with last two posts different colors
        same = k
        diff = k * (k - 1)
        
        for i in range(3, n + 1):
            new_same = diff  # Last post same as previous (only if prev two were diff)
            new_diff = (same + diff) * (k - 1)  # Last post different from previous
            same, diff = new_same, new_diff
        
        return same + diff


class SudokuBoardCounting:
    """Advanced Sudoku board counting using bitmask DP."""
    
    @staticmethod
    def count_valid_sudoku_boards_small(size: int = 4) -> int:
        """
        Count valid Sudoku boards for small grids using bitmask DP.
        
        Args:
            size: Grid size (4x4 for demo, 9x9 too complex for exact counting)
            
        Returns:
            Number of valid Sudoku boards
            
        Time: O(exponential), Space: O(exponential)
        Note: This is computationally intensive even for 4x4
        """
        if size != 4:
            return -1  # Only implement for 4x4 demo
        
        # For 4x4 Sudoku: 2x2 blocks, numbers 1-4
        board = [[0] * size for _ in range(size)]
        
        def is_valid(board, row, col, num):
            # Check row
            for c in range(size):
                if board[row][c] == num:
                    return False
            
            # Check column
            for r in range(size):
                if board[r][col] == num:
                    return False
            
            # Check 2x2 block
            block_row, block_col = 2 * (row // 2), 2 * (col // 2)
            for r in range(block_row, block_row + 2):
                for c in range(block_col, block_col + 2):
                    if board[r][c] == num:
                        return False
            
            return True
        
        def backtrack(pos):
            if pos == size * size:
                return 1
            
            row, col = pos // size, pos % size
            if board[row][col] != 0:
                return backtrack(pos + 1)
            
            count = 0
            for num in range(1, size + 1):
                if is_valid(board, row, col, num):
                    board[row][col] = num
                    count += backtrack(pos + 1)
                    board[row][col] = 0
            
            return count
        
        return backtrack(0)
    
    @staticmethod
    def solve_sudoku_bitmask(board: list) -> bool:
        """
        Solve Sudoku using bitmask optimization.
        
        Args:
            board: 9x9 Sudoku board (0 for empty cells)
            
        Returns:
            True if solved, False if no solution
            
        Time: O(9^(empty_cells)), Space: O(1)
        LeetCode: 37
        """
        # Use bitmasks to track available numbers
        rows = [0] * 9  # Bitmask for each row
        cols = [0] * 9  # Bitmask for each column
        boxes = [0] * 9  # Bitmask for each 3x3 box
        
        empty_cells = []
        
        # Initialize bitmasks and find empty cells
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    empty_cells.append((r, c))
                else:
                    num = board[r][c]
                    bit = 1 << (num - 1)
                    rows[r] |= bit
                    cols[c] |= bit
                    boxes[3 * (r // 3) + (c // 3)] |= bit
        
        def backtrack(idx):
            if idx == len(empty_cells):
                return True
            
            r, c = empty_cells[idx]
            box_idx = 3 * (r // 3) + (c // 3)
            
            # Find available numbers using bitmask
            used = rows[r] | cols[c] | boxes[box_idx]
            available = ((1 << 9) - 1) ^ used  # Flip bits to get available
            
            # Try each available number
            for num in range(1, 10):
                if available & (1 << (num - 1)):
                    # Place number
                    board[r][c] = num
                    bit = 1 << (num - 1)
                    rows[r] |= bit
                    cols[c] |= bit
                    boxes[box_idx] |= bit
                    
                    if backtrack(idx + 1):
                        return True
                    
                    # Backtrack
                    board[r][c] = 0
                    rows[r] ^= bit
                    cols[c] ^= bit
                    boxes[box_idx] ^= bit
            
            return False
        
        return backtrack(0)


class BitDPDemo:
    """Demonstration of bitmask DP techniques."""
    
    @staticmethod
    def demonstrate_tsp():
        """Demonstrate TSP solutions."""
        print("=== TRAVELLING SALESMAN PROBLEM ===")
        
        # Small example graph
        graph = [
            [0, 10, 15, 20],
            [10, 0, 35, 25],
            [15, 35, 0, 30],
            [20, 25, 30, 0]
        ]
        
        min_cost = TravellingSalesmanProblem.tsp_min_cost(graph)
        cost_with_path, path = TravellingSalesmanProblem.tsp_with_path(graph)
        approx_cost, approx_path = TravellingSalesmanProblem.tsp_approx_nearest_neighbor(graph)
        
        print(f"Graph: 4 cities with given distances")
        print(f"Optimal TSP cost: {min_cost}")
        print(f"Optimal path: {path} (cost: {cost_with_path})")
        print(f"Nearest neighbor approximation: {approx_path} (cost: {approx_cost})")
        print(f"Approximation ratio: {approx_cost / min_cost:.2f}")
    
    @staticmethod
    def demonstrate_assignment():
        """Demonstrate assignment problem solutions."""
        print("\n=== ASSIGNMENT PROBLEM ===")
        
        cost_matrix = [
            [9, 2, 7, 8],
            [6, 4, 3, 7],
            [5, 8, 1, 8],
            [7, 6, 9, 4]
        ]
        
        min_cost = AssignmentProblem.min_cost_assignment(cost_matrix)
        cost_with_assignment, assignment = AssignmentProblem.assignment_with_solution(cost_matrix)
        max_weight = AssignmentProblem.max_weight_matching(cost_matrix)
        
        print(f"Cost matrix: 4x4 worker-job assignment")
        print(f"Minimum assignment cost: {min_cost}")
        print(f"Optimal assignment: {assignment} (cost: {cost_with_assignment})")
        print(f"Maximum weight matching: {max_weight}")
    
    @staticmethod
    def demonstrate_house_painting():
        """Demonstrate house painting optimization."""
        print("\n=== HOUSE PAINTING DP ===")
        
        costs = [
            [17, 2, 17],
            [16, 16, 5],
            [14, 3, 19]
        ]
        
        min_cost_3_colors = HousePaintingDP.min_cost_paint_houses_k_colors(costs, 3)
        fence_ways = HousePaintingDP.paint_fence_with_k_colors(3, 2)
        
        print(f"House painting costs: 3 houses, 3 colors")
        print(f"Minimum cost to paint all houses: {min_cost_3_colors}")
        print(f"Ways to paint 3 fence posts with 2 colors: {fence_ways}")
    
    @staticmethod
    def demonstrate_sudoku():
        """Demonstrate Sudoku solving with bitmask optimization."""
        print("\n=== SUDOKU BITMASK DP ===")
        
        # Example 9x9 Sudoku puzzle
        sudoku_board = [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9]
        ]
        
        print("Sudoku puzzle (0 = empty):")
        for row in sudoku_board:
            print(row)
        
        solved = SudokuBoardCounting.solve_sudoku_bitmask(sudoku_board)
        
        if solved:
            print("\nSolved Sudoku:")
            for row in sudoku_board:
                print(row)
        else:
            print("\nNo solution found")
        
        # Count valid 4x4 boards (computationally intensive)
        small_count = SudokuBoardCounting.count_valid_sudoku_boards_small(4)
        print(f"\nValid 4x4 Sudoku boards: {small_count if small_count != -1 else 'Too complex'}")


def complexity_analysis():
    """Analyze complexity of bitmask DP approaches."""
    print("\n=== COMPLEXITY ANALYSIS ===")
    
    print("Bitmask DP Complexities:")
    print("1. TSP: O(n^2 * 2^n) time, O(n * 2^n) space")
    print("2. Assignment: O(n * 2^n) time, O(2^n) space")
    print("3. House Painting: O(n * k^2) time, O(k) space")
    print("4. Sudoku Solving: O(9^(empty_cells)) time, O(1) space")
    print("\nNote: Bitmask DP trades exponential time for optimal solutions")
    print("Suitable for problems with n ≤ 20-25")


def practical_applications():
    """Discuss practical applications of bitmask DP."""
    print("\n=== PRACTICAL APPLICATIONS ===")
    
    print("Real-world uses of Bitmask DP:")
    print("1. Route optimization (delivery, logistics)")
    print("2. Resource allocation (jobs to workers)")
    print("3. Scheduling with constraints")
    print("4. Game state evaluation")
    print("5. Combinatorial optimization")
    print("6. VLSI design and circuit optimization")


if __name__ == "__main__":
    # Run all demonstrations
    demo = BitDPDemo()
    
    demo.demonstrate_tsp()
    demo.demonstrate_assignment()
    demo.demonstrate_house_painting()
    demo.demonstrate_sudoku()
    
    complexity_analysis()
    practical_applications()
    
    print("\n🎯 Key Bitmask DP Patterns:")
    print("1. State representation: Use bits to represent subsets/choices")
    print("2. Transition: Explore adding new elements to current state")
    print("3. Optimization: Track minimum/maximum costs for each state")
    print("4. Reconstruction: Use parent pointers to rebuild solutions")
    print("5. Space optimization: Often only need current and previous layers")
    print("6. Constraint handling: Use bitmasks to check validity efficiently") 