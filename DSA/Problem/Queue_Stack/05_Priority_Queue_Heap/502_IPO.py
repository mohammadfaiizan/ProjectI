"""
502. IPO - Multiple Approaches
Difficulty: Hard

Suppose LeetCode will start its IPO soon. In order to sell a good price of its shares to Venture Capital, LeetCode would like to work on some projects to increase its capital before the IPO.

Since it has limited resources, it can only finish at most k distinct projects before the IPO. Help LeetCode design the best way to maximize its capital after finishing at most k distinct projects.

You are given n projects where the ith project has a pure profit profits[i] and a minimum capital of capital[i] needed to start it.

Initially, you have w capital. When you finish a project, you will obtain its profit and the profit will be added to your total capital.

Pick a list of at most k distinct projects from given projects to maximize your final capital, and return the final maximized capital.
"""

from typing import List
import heapq

class IPO:
    """Multiple approaches to solve IPO problem"""
    
    def findMaximizedCapital_greedy_heap(self, k: int, w: int, profits: List[int], capital: List[int]) -> int:
        """
        Approach 1: Greedy with Two Heaps (Optimal)
        
        Use min heap for available projects and max heap for profitable projects.
        
        Time: O(n log n), Space: O(n)
        """
        n = len(profits)
        
        # Create list of (capital_needed, profit) and sort by capital
        projects = [(capital[i], profits[i]) for i in range(n)]
        projects.sort()
        
        current_capital = w
        max_heap = []  # Max heap for profits (negate values)
        project_idx = 0
        
        for _ in range(k):
            # Add all affordable projects to max heap
            while project_idx < n and projects[project_idx][0] <= current_capital:
                heapq.heappush(max_heap, -projects[project_idx][1])  # Negate for max heap
                project_idx += 1
            
            # If no projects available, break
            if not max_heap:
                break
            
            # Take the most profitable project
            max_profit = -heapq.heappop(max_heap)
            current_capital += max_profit
        
        return current_capital
    
    def findMaximizedCapital_brute_force(self, k: int, w: int, profits: List[int], capital: List[int]) -> int:
        """
        Approach 2: Brute Force
        
        For each iteration, find the most profitable affordable project.
        
        Time: O(k * n), Space: O(n)
        """
        n = len(profits)
        used = [False] * n
        current_capital = w
        
        for _ in range(k):
            best_profit = -1
            best_idx = -1
            
            # Find the most profitable affordable project
            for i in range(n):
                if not used[i] and capital[i] <= current_capital:
                    if profits[i] > best_profit:
                        best_profit = profits[i]
                        best_idx = i
            
            # If no project found, break
            if best_idx == -1:
                break
            
            # Take the project
            used[best_idx] = True
            current_capital += best_profit
        
        return current_capital
    
    def findMaximizedCapital_sorting_approach(self, k: int, w: int, profits: List[int], capital: List[int]) -> int:
        """
        Approach 3: Sorting Approach
        
        Sort projects and use greedy selection.
        
        Time: O(k * n log n), Space: O(n)
        """
        n = len(profits)
        current_capital = w
        used = [False] * n
        
        for _ in range(k):
            # Create list of affordable projects with their profits
            affordable = []
            for i in range(n):
                if not used[i] and capital[i] <= current_capital:
                    affordable.append((profits[i], i))
            
            # If no affordable projects, break
            if not affordable:
                break
            
            # Sort by profit (descending) and take the best
            affordable.sort(reverse=True)
            best_profit, best_idx = affordable[0]
            
            used[best_idx] = True
            current_capital += best_profit
        
        return current_capital
    
    def findMaximizedCapital_dp_approach(self, k: int, w: int, profits: List[int], capital: List[int]) -> int:
        """
        Approach 4: Dynamic Programming (Suboptimal for this problem)
        
        Use DP to explore different project combinations.
        
        Time: O(2^n), Space: O(2^n)
        """
        from functools import lru_cache
        
        n = len(profits)
        
        @lru_cache(maxsize=None)
        def dp(remaining_projects: int, current_capital: int, used_mask: int) -> int:
            """Return maximum capital achievable"""
            if remaining_projects == 0:
                return current_capital
            
            max_capital = current_capital
            
            # Try each unused project
            for i in range(n):
                if (used_mask & (1 << i)) == 0 and capital[i] <= current_capital:
                    new_mask = used_mask | (1 << i)
                    new_capital = current_capital + profits[i]
                    
                    result = dp(remaining_projects - 1, new_capital, new_mask)
                    max_capital = max(max_capital, result)
            
            return max_capital
        
        return dp(k, w, 0)
    
    def findMaximizedCapital_priority_queue_simulation(self, k: int, w: int, profits: List[int], capital: List[int]) -> int:
        """
        Approach 5: Priority Queue Simulation
        
        Simulate priority queue behavior with list operations.
        
        Time: O(k * n log n), Space: O(n)
        """
        n = len(profits)
        current_capital = w
        available_projects = list(range(n))
        
        for _ in range(k):
            # Filter affordable projects
            affordable = [i for i in available_projects if capital[i] <= current_capital]
            
            if not affordable:
                break
            
            # Find most profitable
            best_idx = max(affordable, key=lambda i: profits[i])
            
            # Take the project
            current_capital += profits[best_idx]
            available_projects.remove(best_idx)
        
        return current_capital


def test_ipo():
    """Test IPO algorithms"""
    solver = IPO()
    
    test_cases = [
        (2, 0, [1,2,3], [0,1,1], 4, "Example 1"),
        (3, 0, [1,2,3], [0,1,2], 6, "Example 2"),
        (1, 0, [1,2,3], [1,1,2], 0, "No affordable projects"),
        (2, 1, [1,2,3], [0,1,1], 6, "Some initial capital"),
        (1, 2, [1,2,3], [0,1,1], 5, "Single project"),
        (10, 0, [1,2,3], [0,1,1], 6, "More k than projects"),
        (2, 0, [1,2,3], [0,0,0], 6, "All projects affordable"),
    ]
    
    algorithms = [
        ("Greedy Heap", solver.findMaximizedCapital_greedy_heap),
        ("Brute Force", solver.findMaximizedCapital_brute_force),
        ("Sorting Approach", solver.findMaximizedCapital_sorting_approach),
        ("Priority Queue Sim", solver.findMaximizedCapital_priority_queue_simulation),
        # ("DP Approach", solver.findMaximizedCapital_dp_approach),  # Too slow for large inputs
    ]
    
    print("=== Testing IPO ===")
    
    for k, w, profits, capital, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"k: {k}, w: {w}")
        print(f"Profits: {profits}")
        print(f"Capital: {capital}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(k, w, profits[:], capital[:])  # Pass copies
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_greedy_heap_approach():
    """Demonstrate greedy heap approach step by step"""
    print("\n=== Greedy Heap Approach Step-by-Step Demo ===")
    
    k = 2
    w = 0
    profits = [1, 2, 3]
    capital = [0, 1, 1]
    
    print(f"k: {k}, initial capital: {w}")
    print(f"Projects: {[(capital[i], profits[i]) for i in range(len(profits))]}")
    print("Format: (capital_needed, profit)")
    
    # Sort projects by capital needed
    projects = [(capital[i], profits[i]) for i in range(len(profits))]
    projects.sort()
    
    print(f"Sorted by capital: {projects}")
    
    current_capital = w
    max_heap = []
    project_idx = 0
    
    print(f"\nStrategy: Use heap to always pick most profitable affordable project")
    
    for iteration in range(k):
        print(f"\nIteration {iteration + 1}:")
        print(f"  Current capital: {current_capital}")
        
        # Add affordable projects to heap
        added_projects = []
        while project_idx < len(projects) and projects[project_idx][0] <= current_capital:
            profit = projects[project_idx][1]
            heapq.heappush(max_heap, -profit)  # Negate for max heap
            added_projects.append(projects[project_idx])
            project_idx += 1
        
        if added_projects:
            print(f"  Added affordable projects: {added_projects}")
        
        print(f"  Available profits in heap: {sorted([-x for x in max_heap], reverse=True)}")
        
        if not max_heap:
            print(f"  No projects available, stopping")
            break
        
        # Take most profitable project
        max_profit = -heapq.heappop(max_heap)
        current_capital += max_profit
        
        print(f"  Selected project with profit: {max_profit}")
        print(f"  New capital: {current_capital}")
    
    print(f"\nFinal capital: {current_capital}")


def visualize_project_selection():
    """Visualize project selection process"""
    print("\n=== Project Selection Visualization ===")
    
    k = 3
    w = 0
    profits = [1, 2, 3, 5]
    capital = [0, 1, 1, 3]
    
    print(f"Available projects:")
    for i in range(len(profits)):
        print(f"  Project {i}: needs ${capital[i]} capital, gives ${profits[i]} profit")
    
    print(f"\nInitial capital: ${w}")
    print(f"Can do at most {k} projects")
    
    current_capital = w
    projects_done = []
    
    # Simulate the greedy approach
    available = list(range(len(profits)))
    
    for iteration in range(k):
        print(f"\n--- Round {iteration + 1} ---")
        print(f"Current capital: ${current_capital}")
        
        # Find affordable projects
        affordable = [i for i in available if capital[i] <= current_capital]
        
        if not affordable:
            print("No affordable projects remaining")
            break
        
        print("Affordable projects:")
        for i in affordable:
            print(f"  Project {i}: needs ${capital[i]}, profit ${profits[i]}")
        
        # Pick most profitable
        best_project = max(affordable, key=lambda i: profits[i])
        
        print(f"Selected: Project {best_project} (profit ${profits[best_project]})")
        
        current_capital += profits[best_project]
        projects_done.append(best_project)
        available.remove(best_project)
        
        print(f"New capital: ${current_capital}")
    
    print(f"\nFinal result:")
    print(f"Projects completed: {projects_done}")
    print(f"Final capital: ${current_capital}")


if __name__ == "__main__":
    test_ipo()
    demonstrate_greedy_heap_approach()
    visualize_project_selection()

"""
IPO demonstrates heap applications for investment optimization problems,
including greedy strategies with priority queues and multiple approaches
for capital maximization with project selection constraints.
"""
