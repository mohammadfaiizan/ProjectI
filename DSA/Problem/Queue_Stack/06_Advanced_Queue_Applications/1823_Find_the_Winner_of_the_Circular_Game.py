"""
1823. Find the Winner of the Circular Game - Multiple Approaches
Difficulty: Medium

There are n friends that are playing a game. The friends are sitting in a circle and are numbered from 1 to n in clockwise order. More formally, moving clockwise from the ith friend brings you to the (i+1)th friend for 1 <= i < n, and moving clockwise from the nth friend brings you to the 1st friend.

The rules of the game are as follows:
1. Start at the 1st friend.
2. Count the next k friends in the clockwise direction including the friend you started at. The counting wraps around the circle and may count some friends more than once.
3. The last friend you counted leaves the game.
4. If there is still more than one friend in the game, go back to step 2 starting from the friend immediately clockwise of the friend who just left and repeat.
5. Else, the last friend in the game is the winner.

Given the number of friends, n, and an integer k, return the winner of the game.
"""

from typing import List
from collections import deque

class FindWinnerOfCircularGame:
    """Multiple approaches to solve Josephus problem"""
    
    def findTheWinner_queue_simulation(self, n: int, k: int) -> int:
        """
        Approach 1: Queue Simulation (Optimal for understanding)
        
        Use queue to simulate the circular elimination process.
        
        Time: O(n * k), Space: O(n)
        """
        queue = deque(range(1, n + 1))
        
        while len(queue) > 1:
            # Skip k-1 people
            for _ in range(k - 1):
                queue.append(queue.popleft())
            
            # Remove the kth person
            eliminated = queue.popleft()
        
        return queue[0]
    
    def findTheWinner_josephus_formula(self, n: int, k: int) -> int:
        """
        Approach 2: Josephus Formula (Optimal)
        
        Use mathematical formula for Josephus problem.
        
        Time: O(n), Space: O(1)
        """
        def josephus(n: int, k: int) -> int:
            """Josephus problem with 0-based indexing"""
            if n == 1:
                return 0
            return (josephus(n - 1, k) + k) % n
        
        # Convert to 1-based indexing
        return josephus(n, k) + 1
    
    def findTheWinner_iterative_josephus(self, n: int, k: int) -> int:
        """
        Approach 3: Iterative Josephus Formula
        
        Iterative version of Josephus formula.
        
        Time: O(n), Space: O(1)
        """
        result = 0  # 0-based indexing
        
        for i in range(2, n + 1):
            result = (result + k) % i
        
        return result + 1  # Convert to 1-based
    
    def findTheWinner_list_simulation(self, n: int, k: int) -> int:
        """
        Approach 4: List Simulation
        
        Use list to simulate the elimination process.
        
        Time: O(n²), Space: O(n)
        """
        friends = list(range(1, n + 1))
        current_pos = 0
        
        while len(friends) > 1:
            # Find position to eliminate
            eliminate_pos = (current_pos + k - 1) % len(friends)
            
            # Remove the friend
            friends.pop(eliminate_pos)
            
            # Update current position
            current_pos = eliminate_pos % len(friends)
        
        return friends[0]
    
    def findTheWinner_recursive_simulation(self, n: int, k: int) -> int:
        """
        Approach 5: Recursive Simulation
        
        Use recursion to simulate the process.
        
        Time: O(n * k), Space: O(n)
        """
        def eliminate(friends: List[int], start: int) -> int:
            if len(friends) == 1:
                return friends[0]
            
            # Find elimination position
            eliminate_pos = (start + k - 1) % len(friends)
            
            # Remove friend and recurse
            eliminated = friends.pop(eliminate_pos)
            next_start = eliminate_pos % len(friends)
            
            return eliminate(friends, next_start)
        
        friends = list(range(1, n + 1))
        return eliminate(friends, 0)
    
    def findTheWinner_linked_list_simulation(self, n: int, k: int) -> int:
        """
        Approach 6: Linked List Simulation
        
        Use circular linked list for simulation.
        
        Time: O(n * k), Space: O(n)
        """
        class ListNode:
            def __init__(self, val: int):
                self.val = val
                self.next = None
        
        # Create circular linked list
        head = ListNode(1)
        current = head
        
        for i in range(2, n + 1):
            current.next = ListNode(i)
            current = current.next
        
        current.next = head  # Make it circular
        
        # Simulate elimination
        current = head
        
        while current.next != current:  # More than one node
            # Skip k-1 nodes
            for _ in range(k - 2):
                current = current.next
            
            # Remove next node
            current.next = current.next.next
            current = current.next
        
        return current.val


def test_find_winner_of_circular_game():
    """Test find winner algorithms"""
    solver = FindWinnerOfCircularGame()
    
    test_cases = [
        (5, 2, 3, "Example 1"),
        (6, 5, 1, "Example 2"),
        (1, 1, 1, "Single person"),
        (2, 1, 2, "Two people, k=1"),
        (2, 2, 1, "Two people, k=2"),
        (3, 3, 2, "Three people, k=3"),
        (7, 3, 4, "Seven people, k=3"),
        (10, 4, 5, "Ten people, k=4"),
    ]
    
    algorithms = [
        ("Queue Simulation", solver.findTheWinner_queue_simulation),
        ("Josephus Formula", solver.findTheWinner_josephus_formula),
        ("Iterative Josephus", solver.findTheWinner_iterative_josephus),
        ("List Simulation", solver.findTheWinner_list_simulation),
        ("Recursive Simulation", solver.findTheWinner_recursive_simulation),
        ("Linked List Simulation", solver.findTheWinner_linked_list_simulation),
    ]
    
    print("=== Testing Find Winner of Circular Game ===")
    
    for n, k, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"n: {n}, k: {k}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(n, k)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:25} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:25} | ERROR: {str(e)[:40]}")


def demonstrate_queue_simulation():
    """Demonstrate queue simulation step by step"""
    print("\n=== Queue Simulation Step-by-Step Demo ===")
    
    n, k = 5, 2
    print(f"n = {n}, k = {k}")
    print("Friends numbered 1 to 5 in circle")
    
    queue = deque(range(1, n + 1))
    print(f"Initial queue: {list(queue)}")
    
    round_num = 1
    while len(queue) > 1:
        print(f"\nRound {round_num}:")
        print(f"  Current queue: {list(queue)}")
        
        # Skip k-1 people
        for i in range(k - 1):
            moved = queue.popleft()
            queue.append(moved)
            print(f"  Skip person {moved}: {list(queue)}")
        
        # Eliminate kth person
        eliminated = queue.popleft()
        print(f"  Eliminate person {eliminated}: {list(queue)}")
        
        round_num += 1
    
    winner = queue[0]
    print(f"\nWinner: Person {winner}")


def demonstrate_josephus_formula():
    """Demonstrate Josephus formula derivation"""
    print("\n=== Josephus Formula Demonstration ===")
    
    print("Josephus Problem Formula:")
    print("J(n, k) = (J(n-1, k) + k) % n")
    print("Base case: J(1, k) = 0 (0-indexed)")
    
    n, k = 5, 2
    print(f"\nCalculating J({n}, {k}):")
    
    # Show recursive calculation
    def josephus_with_trace(n: int, k: int, depth: int = 0) -> int:
        indent = "  " * depth
        
        if n == 1:
            print(f"{indent}J(1, {k}) = 0 (base case)")
            return 0
        
        print(f"{indent}J({n}, {k}) = (J({n-1}, {k}) + {k}) % {n}")
        
        prev_result = josephus_with_trace(n - 1, k, depth + 1)
        result = (prev_result + k) % n
        
        print(f"{indent}J({n}, {k}) = ({prev_result} + {k}) % {n} = {result}")
        
        return result
    
    result_0_indexed = josephus_with_trace(n, k)
    result_1_indexed = result_0_indexed + 1
    
    print(f"\nFinal result (0-indexed): {result_0_indexed}")
    print(f"Final result (1-indexed): {result_1_indexed}")


def visualize_elimination_process():
    """Visualize the elimination process"""
    print("\n=== Elimination Process Visualization ===")
    
    n, k = 6, 3
    print(f"n = {n}, k = {k}")
    
    friends = list(range(1, n + 1))
    current_pos = 0
    
    print("Initial circle:")
    print(f"  Friends: {friends}")
    print(f"  Starting at position {current_pos} (friend {friends[current_pos]})")
    
    round_num = 1
    while len(friends) > 1:
        print(f"\nRound {round_num}:")
        print(f"  Current friends: {friends}")
        print(f"  Starting from position {current_pos} (friend {friends[current_pos]})")
        
        # Calculate elimination position
        eliminate_pos = (current_pos + k - 1) % len(friends)
        eliminated_friend = friends[eliminate_pos]
        
        print(f"  Count {k} positions: eliminate position {eliminate_pos} (friend {eliminated_friend})")
        
        # Remove friend
        friends.pop(eliminate_pos)
        
        # Update current position
        if eliminate_pos < len(friends):
            current_pos = eliminate_pos
        else:
            current_pos = 0
        
        if friends:
            print(f"  After elimination: {friends}")
            print(f"  Next starting position: {current_pos} (friend {friends[current_pos]})")
        
        round_num += 1
    
    print(f"\nWinner: Friend {friends[0]}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Load balancing
    print("1. Round-Robin Load Balancing:")
    servers = 5
    skip_factor = 2
    
    solver = FindWinnerOfCircularGame()
    last_server = solver.findTheWinner_iterative_josephus(servers, skip_factor)
    
    print(f"  {servers} servers in rotation")
    print(f"  Skip factor: {skip_factor}")
    print(f"  Last server to handle requests: Server {last_server}")
    
    # Application 2: Task scheduling
    print(f"\n2. Circular Task Scheduling:")
    tasks = 7
    priority_skip = 3
    
    final_task = solver.findTheWinner_iterative_josephus(tasks, priority_skip)
    
    print(f"  {tasks} tasks in circular queue")
    print(f"  Priority elimination every {priority_skip} tasks")
    print(f"  Final high-priority task: Task {final_task}")
    
    # Application 3: Game elimination
    print(f"\n3. Tournament Elimination:")
    players = 8
    elimination_rate = 2
    
    winner = solver.findTheWinner_iterative_josephus(players, elimination_rate)
    
    print(f"  {players} players in tournament")
    print(f"  Elimination every {elimination_rate} positions")
    print(f"  Tournament winner: Player {winner}")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Queue Simulation", "O(n * k)", "O(n)", "Simulate each elimination"),
        ("Josephus Formula", "O(n)", "O(n)", "Recursive formula"),
        ("Iterative Josephus", "O(n)", "O(1)", "Optimal solution"),
        ("List Simulation", "O(n²)", "O(n)", "List operations are O(n)"),
        ("Recursive Simulation", "O(n * k)", "O(n)", "Recursion + elimination"),
        ("Linked List", "O(n * k)", "O(n)", "Traverse k positions each time"),
    ]
    
    print(f"{'Approach':<25} | {'Time':<10} | {'Space':<8} | {'Notes'}")
    print("-" * 70)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<25} | {time_comp:<10} | {space_comp:<8} | {notes}")
    
    print(f"\nIterative Josephus formula is optimal: O(n) time, O(1) space")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = FindWinnerOfCircularGame()
    
    edge_cases = [
        (1, 1, 1, "Single person"),
        (1, 5, 1, "Single person, large k"),
        (2, 1, 2, "Two people, k=1"),
        (2, 2, 1, "Two people, k=2"),
        (2, 100, 1, "Two people, large k"),
        (3, 1, 3, "Three people, k=1"),
        (3, 3, 2, "Three people, k=3"),
        (100, 1, 100, "Large n, k=1"),
        (100, 2, 73, "Large n, k=2"),
    ]
    
    for n, k, expected, description in edge_cases:
        try:
            result = solver.findTheWinner_iterative_josephus(n, k)
            status = "✓" if result == expected else "✗"
            print(f"{description:25} | {status} | n={n}, k={k} -> {result}")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")


def benchmark_approaches():
    """Benchmark different approaches"""
    import time
    
    approaches = [
        ("Queue Simulation", FindWinnerOfCircularGame().findTheWinner_queue_simulation),
        ("Iterative Josephus", FindWinnerOfCircularGame().findTheWinner_iterative_josephus),
        ("List Simulation", FindWinnerOfCircularGame().findTheWinner_list_simulation),
    ]
    
    test_cases = [(100, 3), (1000, 5), (5000, 7)]
    
    print("\n=== Performance Benchmark ===")
    
    for n, k in test_cases:
        print(f"\nn = {n}, k = {k}:")
        
        for name, func in approaches:
            start_time = time.time()
            
            try:
                result = func(n, k)
                end_time = time.time()
                print(f"  {name:20} | Time: {end_time - start_time:.4f}s | Result: {result}")
            except Exception as e:
                print(f"  {name:20} | ERROR: {str(e)[:30]}")


if __name__ == "__main__":
    test_find_winner_of_circular_game()
    demonstrate_queue_simulation()
    demonstrate_josephus_formula()
    visualize_elimination_process()
    demonstrate_real_world_applications()
    analyze_time_complexity()
    test_edge_cases()
    benchmark_approaches()

"""
Find the Winner of the Circular Game demonstrates advanced queue applications
for the classic Josephus problem, including multiple simulation approaches
and mathematical optimization with real-world scheduling applications.
"""
