"""
649. Dota2 Senate - Multiple Approaches
Difficulty: Medium (but categorized as Easy in advanced queue context)

In the world of Dota2, there are two parties: the Radiant and the Dire.

The Dota2 senate consists of senators coming from two parties. Now the senate wants to decide on a change in the Dota2 game. The voting for this change is a round-based procedure. In each round, each senator can exercise one of the two rights:

1. Ban the next senator's right: A senator can make another senator lose all his rights in this and all the following rounds.
2. Announce the victory: If this senator found the senators still having rights to vote are all from the same party, he can announce the victory and decide on the change in the game.

Given a string senate representing each senator's party belonging. The character 'R' and 'D' represent the Radiant and the Dire party respectively. Then if there are n senators, the size of the given string will be n.

The round-based procedure starts from the first senator to the last senator in the given order. This procedure will last until the end of voting. All the senators who have lost their rights will be skipped during the procedure.

Suppose every senator is smart enough to play the optimal strategy, return the party that will finally win the game.
"""

from typing import List
from collections import deque

class Dota2Senate:
    """Multiple approaches to solve Dota2 Senate problem"""
    
    def predictPartyVictory_queue_simulation(self, senate: str) -> str:
        """
        Approach 1: Queue Simulation (Optimal)
        
        Use two queues to simulate the voting process.
        
        Time: O(n), Space: O(n)
        """
        radiant = deque()
        dire = deque()
        n = len(senate)
        
        # Initialize queues with positions
        for i, party in enumerate(senate):
            if party == 'R':
                radiant.append(i)
            else:
                dire.append(i)
        
        # Simulate rounds
        while radiant and dire:
            r_pos = radiant.popleft()
            d_pos = dire.popleft()
            
            # The senator with smaller position acts first
            if r_pos < d_pos:
                # Radiant bans Dire, Radiant gets to vote again in next round
                radiant.append(r_pos + n)
            else:
                # Dire bans Radiant, Dire gets to vote again in next round
                dire.append(d_pos + n)
        
        return "Radiant" if radiant else "Dire"
    
    def predictPartyVictory_greedy_approach(self, senate: str) -> str:
        """
        Approach 2: Greedy Approach
        
        Use greedy strategy to ban the next opponent.
        
        Time: O(n²), Space: O(n)
        """
        senate_list = list(senate)
        
        while True:
            # Count remaining senators
            radiant_count = senate_list.count('R')
            dire_count = senate_list.count('D')
            
            if radiant_count == 0:
                return "Dire"
            if dire_count == 0:
                return "Radiant"
            
            # Process one round
            new_senate = []
            radiant_bans = 0
            dire_bans = 0
            
            for senator in senate_list:
                if senator == 'R':
                    if dire_bans > 0:
                        dire_bans -= 1  # This Radiant is banned
                    else:
                        new_senate.append('R')
                        radiant_bans += 1  # This Radiant bans next Dire
                else:  # senator == 'D'
                    if radiant_bans > 0:
                        radiant_bans -= 1  # This Dire is banned
                    else:
                        new_senate.append('D')
                        dire_bans += 1  # This Dire bans next Radiant
            
            senate_list = new_senate
    
    def predictPartyVictory_circular_queue(self, senate: str) -> str:
        """
        Approach 3: Circular Queue
        
        Use circular queue to handle the round-based nature.
        
        Time: O(n), Space: O(n)
        """
        from collections import deque
        
        queue = deque(senate)
        radiant_count = senate.count('R')
        dire_count = senate.count('D')
        
        radiant_floating_bans = 0
        dire_floating_bans = 0
        
        while radiant_count > 0 and dire_count > 0:
            current = queue.popleft()
            
            if current == 'R':
                if dire_floating_bans > 0:
                    # This Radiant is banned
                    dire_floating_bans -= 1
                    radiant_count -= 1
                else:
                    # This Radiant survives and bans a Dire
                    radiant_floating_bans += 1
                    queue.append('R')
            else:  # current == 'D'
                if radiant_floating_bans > 0:
                    # This Dire is banned
                    radiant_floating_bans -= 1
                    dire_count -= 1
                else:
                    # This Dire survives and bans a Radiant
                    dire_floating_bans += 1
                    queue.append('D')
        
        return "Radiant" if radiant_count > 0 else "Dire"
    
    def predictPartyVictory_state_machine(self, senate: str) -> str:
        """
        Approach 4: State Machine
        
        Use state machine to track the voting process.
        
        Time: O(n), Space: O(n)
        """
        senate_list = list(senate)
        n = len(senate_list)
        
        # State: (radiant_active, dire_active, radiant_bans, dire_bans)
        radiant_active = senate.count('R')
        dire_active = senate.count('D')
        radiant_bans = 0
        dire_bans = 0
        
        i = 0
        while radiant_active > 0 and dire_active > 0:
            if senate_list[i % n] == 'R':
                if dire_bans > 0:
                    dire_bans -= 1
                    radiant_active -= 1
                    senate_list[i % n] = 'X'  # Mark as banned
                else:
                    radiant_bans += 1
            else:  # senate_list[i % n] == 'D'
                if radiant_bans > 0:
                    radiant_bans -= 1
                    dire_active -= 1
                    senate_list[i % n] = 'X'  # Mark as banned
                else:
                    dire_bans += 1
            
            i += 1
            
            # Skip banned senators
            while i < len(senate_list) * 10 and senate_list[i % n] == 'X':
                i += 1
        
        return "Radiant" if radiant_active > 0 else "Dire"
    
    def predictPartyVictory_priority_queue(self, senate: str) -> str:
        """
        Approach 5: Priority Queue
        
        Use priority queue to handle voting order.
        
        Time: O(n log n), Space: O(n)
        """
        import heapq
        
        radiant_queue = []
        dire_queue = []
        
        # Initialize priority queues with positions
        for i, party in enumerate(senate):
            if party == 'R':
                heapq.heappush(radiant_queue, i)
            else:
                heapq.heappush(dire_queue, i)
        
        round_offset = len(senate)
        
        while radiant_queue and dire_queue:
            r_pos = heapq.heappop(radiant_queue)
            d_pos = heapq.heappop(dire_queue)
            
            if r_pos < d_pos:
                # Radiant acts first, bans Dire
                heapq.heappush(radiant_queue, r_pos + round_offset)
            else:
                # Dire acts first, bans Radiant
                heapq.heappush(dire_queue, d_pos + round_offset)
        
        return "Radiant" if radiant_queue else "Dire"


def test_dota2_senate():
    """Test Dota2 Senate algorithms"""
    solver = Dota2Senate()
    
    test_cases = [
        ("RD", "Radiant", "Example 1"),
        ("RDD", "Dire", "Example 2"),
        ("DRDRDR", "Dire", "Alternating pattern"),
        ("R", "Radiant", "Single Radiant"),
        ("D", "Dire", "Single Dire"),
        ("RRR", "Radiant", "All Radiant"),
        ("DDD", "Dire", "All Dire"),
        ("RDRD", "Radiant", "Even split, Radiant first"),
        ("DRDR", "Dire", "Even split, Dire first"),
        ("RRDDD", "Dire", "Dire majority"),
    ]
    
    algorithms = [
        ("Queue Simulation", solver.predictPartyVictory_queue_simulation),
        ("Greedy Approach", solver.predictPartyVictory_greedy_approach),
        ("Circular Queue", solver.predictPartyVictory_circular_queue),
        ("State Machine", solver.predictPartyVictory_state_machine),
        ("Priority Queue", solver.predictPartyVictory_priority_queue),
    ]
    
    print("=== Testing Dota2 Senate ===")
    
    for senate, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Senate: '{senate}'")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(senate)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_queue_simulation():
    """Demonstrate queue simulation approach step by step"""
    print("\n=== Queue Simulation Step-by-Step Demo ===")
    
    senate = "RDD"
    print(f"Senate: '{senate}'")
    print("Strategy: Use two queues to track Radiant and Dire positions")
    
    radiant = deque()
    dire = deque()
    n = len(senate)
    
    # Initialize queues
    for i, party in enumerate(senate):
        if party == 'R':
            radiant.append(i)
        else:
            dire.append(i)
    
    print(f"\nInitial state:")
    print(f"  Radiant positions: {list(radiant)}")
    print(f"  Dire positions: {list(dire)}")
    
    round_num = 1
    while radiant and dire:
        print(f"\nRound {round_num}:")
        
        r_pos = radiant.popleft()
        d_pos = dire.popleft()
        
        print(f"  Radiant senator at position {r_pos} vs Dire senator at position {d_pos}")
        
        if r_pos < d_pos:
            radiant.append(r_pos + n)
            print(f"  Radiant acts first, bans Dire senator")
            print(f"  Radiant senator gets position {r_pos + n} for next round")
        else:
            dire.append(d_pos + n)
            print(f"  Dire acts first, bans Radiant senator")
            print(f"  Dire senator gets position {d_pos + n} for next round")
        
        print(f"  Remaining Radiant: {list(radiant)}")
        print(f"  Remaining Dire: {list(dire)}")
        
        round_num += 1
    
    winner = "Radiant" if radiant else "Dire"
    print(f"\nWinner: {winner}")


def visualize_voting_process():
    """Visualize the voting process"""
    print("\n=== Voting Process Visualization ===")
    
    senate = "DRDRDR"
    print(f"Senate: '{senate}'")
    print("Voting order: left to right, then repeat")
    
    # Show initial state
    print(f"\nInitial senators:")
    for i, party in enumerate(senate):
        party_name = "Radiant" if party == 'R' else "Dire"
        print(f"  Position {i}: {party_name}")
    
    # Simulate with detailed tracking
    solver = Dota2Senate()
    
    # Manual simulation for visualization
    radiant_positions = [i for i, p in enumerate(senate) if p == 'R']
    dire_positions = [i for i, p in enumerate(senate) if p == 'D']
    
    print(f"\nRadiant senators at positions: {radiant_positions}")
    print(f"Dire senators at positions: {dire_positions}")
    
    print(f"\nOptimal strategy: Each senator bans the next opponent in line")
    
    result = solver.predictPartyVictory_queue_simulation(senate)
    print(f"Final winner: {result}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Committee voting
    print("1. Committee Voting System:")
    committee = "RRDDRD"  # R = Republican, D = Democrat
    
    solver = Dota2Senate()
    winner = solver.predictPartyVictory_queue_simulation(committee)
    
    print(f"  Committee composition: {committee}")
    print(f"  Voting rule: Each member can block the next opponent")
    print(f"  Winning party: {winner}")
    
    # Application 2: Resource allocation
    print(f"\n2. Resource Allocation Game:")
    players = "ABBAAB"  # A and B are competing teams
    
    # Convert to R/D format for simulation
    converted = players.replace('A', 'R').replace('B', 'D')
    winner = solver.predictPartyVictory_queue_simulation(converted)
    original_winner = winner.replace('Radiant', 'Team A').replace('Dire', 'Team B')
    
    print(f"  Player order: {players}")
    print(f"  Rule: Each player can eliminate next opponent")
    print(f"  Winning team: {original_winner}")


if __name__ == "__main__":
    test_dota2_senate()
    demonstrate_queue_simulation()
    visualize_voting_process()
    demonstrate_real_world_applications()

"""
Dota2 Senate demonstrates advanced queue applications for game theory
and voting systems, including circular queue simulation and multiple
approaches for strategic decision-making algorithms.
"""
