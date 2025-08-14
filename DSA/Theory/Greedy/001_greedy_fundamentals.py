"""
Greedy Algorithm Fundamentals - Core Concepts and Principles
===========================================================

Topics: Greedy definition, properties, design principles, proof techniques
Companies: All tech companies test greedy algorithm understanding
Difficulty: Easy to Medium (concepts), Hard (proofs)
Time Complexity: Usually O(n log n) due to sorting
Space Complexity: O(1) to O(n) depending on problem
"""

from typing import List, Tuple, Optional, Dict, Any, Callable
import math

class GreedyFundamentals:
    
    def __init__(self):
        """Initialize with demonstration tracking"""
        self.operation_count = 0
        self.demo_steps = []
    
    # ==========================================
    # 1. WHAT IS A GREEDY ALGORITHM?
    # ==========================================
    
    def explain_greedy_concept(self) -> None:
        """
        Explain the fundamental concept of greedy algorithms
        
        Greedy algorithms make locally optimal choices at each step
        hoping to find a global optimum
        """
        print("=== WHAT IS A GREEDY ALGORITHM? ===")
        print("A Greedy Algorithm is a problem-solving technique with these characteristics:")
        print()
        print("KEY PRINCIPLES:")
        print("• Makes the locally optimal choice at each step")
        print("• Never reconsiders previous choices (no backtracking)")
        print("• Hopes that local optimality leads to global optimality")
        print("• Follows a 'greedy choice property'")
        print("• Often involves sorting and making sequential decisions")
        print()
        print("GREEDY CHOICE PROPERTY:")
        print("• A global optimum can be arrived at by making locally optimal choices")
        print("• The choice made at each step is the best available option")
        print("• Once a choice is made, it cannot be undone")
        print()
        print("OPTIMAL SUBSTRUCTURE:")
        print("• An optimal solution contains optimal solutions to subproblems")
        print("• Similar to dynamic programming, but without overlapping subproblems")
        print("• Each subproblem is solved exactly once")
        print()
        print("GENERAL GREEDY ALGORITHM STRUCTURE:")
        print("1. Sort the input based on some criteria")
        print("2. Initialize empty solution")
        print("3. For each element in sorted order:")
        print("   - If adding element improves the solution")
        print("   - Add element to solution")
        print("4. Return the solution")
        print()
        print("Real-world Analogies:")
        print("• Making change: Always give largest denomination possible")
        print("• Traffic routing: Always take the fastest current route")
        print("• Resource allocation: Assign to highest priority task first")
        print("• Eating at buffet: Take the most appealing food first")
    
    def greedy_vs_other_paradigms(self) -> None:
        """Compare greedy with other algorithmic paradigms"""
        print("=== GREEDY VS OTHER ALGORITHMIC PARADIGMS ===")
        print()
        print("Greedy vs Dynamic Programming:")
        print("  Greedy: Makes irrevocable choices based on current information")
        print("  DP: Considers all possible choices and remembers results")
        print("  Greedy: Faster (usually O(n log n)) but doesn't always work")
        print("  DP: Slower (usually O(n²) or higher) but guaranteed optimal")
        print("  Use Greedy when: Greedy choice property holds")
        print()
        print("Greedy vs Divide and Conquer:")
        print("  Greedy: Solves by making a series of choices")
        print("  D&C: Breaks problem into subproblems and combines solutions")
        print("  Greedy: Sequential decision making")
        print("  D&C: Recursive problem decomposition")
        print()
        print("Greedy vs Brute Force:")
        print("  Greedy: Makes smart choices to avoid exploring all possibilities")
        print("  Brute Force: Explores all possible solutions")
        print("  Greedy: Much faster but may miss optimal solution")
        print("  Brute Force: Slower but guaranteed to find optimal")
        print()
        print("When to Use Greedy:")
        print("• Problem exhibits greedy choice property")
        print("• Optimal substructure exists")
        print("• Need fast solution and greedy gives optimal result")
        print("• Activity selection, scheduling problems")
        print("• Minimum spanning tree, shortest path")
        print("• Huffman coding, fractional knapsack")
    
    # ==========================================
    # 2. GREEDY ALGORITHM PROPERTIES
    # ==========================================
    
    def demonstrate_greedy_properties(self) -> None:
        """
        Demonstrate the key properties of greedy algorithms
        """
        print("=== GREEDY ALGORITHM PROPERTIES ===")
        print()
        
        # Property 1: Greedy Choice Property
        print("1. GREEDY CHOICE PROPERTY")
        print("   Definition: A global optimum can be reached by making locally optimal choices")
        print("   Example: Activity Selection Problem")
        print()
        
        self._demo_activity_selection_greedy_choice()
        
        print("\n" + "="*50 + "\n")
        
        # Property 2: Optimal Substructure
        print("2. OPTIMAL SUBSTRUCTURE PROPERTY")
        print("   Definition: Optimal solution contains optimal solutions to subproblems")
        print("   Example: Fractional Knapsack Problem")
        print()
        
        self._demo_fractional_knapsack_optimal_substructure()
    
    def _demo_activity_selection_greedy_choice(self) -> None:
        """Demonstrate greedy choice property with activity selection"""
        activities = [
            ("A1", 1, 4),   # (name, start, end)
            ("A2", 3, 5),
            ("A3", 0, 6),
            ("A4", 5, 7),
            ("A5", 8, 9),
            ("A6", 5, 9),
            ("A7", 6, 10),
            ("A8", 8, 11),
            ("A9", 8, 12),
            ("A10", 2, 13),
            ("A11", 12, 14)
        ]
        
        print("Activity Selection - Greedy Choice Property Demo")
        print("Activities (name, start_time, end_time):")
        for name, start, end in activities:
            print(f"   {name}: [{start}, {end}]")
        print()
        
        # Sort by end time (greedy choice: always pick activity that ends earliest)
        sorted_activities = sorted(activities, key=lambda x: x[2])
        
        print("Sorted by end time (greedy choice criterion):")
        for name, start, end in sorted_activities:
            print(f"   {name}: [{start}, {end}]")
        print()
        
        selected = []
        last_end_time = -1
        
        print("Greedy selection process:")
        for name, start, end in sorted_activities:
            print(f"Considering {name}: [{start}, {end}]")
            
            if start >= last_end_time:
                selected.append((name, start, end))
                last_end_time = end
                print(f"   ✓ Selected {name} (no conflict)")
                print(f"   Current selection: {[n for n, s, e in selected]}")
            else:
                print(f"   ✗ Rejected {name} (conflicts with last selected activity)")
            print()
        
        print(f"Final selection: {[name for name, start, end in selected]}")
        print(f"Total activities selected: {len(selected)}")
        print()
        print("Greedy Choice Property Verified:")
        print("• Always choosing the activity that ends earliest")
        print("• This local choice leads to globally optimal solution")
        print("• No need to reconsider previous choices")
    
    def _demo_fractional_knapsack_optimal_substructure(self) -> None:
        """Demonstrate optimal substructure with fractional knapsack"""
        items = [
            ("Item1", 60, 10),  # (name, value, weight)
            ("Item2", 100, 20),
            ("Item3", 120, 30)
        ]
        capacity = 50
        
        print("Fractional Knapsack - Optimal Substructure Demo")
        print(f"Knapsack capacity: {capacity}")
        print("Items (name, value, weight, value/weight ratio):")
        
        # Calculate value-to-weight ratio
        items_with_ratio = []
        for name, value, weight in items:
            ratio = value / weight
            items_with_ratio.append((name, value, weight, ratio))
            print(f"   {name}: value={value}, weight={weight}, ratio={ratio:.2f}")
        print()
        
        # Sort by value-to-weight ratio (greedy choice)
        items_with_ratio.sort(key=lambda x: x[3], reverse=True)
        
        print("Sorted by value/weight ratio (greedy choice):")
        for name, value, weight, ratio in items_with_ratio:
            print(f"   {name}: ratio={ratio:.2f}")
        print()
        
        total_value = 0
        remaining_capacity = capacity
        solution = []
        
        print("Greedy filling process:")
        for name, value, weight, ratio in items_with_ratio:
            print(f"Considering {name}: value={value}, weight={weight}")
            
            if weight <= remaining_capacity:
                # Take full item
                solution.append((name, 1.0, value))
                total_value += value
                remaining_capacity -= weight
                print(f"   ✓ Take 100% of {name}")
            else:
                # Take fractional item
                fraction = remaining_capacity / weight
                fractional_value = value * fraction
                solution.append((name, fraction, fractional_value))
                total_value += fractional_value
                remaining_capacity = 0
                print(f"   ✓ Take {fraction:.2%} of {name}")
            
            print(f"   Total value so far: {total_value:.2f}")
            print(f"   Remaining capacity: {remaining_capacity}")
            print()
            
            if remaining_capacity == 0:
                break
        
        print("Final solution:")
        for name, fraction, value in solution:
            print(f"   {name}: {fraction:.2%} taken, value = {value:.2f}")
        print(f"Total value: {total_value:.2f}")
        print()
        print("Optimal Substructure Verified:")
        print("• After choosing items greedily, remaining problem is still optimal")
        print("• Each choice creates a subproblem that is solved optimally")
    
    # ==========================================
    # 3. PROVING GREEDY ALGORITHMS
    # ==========================================
    
    def greedy_proof_techniques(self) -> None:
        """
        Explain techniques for proving greedy algorithms are correct
        """
        print("=== TECHNIQUES FOR PROVING GREEDY ALGORITHMS ===")
        print()
        
        print("1. EXCHANGE ARGUMENT (MOST COMMON)")
        print("   • Assume there's an optimal solution different from greedy")
        print("   • Show that you can 'exchange' choices to make it more like greedy")
        print("   • Prove the exchange doesn't worsen the solution")
        print("   • Conclude greedy is optimal")
        print()
        
        print("2. GREEDY STAYS AHEAD")
        print("   • Show that greedy algorithm's solution is always ahead")
        print("   • At each step, greedy is at least as good as any other algorithm")
        print("   • Prove by induction that greedy maintains advantage")
        print()
        
        print("3. STRUCTURAL INDUCTION")
        print("   • Prove optimal substructure exists")
        print("   • Show greedy choice leads to subproblem")
        print("   • Prove subproblem has same optimal structure")
        print("   • Combine to show overall optimality")
        print()
        
        self._demo_exchange_argument()
    
    def _demo_exchange_argument(self) -> None:
        """Demonstrate exchange argument proof technique"""
        print("EXCHANGE ARGUMENT EXAMPLE: Activity Selection")
        print()
        print("Theorem: Selecting activity with earliest end time is optimal")
        print()
        print("Proof by Exchange Argument:")
        print("1. Let A = {a1, a2, ..., ak} be optimal solution")
        print("2. Let g be the activity with earliest end time")
        print("3. If g ∈ A, we're done")
        print("4. If g ∉ A, let a1 be the first activity in A")
        print("5. Since g has earliest end time: end(g) ≤ end(a1)")
        print("6. Replace a1 with g in A to get A' = {g, a2, ..., ak}")
        print("7. A' is still feasible because:")
        print("   • g ends before a1, so g doesn't conflict with a2")
        print("   • |A'| = |A|, so A' is also optimal")
        print("8. This shows greedy choice is safe")
        print("9. Apply recursively to remaining activities")
        print("10. Therefore, greedy algorithm is optimal")
        print()
        print("Key Insight: We can always exchange non-greedy choice")
        print("with greedy choice without losing optimality")
    
    # ==========================================
    # 4. GREEDY ALGORITHM DESIGN PROCESS
    # ==========================================
    
    def greedy_design_process(self) -> None:
        """
        Demonstrate the systematic process for designing greedy algorithms
        """
        print("=== GREEDY ALGORITHM DESIGN PROCESS ===")
        print()
        
        print("STEP 1: IDENTIFY THE GREEDY CHOICE")
        print("• What locally optimal choice can we make?")
        print("• What criteria should we use for selection?")
        print("• Examples:")
        print("  - Activity selection: Choose earliest ending activity")
        print("  - Fractional knapsack: Choose highest value/weight ratio")
        print("  - Huffman coding: Choose two least frequent symbols")
        print()
        
        print("STEP 2: PROVE GREEDY CHOICE PROPERTY")
        print("• Show that greedy choice leads to optimal solution")
        print("• Use exchange argument or other proof techniques")
        print("• Ensure the choice is always safe to make")
        print()
        
        print("STEP 3: SHOW OPTIMAL SUBSTRUCTURE")
        print("• After making greedy choice, remaining problem should be similar")
        print("• Optimal solution to original contains optimal solution to subproblem")
        print("• This allows recursive or iterative solution")
        print()
        
        print("STEP 4: IMPLEMENT THE ALGORITHM")
        print("• Usually involves sorting based on greedy criteria")
        print("• Iterate through sorted elements")
        print("• Make greedy choice at each step")
        print("• Accumulate the solution")
        print()
        
        self._demo_design_process_example()
    
    def _demo_design_process_example(self) -> None:
        """Demonstrate design process with interval scheduling example"""
        print("DESIGN PROCESS EXAMPLE: Interval Scheduling")
        print()
        
        print("Problem: Schedule maximum number of non-overlapping intervals")
        intervals = [(1, 3), (2, 4), (3, 6), (5, 7), (8, 9)]
        
        print(f"Input intervals: {intervals}")
        print()
        
        print("STEP 1: Identify Greedy Choice")
        print("• Consider different strategies:")
        print("  - Choose shortest interval? No, suboptimal")
        print("  - Choose earliest start? No, suboptimal")
        print("  - Choose earliest end? YES, this works!")
        print("• Greedy choice: Always pick interval that ends earliest")
        print()
        
        print("STEP 2: Prove Greedy Choice (Exchange Argument)")
        print("• Earliest-ending choice is always safe")
        print("• Can replace any other choice with earliest-ending")
        print("• This replacement maintains feasibility and optimality")
        print()
        
        print("STEP 3: Show Optimal Substructure")
        print("• After choosing interval [a, b], remove all conflicting intervals")
        print("• Remaining problem is identical to original problem")
        print("• Optimal solution to subproblem + greedy choice = optimal overall")
        print()
        
        print("STEP 4: Implementation")
        # Sort by end time
        sorted_intervals = sorted(intervals, key=lambda x: x[1])
        
        print("a) Sort by end time:")
        print(f"   Sorted intervals: {sorted_intervals}")
        
        print("b) Greedy selection:")
        selected = []
        last_end = -1
        
        for start, end in sorted_intervals:
            print(f"   Consider interval ({start}, {end})")
            if start >= last_end:
                selected.append((start, end))
                last_end = end
                print(f"     ✓ Selected (no conflict)")
            else:
                print(f"     ✗ Rejected (conflicts with last selected)")
        
        print(f"c) Final result: {selected}")
        print(f"   Maximum non-overlapping intervals: {len(selected)}")
    
    # ==========================================
    # 5. COMMON GREEDY PATTERNS
    # ==========================================
    
    def common_greedy_patterns(self) -> None:
        """
        Identify and explain common patterns in greedy algorithms
        """
        print("=== COMMON GREEDY ALGORITHM PATTERNS ===")
        print()
        
        print("PATTERN 1: EARLIEST DEADLINE FIRST")
        print("• Sort by deadline/end time")
        print("• Process items in order of deadline")
        print("• Examples: Activity selection, interval scheduling")
        print("• Template:")
        print("  1. Sort by end time")
        print("  2. Select if no conflict with previous selection")
        print()
        
        print("PATTERN 2: HIGHEST RATIO FIRST")
        print("• Calculate benefit/cost ratio for each item")
        print("• Sort by ratio in descending order")
        print("• Examples: Fractional knapsack, job scheduling")
        print("• Template:")
        print("  1. Calculate value/weight or profit/time ratio")
        print("  2. Sort by ratio descending")
        print("  3. Select greedily")
        print()
        
        print("PATTERN 3: SMALLEST/LARGEST FIRST")
        print("• Sort by size/magnitude")
        print("• Choose smallest or largest based on problem")
        print("• Examples: Huffman coding, minimum spanning tree")
        print("• Template:")
        print("  1. Sort by appropriate criteria")
        print("  2. Always pick smallest/largest available")
        print()
        
        print("PATTERN 4: EXCHANGE/SWAP OPTIMIZATION")
        print("• Look for beneficial swaps in current solution")
        print("• Make swaps that improve objective function")
        print("• Examples: 2-opt for TSP, local search")
        print("• Template:")
        print("  1. Start with initial solution")
        print("  2. Find beneficial swaps")
        print("  3. Make swaps until no improvement possible")
        print()
        
        self._demo_greedy_patterns()
    
    def _demo_greedy_patterns(self) -> None:
        """Demonstrate different greedy patterns with examples"""
        print("PATTERN DEMONSTRATION:")
        print()
        
        # Pattern 1: Earliest Deadline First
        print("1. EARLIEST DEADLINE FIRST - Meeting Scheduling")
        meetings = [("M1", 9, 11), ("M2", 10, 12), ("M3", 11, 13)]
        sorted_meetings = sorted(meetings, key=lambda x: x[2])
        
        print(f"   Meetings: {meetings}")
        print(f"   Sorted by end time: {sorted_meetings}")
        print("   Select: M1 (ends earliest), then M3 (no conflict)")
        print()
        
        # Pattern 2: Highest Ratio First
        print("2. HIGHEST RATIO FIRST - Item Selection")
        items = [("A", 60, 10), ("B", 100, 20), ("C", 120, 30)]
        items_with_ratio = [(name, val, wt, val/wt) for name, val, wt in items]
        items_with_ratio.sort(key=lambda x: x[3], reverse=True)
        
        print(f"   Items (name, value, weight): {items}")
        print("   With ratios:", [(name, f"{ratio:.2f}") for name, val, wt, ratio in items_with_ratio])
        print("   Order: A (6.0), B (5.0), C (4.0)")
        print()
        
        # Pattern 3: Smallest First
        print("3. SMALLEST FIRST - Optimal Merge Pattern")
        files = [20, 30, 10, 40]
        print(f"   File sizes: {files}")
        print("   Strategy: Always merge two smallest files")
        print("   Merge order: (10,20)→30, (30,30)→60, (40,60)→100")
        print("   Total cost: 30 + 60 + 100 = 190")
    
    # ==========================================
    # 6. WHEN GREEDY FAILS
    # ==========================================
    
    def when_greedy_fails(self) -> None:
        """
        Demonstrate cases where greedy algorithms fail
        """
        print("=== WHEN GREEDY ALGORITHMS FAIL ===")
        print()
        
        print("Greedy algorithms don't always give optimal solutions!")
        print("Here are common scenarios where greedy fails:")
        print()
        
        # Example 1: 0/1 Knapsack
        print("1. 0/1 KNAPSACK PROBLEM")
        print("   Items: (value, weight)")
        print("   Item 1: (60, 10) - ratio 6.0")
        print("   Item 2: (100, 20) - ratio 5.0") 
        print("   Item 3: (120, 30) - ratio 4.0")
        print("   Capacity: 50")
        print()
        print("   Greedy (by ratio): Take items 1 and 2")
        print("   Value: 60 + 100 = 160")
        print("   Weight: 10 + 20 = 30")
        print()
        print("   Optimal: Take items 2 and 3")
        print("   Value: 100 + 120 = 220")
        print("   Weight: 20 + 30 = 50")
        print()
        print("   Why greedy fails: Can't take fractions")
        print()
        
        # Example 2: Coin Change
        print("2. COIN CHANGE PROBLEM")
        print("   Coins: [1, 3, 4]")
        print("   Amount: 6")
        print()
        print("   Greedy: 4 + 1 + 1 = 3 coins")
        print("   Optimal: 3 + 3 = 2 coins")
        print()
        print("   Why greedy fails: Largest coin first isn't always best")
        print()
        
        # Example 3: Graph Coloring
        print("3. GRAPH COLORING")
        print("   Greedy: Color vertices in arbitrary order")
        print("   May use more colors than necessary")
        print("   Optimal requires considering global structure")
        print()
        
        print("SIGNS THAT GREEDY MIGHT FAIL:")
        print("• Problem asks for exact optimization (like 0/1 knapsack)")
        print("• Local choices can block better global choices")
        print("• Problem has complex dependencies")
        print("• Multiple constraints must be satisfied simultaneously")
        print("• Solution space has multiple local optima")
        print()
        
        print("ALTERNATIVES WHEN GREEDY FAILS:")
        print("• Dynamic Programming: For optimal substructure problems")
        print("• Branch and Bound: For exact solutions to complex problems")
        print("• Approximation Algorithms: For NP-hard problems")
        print("• Heuristics: For practical solutions to hard problems")


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_greedy_fundamentals():
    """Demonstrate all greedy fundamental concepts"""
    print("=== GREEDY ALGORITHM FUNDAMENTALS DEMONSTRATION ===\n")
    
    fundamentals = GreedyFundamentals()
    
    # 1. Core concept explanation
    fundamentals.explain_greedy_concept()
    print("\n" + "="*60 + "\n")
    
    # 2. Comparison with other paradigms
    fundamentals.greedy_vs_other_paradigms()
    print("\n" + "="*60 + "\n")
    
    # 3. Greedy algorithm properties
    fundamentals.demonstrate_greedy_properties()
    print("\n" + "="*60 + "\n")
    
    # 4. Proof techniques
    fundamentals.greedy_proof_techniques()
    print("\n" + "="*60 + "\n")
    
    # 5. Design process
    fundamentals.greedy_design_process()
    print("\n" + "="*60 + "\n")
    
    # 6. Common patterns
    fundamentals.common_greedy_patterns()
    print("\n" + "="*60 + "\n")
    
    # 7. When greedy fails
    fundamentals.when_greedy_fails()


if __name__ == "__main__":
    demonstrate_greedy_fundamentals()
    
    print("\n" + "="*60)
    print("=== GREEDY ALGORITHM MASTERY GUIDE ===")
    print("="*60)
    
    print("\n🎯 WHEN TO USE GREEDY ALGORITHMS:")
    print("✅ Problem has greedy choice property")
    print("✅ Optimal substructure exists")
    print("✅ Need fast solution (O(n log n) typically)")
    print("✅ Activity/interval scheduling problems")
    print("✅ Minimum spanning tree algorithms")
    print("✅ Shortest path algorithms (Dijkstra)")
    print("✅ Huffman coding and compression")
    print("✅ Fractional optimization problems")
    
    print("\n📋 GREEDY ALGORITHM CHECKLIST:")
    print("1. Identify the greedy choice criterion")
    print("2. Prove the greedy choice property")
    print("3. Verify optimal substructure")
    print("4. Design the algorithm (usually with sorting)")
    print("5. Analyze time and space complexity")
    print("6. Test with counterexamples")
    
    print("\n⚡ OPTIMIZATION STRATEGIES:")
    print("• Choose the right sorting criteria")
    print("• Use efficient data structures for selection")
    print("• Consider preprocessing to speed up greedy choices")
    print("• Combine with other techniques when pure greedy fails")
    print("• Use approximation when exact greedy doesn't work")
    
    print("\n🚨 COMMON PITFALLS:")
    print("• Assuming greedy works without proof")
    print("• Using wrong greedy criteria")
    print("• Ignoring problem constraints")
    print("• Not considering all possible greedy strategies")
    print("• Confusing with dynamic programming problems")
    
    print("\n🎓 LEARNING PROGRESSION:")
    print("1. Master the fundamental concepts and properties")
    print("2. Learn to identify greedy choice opportunities")
    print("3. Practice proving greedy algorithms correct")
    print("4. Solve classic greedy problems")
    print("5. Study advanced applications and variations")
    
    print("\n📚 PROBLEM CATEGORIES TO PRACTICE:")
    print("• Activity and interval scheduling")
    print("• Fractional optimization (knapsack, etc.)")
    print("• Graph algorithms (MST, shortest path)")
    print("• String and array manipulation")
    print("• Mathematical optimization")
    print("• Approximation algorithms")
    
    print("\n🔍 PROOF TECHNIQUES TO MASTER:")
    print("• Exchange argument (most important)")
    print("• Greedy stays ahead")
    print("• Structural induction")
    print("• Cut property (for graph algorithms)")
    print("• Matroid theory (advanced)")
    
    print("\n💡 SUCCESS TIPS:")
    print("• Always try to prove your greedy algorithm is correct")
    print("• Think about why other greedy choices would be worse")
    print("• Use counterexamples to test your approach")
    print("• Consider multiple greedy strategies for each problem")
    print("• Understand when to use greedy vs DP vs other approaches")
