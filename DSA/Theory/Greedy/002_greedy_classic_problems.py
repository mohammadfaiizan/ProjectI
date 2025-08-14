"""
Classic Greedy Problems - Fundamental Examples
==============================================

Topics: Activity selection, fractional knapsack, coin change, job scheduling
Companies: Google, Amazon, Microsoft, Facebook, Apple
Difficulty: Easy to Medium
Time Complexity: O(n log n) for sorting, O(n) for selection
Space Complexity: O(1) to O(n) depending on problem
"""

from typing import List, Tuple, Optional, Dict, Any
from collections import Counter
import heapq

class ClassicGreedyProblems:
    
    def __init__(self):
        """Initialize with problem tracking"""
        self.solution_steps = []
        self.problem_count = 0
    
    # ==========================================
    # 1. ACTIVITY SELECTION PROBLEM
    # ==========================================
    
    def activity_selection(self, activities: List[Tuple[str, int, int]]) -> List[str]:
        """
        Activity Selection Problem - The Classic Greedy Example
        
        Company: Google, Amazon, Microsoft (fundamental greedy problem)
        Difficulty: Medium
        Time: O(n log n), Space: O(1)
        
        Problem: Select maximum number of non-overlapping activities
        Greedy Strategy: Always choose activity that ends earliest
        
        Args:
            activities: List of (name, start_time, end_time)
        
        Returns:
            List of selected activity names
        """
        print("=== ACTIVITY SELECTION PROBLEM ===")
        print("Problem: Select maximum number of non-overlapping activities")
        print("Greedy Strategy: Always choose activity that ends earliest")
        print()
        
        print("Input activities:")
        for name, start, end in activities:
            print(f"   {name}: [{start}, {end}]")
        print()
        
        # Step 1: Sort by end time (greedy choice criterion)
        sorted_activities = sorted(activities, key=lambda x: x[2])
        
        print("Step 1: Sort by end time (greedy criterion)")
        for name, start, end in sorted_activities:
            print(f"   {name}: [{start}, {end}]")
        print()
        
        # Step 2: Greedy selection
        selected = []
        last_end_time = -1
        
        print("Step 2: Greedy selection process")
        for i, (name, start, end) in enumerate(sorted_activities):
            print(f"Iteration {i+1}: Consider {name} [{start}, {end}]")
            
            if start >= last_end_time:
                selected.append(name)
                last_end_time = end
                print(f"   ✓ Selected {name} (start {start} >= last_end {last_end_time if last_end_time != end else 'none'})")
                print(f"   Current selection: {selected}")
            else:
                print(f"   ✗ Rejected {name} (start {start} < last_end {last_end_time})")
            
            print(f"   Last end time: {last_end_time}")
            print()
        
        print(f"Final result: {len(selected)} activities selected")
        print(f"Selected activities: {selected}")
        
        # Verification
        print("\nVerification (check for overlaps):")
        activity_times = {name: (start, end) for name, start, end in activities}
        for i in range(len(selected) - 1):
            curr_name = selected[i]
            next_name = selected[i + 1]
            curr_end = activity_times[curr_name][1]
            next_start = activity_times[next_name][0]
            
            if curr_end <= next_start:
                print(f"   ✓ {curr_name} ends at {curr_end}, {next_name} starts at {next_start}")
            else:
                print(f"   ✗ Overlap detected!")
        
        return selected
    
    def prove_activity_selection_optimality(self) -> None:
        """
        Prove that activity selection greedy algorithm is optimal
        """
        print("=== PROOF OF OPTIMALITY: Activity Selection ===")
        print()
        
        print("THEOREM: The greedy algorithm that selects activities by earliest")
        print("end time produces an optimal solution.")
        print()
        
        print("PROOF BY EXCHANGE ARGUMENT:")
        print()
        print("1. Let S = {a1, a2, ..., ak} be an optimal solution")
        print("   where activities are sorted by end time.")
        print()
        print("2. Let G = {g1, g2, ..., gm} be the greedy solution")
        print("   where g1 is the activity with earliest end time.")
        print()
        print("3. Claim: We can transform S into a solution that starts with g1")
        print("   without decreasing the number of activities.")
        print()
        print("4. Case 1: If a1 = g1, then S already starts with greedy choice.")
        print()
        print("5. Case 2: If a1 ≠ g1, then end(g1) ≤ end(a1)")
        print("   (since g1 has earliest end time)")
        print()
        print("6. Replace a1 with g1 in S to get S' = {g1, a2, ..., ak}")
        print()
        print("7. S' is feasible because:")
        print("   • g1 doesn't overlap with any ai (i ≥ 2)")
        print("   • Since end(g1) ≤ end(a1) and a1 didn't overlap with a2,")
        print("     g1 won't overlap with a2 either")
        print()
        print("8. |S'| = |S|, so S' is also optimal")
        print()
        print("9. By induction, we can continue this process for the")
        print("   remaining activities, showing greedy is optimal.")
        print()
        print("QED: The greedy choice is always safe to make.")
    
    # ==========================================
    # 2. FRACTIONAL KNAPSACK PROBLEM
    # ==========================================
    
    def fractional_knapsack(self, items: List[Tuple[str, float, float]], capacity: float) -> Tuple[List[Tuple[str, float, float]], float]:
        """
        Fractional Knapsack Problem
        
        Company: Amazon, Google, Facebook
        Difficulty: Medium
        Time: O(n log n), Space: O(n)
        
        Problem: Fill knapsack to maximize value (can take fractions)
        Greedy Strategy: Always choose item with highest value/weight ratio
        
        Args:
            items: List of (name, value, weight)
            capacity: Knapsack capacity
        
        Returns:
            (selected_items_with_fractions, total_value)
        """
        print("=== FRACTIONAL KNAPSACK PROBLEM ===")
        print("Problem: Maximize value in knapsack (fractions allowed)")
        print("Greedy Strategy: Choose highest value/weight ratio first")
        print()
        
        print(f"Knapsack capacity: {capacity}")
        print("Items (name, value, weight, ratio):")
        
        # Calculate value-to-weight ratios
        items_with_ratio = []
        for name, value, weight in items:
            ratio = value / weight if weight > 0 else float('inf')
            items_with_ratio.append((name, value, weight, ratio))
            print(f"   {name}: value={value}, weight={weight}, ratio={ratio:.3f}")
        print()
        
        # Sort by ratio in descending order
        items_with_ratio.sort(key=lambda x: x[3], reverse=True)
        
        print("Sorted by value/weight ratio (greedy order):")
        for name, value, weight, ratio in items_with_ratio:
            print(f"   {name}: ratio={ratio:.3f}")
        print()
        
        # Greedy selection
        selected = []
        total_value = 0.0
        remaining_capacity = capacity
        
        print("Greedy selection process:")
        for i, (name, value, weight, ratio) in enumerate(items_with_ratio):
            print(f"Step {i+1}: Consider {name}")
            print(f"   Available: value={value}, weight={weight}")
            print(f"   Remaining capacity: {remaining_capacity}")
            
            if weight <= remaining_capacity:
                # Take entire item
                selected.append((name, 1.0, value))
                total_value += value
                remaining_capacity -= weight
                print(f"   ✓ Take 100% of {name}")
                print(f"   Added value: {value}")
            elif remaining_capacity > 0:
                # Take fractional item
                fraction = remaining_capacity / weight
                fractional_value = value * fraction
                selected.append((name, fraction, fractional_value))
                total_value += fractional_value
                remaining_capacity = 0
                print(f"   ✓ Take {fraction:.1%} of {name}")
                print(f"   Added value: {fractional_value:.2f}")
            else:
                print(f"   ✗ No remaining capacity for {name}")
            
            print(f"   Total value so far: {total_value:.2f}")
            print(f"   Remaining capacity: {remaining_capacity}")
            print()
            
            if remaining_capacity == 0:
                print("Knapsack is full!")
                break
        
        print("Final solution:")
        total_weight = 0
        for name, fraction, value in selected:
            # Find original weight
            original_weight = next(w for n, v, w in items if n == name)
            used_weight = original_weight * fraction
            total_weight += used_weight
            print(f"   {name}: {fraction:.1%} taken, value={value:.2f}, weight={used_weight:.2f}")
        
        print(f"\nSummary:")
        print(f"   Total value: {total_value:.2f}")
        print(f"   Total weight: {total_weight:.2f}")
        print(f"   Capacity utilization: {(total_weight/capacity):.1%}")
        
        return selected, total_value
    
    # ==========================================
    # 3. COIN CHANGE GREEDY (CANONICAL SYSTEMS)
    # ==========================================
    
    def coin_change_greedy(self, coins: List[int], amount: int) -> Tuple[List[int], int]:
        """
        Coin Change using Greedy (works for canonical coin systems)
        
        Company: Amazon, Microsoft
        Difficulty: Easy to Medium
        Time: O(n log n + amount/largest_coin), Space: O(1)
        
        Problem: Make change using minimum number of coins
        Greedy Strategy: Always use largest possible coin
        
        Note: Only works for canonical coin systems (like standard currency)
        """
        print("=== COIN CHANGE PROBLEM (GREEDY) ===")
        print("Problem: Make change using minimum number of coins")
        print("Greedy Strategy: Always use largest possible coin")
        print()
        
        print(f"Available coins: {coins}")
        print(f"Amount to make: {amount}")
        print()
        
        # Sort coins in descending order
        sorted_coins = sorted(coins, reverse=True)
        
        print(f"Coins sorted (largest first): {sorted_coins}")
        print()
        
        result = []
        remaining = amount
        total_coins = 0
        
        print("Greedy selection process:")
        for i, coin in enumerate(sorted_coins):
            if remaining == 0:
                break
            
            count = remaining // coin
            
            print(f"Step {i+1}: Using coin {coin}")
            print(f"   Remaining amount: {remaining}")
            print(f"   Can use {count} coins of value {coin}")
            
            if count > 0:
                result.extend([coin] * count)
                remaining -= coin * count
                total_coins += count
                print(f"   ✓ Used {count} × {coin} = {coin * count}")
                print(f"   Coins used so far: {result}")
            else:
                print(f"   ✗ Cannot use coin {coin} (too large)")
            
            print(f"   Remaining amount: {remaining}")
            print()
        
        if remaining > 0:
            print(f"Cannot make exact change! Remaining: {remaining}")
            return [], -1
        
        print("Final result:")
        coin_count = Counter(result)
        for coin in sorted(coin_count.keys(), reverse=True):
            count = coin_count[coin]
            print(f"   {count} × {coin} = {count * coin}")
        
        print(f"Total coins used: {total_coins}")
        print(f"Verification: {sum(result)} = {amount} ✓")
        
        return result, total_coins
    
    def coin_change_greedy_fails(self) -> None:
        """
        Demonstrate when greedy coin change fails
        """
        print("=== WHEN GREEDY COIN CHANGE FAILS ===")
        print()
        
        # Example where greedy fails
        coins = [1, 3, 4]
        amount = 6
        
        print(f"Coins: {coins}")
        print(f"Amount: {amount}")
        print()
        
        print("Greedy approach:")
        print("1. Use largest coin 4: remaining = 6 - 4 = 2")
        print("2. Cannot use 3 (too large)")
        print("3. Use coin 1: remaining = 2 - 1 = 1")
        print("4. Use coin 1: remaining = 1 - 1 = 0")
        print("Greedy result: [4, 1, 1] = 3 coins")
        print()
        
        print("Optimal approach:")
        print("1. Use coin 3: remaining = 6 - 3 = 3")
        print("2. Use coin 3: remaining = 3 - 3 = 0")
        print("Optimal result: [3, 3] = 2 coins")
        print()
        
        print("Conclusion: Greedy doesn't always work for coin change!")
        print("It only works for canonical coin systems like [1, 5, 10, 25]")
    
    # ==========================================
    # 4. JOB SCHEDULING PROBLEMS
    # ==========================================
    
    def job_scheduling_deadline(self, jobs: List[Tuple[str, int, int]]) -> Tuple[List[str], int]:
        """
        Job Scheduling with Deadlines and Penalties
        
        Company: Google, Amazon, Microsoft
        Difficulty: Medium to Hard
        Time: O(n²), Space: O(n)
        
        Problem: Schedule jobs to minimize total penalty
        Greedy Strategy: Schedule jobs by profit/penalty ratio
        
        Args:
            jobs: List of (job_name, deadline, penalty)
        
        Returns:
            (scheduled_jobs, total_penalty)
        """
        print("=== JOB SCHEDULING WITH DEADLINES ===")
        print("Problem: Schedule jobs to minimize penalty for late completion")
        print("Greedy Strategy: Sort by penalty (highest first)")
        print()
        
        print("Jobs (name, deadline, penalty):")
        for name, deadline, penalty in jobs:
            print(f"   {name}: deadline={deadline}, penalty={penalty}")
        print()
        
        # Sort by penalty in descending order
        sorted_jobs = sorted(jobs, key=lambda x: x[2], reverse=True)
        
        print("Jobs sorted by penalty (highest first):")
        for name, deadline, penalty in sorted_jobs:
            print(f"   {name}: deadline={deadline}, penalty={penalty}")
        print()
        
        # Find maximum deadline to create time slots
        max_deadline = max(deadline for _, deadline, _ in jobs)
        
        # Initialize schedule (None means empty slot)
        schedule = [None] * max_deadline
        scheduled_jobs = []
        total_penalty = 0
        
        print(f"Available time slots: {max_deadline} (from 1 to {max_deadline})")
        print()
        
        print("Greedy scheduling process:")
        for i, (name, deadline, penalty) in enumerate(sorted_jobs):
            print(f"Step {i+1}: Schedule job {name}")
            print(f"   Deadline: {deadline}, Penalty: {penalty}")
            
            # Try to schedule in latest possible slot before deadline
            scheduled = False
            for slot in range(min(deadline, max_deadline) - 1, -1, -1):
                if schedule[slot] is None:
                    schedule[slot] = name
                    scheduled_jobs.append(name)
                    scheduled = True
                    print(f"   ✓ Scheduled {name} in slot {slot + 1}")
                    break
            
            if not scheduled:
                total_penalty += penalty
                print(f"   ✗ Cannot schedule {name} before deadline")
                print(f"   Added penalty: {penalty}")
            
            print(f"   Current schedule: {schedule}")
            print(f"   Total penalty so far: {total_penalty}")
            print()
        
        print("Final schedule:")
        for i, job in enumerate(schedule):
            if job:
                print(f"   Slot {i+1}: {job}")
            else:
                print(f"   Slot {i+1}: (empty)")
        
        print(f"\nSummary:")
        print(f"   Scheduled jobs: {scheduled_jobs}")
        print(f"   Total penalty: {total_penalty}")
        
        return scheduled_jobs, total_penalty
    
    def job_scheduling_shortest_processing_time(self, jobs: List[Tuple[str, int]]) -> Tuple[List[str], float]:
        """
        Job Scheduling - Shortest Processing Time First
        
        Company: Operating Systems, Google
        Difficulty: Easy to Medium  
        Time: O(n log n), Space: O(1)
        
        Problem: Minimize average completion time
        Greedy Strategy: Process shortest jobs first
        """
        print("=== SHORTEST PROCESSING TIME FIRST ===")
        print("Problem: Minimize average completion time")
        print("Greedy Strategy: Process shortest jobs first")
        print()
        
        print("Jobs (name, processing_time):")
        for name, time in jobs:
            print(f"   {name}: {time} units")
        print()
        
        # Sort by processing time
        sorted_jobs = sorted(jobs, key=lambda x: x[1])
        
        print("Jobs sorted by processing time:")
        for name, time in sorted_jobs:
            print(f"   {name}: {time} units")
        print()
        
        # Calculate completion times
        current_time = 0
        completion_times = []
        schedule = []
        
        print("Execution schedule:")
        for i, (name, time) in enumerate(sorted_jobs):
            start_time = current_time
            current_time += time
            completion_times.append(current_time)
            schedule.append(name)
            
            print(f"Job {i+1}: {name}")
            print(f"   Start time: {start_time}")
            print(f"   Processing time: {time}")
            print(f"   Completion time: {current_time}")
            print()
        
        avg_completion_time = sum(completion_times) / len(completion_times)
        
        print("Summary:")
        print(f"   Execution order: {schedule}")
        print(f"   Completion times: {completion_times}")
        print(f"   Average completion time: {avg_completion_time:.2f}")
        
        return schedule, avg_completion_time
    
    # ==========================================
    # 5. HUFFMAN CODING
    # ==========================================
    
    def huffman_coding_basic(self, text: str) -> Tuple[Dict[str, str], str]:
        """
        Huffman Coding - Basic Implementation
        
        Company: Google, Amazon, Compression algorithms
        Difficulty: Medium to Hard
        Time: O(n log n), Space: O(n)
        
        Problem: Create optimal prefix-free code
        Greedy Strategy: Merge two least frequent nodes
        """
        print("=== HUFFMAN CODING ===")
        print("Problem: Create optimal prefix-free code for compression")
        print("Greedy Strategy: Always merge two least frequent symbols")
        print()
        
        # Count character frequencies
        from collections import Counter
        freq_count = Counter(text)
        
        print(f"Input text: '{text}'")
        print(f"Character frequencies:")
        for char, freq in sorted(freq_count.items()):
            print(f"   '{char}': {freq}")
        print()
        
        # Create priority queue with frequencies
        import heapq
        
        # Format: (frequency, unique_id, node_data)
        heap = []
        node_id = 0
        
        for char, freq in freq_count.items():
            heapq.heappush(heap, (freq, node_id, {'char': char, 'left': None, 'right': None}))
            node_id += 1
        
        print("Building Huffman tree:")
        step = 1
        
        # Build tree by merging nodes
        while len(heap) > 1:
            # Get two nodes with smallest frequencies
            freq1, id1, node1 = heapq.heappop(heap)
            freq2, id2, node2 = heapq.heappop(heap)
            
            print(f"Step {step}: Merge frequencies {freq1} and {freq2}")
            
            # Create new internal node
            merged_freq = freq1 + freq2
            merged_node = {
                'char': None,
                'left': node1,
                'right': node2
            }
            
            heapq.heappush(heap, (merged_freq, node_id, merged_node))
            print(f"   Created internal node with frequency {merged_freq}")
            
            node_id += 1
            step += 1
        
        # Generate codes from tree
        if heap:
            _, _, root = heap[0]
        else:
            # Single character case
            root = {'char': text[0], 'left': None, 'right': None}
        
        def generate_codes(node, code="", codes={}):
            if node['char'] is not None:
                # Leaf node
                codes[node['char']] = code if code else "0"
            else:
                # Internal node
                if node['left']:
                    generate_codes(node['left'], code + "0", codes)
                if node['right']:
                    generate_codes(node['right'], code + "1", codes)
            return codes
        
        codes = generate_codes(root)
        
        print("\nGenerated Huffman codes:")
        for char in sorted(codes.keys()):
            print(f"   '{char}': {codes[char]}")
        
        # Encode the text
        encoded = ''.join(codes[char] for char in text)
        
        print(f"\nOriginal text: '{text}'")
        print(f"Encoded text: '{encoded}'")
        print(f"Original bits (8 per char): {len(text) * 8}")
        print(f"Encoded bits: {len(encoded)}")
        print(f"Compression ratio: {len(encoded) / (len(text) * 8):.2%}")
        
        return codes, encoded


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_classic_greedy_problems():
    """Demonstrate all classic greedy problems"""
    print("=== CLASSIC GREEDY PROBLEMS DEMONSTRATION ===\n")
    
    problems = ClassicGreedyProblems()
    
    # 1. Activity Selection
    print("1. ACTIVITY SELECTION PROBLEM")
    activities = [
        ("A1", 1, 4),
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
    problems.activity_selection(activities)
    print("\n" + "-"*60 + "\n")
    
    problems.prove_activity_selection_optimality()
    print("\n" + "="*60 + "\n")
    
    # 2. Fractional Knapsack
    print("2. FRACTIONAL KNAPSACK PROBLEM")
    items = [
        ("Gold", 100, 10),
        ("Silver", 60, 20),
        ("Bronze", 40, 30)
    ]
    problems.fractional_knapsack(items, 50)
    print("\n" + "="*60 + "\n")
    
    # 3. Coin Change
    print("3. COIN CHANGE PROBLEM")
    problems.coin_change_greedy([1, 5, 10, 25], 67)
    print("\n" + "-"*40 + "\n")
    
    problems.coin_change_greedy_fails()
    print("\n" + "="*60 + "\n")
    
    # 4. Job Scheduling
    print("4. JOB SCHEDULING PROBLEMS")
    
    print("a) Job Scheduling with Deadlines:")
    jobs_deadline = [
        ("J1", 2, 60),
        ("J2", 1, 50),
        ("J3", 3, 20),
        ("J4", 2, 30)
    ]
    problems.job_scheduling_deadline(jobs_deadline)
    print("\n" + "-"*40 + "\n")
    
    print("b) Shortest Processing Time First:")
    jobs_spt = [
        ("Task1", 6),
        ("Task2", 2),
        ("Task3", 8),
        ("Task4", 3),
        ("Task5", 4)
    ]
    problems.job_scheduling_shortest_processing_time(jobs_spt)
    print("\n" + "="*60 + "\n")
    
    # 5. Huffman Coding
    print("5. HUFFMAN CODING")
    problems.huffman_coding_basic("ABRACADABRA")


if __name__ == "__main__":
    demonstrate_classic_greedy_problems()
    
    print("\n=== CLASSIC GREEDY PROBLEMS MASTERY GUIDE ===")
    
    print("\n🎯 PROBLEM PATTERNS:")
    print("• Activity Selection: Earliest deadline first")
    print("• Fractional Knapsack: Highest value/weight ratio first")
    print("• Coin Change: Largest denomination first (canonical systems)")
    print("• Job Scheduling: Various criteria (deadline, penalty, duration)")
    print("• Huffman Coding: Merge least frequent symbols")
    
    print("\n📊 COMPLEXITY ANALYSIS:")
    print("• Activity Selection: O(n log n) for sorting + O(n) for selection")
    print("• Fractional Knapsack: O(n log n) for sorting + O(n) for packing")
    print("• Coin Change: O(n log n) for sorting + O(amount/largest_coin)")
    print("• Job Scheduling: O(n log n) to O(n²) depending on variant")
    print("• Huffman Coding: O(n log n) for heap operations")
    
    print("\n⚡ OPTIMIZATION STRATEGIES:")
    print("• Pre-sort data according to greedy criterion")
    print("• Use appropriate data structures (heaps, priority queues)")
    print("• Consider preprocessing to identify greedy choices")
    print("• Implement efficient selection and update operations")
    
    print("\n🔧 IMPLEMENTATION TIPS:")
    print("• Always sort first according to greedy criterion")
    print("• Use clear variable names for greedy choices")
    print("• Add verification steps to check solution validity")
    print("• Handle edge cases (empty input, single item)")
    print("• Consider tie-breaking rules for equal priorities")
    
    print("\n🏆 REAL-WORLD APPLICATIONS:")
    print("• Operating Systems: CPU scheduling, memory allocation")
    print("• Network Systems: Bandwidth allocation, routing")
    print("• Manufacturing: Production scheduling, resource planning")
    print("• Finance: Portfolio optimization, task prioritization")
    print("• Compression: Data encoding, file compression")
    
    print("\n🎓 LEARNING OBJECTIVES:")
    print("• Master the fundamental greedy patterns")
    print("• Understand when greedy works vs when it fails")
    print("• Learn to prove greedy algorithm correctness")
    print("• Practice identifying greedy choice criteria")
    print("• Study variations and extensions of classic problems")
