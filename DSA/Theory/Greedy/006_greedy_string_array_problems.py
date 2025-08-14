"""
Greedy String and Array Problems
================================

Topics: String manipulation, array optimization, pattern matching
Companies: Google, Amazon, Microsoft, Facebook, Apple, Netflix
Difficulty: Easy to Hard
Time Complexity: O(n) to O(n log n) depending on problem
Space Complexity: O(1) to O(n) for auxiliary storage
"""

from typing import List, Tuple, Optional, Dict, Any, Set
from collections import Counter, defaultdict
import heapq

class GreedyStringArrayProblems:
    
    def __init__(self):
        """Initialize with problem tracking"""
        self.solution_steps = []
        self.problem_count = 0
    
    # ==========================================
    # 1. STRING RECONSTRUCTION PROBLEMS
    # ==========================================
    
    def reorganize_string(self, s: str) -> str:
        """
        Reorganize String (No Adjacent Duplicates)
        
        Company: Google, Facebook, Amazon
        Difficulty: Medium
        Time: O(n log k), Space: O(k) where k is unique characters
        
        Problem: Rearrange characters so no two adjacent characters are same
        Greedy Strategy: Always place most frequent character in available positions
        
        LeetCode 767
        """
        print("=== REORGANIZE STRING ===")
        print("Problem: Rearrange so no adjacent characters are same")
        print("Greedy Strategy: Place most frequent character first")
        print()
        
        print(f"Input string: '{s}'")
        
        # Count character frequencies
        char_count = Counter(s)
        print(f"Character frequencies: {dict(char_count)}")
        
        # Check if reorganization is possible
        max_freq = max(char_count.values())
        if max_freq > (len(s) + 1) // 2:
            print(f"Impossible: max frequency {max_freq} > {(len(s) + 1) // 2}")
            return ""
        
        print("Reorganization is possible!")
        print()
        
        # Use max heap to always get most frequent character
        max_heap = [(-count, char) for char, count in char_count.items()]
        heapq.heapify(max_heap)
        
        print("Initial character heap (frequency, character):")
        for count, char in sorted(max_heap):
            print(f"   '{char}': {-count}")
        print()
        
        result = []
        prev_count = 0
        prev_char = None
        
        print("Greedy reorganization process:")
        step = 1
        
        while max_heap:
            print(f"Step {step}:")
            
            # Get most frequent available character
            count, char = heapq.heappop(max_heap)
            count = -count  # Convert back to positive
            
            result.append(char)
            print(f"   Added '{char}' (frequency was {count})")
            print(f"   Result so far: '{''.join(result)}'")
            
            # Add back previous character if it still has occurrences
            if prev_count > 0:
                heapq.heappush(max_heap, (-prev_count, prev_char))
                print(f"   Added back '{prev_char}' with count {prev_count}")
            
            # Update previous character info
            prev_char = char
            prev_count = count - 1
            
            if max_heap:
                next_chars = [(-c, ch) for c, ch in sorted(max_heap)[:3]]
                print(f"   Next available: {next_chars}")
            
            print()
            step += 1
        
        final_result = ''.join(result)
        print(f"Final reorganized string: '{final_result}'")
        
        # Verify no adjacent duplicates
        valid = True
        for i in range(len(final_result) - 1):
            if final_result[i] == final_result[i + 1]:
                print(f"✗ Validation failed: adjacent duplicates at position {i}")
                valid = False
                break
        
        if valid:
            print("✓ Validation passed: no adjacent duplicates")
        
        return final_result
    
    def remove_duplicate_letters(self, s: str) -> str:
        """
        Remove Duplicate Letters (Lexicographically Smallest)
        
        Company: Google, Facebook
        Difficulty: Medium
        Time: O(n), Space: O(1) - at most 26 characters
        
        Problem: Remove duplicate letters to get lexicographically smallest result
        Greedy Strategy: Use stack, remove larger characters if they appear later
        
        LeetCode 316
        """
        print("=== REMOVE DUPLICATE LETTERS ===")
        print("Problem: Remove duplicates to get lexicographically smallest string")
        print("Greedy Strategy: Remove larger characters that appear later")
        print()
        
        print(f"Input string: '{s}'")
        
        # Count occurrences of each character
        char_count = Counter(s)
        print(f"Character frequencies: {dict(char_count)}")
        print()
        
        # Track characters in result and use stack for result
        in_result = set()
        stack = []
        
        print("Greedy removal process:")
        for i, char in enumerate(s):
            print(f"Step {i+1}: Process '{char}'")
            print(f"   Current stack: {''.join(stack)}")
            
            # Decrease count for current character
            char_count[char] -= 1
            print(f"   Remaining '{char}': {char_count[char]}")
            
            if char in in_result:
                print(f"   '{char}' already in result, skip")
                print()
                continue
            
            # Remove characters from stack that are:
            # 1. Lexicographically larger than current character
            # 2. Will appear later in the string
            while (stack and 
                   stack[-1] > char and 
                   char_count[stack[-1]] > 0):
                
                removed = stack.pop()
                in_result.remove(removed)
                print(f"   Removed '{removed}' (appears later and > '{char}')")
            
            # Add current character
            stack.append(char)
            in_result.add(char)
            print(f"   Added '{char}' to stack")
            print(f"   Current result: '{''.join(stack)}'")
            print()
        
        result = ''.join(stack)
        print(f"Final result: '{result}'")
        
        # Verify all characters included exactly once
        result_chars = set(result)
        original_chars = set(s)
        if result_chars == original_chars:
            print("✓ All unique characters included exactly once")
        else:
            print(f"✗ Missing characters: {original_chars - result_chars}")
        
        return result
    
    # ==========================================
    # 2. ARRAY OPTIMIZATION PROBLEMS
    # ==========================================
    
    def maximum_swap(self, num: int) -> int:
        """
        Maximum Swap (Single Swap to Maximize Number)
        
        Company: Facebook, Amazon
        Difficulty: Medium
        Time: O(n), Space: O(n) where n is number of digits
        
        Problem: Maximize number by swapping at most one pair of digits
        Greedy Strategy: Find rightmost larger digit for leftmost smaller digit
        
        LeetCode 670
        """
        print("=== MAXIMUM SWAP ===")
        print("Problem: Maximize number with at most one swap")
        print("Greedy Strategy: Swap leftmost small digit with rightmost larger digit")
        print()
        
        digits = list(str(num))
        n = len(digits)
        
        print(f"Input number: {num}")
        print(f"Digits: {digits}")
        print()
        
        # Find the last occurrence of each digit (0-9)
        last_occurrence = {}
        for i, digit in enumerate(digits):
            last_occurrence[digit] = i
        
        print("Last occurrence of each digit:")
        for digit in sorted(last_occurrence.keys()):
            print(f"   '{digit}': position {last_occurrence[digit]}")
        print()
        
        print("Finding optimal swap:")
        for i in range(n):
            # For current position, try to find a larger digit that appears later
            for digit in '9876543210':
                if digit > digits[i] and digit in last_occurrence:
                    j = last_occurrence[digit]
                    if j > i:
                        print(f"   Position {i}: current digit '{digits[i]}'")
                        print(f"   Found larger digit '{digit}' at position {j}")
                        print(f"   ✓ Optimal swap: position {i} ↔ position {j}")
                        
                        # Perform the swap
                        digits[i], digits[j] = digits[j], digits[i]
                        
                        result = int(''.join(digits))
                        print(f"   After swap: {result}")
                        return result
        
        print("   No beneficial swap found")
        print(f"   Result: {num} (unchanged)")
        return num
    
    def jump_game_ii(self, nums: List[int]) -> int:
        """
        Jump Game II (Minimum Jumps to Reach End)
        
        Company: Google, Amazon, Microsoft
        Difficulty: Medium
        Time: O(n), Space: O(1)
        
        Problem: Find minimum jumps to reach last index
        Greedy Strategy: Always jump to position that allows furthest reach
        
        LeetCode 45
        """
        print("=== JUMP GAME II ===")
        print("Problem: Find minimum jumps to reach the end")
        print("Greedy Strategy: Jump to position allowing furthest reach")
        print()
        
        if len(nums) <= 1:
            return 0
        
        print(f"Array: {nums}")
        print("Positions and jump ranges:")
        for i, jump in enumerate(nums):
            max_reach = min(i + jump, len(nums) - 1)
            print(f"   Position {i}: can jump {jump} steps, reach up to {max_reach}")
        print()
        
        jumps = 0
        current_end = 0
        farthest = 0
        
        print("Greedy jump selection:")
        for i in range(len(nums) - 1):  # Don't need to jump from last position
            farthest = max(farthest, i + nums[i])
            
            print(f"Position {i}: value={nums[i]}, farthest_reachable={farthest}")
            
            if i == current_end:
                jumps += 1
                current_end = farthest
                print(f"   ✓ Jump #{jumps}: new range end = {current_end}")
                
                if current_end >= len(nums) - 1:
                    print(f"   Can reach the end!")
                    break
            else:
                print(f"   Continue in current jump range")
        
        print(f"\nMinimum jumps needed: {jumps}")
        return jumps
    
    def gas_station(self, gas: List[int], cost: List[int]) -> int:
        """
        Gas Station Problem
        
        Company: Amazon, Google
        Difficulty: Medium
        Time: O(n), Space: O(1)
        
        Problem: Find starting gas station to complete circular route
        Greedy Strategy: Start from station where tank first becomes positive
        
        LeetCode 134
        """
        print("=== GAS STATION PROBLEM ===")
        print("Problem: Find starting station to complete circular route")
        print("Greedy Strategy: Start where cumulative gas deficit ends")
        print()
        
        n = len(gas)
        print(f"Gas stations: {list(range(n))}")
        print(f"Gas available: {gas}")
        print(f"Gas needed: {cost}")
        print()
        
        # Check if solution exists
        total_gas = sum(gas)
        total_cost = sum(cost)
        
        print(f"Total gas available: {total_gas}")
        print(f"Total gas needed: {total_cost}")
        
        if total_gas < total_cost:
            print("✗ Insufficient gas to complete the route")
            return -1
        
        print("✓ Sufficient gas exists, finding starting point")
        print()
        
        # Find starting point using greedy approach
        current_tank = 0
        starting_station = 0
        
        print("Analyzing each station:")
        for i in range(n):
            net_gas = gas[i] - cost[i]
            current_tank += net_gas
            
            print(f"Station {i}: gas={gas[i]}, cost={cost[i]}, net={net_gas}")
            print(f"   Tank after station {i}: {current_tank}")
            
            if current_tank < 0:
                # Can't reach next station, reset starting point
                print(f"   ✗ Cannot reach station {(i + 1) % n}")
                print(f"   Reset starting point to station {i + 1}")
                starting_station = i + 1
                current_tank = 0
            else:
                print(f"   ✓ Can reach station {(i + 1) % n}")
            print()
        
        print(f"Starting station: {starting_station}")
        
        # Verify the solution
        if starting_station < n:
            print("\nVerification - simulating complete route:")
            tank = 0
            for i in range(n):
                station = (starting_station + i) % n
                tank += gas[station] - cost[station]
                print(f"   After station {station}: tank = {tank}")
                
                if tank < 0:
                    print(f"   ✗ Failed at station {station}")
                    return -1
            
            print("   ✓ Successfully completed the route")
        
        return starting_station if starting_station < n else -1
    
    # ==========================================
    # 3. GREEDY PARTITIONING PROBLEMS
    # ==========================================
    
    def partition_labels(self, s: str) -> List[int]:
        """
        Partition Labels
        
        Company: Amazon, Google
        Difficulty: Medium
        Time: O(n), Space: O(1) - at most 26 characters
        
        Problem: Partition string so each character appears in at most one part
        Greedy Strategy: Extend partition to include last occurrence of all seen characters
        
        LeetCode 763
        """
        print("=== PARTITION LABELS ===")
        print("Problem: Partition so each character appears in at most one part")
        print("Greedy Strategy: Extend partition to last occurrence of seen characters")
        print()
        
        print(f"Input string: '{s}'")
        
        # Find last occurrence of each character
        last_occurrence = {}
        for i, char in enumerate(s):
            last_occurrence[char] = i
        
        print("Last occurrence of each character:")
        for char in sorted(set(s)):
            print(f"   '{char}': position {last_occurrence[char]}")
        print()
        
        partitions = []
        start = 0
        end = 0
        
        print("Greedy partitioning process:")
        for i, char in enumerate(s):
            # Extend current partition to include last occurrence of current character
            end = max(end, last_occurrence[char])
            
            print(f"Position {i}: character '{char}'")
            print(f"   Last occurrence of '{char}': {last_occurrence[char]}")
            print(f"   Current partition end: {end}")
            
            if i == end:
                # We've reached the end of current partition
                partition_length = end - start + 1
                partitions.append(partition_length)
                
                partition_str = s[start:end+1]
                print(f"   ✓ Complete partition: '{partition_str}' (length {partition_length})")
                
                start = i + 1
                print(f"   Next partition starts at position {start}")
            else:
                print(f"   Continue current partition")
            print()
        
        print(f"Final partitions: {partitions}")
        print("Partition breakdown:")
        
        pos = 0
        for i, length in enumerate(partitions):
            partition_str = s[pos:pos + length]
            unique_chars = set(partition_str)
            print(f"   Part {i+1}: '{partition_str}' (length {length}, chars: {sorted(unique_chars)})")
            pos += length
        
        return partitions
    
    def candy_distribution(self, ratings: List[int]) -> int:
        """
        Candy Distribution Problem
        
        Company: Google, Amazon
        Difficulty: Hard
        Time: O(n), Space: O(n)
        
        Problem: Distribute candies such that higher rated children get more candies
        Greedy Strategy: Two passes - left to right, then right to left
        
        LeetCode 135
        """
        print("=== CANDY DISTRIBUTION ===")
        print("Problem: Higher rated children must get more candies")
        print("Greedy Strategy: Two passes to satisfy all constraints")
        print()
        
        if not ratings:
            return 0
        
        n = len(ratings)
        print(f"Children ratings: {ratings}")
        print()
        
        # Initialize each child with 1 candy
        candies = [1] * n
        
        print("Initial candy distribution (everyone gets 1):")
        print(f"   Candies: {candies}")
        print()
        
        # Left to right pass
        print("Pass 1: Left to right (handle increasing ratings)")
        for i in range(1, n):
            if ratings[i] > ratings[i-1]:
                candies[i] = candies[i-1] + 1
                print(f"   Position {i}: rating {ratings[i]} > {ratings[i-1]}")
                print(f"   Give {candies[i]} candies (was {candies[i-1]})")
            else:
                print(f"   Position {i}: rating {ratings[i]} ≤ {ratings[i-1]}, keep 1 candy")
        
        print(f"   After pass 1: {candies}")
        print()
        
        # Right to left pass
        print("Pass 2: Right to left (handle decreasing ratings)")
        for i in range(n-2, -1, -1):
            if ratings[i] > ratings[i+1]:
                needed = candies[i+1] + 1
                if candies[i] < needed:
                    candies[i] = needed
                    print(f"   Position {i}: rating {ratings[i]} > {ratings[i+1]}")
                    print(f"   Increase to {candies[i]} candies (neighbor has {candies[i+1]})")
                else:
                    print(f"   Position {i}: rating {ratings[i]} > {ratings[i+1]}")
                    print(f"   Keep {candies[i]} candies (already sufficient)")
            else:
                print(f"   Position {i}: rating {ratings[i]} ≤ {ratings[i+1]}, no change needed")
        
        print(f"   After pass 2: {candies}")
        print()
        
        total_candies = sum(candies)
        
        print("Final distribution:")
        for i in range(n):
            print(f"   Child {i}: rating={ratings[i]}, candies={candies[i]}")
        
        print(f"\nTotal candies needed: {total_candies}")
        
        # Verify constraints
        print("\nConstraint verification:")
        valid = True
        for i in range(n-1):
            if ratings[i] > ratings[i+1] and candies[i] <= candies[i+1]:
                print(f"   ✗ Violation: child {i} (rating {ratings[i]}) should have more than child {i+1} (rating {ratings[i+1]})")
                valid = False
            elif ratings[i] < ratings[i+1] and candies[i] >= candies[i+1]:
                print(f"   ✗ Violation: child {i+1} (rating {ratings[i+1]}) should have more than child {i} (rating {ratings[i]})")
                valid = False
        
        if valid:
            print("   ✓ All constraints satisfied")
        
        return total_candies


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_greedy_string_array_problems():
    """Demonstrate all greedy string and array problems"""
    print("=== GREEDY STRING AND ARRAY PROBLEMS DEMONSTRATION ===\n")
    
    problems = GreedyStringArrayProblems()
    
    # 1. String Reconstruction
    print("1. STRING RECONSTRUCTION PROBLEMS")
    
    print("a) Reorganize String:")
    problems.reorganize_string("aab")
    print("\n" + "-"*40 + "\n")
    
    problems.reorganize_string("aaab")
    print("\n" + "-"*40 + "\n")
    
    print("b) Remove Duplicate Letters:")
    problems.remove_duplicate_letters("bcabc")
    print("\n" + "-"*40 + "\n")
    
    problems.remove_duplicate_letters("cbacdcbc")
    print("\n" + "="*60 + "\n")
    
    # 2. Array Optimization
    print("2. ARRAY OPTIMIZATION PROBLEMS")
    
    print("a) Maximum Swap:")
    problems.maximum_swap(2736)
    print("\n" + "-"*40 + "\n")
    
    problems.maximum_swap(9973)
    print("\n" + "-"*40 + "\n")
    
    print("b) Jump Game II:")
    problems.jump_game_ii([2, 3, 1, 1, 4])
    print("\n" + "-"*40 + "\n")
    
    print("c) Gas Station:")
    problems.gas_station([1, 2, 3, 4, 5], [3, 4, 5, 1, 2])
    print("\n" + "="*60 + "\n")
    
    # 3. Partitioning Problems
    print("3. GREEDY PARTITIONING PROBLEMS")
    
    print("a) Partition Labels:")
    problems.partition_labels("ababcbacadefegdehijhklij")
    print("\n" + "-"*40 + "\n")
    
    print("b) Candy Distribution:")
    problems.candy_distribution([1, 0, 2])
    print("\n" + "-"*40 + "\n")
    
    problems.candy_distribution([1, 2, 2])


if __name__ == "__main__":
    demonstrate_greedy_string_array_problems()
    
    print("\n=== STRING AND ARRAY PROBLEMS MASTERY GUIDE ===")
    
    print("\n🎯 PROBLEM PATTERNS:")
    print("• String Reconstruction: Use frequency + heap for optimal ordering")
    print("• Character Removal: Use stack with look-ahead for lexicographic optimization")
    print("• Array Optimization: Find optimal positions for swaps/jumps")
    print("• Partitioning: Extend boundaries based on constraints")
    print("• Distribution: Multi-pass algorithms to satisfy all constraints")
    
    print("\n📊 COMPLEXITY ANALYSIS:")
    print("• String problems: O(n log k) where k is unique characters")
    print("• Array optimization: O(n) for single-pass, O(n log n) for sorting")
    print("• Partitioning: O(n) for linear scan algorithms")
    print("• Distribution: O(n) for multi-pass greedy algorithms")
    
    print("\n⚡ KEY STRATEGIES:")
    print("• Frequency-based greedy: Use heaps for priority ordering")
    print("• Stack-based greedy: Remove suboptimal choices when better ones arrive")
    print("• Two-pass greedy: Handle bidirectional constraints separately")
    print("• Boundary extension: Grow partitions to satisfy all constraints")
    print("• Look-ahead optimization: Make decisions based on future availability")
    
    print("\n🔧 IMPLEMENTATION TECHNIQUES:")
    print("• Use Counter for frequency tracking")
    print("• Use heaps for dynamic priority selection")
    print("• Use stacks for maintaining optimal subsequences")
    print("• Use last occurrence maps for boundary determination")
    print("• Implement constraint validation for verification")
    
    print("\n🏆 REAL-WORLD APPLICATIONS:")
    print("• Text Processing: Document formatting, spell checking")
    print("• Resource Scheduling: Task ordering, priority queues")
    print("• Game Development: Optimal moves, score maximization")
    print("• Data Compression: Optimal encoding schemes")
    print("• Network Routing: Path optimization, load balancing")
    
    print("\n🎓 ADVANCED CONCEPTS:")
    print("• Multi-criteria optimization in string problems")
    print("• Online algorithms for streaming data")
    print("• Approximation algorithms for NP-hard variants")
    print("• Parallel algorithms for large-scale processing")
    print("• Dynamic programming vs greedy trade-offs")
    
    print("\n💡 PROBLEM-SOLVING TIPS:")
    print("• Identify the greedy choice criterion early")
    print("• Consider multiple passes for complex constraints")
    print("• Use appropriate data structures for efficient selection")
    print("• Verify solutions against all problem constraints")
    print("• Think about edge cases and boundary conditions")
