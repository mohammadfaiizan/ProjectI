"""
Combinatorial Backtracking Problems
==================================

Topics: Permutations, combinations, subsets, partitions, arrangements
Companies: Google, Amazon, Microsoft, Facebook, Apple, Netflix, Uber
Difficulty: Medium to Hard
Time Complexity: O(2^n) to O(n!) depending on problem
Space Complexity: O(n) to O(n!) for storing solutions
"""

from typing import List, Set, Dict, Tuple, Optional
import itertools

class CombinatorialBacktracking:
    
    def __init__(self):
        """Initialize with solution tracking and performance metrics"""
        self.solutions = []
        self.call_count = 0
        self.pruned_branches = 0
    
    # ==========================================
    # 1. PERMUTATIONS - ALL ARRANGEMENTS
    # ==========================================
    
    def generate_permutations(self, nums: List[int]) -> List[List[int]]:
        """
        Generate all permutations of given numbers
        
        Company: Google, Amazon, Microsoft
        Difficulty: Medium
        
        Time: O(n! * n), Space: O(n!)
        """
        self.solutions = []
        self.call_count = 0
        
        def backtrack(current_permutation: List[int], used: Set[int]) -> None:
            self.call_count += 1
            
            print(f"{'  ' * len(current_permutation)}Building: {current_permutation}, Used: {used}")
            
            # BASE CASE: Permutation is complete
            if len(current_permutation) == len(nums):
                self.solutions.append(current_permutation[:])  # Make a copy
                print(f"{'  ' * len(current_permutation)}✓ Complete permutation: {current_permutation}")
                return
            
            # RECURSIVE CASE: Try each unused number
            for num in nums:
                if num not in used:
                    # MAKE CHOICE
                    current_permutation.append(num)
                    used.add(num)
                    
                    # RECURSE
                    backtrack(current_permutation, used)
                    
                    # BACKTRACK
                    current_permutation.pop()
                    used.remove(num)
                    print(f"{'  ' * len(current_permutation)}← Backtracking from {current_permutation + [num]}")
        
        print(f"Generating all permutations of {nums}:")
        backtrack([], set())
        return self.solutions
    
    def permutations_with_swapping(self, nums: List[int]) -> List[List[int]]:
        """
        Generate permutations using swapping technique (more efficient)
        
        Time: O(n!), Space: O(n!)
        """
        result = []
        
        def backtrack(start: int) -> None:
            # BASE CASE: Complete permutation
            if start == len(nums):
                result.append(nums[:])  # Make a copy
                print(f"Generated permutation: {nums}")
                return
            
            # TRY: Each element from start to end as the start element
            for i in range(start, len(nums)):
                # MAKE CHOICE: Swap current element to start position
                nums[start], nums[i] = nums[i], nums[start]
                print(f"{'  ' * start}Swapped {nums[i]} to position {start}: {nums}")
                
                # RECURSE: Generate permutations for remaining positions
                backtrack(start + 1)
                
                # BACKTRACK: Restore original order
                nums[start], nums[i] = nums[i], nums[start]
                print(f"{'  ' * start}← Restored: {nums}")
        
        print(f"Generating permutations of {nums} using swapping:")
        backtrack(0)
        return result
    
    def permutations_with_duplicates(self, nums: List[int]) -> List[List[int]]:
        """
        Generate unique permutations when array contains duplicates
        
        Company: Facebook, Amazon
        Difficulty: Medium
        
        Time: O(n! * n), Space: O(n!)
        """
        nums.sort()  # Sort to group duplicates together
        result = []
        used = [False] * len(nums)
        
        def backtrack(current: List[int]) -> None:
            # BASE CASE: Complete permutation
            if len(current) == len(nums):
                result.append(current[:])
                print(f"Unique permutation: {current}")
                return
            
            for i in range(len(nums)):
                # SKIP: Already used element
                if used[i]:
                    continue
                
                # SKIP: Duplicate element (ensure we use duplicates in order)
                if i > 0 and nums[i] == nums[i-1] and not used[i-1]:
                    continue
                
                # MAKE CHOICE
                current.append(nums[i])
                used[i] = True
                
                # RECURSE
                backtrack(current)
                
                # BACKTRACK
                current.pop()
                used[i] = False
        
        print(f"Generating unique permutations of {nums}:")
        backtrack([])
        return result
    
    # ==========================================
    # 2. COMBINATIONS - SELECTIONS
    # ==========================================
    
    def generate_combinations(self, n: int, k: int) -> List[List[int]]:
        """
        Generate all combinations of k numbers from 1 to n
        
        Company: Google, Microsoft
        Difficulty: Medium
        
        Time: O(C(n,k) * k), Space: O(C(n,k) * k)
        """
        result = []
        
        def backtrack(start: int, current_combination: List[int]) -> None:
            print(f"{'  ' * len(current_combination)}Building: {current_combination}, start={start}")
            
            # BASE CASE: Combination is complete
            if len(current_combination) == k:
                result.append(current_combination[:])
                print(f"{'  ' * len(current_combination)}✓ Complete combination: {current_combination}")
                return
            
            # RECURSIVE CASE: Try numbers from start to n
            for i in range(start, n + 1):
                # MAKE CHOICE
                current_combination.append(i)
                
                # RECURSE: Start from i+1 to avoid duplicates
                backtrack(i + 1, current_combination)
                
                # BACKTRACK
                current_combination.pop()
                print(f"{'  ' * len(current_combination)}← Backtracking from {current_combination + [i]}")
        
        print(f"Generating combinations C({n},{k}):")
        backtrack(1, [])
        return result
    
    def combination_sum(self, candidates: List[int], target: int) -> List[List[int]]:
        """
        Find all combinations where candidates sum to target
        
        Company: Amazon, Facebook
        Difficulty: Medium
        
        Numbers can be reused
        Time: O(2^target), Space: O(target)
        """
        candidates.sort()  # Sort for optimization
        result = []
        
        def backtrack(start: int, current: List[int], remaining: int) -> None:
            print(f"{'  ' * len(current)}Current: {current}, remaining: {remaining}, start: {start}")
            
            # BASE CASE: Found target sum
            if remaining == 0:
                result.append(current[:])
                print(f"{'  ' * len(current)}✓ Found combination: {current}")
                return
            
            # PRUNING: If remaining is negative, stop
            if remaining < 0:
                return
            
            # TRY: Each candidate from start index
            for i in range(start, len(candidates)):
                # PRUNING: If current candidate > remaining, no point continuing
                if candidates[i] > remaining:
                    break
                
                # MAKE CHOICE
                current.append(candidates[i])
                
                # RECURSE: Can reuse same number (start=i), or use next (start=i+1) for unique
                backtrack(i, current, remaining - candidates[i])  # Allow reuse
                
                # BACKTRACK
                current.pop()
        
        print(f"Finding combinations that sum to {target} from {candidates}:")
        backtrack(0, [], target)
        return result
    
    def combination_sum_unique(self, candidates: List[int], target: int) -> List[List[int]]:
        """
        Find combinations that sum to target (each number used at most once)
        
        Company: Google, Amazon
        Difficulty: Medium
        
        Time: O(2^n), Space: O(target)
        """
        candidates.sort()
        result = []
        
        def backtrack(start: int, current: List[int], remaining: int) -> None:
            # BASE CASE: Found target sum
            if remaining == 0:
                result.append(current[:])
                print(f"Found unique combination: {current}")
                return
            
            for i in range(start, len(candidates)):
                # PRUNING: Skip if candidate > remaining
                if candidates[i] > remaining:
                    break
                
                # SKIP DUPLICATES: If current == previous and we haven't used previous
                if i > start and candidates[i] == candidates[i-1]:
                    continue
                
                # MAKE CHOICE
                current.append(candidates[i])
                
                # RECURSE: Move to next index (no reuse)
                backtrack(i + 1, current, remaining - candidates[i])
                
                # BACKTRACK
                current.pop()
        
        print(f"Finding unique combinations that sum to {target} from {candidates}:")
        backtrack(0, [], target)
        return result
    
    # ==========================================
    # 3. SUBSETS - POWER SET GENERATION
    # ==========================================
    
    def generate_subsets(self, nums: List[int]) -> List[List[int]]:
        """
        Generate all subsets (power set) of given array
        
        Company: Facebook, Google, Amazon
        Difficulty: Medium
        
        Time: O(2^n * n), Space: O(2^n * n)
        """
        result = []
        
        def backtrack(start: int, current_subset: List[int]) -> None:
            # ADD: Current subset to result (every state is valid)
            result.append(current_subset[:])
            print(f"{'  ' * len(current_subset)}Added subset: {current_subset}")
            
            # TRY: Adding each remaining element
            for i in range(start, len(nums)):
                # MAKE CHOICE: Include nums[i]
                current_subset.append(nums[i])
                
                # RECURSE: Start from next index
                backtrack(i + 1, current_subset)
                
                # BACKTRACK: Exclude nums[i]
                current_subset.pop()
        
        print(f"Generating all subsets of {nums}:")
        backtrack(0, [])
        return result
    
    def generate_subsets_bit_approach(self, nums: List[int]) -> List[List[int]]:
        """
        Generate subsets using bit manipulation approach
        
        Each subset corresponds to a binary number
        Time: O(2^n * n), Space: O(2^n * n)
        """
        result = []
        n = len(nums)
        
        # Generate all numbers from 0 to 2^n - 1
        for i in range(2**n):
            subset = []
            
            # Check each bit position
            for j in range(n):
                # If jth bit is set, include nums[j]
                if i & (1 << j):
                    subset.append(nums[j])
            
            result.append(subset)
            print(f"Binary {bin(i)[2:].zfill(n)}: {subset}")
        
        return result
    
    def subsets_with_duplicates(self, nums: List[int]) -> List[List[int]]:
        """
        Generate unique subsets when array contains duplicates
        
        Company: Amazon, Microsoft
        Difficulty: Medium
        
        Time: O(2^n * n), Space: O(2^n * n)
        """
        nums.sort()  # Sort to group duplicates
        result = []
        
        def backtrack(start: int, current: List[int]) -> None:
            # ADD: Current subset
            result.append(current[:])
            
            for i in range(start, len(nums)):
                # SKIP DUPLICATES: Only use first occurrence at each level
                if i > start and nums[i] == nums[i-1]:
                    continue
                
                # MAKE CHOICE
                current.append(nums[i])
                
                # RECURSE
                backtrack(i + 1, current)
                
                # BACKTRACK
                current.pop()
        
        print(f"Generating unique subsets of {nums}:")
        backtrack(0, [])
        return result
    
    # ==========================================
    # 4. PARTITIONING PROBLEMS
    # ==========================================
    
    def partition_equal_subset_sum(self, nums: List[int]) -> bool:
        """
        Check if array can be partitioned into two equal sum subsets
        
        Company: Amazon, Facebook
        Difficulty: Medium
        
        Time: O(2^n), Space: O(n)
        """
        total_sum = sum(nums)
        
        # If sum is odd, can't partition equally
        if total_sum % 2 != 0:
            return False
        
        target = total_sum // 2
        
        def backtrack(index: int, current_sum: int) -> bool:
            print(f"{'  ' * index}Checking index {index}, current_sum={current_sum}, target={target}")
            
            # BASE CASE: Found target sum
            if current_sum == target:
                return True
            
            # BASE CASE: Exceeded target or no more elements
            if current_sum > target or index >= len(nums):
                return False
            
            # CHOICE 1: Include current element
            if backtrack(index + 1, current_sum + nums[index]):
                return True
            
            # CHOICE 2: Exclude current element
            if backtrack(index + 1, current_sum):
                return True
            
            return False
        
        print(f"Checking if {nums} can be partitioned into equal subsets:")
        return backtrack(0, 0)
    
    def partition_k_equal_subsets(self, nums: List[int], k: int) -> bool:
        """
        Partition array into k subsets with equal sum
        
        Company: Google, Amazon
        Difficulty: Hard
        
        Time: O(k^n), Space: O(n)
        """
        total_sum = sum(nums)
        
        if total_sum % k != 0:
            return False
        
        target = total_sum // k
        nums.sort(reverse=True)  # Sort descending for better pruning
        used = [False] * len(nums)
        
        def backtrack(group: int, current_sum: int, start: int) -> bool:
            print(f"Group {group}, sum={current_sum}, start={start}")
            
            # BASE CASE: All groups formed
            if group == k:
                return True
            
            # BASE CASE: Current group sum equals target
            if current_sum == target:
                return backtrack(group + 1, 0, 0)
            
            # TRY: Each unused number
            for i in range(start, len(nums)):
                if used[i] or current_sum + nums[i] > target:
                    continue
                
                # MAKE CHOICE
                used[i] = True
                
                # RECURSE
                if backtrack(group, current_sum + nums[i], i + 1):
                    return True
                
                # BACKTRACK
                used[i] = False
                
                # PRUNING: If this was the first element we tried, no point trying others
                if current_sum == 0:
                    break
            
            return False
        
        print(f"Partitioning {nums} into {k} equal sum subsets:")
        return backtrack(0, 0, 0)
    
    def palindrome_partitioning(self, s: str) -> List[List[str]]:
        """
        Partition string such that every substring is a palindrome
        
        Company: Amazon, Microsoft
        Difficulty: Medium
        
        Time: O(2^n * n), Space: O(n)
        """
        result = []
        
        def is_palindrome(string: str, start: int, end: int) -> bool:
            """Check if substring is palindrome"""
            while start < end:
                if string[start] != string[end]:
                    return False
                start += 1
                end -= 1
            return True
        
        def backtrack(start: int, current_partition: List[str]) -> None:
            print(f"{'  ' * len(current_partition)}Partitioning from index {start}: {current_partition}")
            
            # BASE CASE: Reached end of string
            if start >= len(s):
                result.append(current_partition[:])
                print(f"{'  ' * len(current_partition)}✓ Valid partition: {current_partition}")
                return
            
            # TRY: All possible substrings starting from start
            for end in range(start, len(s)):
                substring = s[start:end + 1]
                
                # CHECK: If current substring is palindrome
                if is_palindrome(s, start, end):
                    # MAKE CHOICE
                    current_partition.append(substring)
                    
                    # RECURSE
                    backtrack(end + 1, current_partition)
                    
                    # BACKTRACK
                    current_partition.pop()
        
        print(f"Finding palindromic partitions of '{s}':")
        backtrack(0, [])
        return result
    
    # ==========================================
    # 5. ADVANCED COMBINATORIAL PROBLEMS
    # ==========================================
    
    def letter_combinations_phone(self, digits: str) -> List[str]:
        """
        Generate all letter combinations from phone number digits
        
        Company: Amazon, Facebook, Google
        Difficulty: Medium
        
        Time: O(4^n), Space: O(4^n)
        """
        if not digits:
            return []
        
        # Phone number mapping
        phone_map = {
            '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
            '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
        }
        
        result = []
        
        def backtrack(index: int, current_combination: str) -> None:
            print(f"{'  ' * len(current_combination)}Building: '{current_combination}', digit index: {index}")
            
            # BASE CASE: All digits processed
            if index >= len(digits):
                result.append(current_combination)
                print(f"{'  ' * len(current_combination)}✓ Complete: '{current_combination}'")
                return
            
            # GET: Letters for current digit
            current_digit = digits[index]
            letters = phone_map.get(current_digit, '')
            
            # TRY: Each letter for current digit
            for letter in letters:
                # RECURSE: Move to next digit
                backtrack(index + 1, current_combination + letter)
        
        print(f"Generating letter combinations for '{digits}':")
        backtrack(0, '')
        return result
    
    def generate_ip_addresses(self, s: str) -> List[str]:
        """
        Generate all valid IP addresses from string
        
        Company: Google, Facebook
        Difficulty: Medium
        
        Time: O(1) - constant since max 12 digits, Space: O(1)
        """
        result = []
        
        def is_valid_part(part: str) -> bool:
            """Check if string is valid IP part"""
            if not part or len(part) > 3:
                return False
            if len(part) > 1 and part[0] == '0':  # No leading zeros
                return False
            return 0 <= int(part) <= 255
        
        def backtrack(start: int, parts: List[str]) -> None:
            print(f"{'  ' * len(parts)}Building IP: {'.'.join(parts)}, start: {start}")
            
            # BASE CASE: Have 4 parts and used all characters
            if len(parts) == 4:
                if start == len(s):
                    ip = '.'.join(parts)
                    result.append(ip)
                    print(f"{'  ' * len(parts)}✓ Valid IP: {ip}")
                return
            
            # TRY: Different lengths for next part (1, 2, or 3 digits)
            for length in range(1, 4):
                if start + length > len(s):
                    break
                
                part = s[start:start + length]
                
                if is_valid_part(part):
                    # MAKE CHOICE
                    parts.append(part)
                    
                    # RECURSE
                    backtrack(start + length, parts)
                    
                    # BACKTRACK
                    parts.pop()
        
        print(f"Generating valid IP addresses from '{s}':")
        backtrack(0, [])
        return result
    
    # ==========================================
    # 6. PERFORMANCE OPTIMIZATION TECHNIQUES
    # ==========================================
    
    def demonstrate_optimization_techniques(self) -> None:
        """
        Demonstrate various optimization techniques for combinatorial problems
        """
        print("=== COMBINATORIAL OPTIMIZATION TECHNIQUES ===")
        print()
        
        print("1. PRUNING STRATEGIES:")
        print("   • Early termination when constraints violated")
        print("   • Skip duplicate elements systematically")
        print("   • Bound checking (sum limits, size limits)")
        print("   • Symmetry breaking to avoid redundant solutions")
        print()
        
        print("2. ORDERING OPTIMIZATIONS:")
        print("   • Sort input for better pruning opportunities")
        print("   • Try larger elements first (for sum problems)")
        print("   • Use frequency sorting for duplicate handling")
        print()
        
        print("3. STATE REPRESENTATION:")
        print("   • Use bitmasks for subset problems")
        print("   • Efficient data structures (sets vs lists)")
        print("   • Minimal state tracking")
        print()
        
        print("4. MEMORY OPTIMIZATIONS:")
        print("   • Generate solutions on-demand (iterators)")
        print("   • Avoid copying large data structures")
        print("   • Use in-place modifications where possible")

# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_combinatorial_problems():
    """Demonstrate all combinatorial backtracking problems"""
    print("=== COMBINATORIAL BACKTRACKING DEMONSTRATION ===\n")
    
    cb = CombinatorialBacktracking()
    
    # 1. Permutations
    print("=== PERMUTATIONS ===")
    
    # Basic permutations
    print("1. Basic Permutations of [1,2,3]:")
    perms = cb.generate_permutations([1, 2, 3])
    print(f"Generated {len(perms)} permutations: {perms}")
    print(f"Function calls: {cb.call_count}")
    print()
    
    # Permutations with swapping
    print("2. Permutations using swapping [1,2,3]:")
    perms_swap = cb.permutations_with_swapping([1, 2, 3])
    print(f"Generated {len(perms_swap)} permutations: {perms_swap}")
    print()
    
    # Permutations with duplicates
    print("3. Unique Permutations of [1,1,2]:")
    unique_perms = cb.permutations_with_duplicates([1, 1, 2])
    print(f"Generated {len(unique_perms)} unique permutations: {unique_perms}")
    print()
    
    # 2. Combinations
    print("=== COMBINATIONS ===")
    
    # Basic combinations
    print("1. Combinations C(4,2):")
    combs = cb.generate_combinations(4, 2)
    print(f"Generated {len(combs)} combinations: {combs}")
    print()
    
    # Combination sum
    print("2. Combination Sum (target=7, candidates=[2,3,6,7]):")
    comb_sum = cb.combination_sum([2, 3, 6, 7], 7)
    print(f"Found {len(comb_sum)} combinations: {comb_sum}")
    print()
    
    # 3. Subsets
    print("=== SUBSETS ===")
    
    # Basic subsets
    print("1. All Subsets of [1,2,3]:")
    subsets = cb.generate_subsets([1, 2, 3])
    print(f"Generated {len(subsets)} subsets: {subsets}")
    print()
    
    # Subsets with bit approach
    print("2. Subsets using bit manipulation [1,2]:")
    bit_subsets = cb.generate_subsets_bit_approach([1, 2])
    print(f"Generated {len(bit_subsets)} subsets: {bit_subsets}")
    print()
    
    # 4. Partitioning
    print("=== PARTITIONING PROBLEMS ===")
    
    # Equal subset sum
    print("1. Partition Equal Subset Sum [1,5,11,5]:")
    can_partition = cb.partition_equal_subset_sum([1, 5, 11, 5])
    print(f"Can partition into equal subsets: {can_partition}")
    print()
    
    # Palindrome partitioning
    print("2. Palindrome Partitioning 'aab':")
    palindrome_parts = cb.palindrome_partitioning('aab')
    print(f"Found {len(palindrome_parts)} palindromic partitions: {palindrome_parts}")
    print()
    
    # 5. Advanced Problems
    print("=== ADVANCED COMBINATORIAL PROBLEMS ===")
    
    # Phone number combinations
    print("1. Letter Combinations of Phone Number '23':")
    phone_combos = cb.letter_combinations_phone('23')
    print(f"Generated {len(phone_combos)} combinations: {phone_combos}")
    print()
    
    # IP addresses
    print("2. Generate IP Addresses from '25525511135':")
    ip_addresses = cb.generate_ip_addresses('25525511135')
    print(f"Found {len(ip_addresses)} valid IP addresses: {ip_addresses}")
    print()
    
    # Optimization techniques
    cb.demonstrate_optimization_techniques()

if __name__ == "__main__":
    demonstrate_combinatorial_problems()
    
    print("\n=== COMBINATORIAL PROBLEM SOLVING GUIDE ===")
    
    print("\n🎯 PROBLEM IDENTIFICATION:")
    print("• Permutations: All arrangements/orderings")
    print("• Combinations: Selections without regard to order")
    print("• Subsets: All possible collections (power set)")
    print("• Partitions: Divide into groups with constraints")
    
    print("\n📋 SOLUTION APPROACH:")
    print("1. Identify the type of combinatorial problem")
    print("2. Determine constraints and valid conditions")
    print("3. Choose appropriate backtracking pattern")
    print("4. Implement with proper pruning")
    print("5. Optimize for specific problem characteristics")
    
    print("\n⚡ OPTIMIZATION STRATEGIES:")
    print("• Sort input for better pruning")
    print("• Skip duplicates systematically")
    print("• Use early termination conditions")
    print("• Choose efficient data structures")
    print("• Consider bit manipulation for subsets")
    
    print("\n🎓 COMMON PATTERNS:")
    print("• Include/Exclude: For subset problems")
    print("• Position-based: For permutation problems")
    print("• Start-index: For combination problems")
    print("• Constraint-driven: For partition problems")
    
    print("\n📊 COMPLEXITY ANALYSIS:")
    print("• Permutations: O(n!) time, O(n) space")
    print("• Combinations: O(C(n,k)) time, O(k) space")
    print("• Subsets: O(2^n) time, O(n) space")
    print("• Depends on pruning effectiveness in practice")
