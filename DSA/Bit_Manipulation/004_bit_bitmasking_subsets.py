"""
🔐 BIT MANIPULATION - BITMASKING TECHNIQUES
==========================================

This module covers bitmasking techniques for subset generation and manipulation.
Bitmasking is a powerful technique for solving problems involving subsets.

Topics Covered:
1. Generate All Subsets (Power Set)
2. Subsets with Given XOR
3. Subset Sum using Bitmask
4. Count Subsets with Given Property

Author: Interview Preparation Collection
LeetCode Problems: 78, 90, 416, 698, 1125, 1434
"""

class SubsetGeneration:
    """Generate subsets using bitmasking techniques."""
    
    @staticmethod
    def generate_all_subsets_iterative(nums: list) -> list:
        """
        Generate all subsets using iterative bitmasking.
        
        Key Insight: Each bit in a number represents whether
        to include the corresponding element in the subset.
        
        Args:
            nums: Input array
            
        Returns:
            List of all subsets
            
        Time: O(n * 2^n), Space: O(n * 2^n)
        LeetCode: 78
        """
        n = len(nums)
        all_subsets = []
        
        # Generate all numbers from 0 to 2^n - 1
        for mask in range(1 << n):  # 2^n possibilities
            subset = []
            
            # Check each bit in the mask
            for i in range(n):
                if mask & (1 << i):  # If ith bit is set
                    subset.append(nums[i])
            
            all_subsets.append(subset)
        
        return all_subsets
    
    @staticmethod
    def generate_all_subsets_recursive(nums: list) -> list:
        """
        Generate all subsets using recursive approach.
        
        Args:
            nums: Input array
            
        Returns:
            List of all subsets
            
        Time: O(n * 2^n), Space: O(n * 2^n)
        """
        def backtrack(start, current_subset):
            # Add current subset to result
            result.append(current_subset[:])
            
            # Try adding each remaining element
            for i in range(start, len(nums)):
                current_subset.append(nums[i])
                backtrack(i + 1, current_subset)
                current_subset.pop()
        
        result = []
        backtrack(0, [])
        return result
    
    @staticmethod
    def generate_subsets_with_duplicates(nums: list) -> list:
        """
        Generate subsets from array with duplicates.
        
        Args:
            nums: Input array (may contain duplicates)
            
        Returns:
            List of unique subsets
            
        Time: O(n * 2^n), Space: O(n * 2^n)
        LeetCode: 90
        """
        nums.sort()  # Sort to handle duplicates
        n = len(nums)
        all_subsets = []
        
        for mask in range(1 << n):
            subset = []
            
            for i in range(n):
                if mask & (1 << i):
                    subset.append(nums[i])
            
            # Add only if not already present
            if subset not in all_subsets:
                all_subsets.append(subset)
        
        return all_subsets
    
    @staticmethod
    def count_subsets(nums: list) -> int:
        """
        Count total number of subsets.
        
        Args:
            nums: Input array
            
        Returns:
            Number of subsets (2^n)
            
        Time: O(1), Space: O(1)
        """
        return 1 << len(nums)  # 2^n
    
    @staticmethod
    def generate_k_size_subsets(nums: list, k: int) -> list:
        """
        Generate all subsets of size k.
        
        Args:
            nums: Input array
            k: Size of subsets
            
        Returns:
            List of k-size subsets
            
        Time: O(C(n,k) * k), Space: O(C(n,k) * k)
        """
        n = len(nums)
        k_subsets = []
        
        # Generate all masks with exactly k bits set
        for mask in range(1 << n):
            if bin(mask).count('1') == k:  # Check if exactly k bits are set
                subset = []
                
                for i in range(n):
                    if mask & (1 << i):
                        subset.append(nums[i])
                
                k_subsets.append(subset)
        
        return k_subsets


class XORSubsetProblems:
    """Problems involving XOR operations on subsets."""
    
    @staticmethod
    def subsets_with_given_xor(nums: list, target_xor: int) -> list:
        """
        Find all subsets with given XOR value.
        
        Args:
            nums: Input array
            target_xor: Target XOR value
            
        Returns:
            List of subsets with target XOR
            
        Time: O(n * 2^n), Space: O(2^n)
        """
        n = len(nums)
        valid_subsets = []
        
        for mask in range(1 << n):
            subset = []
            current_xor = 0
            
            for i in range(n):
                if mask & (1 << i):
                    subset.append(nums[i])
                    current_xor ^= nums[i]
            
            if current_xor == target_xor:
                valid_subsets.append(subset)
        
        return valid_subsets
    
    @staticmethod
    def count_subsets_with_xor(nums: list, target_xor: int) -> int:
        """
        Count subsets with given XOR value.
        
        Args:
            nums: Input array
            target_xor: Target XOR value
            
        Returns:
            Count of valid subsets
            
        Time: O(n * 2^n), Space: O(1)
        """
        n = len(nums)
        count = 0
        
        for mask in range(1 << n):
            current_xor = 0
            
            for i in range(n):
                if mask & (1 << i):
                    current_xor ^= nums[i]
            
            if current_xor == target_xor:
                count += 1
        
        return count
    
    @staticmethod
    def maximum_xor_subset(nums: list) -> int:
        """
        Find maximum XOR value among all subsets.
        
        Args:
            nums: Input array
            
        Returns:
            Maximum XOR value
            
        Time: O(n * 2^n), Space: O(1)
        """
        n = len(nums)
        max_xor = 0
        
        for mask in range(1 << n):
            current_xor = 0
            
            for i in range(n):
                if mask & (1 << i):
                    current_xor ^= nums[i]
            
            max_xor = max(max_xor, current_xor)
        
        return max_xor
    
    @staticmethod
    def xor_of_all_subsets(nums: list) -> int:
        """
        Calculate XOR of XOR values of all subsets.
        
        Key Insight: Each element appears in exactly 2^(n-1) subsets.
        If n > 1, each element contributes 0 to final XOR.
        If n = 1, the element itself is the answer.
        
        Args:
            nums: Input array
            
        Returns:
            XOR of all subset XORs
            
        Time: O(1), Space: O(1)
        """
        n = len(nums)
        
        if n == 1:
            return nums[0]
        
        # For n > 1, each element appears in 2^(n-1) subsets
        # Since 2^(n-1) is even for n > 1, XOR becomes 0
        return 0


class SubsetSumProblems:
    """Subset sum problems using bitmasking."""
    
    @staticmethod
    def subset_sum_exists(nums: list, target: int) -> bool:
        """
        Check if subset with given sum exists.
        
        Args:
            nums: Input array
            target: Target sum
            
        Returns:
            True if subset exists
            
        Time: O(n * 2^n), Space: O(1)
        """
        n = len(nums)
        
        for mask in range(1 << n):
            current_sum = 0
            
            for i in range(n):
                if mask & (1 << i):
                    current_sum += nums[i]
            
            if current_sum == target:
                return True
        
        return False
    
    @staticmethod
    def find_subset_with_sum(nums: list, target: int) -> list:
        """
        Find a subset with given sum.
        
        Args:
            nums: Input array
            target: Target sum
            
        Returns:
            Subset with target sum (empty if not found)
            
        Time: O(n * 2^n), Space: O(n)
        """
        n = len(nums)
        
        for mask in range(1 << n):
            subset = []
            current_sum = 0
            
            for i in range(n):
                if mask & (1 << i):
                    subset.append(nums[i])
                    current_sum += nums[i]
            
            if current_sum == target:
                return subset
        
        return []
    
    @staticmethod
    def count_subsets_with_sum(nums: list, target: int) -> int:
        """
        Count subsets with given sum.
        
        Args:
            nums: Input array
            target: Target sum
            
        Returns:
            Count of valid subsets
            
        Time: O(n * 2^n), Space: O(1)
        """
        n = len(nums)
        count = 0
        
        for mask in range(1 << n):
            current_sum = 0
            
            for i in range(n):
                if mask & (1 << i):
                    current_sum += nums[i]
            
            if current_sum == target:
                count += 1
        
        return count
    
    @staticmethod
    def partition_equal_subset_sum(nums: list) -> bool:
        """
        Check if array can be partitioned into two equal sum subsets.
        
        Args:
            nums: Input array
            
        Returns:
            True if partition possible
            
        Time: O(n * 2^n), Space: O(1)
        LeetCode: 416
        """
        total_sum = sum(nums)
        
        # If total sum is odd, partition is impossible
        if total_sum % 2 != 0:
            return False
        
        target = total_sum // 2
        return SubsetSumProblems.subset_sum_exists(nums, target)
    
    @staticmethod
    def minimum_subset_sum_difference(nums: list) -> int:
        """
        Find minimum difference between two subset sums.
        
        Args:
            nums: Input array
            
        Returns:
            Minimum difference
            
        Time: O(n * 2^n), Space: O(1)
        """
        n = len(nums)
        total_sum = sum(nums)
        min_diff = float('inf')
        
        # Try all possible subsets
        for mask in range(1 << n):
            subset_sum = 0
            
            for i in range(n):
                if mask & (1 << i):
                    subset_sum += nums[i]
            
            other_sum = total_sum - subset_sum
            diff = abs(subset_sum - other_sum)
            min_diff = min(min_diff, diff)
        
        return min_diff


class AdvancedBitmaskingProblems:
    """Advanced bitmasking problems and techniques."""
    
    @staticmethod
    def can_partition_k_subsets(nums: list, k: int) -> bool:
        """
        Check if array can be partitioned into k equal sum subsets.
        
        Args:
            nums: Input array
            k: Number of subsets
            
        Returns:
            True if partition possible
            
        Time: O(k * 2^n), Space: O(2^n)
        LeetCode: 698
        """
        total_sum = sum(nums)
        
        if total_sum % k != 0:
            return False
        
        target = total_sum // k
        n = len(nums)
        
        # dp[mask] = True if subset represented by mask has sum = target
        dp = [False] * (1 << n)
        subset_sums = [0] * (1 << n)
        
        # Calculate sum for each subset
        for mask in range(1 << n):
            for i in range(n):
                if mask & (1 << i):
                    subset_sums[mask] += nums[i]
        
        # Mark subsets with target sum
        for mask in range(1 << n):
            if subset_sums[mask] == target:
                dp[mask] = True
        
        # Try to find k disjoint subsets
        def backtrack(used_mask, groups_formed):
            if groups_formed == k:
                return True
            
            # Find next valid subset
            for mask in range(1 << n):
                if dp[mask] and (used_mask & mask) == 0:
                    if backtrack(used_mask | mask, groups_formed + 1):
                        return True
            
            return False
        
        return backtrack(0, 0)
    
    @staticmethod
    def shortest_superstring_bitmask(words: list) -> str:
        """
        Find shortest superstring containing all words (using bitmask DP).
        
        Args:
            words: List of words
            
        Returns:
            Shortest superstring
            
        Time: O(n^2 * 2^n), Space: O(n * 2^n)
        """
        n = len(words)
        
        # Calculate overlap between words
        overlap = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    max_overlap = 0
                    for k in range(1, min(len(words[i]), len(words[j])) + 1):
                        if words[i][-k:] == words[j][:k]:
                            max_overlap = k
                    overlap[i][j] = max_overlap
        
        # DP: dp[mask][i] = minimum length to visit all words in mask ending at word i
        dp = [[float('inf')] * n for _ in range(1 << n)]
        parent = [[-1] * n for _ in range(1 << n)]
        
        # Initialize: single words
        for i in range(n):
            dp[1 << i][i] = len(words[i])
        
        # Fill DP table
        for mask in range(1 << n):
            for i in range(n):
                if not (mask & (1 << i)) or dp[mask][i] == float('inf'):
                    continue
                
                for j in range(n):
                    if mask & (1 << j):
                        continue
                    
                    new_mask = mask | (1 << j)
                    new_length = dp[mask][i] + len(words[j]) - overlap[i][j]
                    
                    if new_length < dp[new_mask][j]:
                        dp[new_mask][j] = new_length
                        parent[new_mask][j] = i
        
        # Find minimum length solution
        final_mask = (1 << n) - 1
        min_length = float('inf')
        last_word = -1
        
        for i in range(n):
            if dp[final_mask][i] < min_length:
                min_length = dp[final_mask][i]
                last_word = i
        
        # Reconstruct path
        path = []
        mask = final_mask
        current = last_word
        
        while current != -1:
            path.append(current)
            next_word = parent[mask][current]
            mask ^= (1 << current)
            current = next_word
        
        path.reverse()
        
        # Build result string
        result = words[path[0]]
        for i in range(1, len(path)):
            prev_word = path[i - 1]
            curr_word = path[i]
            result += words[curr_word][overlap[prev_word][curr_word]:]
        
        return result
    
    @staticmethod
    def number_of_ways_to_wear_hats(hats: list) -> int:
        """
        Count ways to assign hats to people (each person gets exactly one hat).
        
        Args:
            hats: List of hat preferences for each person
            
        Returns:
            Number of ways
            
        Time: O(2^n * m), Space: O(2^n)
        LeetCode: 1434
        """
        MOD = 10**9 + 7
        n = len(hats)
        
        # Create hat to people mapping
        hat_to_people = {}
        for person in range(n):
            for hat in hats[person]:
                if hat not in hat_to_people:
                    hat_to_people[hat] = []
                hat_to_people[hat].append(person)
        
        all_hats = sorted(hat_to_people.keys())
        
        # dp[mask] = number of ways to assign hats to people in mask
        dp = [0] * (1 << n)
        dp[0] = 1  # Base case: no people assigned
        
        # Process each hat
        for hat in all_hats:
            # Process in reverse to avoid using updated values
            for mask in range((1 << n) - 1, -1, -1):
                if dp[mask] == 0:
                    continue
                
                # Try assigning this hat to each person who likes it
                for person in hat_to_people[hat]:
                    if not (mask & (1 << person)):  # Person not yet assigned
                        new_mask = mask | (1 << person)
                        dp[new_mask] = (dp[new_mask] + dp[mask]) % MOD
        
        return dp[(1 << n) - 1]


class BitmaskingDemo:
    """Demonstration of bitmasking techniques."""
    
    @staticmethod
    def demonstrate_subset_generation():
        """Demonstrate subset generation techniques."""
        print("=== SUBSET GENERATION ===")
        
        nums = [1, 2, 3]
        
        # All subsets
        iterative_subsets = SubsetGeneration.generate_all_subsets_iterative(nums)
        recursive_subsets = SubsetGeneration.generate_all_subsets_recursive(nums)
        
        print(f"Array: {nums}")
        print(f"All subsets (iterative): {iterative_subsets}")
        print(f"All subsets (recursive): {recursive_subsets}")
        print(f"Total subsets: {SubsetGeneration.count_subsets(nums)}")
        
        # K-size subsets
        k = 2
        k_subsets = SubsetGeneration.generate_k_size_subsets(nums, k)
        print(f"Subsets of size {k}: {k_subsets}")
    
    @staticmethod
    def demonstrate_xor_problems():
        """Demonstrate XOR subset problems."""
        print("\n=== XOR SUBSET PROBLEMS ===")
        
        nums = [1, 2, 3, 4]
        target_xor = 3
        
        # Subsets with given XOR
        xor_subsets = XORSubsetProblems.subsets_with_given_xor(nums, target_xor)
        count_xor = XORSubsetProblems.count_subsets_with_xor(nums, target_xor)
        
        print(f"Array: {nums}")
        print(f"Subsets with XOR {target_xor}: {xor_subsets}")
        print(f"Count: {count_xor}")
        
        # Maximum XOR
        max_xor = XORSubsetProblems.maximum_xor_subset(nums)
        print(f"Maximum XOR among all subsets: {max_xor}")
        
        # XOR of all subsets
        all_xor = XORSubsetProblems.xor_of_all_subsets(nums)
        print(f"XOR of all subset XORs: {all_xor}")
    
    @staticmethod
    def demonstrate_subset_sum():
        """Demonstrate subset sum problems."""
        print("\n=== SUBSET SUM PROBLEMS ===")
        
        nums = [3, 34, 4, 12, 5, 2]
        target = 9
        
        # Subset sum existence
        exists = SubsetSumProblems.subset_sum_exists(nums, target)
        subset = SubsetSumProblems.find_subset_with_sum(nums, target)
        count = SubsetSumProblems.count_subsets_with_sum(nums, target)
        
        print(f"Array: {nums}")
        print(f"Subset with sum {target} exists: {exists}")
        print(f"One such subset: {subset}")
        print(f"Count of subsets with sum {target}: {count}")
        
        # Partition problems
        partition_nums = [1, 5, 11, 5]
        can_partition = SubsetSumProblems.partition_equal_subset_sum(partition_nums)
        min_diff = SubsetSumProblems.minimum_subset_sum_difference(partition_nums)
        
        print(f"\nPartition array {partition_nums}:")
        print(f"Can partition into equal sum subsets: {can_partition}")
        print(f"Minimum subset sum difference: {min_diff}")
    
    @staticmethod
    def demonstrate_advanced_problems():
        """Demonstrate advanced bitmasking problems."""
        print("\n=== ADVANCED BITMASKING ===")
        
        # K-subset partition
        nums = [4, 3, 2, 3, 5, 2, 1]
        k = 4
        can_partition_k = AdvancedBitmaskingProblems.can_partition_k_subsets(nums, k)
        print(f"Can partition {nums} into {k} equal sum subsets: {can_partition_k}")
        
        # Shortest superstring
        words = ["catg", "ctaagt", "gcta", "ttca", "atgcatc"]
        superstring = AdvancedBitmaskingProblems.shortest_superstring_bitmask(words[:3])  # Use first 3 for demo
        print(f"Shortest superstring for {words[:3]}: {superstring}")


def complexity_analysis():
    """Analyze complexity of bitmasking approaches."""
    print("\n=== COMPLEXITY ANALYSIS ===")
    
    print("Bitmasking Complexities:")
    print("1. Generate all subsets: O(n * 2^n) time, O(n * 2^n) space")
    print("2. Subset sum (brute force): O(n * 2^n) time, O(1) space")
    print("3. K-subset partition: O(k * 2^n) time, O(2^n) space")
    print("4. Shortest superstring: O(n^2 * 2^n) time, O(n * 2^n) space")
    print("\nNote: Bitmasking is exponential but often the best approach")
    print("for small n (typically n ≤ 20-25)")


def practical_tips():
    """Provide practical tips for bitmasking problems."""
    print("\n=== PRACTICAL TIPS ===")
    
    print("Bitmasking Best Practices:")
    print("1. Use when n ≤ 20-25 (2^n becomes too large beyond this)")
    print("2. Each bit represents inclusion/exclusion of an element")
    print("3. Iterate through all 2^n possible masks")
    print("4. Use (mask & (1 << i)) to check if ith element is included")
    print("5. Combine with DP for optimization (bitmask DP)")
    print("6. Consider space-time tradeoffs")
    print("7. Handle edge cases (empty subset, single element)")


if __name__ == "__main__":
    # Run all demonstrations
    demo = BitmaskingDemo()
    
    demo.demonstrate_subset_generation()
    demo.demonstrate_xor_problems()
    demo.demonstrate_subset_sum()
    demo.demonstrate_advanced_problems()
    
    complexity_analysis()
    practical_tips()
    
    print("\n🎯 Key Bitmasking Patterns:")
    print("1. Subset enumeration: Iterate through all 2^n masks")
    print("2. Inclusion check: Use (mask & (1 << i)) != 0")
    print("3. Subset operations: XOR, sum, product, etc.")
    print("4. DP optimization: Use masks as DP states")
    print("5. Constraint satisfaction: Check properties of subsets")
    print("6. Combinatorial optimization: Find optimal subset") 