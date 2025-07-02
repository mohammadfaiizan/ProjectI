"""
Array Cyclic Sort Pattern - Missing Numbers Problems
====================================================

Topics: Cyclic sort, missing numbers, duplicate detection
Companies: Google, Facebook, Amazon, Microsoft
Difficulty: Easy to Medium
"""

from typing import List

class ArrayCyclicSort:
    
    # ==========================================
    # 1. BASIC CYCLIC SORT
    # ==========================================
    
    def cyclic_sort(self, nums: List[int]) -> List[int]:
        """Basic cyclic sort for numbers 1 to n
        Time: O(n), Space: O(1)
        """
        i = 0
        while i < len(nums):
            correct_index = nums[i] - 1
            if nums[i] != nums[correct_index]:
                nums[i], nums[correct_index] = nums[correct_index], nums[i]
            else:
                i += 1
        return nums
    
    def find_missing_number(self, nums: List[int]) -> int:
        """LC 268: Missing Number (0 to n)
        Time: O(n), Space: O(1)
        """
        i = 0
        n = len(nums)
        
        # Place each number at its correct position
        while i < n:
            if nums[i] < n and nums[i] != nums[nums[i]]:
                nums[nums[i]], nums[i] = nums[i], nums[nums[i]]
            else:
                i += 1
        
        # Find missing number
        for i in range(n):
            if nums[i] != i:
                return i
        
        return n
    
    def find_missing_numbers(self, nums: List[int]) -> List[int]:
        """LC 448: Find All Numbers Disappeared (1 to n)
        Time: O(n), Space: O(1)
        """
        i = 0
        while i < len(nums):
            correct_index = nums[i] - 1
            if nums[i] != nums[correct_index]:
                nums[i], nums[correct_index] = nums[correct_index], nums[i]
            else:
                i += 1
        
        missing = []
        for i in range(len(nums)):
            if nums[i] != i + 1:
                missing.append(i + 1)
        
        return missing
    
    # ==========================================
    # 2. DUPLICATE FINDING
    # ==========================================
    
    def find_duplicate(self, nums: List[int]) -> int:
        """LC 287: Find Duplicate Number (1 to n)
        Time: O(n), Space: O(1)
        """
        i = 0
        while i < len(nums):
            if nums[i] != i + 1:
                correct_index = nums[i] - 1
                if nums[i] != nums[correct_index]:
                    nums[i], nums[correct_index] = nums[correct_index], nums[i]
                else:
                    return nums[i]  # Found duplicate
            else:
                i += 1
        
        return -1
    
    def find_all_duplicates(self, nums: List[int]) -> List[int]:
        """LC 442: Find All Duplicates (1 to n)
        Time: O(n), Space: O(1)
        """
        i = 0
        while i < len(nums):
            correct_index = nums[i] - 1
            if nums[i] != nums[correct_index]:
                nums[i], nums[correct_index] = nums[correct_index], nums[i]
            else:
                i += 1
        
        duplicates = []
        for i in range(len(nums)):
            if nums[i] != i + 1:
                duplicates.append(nums[i])
        
        return duplicates
    
    # ==========================================
    # 3. ADVANCED PROBLEMS
    # ==========================================
    
    def find_first_missing_positive(self, nums: List[int]) -> int:
        """LC 41: First Missing Positive
        Time: O(n), Space: O(1)
        """
        n = len(nums)
        
        # Place each positive number at its correct position
        i = 0
        while i < n:
            if 1 <= nums[i] <= n and nums[i] != nums[nums[i] - 1]:
                correct_index = nums[i] - 1
                nums[i], nums[correct_index] = nums[correct_index], nums[i]
            else:
                i += 1
        
        # Find first missing positive
        for i in range(n):
            if nums[i] != i + 1:
                return i + 1
        
        return n + 1
    
    def find_corrupt_pair(self, nums: List[int]) -> List[int]:
        """Find corrupt number pair (missing and duplicate)
        Time: O(n), Space: O(1)
        """
        i = 0
        while i < len(nums):
            correct_index = nums[i] - 1
            if nums[i] != nums[correct_index]:
                nums[i], nums[correct_index] = nums[correct_index], nums[i]
            else:
                i += 1
        
        for i in range(len(nums)):
            if nums[i] != i + 1:
                return [nums[i], i + 1]  # [duplicate, missing]
        
        return [-1, -1]
    
    def find_smallest_missing_positive_k(self, nums: List[int], k: int) -> List[int]:
        """Find k smallest missing positive numbers
        Time: O(n), Space: O(1)
        """
        n = len(nums)
        
        # Cyclic sort for positive numbers
        i = 0
        while i < n:
            if 1 <= nums[i] <= n and nums[i] != nums[nums[i] - 1]:
                correct_index = nums[i] - 1
                nums[i], nums[correct_index] = nums[correct_index], nums[i]
            else:
                i += 1
        
        missing = []
        extras = set()
        
        # Find missing numbers in range [1, n]
        for i in range(n):
            if len(missing) < k:
                if nums[i] != i + 1:
                    missing.append(i + 1)
                    extras.add(nums[i])
        
        # Add more missing numbers beyond n
        candidate = n + 1
        while len(missing) < k:
            if candidate not in extras:
                missing.append(candidate)
            candidate += 1
        
        return missing

# Test Examples
def run_examples():
    acs = ArrayCyclicSort()
    
    print("=== ARRAY CYCLIC SORT EXAMPLES ===\n")
    
    # Basic cyclic sort
    print("1. BASIC CYCLIC SORT:")
    nums = [3, 1, 5, 4, 2]
    print(f"Before: {nums}")
    result = acs.cyclic_sort(nums.copy())
    print(f"After: {result}")
    
    # Missing number
    print("\n2. MISSING NUMBER:")
    nums = [4, 0, 3, 1]
    missing = acs.find_missing_number(nums.copy())
    print(f"Array: {nums}, Missing: {missing}")
    
    # Missing numbers
    print("\n3. FIND ALL MISSING:")
    nums = [4, 3, 2, 7, 8, 2, 3, 1]
    missing = acs.find_missing_numbers(nums.copy())
    print(f"Array: {nums}, Missing: {missing}")
    
    # Find duplicate
    print("\n4. FIND DUPLICATE:")
    nums = [1, 3, 4, 2, 2]
    duplicate = acs.find_duplicate(nums.copy())
    print(f"Array: {nums}, Duplicate: {duplicate}")
    
    # All duplicates
    print("\n5. FIND ALL DUPLICATES:")
    nums = [4, 3, 2, 7, 8, 2, 3, 1]
    duplicates = acs.find_all_duplicates(nums.copy())
    print(f"Array: {nums}, Duplicates: {duplicates}")
    
    # First missing positive
    print("\n6. FIRST MISSING POSITIVE:")
    nums = [3, 4, -1, 1]
    first_missing = acs.find_first_missing_positive(nums.copy())
    print(f"Array: {nums}, First missing positive: {first_missing}")
    
    # Corrupt pair
    print("\n7. CORRUPT PAIR:")
    nums = [3, 1, 2, 5, 2]
    corrupt = acs.find_corrupt_pair(nums.copy())
    print(f"Array: {nums}, Corrupt pair [duplicate, missing]: {corrupt}")

if __name__ == "__main__":
    run_examples() 