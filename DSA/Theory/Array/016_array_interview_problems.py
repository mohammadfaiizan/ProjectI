"""
Array Interview Problems - Company Specific Questions
====================================================

Real interview questions from top tech companies
Companies: Google, Facebook/Meta, Amazon, Microsoft, Apple, Netflix
Difficulty: Easy to Hard
"""

from typing import List, Dict, Tuple
from collections import defaultdict, Counter
import heapq

class ArrayInterviewProblems:
    
    # ==========================================
    # GOOGLE PROBLEMS
    # ==========================================
    
    def next_permutation(self, nums: List[int]) -> None:
        """LC 31: Next Permutation (Google)
        Time: O(n), Space: O(1)
        """
        # Find the largest index i such that nums[i] < nums[i + 1]
        i = len(nums) - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        
        if i >= 0:
            # Find the largest index j such that nums[i] < nums[j]
            j = len(nums) - 1
            while nums[j] <= nums[i]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]
        
        # Reverse the suffix starting at nums[i + 1]
        nums[i + 1:] = reversed(nums[i + 1:])
    
    def candy(self, ratings: List[int]) -> int:
        """LC 135: Candy (Google)
        Time: O(n), Space: O(n)
        """
        n = len(ratings)
        candies = [1] * n
        
        # Left to right pass
        for i in range(1, n):
            if ratings[i] > ratings[i - 1]:
                candies[i] = candies[i - 1] + 1
        
        # Right to left pass
        for i in range(n - 2, -1, -1):
            if ratings[i] > ratings[i + 1]:
                candies[i] = max(candies[i], candies[i + 1] + 1)
        
        return sum(candies)
    
    def trap_rain_water(self, height: List[int]) -> int:
        """LC 42: Trapping Rain Water (Google)
        Time: O(n), Space: O(1)
        """
        if not height:
            return 0
        
        left, right = 0, len(height) - 1
        left_max = right_max = 0
        water = 0
        
        while left < right:
            if height[left] < height[right]:
                if height[left] >= left_max:
                    left_max = height[left]
                else:
                    water += left_max - height[left]
                left += 1
            else:
                if height[right] >= right_max:
                    right_max = height[right]
                else:
                    water += right_max - height[right]
                right -= 1
        
        return water
    
    # ==========================================
    # FACEBOOK/META PROBLEMS
    # ==========================================
    
    def valid_parentheses(self, s: str) -> bool:
        """LC 20: Valid Parentheses (Facebook)
        Time: O(n), Space: O(n)
        """
        stack = []
        mapping = {")": "(", "}": "{", "]": "["}
        
        for char in s:
            if char in mapping:
                if not stack or stack.pop() != mapping[char]:
                    return False
            else:
                stack.append(char)
        
        return not stack
    
    def merge_intervals(self, intervals: List[List[int]]) -> List[List[int]]:
        """LC 56: Merge Intervals (Facebook)
        Time: O(n log n), Space: O(1)
        """
        if not intervals:
            return []
        
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        
        for current in intervals[1:]:
            if current[0] <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], current[1])
            else:
                merged.append(current)
        
        return merged
    
    def subarray_sum_equals_k(self, nums: List[int], k: int) -> int:
        """LC 560: Subarray Sum Equals K (Facebook)
        Time: O(n), Space: O(n)
        """
        count = 0
        prefix_sum = 0
        sum_count = {0: 1}
        
        for num in nums:
            prefix_sum += num
            count += sum_count.get(prefix_sum - k, 0)
            sum_count[prefix_sum] = sum_count.get(prefix_sum, 0) + 1
        
        return count
    
    # ==========================================
    # AMAZON PROBLEMS
    # ==========================================
    
    def meeting_rooms_ii(self, intervals: List[List[int]]) -> int:
        """LC 253: Meeting Rooms II (Amazon)
        Time: O(n log n), Space: O(n)
        """
        if not intervals:
            return 0
        
        start_times = sorted([i[0] for i in intervals])
        end_times = sorted([i[1] for i in intervals])
        
        start = end = 0
        rooms = max_rooms = 0
        
        while start < len(intervals):
            if start_times[start] < end_times[end]:
                rooms += 1
                start += 1
            else:
                rooms -= 1
                end += 1
            
            max_rooms = max(max_rooms, rooms)
        
        return max_rooms
    
    def top_k_frequent(self, nums: List[int], k: int) -> List[int]:
        """LC 347: Top K Frequent Elements (Amazon)
        Time: O(n log k), Space: O(n)
        """
        count = Counter(nums)
        return heapq.nlargest(k, count.keys(), key=count.get)
    
    def rotate_array(self, nums: List[int], k: int) -> None:
        """LC 189: Rotate Array (Amazon)
        Time: O(n), Space: O(1)
        """
        n = len(nums)
        k = k % n
        
        def reverse(start: int, end: int):
            while start < end:
                nums[start], nums[end] = nums[end], nums[start]
                start += 1
                end -= 1
        
        reverse(0, n - 1)
        reverse(0, k - 1)
        reverse(k, n - 1)
    
    # ==========================================
    # MICROSOFT PROBLEMS
    # ==========================================
    
    def longest_consecutive(self, nums: List[int]) -> int:
        """LC 128: Longest Consecutive Sequence (Microsoft)
        Time: O(n), Space: O(n)
        """
        if not nums:
            return 0
        
        num_set = set(nums)
        max_length = 0
        
        for num in num_set:
            if num - 1 not in num_set:  # Start of sequence
                current_num = num
                current_length = 1
                
                while current_num + 1 in num_set:
                    current_num += 1
                    current_length += 1
                
                max_length = max(max_length, current_length)
        
        return max_length
    
    def find_peak_element(self, nums: List[int]) -> int:
        """LC 162: Find Peak Element (Microsoft)
        Time: O(log n), Space: O(1)
        """
        left, right = 0, len(nums) - 1
        
        while left < right:
            mid = (left + right) // 2
            
            if nums[mid] > nums[mid + 1]:
                right = mid
            else:
                left = mid + 1
        
        return left
    
    # ==========================================
    # APPLE PROBLEMS
    # ==========================================
    
    def majority_element(self, nums: List[int]) -> int:
        """LC 169: Majority Element (Apple)
        Time: O(n), Space: O(1) - Boyer-Moore Algorithm
        """
        candidate = count = 0
        
        for num in nums:
            if count == 0:
                candidate = num
            count += 1 if num == candidate else -1
        
        return candidate
    
    def remove_duplicates_sorted_ii(self, nums: List[int]) -> int:
        """LC 80: Remove Duplicates II (Apple)
        Time: O(n), Space: O(1)
        """
        if len(nums) <= 2:
            return len(nums)
        
        write = 2
        for read in range(2, len(nums)):
            if nums[read] != nums[write - 2]:
                nums[write] = nums[read]
                write += 1
        
        return write
    
    # ==========================================
    # NETFLIX PROBLEMS
    # ==========================================
    
    def daily_temperatures(self, temperatures: List[int]) -> List[int]:
        """LC 739: Daily Temperatures (Netflix)
        Time: O(n), Space: O(n)
        """
        result = [0] * len(temperatures)
        stack = []
        
        for i, temp in enumerate(temperatures):
            while stack and temperatures[stack[-1]] < temp:
                idx = stack.pop()
                result[idx] = i - idx
            stack.append(i)
        
        return result
    
    def asteroid_collision(self, asteroids: List[int]) -> List[int]:
        """LC 735: Asteroid Collision (Netflix)
        Time: O(n), Space: O(n)
        """
        stack = []
        
        for asteroid in asteroids:
            while stack and asteroid < 0 < stack[-1]:
                if stack[-1] < -asteroid:
                    stack.pop()
                    continue
                elif stack[-1] == -asteroid:
                    stack.pop()
                break
            else:
                stack.append(asteroid)
        
        return stack
    
    # ==========================================
    # MIXED COMPANY PROBLEMS
    # ==========================================
    
    def first_missing_positive(self, nums: List[int]) -> int:
        """LC 41: First Missing Positive (Multiple)
        Time: O(n), Space: O(1)
        """
        n = len(nums)
        
        # Place each number in its correct position
        for i in range(n):
            while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
                nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
        
        # Find first missing positive
        for i in range(n):
            if nums[i] != i + 1:
                return i + 1
        
        return n + 1
    
    def jump_game_ii(self, nums: List[int]) -> int:
        """LC 45: Jump Game II (Multiple)
        Time: O(n), Space: O(1)
        """
        jumps = 0
        current_end = 0
        farthest = 0
        
        for i in range(len(nums) - 1):
            farthest = max(farthest, i + nums[i])
            
            if i == current_end:
                jumps += 1
                current_end = farthest
        
        return jumps

# Test Examples
def run_examples():
    aip = ArrayInterviewProblems()
    
    print("=== ARRAY INTERVIEW PROBLEMS EXAMPLES ===\n")
    
    # Google problems
    print("1. GOOGLE PROBLEMS:")
    nums = [1, 2, 3]
    print(f"Next permutation of {nums}:", end=" ")
    aip.next_permutation(nums)
    print(nums)
    
    ratings = [1, 0, 2]
    print(f"Candy distribution for {ratings}: {aip.candy(ratings)}")
    
    # Facebook problems
    print("\n2. FACEBOOK PROBLEMS:")
    intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]
    merged = aip.merge_intervals(intervals)
    print(f"Merged intervals: {merged}")
    
    # Amazon problems
    print("\n3. AMAZON PROBLEMS:")
    nums = [1, 1, 1, 2, 2, 3]
    k = 2
    top_k = aip.top_k_frequent(nums, k)
    print(f"Top {k} frequent in {nums}: {top_k}")
    
    # Microsoft problems
    print("\n4. MICROSOFT PROBLEMS:")
    nums = [100, 4, 200, 1, 3, 2]
    longest = aip.longest_consecutive(nums)
    print(f"Longest consecutive in {nums}: {longest}")
    
    # Apple problems
    print("\n5. APPLE PROBLEMS:")
    nums = [3, 2, 3]
    majority = aip.majority_element(nums)
    print(f"Majority element in {nums}: {majority}")
    
    # Netflix problems
    print("\n6. NETFLIX PROBLEMS:")
    temps = [73, 74, 75, 71, 69, 72, 76, 73]
    result = aip.daily_temperatures(temps)
    print(f"Daily temperatures wait days: {result}")

if __name__ == "__main__":
    run_examples() 