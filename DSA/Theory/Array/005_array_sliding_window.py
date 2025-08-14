"""
Array Sliding Window Techniques - Complete Implementation
========================================================

Topics: Fixed window, variable window, maximum/minimum in window
Companies: Google, Facebook, Amazon, Microsoft, Apple
Difficulty: Medium to Hard
"""

from typing import List, Dict
from collections import deque, defaultdict

class ArraySlidingWindow:
    
    # ==========================================
    # 1. FIXED SIZE SLIDING WINDOW
    # ==========================================
    
    def max_sum_subarray_size_k(self, arr: List[int], k: int) -> int:
        """Maximum sum of subarray of size k
        Time: O(n), Space: O(1)
        """
        if len(arr) < k:
            return 0
        
        # Calculate sum of first window
        window_sum = sum(arr[:k])
        max_sum = window_sum
        
        # Slide the window
        for i in range(k, len(arr)):
            window_sum = window_sum - arr[i - k] + arr[i]
            max_sum = max(max_sum, window_sum)
        
        return max_sum
    
    def average_of_subarrays_k(self, arr: List[int], k: int) -> List[float]:
        """LC 643: Maximum Average Subarray I
        Time: O(n), Space: O(n)
        """
        if len(arr) < k:
            return []
        
        result = []
        window_sum = sum(arr[:k])
        result.append(window_sum / k)
        
        for i in range(k, len(arr)):
            window_sum = window_sum - arr[i - k] + arr[i]
            result.append(window_sum / k)
        
        return result
    
    def sliding_window_maximum(self, nums: List[int], k: int) -> List[int]:
        """LC 239: Sliding Window Maximum - Using Deque
        Time: O(n), Space: O(k)
        """
        if not nums or k == 0:
            return []
        
        dq = deque()  # Store indices
        result = []
        
        for i in range(len(nums)):
            # Remove indices outside current window
            while dq and dq[0] <= i - k:
                dq.popleft()
            
            # Remove indices of elements smaller than current element
            while dq and nums[dq[-1]] < nums[i]:
                dq.pop()
            
            dq.append(i)
            
            # Add maximum of current window to result
            if i >= k - 1:
                result.append(nums[dq[0]])
        
        return result
    
    def sliding_window_minimum(self, nums: List[int], k: int) -> List[int]:
        """Sliding Window Minimum - Using Deque
        Time: O(n), Space: O(k)
        """
        if not nums or k == 0:
            return []
        
        dq = deque()
        result = []
        
        for i in range(len(nums)):
            # Remove indices outside current window
            while dq and dq[0] <= i - k:
                dq.popleft()
            
            # Remove indices of elements greater than current element
            while dq and nums[dq[-1]] > nums[i]:
                dq.pop()
            
            dq.append(i)
            
            if i >= k - 1:
                result.append(nums[dq[0]])
        
        return result
    
    # ==========================================
    # 2. VARIABLE SIZE SLIDING WINDOW
    # ==========================================
    
    def longest_substring_without_repeating(self, s: str) -> int:
        """LC 3: Longest Substring Without Repeating Characters
        Time: O(n), Space: O(min(m,n)) where m is charset size
        """
        char_map = {}
        left = 0
        max_length = 0
        
        for right in range(len(s)):
            if s[right] in char_map and char_map[s[right]] >= left:
                left = char_map[s[right]] + 1
            
            char_map[s[right]] = right
            max_length = max(max_length, right - left + 1)
        
        return max_length
    
    def longest_subarray_sum_k(self, arr: List[int], k: int) -> int:
        """Longest subarray with sum equals k (positive numbers)
        Time: O(n), Space: O(1)
        """
        left = 0
        current_sum = 0
        max_length = 0
        
        for right in range(len(arr)):
            current_sum += arr[right]
            
            while current_sum > k and left <= right:
                current_sum -= arr[left]
                left += 1
            
            if current_sum == k:
                max_length = max(max_length, right - left + 1)
        
        return max_length
    
    def smallest_subarray_sum_k(self, arr: List[int], k: int) -> int:
        """LC 209: Minimum Size Subarray Sum
        Time: O(n), Space: O(1)
        """
        left = 0
        current_sum = 0
        min_length = float('inf')
        
        for right in range(len(arr)):
            current_sum += arr[right]
            
            while current_sum >= k:
                min_length = min(min_length, right - left + 1)
                current_sum -= arr[left]
                left += 1
        
        return min_length if min_length != float('inf') else 0
    
    def max_consecutive_ones_ii(self, nums: List[int], k: int) -> int:
        """LC 1004: Max Consecutive Ones III
        Time: O(n), Space: O(1)
        """
        left = 0
        zero_count = 0
        max_length = 0
        
        for right in range(len(nums)):
            if nums[right] == 0:
                zero_count += 1
            
            while zero_count > k:
                if nums[left] == 0:
                    zero_count -= 1
                left += 1
            
            max_length = max(max_length, right - left + 1)
        
        return max_length
    
    # ==========================================
    # 3. STRING PATTERN SLIDING WINDOW
    # ==========================================
    
    def min_window_substring(self, s: str, t: str) -> str:
        """LC 76: Minimum Window Substring
        Time: O(|s| + |t|), Space: O(|s| + |t|)
        """
        if not s or not t:
            return ""
        
        dict_t = {}
        for char in t:
            dict_t[char] = dict_t.get(char, 0) + 1
        
        required = len(dict_t)
        formed = 0
        window_counts = {}
        
        l, r = 0, 0
        ans = float("inf"), None, None
        
        while r < len(s):
            character = s[r]
            window_counts[character] = window_counts.get(character, 0) + 1
            
            if character in dict_t and window_counts[character] == dict_t[character]:
                formed += 1
            
            while l <= r and formed == required:
                character = s[l]
                
                if r - l + 1 < ans[0]:
                    ans = (r - l + 1, l, r)
                
                window_counts[character] -= 1
                if character in dict_t and window_counts[character] < dict_t[character]:
                    formed -= 1
                
                l += 1
            
            r += 1
        
        return "" if ans[0] == float("inf") else s[ans[1]:ans[2] + 1]
    
    def find_anagrams(self, s: str, p: str) -> List[int]:
        """LC 438: Find All Anagrams in a String
        Time: O(|s| + |p|), Space: O(1)
        """
        if len(p) > len(s):
            return []
        
        result = []
        p_count = [0] * 26
        s_count = [0] * 26
        
        # Count characters in p and first window of s
        for i in range(len(p)):
            p_count[ord(p[i]) - ord('a')] += 1
            s_count[ord(s[i]) - ord('a')] += 1
        
        if p_count == s_count:
            result.append(0)
        
        # Sliding window
        for i in range(len(p), len(s)):
            # Add new character
            s_count[ord(s[i]) - ord('a')] += 1
            # Remove old character
            s_count[ord(s[i - len(p)]) - ord('a')] -= 1
            
            if p_count == s_count:
                result.append(i - len(p) + 1)
        
        return result
    
    def permutation_in_string(self, s1: str, s2: str) -> bool:
        """LC 567: Permutation in String
        Time: O(|s1| + |s2|), Space: O(1)
        """
        if len(s1) > len(s2):
            return False
        
        s1_count = [0] * 26
        s2_count = [0] * 26
        
        for i in range(len(s1)):
            s1_count[ord(s1[i]) - ord('a')] += 1
            s2_count[ord(s2[i]) - ord('a')] += 1
        
        if s1_count == s2_count:
            return True
        
        for i in range(len(s1), len(s2)):
            s2_count[ord(s2[i]) - ord('a')] += 1
            s2_count[ord(s2[i - len(s1)]) - ord('a')] -= 1
            
            if s1_count == s2_count:
                return True
        
        return False
    
    # ==========================================
    # 4. ADVANCED SLIDING WINDOW PROBLEMS
    # ==========================================
    
    def longest_substring_k_distinct(self, s: str, k: int) -> int:
        """LC 340: Longest Substring with At Most K Distinct Characters
        Time: O(n), Space: O(k)
        """
        if not s or k == 0:
            return 0
        
        left = 0
        max_length = 0
        char_count = defaultdict(int)
        
        for right in range(len(s)):
            char_count[s[right]] += 1
            
            while len(char_count) > k:
                char_count[s[left]] -= 1
                if char_count[s[left]] == 0:
                    del char_count[s[left]]
                left += 1
            
            max_length = max(max_length, right - left + 1)
        
        return max_length
    
    def character_replacement(self, s: str, k: int) -> int:
        """LC 424: Longest Repeating Character Replacement
        Time: O(n), Space: O(1)
        """
        left = 0
        max_length = 0
        max_count = 0
        char_count = defaultdict(int)
        
        for right in range(len(s)):
            char_count[s[right]] += 1
            max_count = max(max_count, char_count[s[right]])
            
            # If window size - max_count > k, shrink window
            if right - left + 1 - max_count > k:
                char_count[s[left]] -= 1
                left += 1
            
            max_length = max(max_length, right - left + 1)
        
        return max_length
    
    def subarrays_with_k_different(self, nums: List[int], k: int) -> int:
        """LC 992: Subarrays with K Different Integers
        Time: O(n), Space: O(k)
        """
        def at_most_k(nums: List[int], k: int) -> int:
            count = defaultdict(int)
            left = 0
            result = 0
            
            for right in range(len(nums)):
                count[nums[right]] += 1
                
                while len(count) > k:
                    count[nums[left]] -= 1
                    if count[nums[left]] == 0:
                        del count[nums[left]]
                    left += 1
                
                result += right - left + 1
            
            return result
        
        return at_most_k(nums, k) - at_most_k(nums, k - 1)
    
    def max_fruits_in_baskets(self, fruits: List[int]) -> int:
        """LC 904: Fruit Into Baskets (at most 2 types)
        Time: O(n), Space: O(1)
        """
        return self.longest_substring_k_distinct([str(f) for f in fruits], 2)

# Test Examples
def run_examples():
    asw = ArraySlidingWindow()
    
    print("=== ARRAY SLIDING WINDOW EXAMPLES ===\n")
    
    # Fixed size window
    print("1. FIXED SIZE WINDOW:")
    arr = [2, 1, 5, 1, 3, 2]
    k = 3
    print(f"Array: {arr}, k={k}")
    print(f"Max sum of size k: {asw.max_sum_subarray_size_k(arr, k)}")
    
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    print(f"Sliding window maximum: {asw.sliding_window_maximum(nums, k)}")
    
    # Variable size window
    print("\n2. VARIABLE SIZE WINDOW:")
    s = "abcabcbb"
    print(f"Longest substring without repeating: '{s}' -> {asw.longest_substring_without_repeating(s)}")
    
    arr = [2, 1, 2, 4, 3, 1]
    k = 7
    print(f"Smallest subarray sum >= {k}: {asw.smallest_subarray_sum_k(arr, k)}")
    
    # String patterns
    print("\n3. STRING PATTERN WINDOW:")
    s = "ADOBECODEBANC"
    t = "ABC"
    print(f"Min window substring: '{s}' contains '{t}' -> '{asw.min_window_substring(s, t)}'")
    
    s = "cbaebabacd"
    p = "abc"
    print(f"Find anagrams of '{p}' in '{s}': {asw.find_anagrams(s, p)}")
    
    # Advanced problems
    print("\n4. ADVANCED PROBLEMS:")
    s = "eceba"
    k = 2
    print(f"Longest substring with at most {k} distinct: '{s}' -> {asw.longest_substring_k_distinct(s, k)}")
    
    s = "ABAB"
    k = 2
    print(f"Longest repeating char replacement: '{s}', k={k} -> {asw.character_replacement(s, k)}")

if __name__ == "__main__":
    run_examples() 