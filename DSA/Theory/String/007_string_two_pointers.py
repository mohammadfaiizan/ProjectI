"""
String Two Pointers - Advanced Techniques
=========================================

Topics: Left-right pointers, fast-slow pointers, palindromes
Companies: Microsoft, Google, Amazon, Facebook
Difficulty: Easy to Hard
"""

from typing import List

class StringTwoPointers:
    
    # ==========================================
    # 1. BASIC TWO POINTERS
    # ==========================================
    
    def reverse_string(self, s: List[str]) -> None:
        """LC 344: Reverse String - Time: O(n), Space: O(1)"""
        left, right = 0, len(s) - 1
        
        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1
    
    def reverse_vowels(self, s: str) -> str:
        """LC 345: Reverse Vowels - Time: O(n), Space: O(1)"""
        vowels = set('aeiouAEIOU')
        chars = list(s)
        left, right = 0, len(chars) - 1
        
        while left < right:
            if chars[left] not in vowels:
                left += 1
            elif chars[right] not in vowels:
                right -= 1
            else:
                chars[left], chars[right] = chars[right], chars[left]
                left += 1
                right -= 1
        
        return ''.join(chars)
    
    def move_zeros_to_end(self, s: str) -> str:
        """Move all zeros to end while maintaining order"""
        chars = list(s)
        write_pos = 0
        
        # Move non-zero characters to front
        for read_pos in range(len(chars)):
            if chars[read_pos] != '0':
                chars[write_pos] = chars[read_pos]
                write_pos += 1
        
        # Fill remaining positions with zeros
        while write_pos < len(chars):
            chars[write_pos] = '0'
            write_pos += 1
        
        return ''.join(chars)
    
    # ==========================================
    # 2. PALINDROME PROBLEMS
    # ==========================================
    
    def is_palindrome(self, s: str) -> bool:
        """LC 125: Valid Palindrome - Time: O(n), Space: O(1)"""
        left, right = 0, len(s) - 1
        
        while left < right:
            while left < right and not s[left].isalnum():
                left += 1
            while left < right and not s[right].isalnum():
                right -= 1
            
            if s[left].lower() != s[right].lower():
                return False
            
            left += 1
            right -= 1
        
        return True
    
    def valid_palindrome_ii(self, s: str) -> bool:
        """LC 680: Valid Palindrome II - Time: O(n), Space: O(1)"""
        def is_palindrome_range(left: int, right: int) -> bool:
            while left < right:
                if s[left] != s[right]:
                    return False
                left += 1
                right -= 1
            return True
        
        left, right = 0, len(s) - 1
        
        while left < right:
            if s[left] != s[right]:
                return (is_palindrome_range(left + 1, right) or 
                       is_palindrome_range(left, right - 1))
            left += 1
            right -= 1
        
        return True
    
    def shortest_palindrome(self, s: str) -> str:
        """LC 214: Shortest Palindrome - Time: O(n), Space: O(n)"""
        if not s:
            return s
        
        # Find longest palindromic prefix
        rev_s = s[::-1]
        
        # Use KMP-like approach
        for i in range(len(s)):
            if s.startswith(rev_s[i:]):
                return rev_s[:i] + s
        
        return rev_s + s
    
    # ==========================================
    # 3. SUBSTRING PROBLEMS
    # ==========================================
    
    def two_sum_strings(self, s: str, target: str) -> List[int]:
        """Find two characters that sum to target (by ASCII values)"""
        char_map = {}
        target_val = ord(target)
        
        for i, char in enumerate(s):
            complement = target_val - ord(char)
            if complement in char_map:
                return [char_map[complement], i]
            char_map[ord(char)] = i
        
        return []
    
    def three_sum_closest_string(self, s: str, target: int) -> int:
        """Find three characters whose ASCII sum is closest to target"""
        if len(s) < 3:
            return 0
        
        s_sorted = sorted(s)
        n = len(s_sorted)
        closest_sum = ord(s_sorted[0]) + ord(s_sorted[1]) + ord(s_sorted[2])
        
        for i in range(n - 2):
            left, right = i + 1, n - 1
            
            while left < right:
                current_sum = ord(s_sorted[i]) + ord(s_sorted[left]) + ord(s_sorted[right])
                
                if abs(current_sum - target) < abs(closest_sum - target):
                    closest_sum = current_sum
                
                if current_sum < target:
                    left += 1
                elif current_sum > target:
                    right -= 1
                else:
                    return current_sum
        
        return closest_sum
    
    def container_with_most_water_chars(self, heights: List[int]) -> int:
        """LC 11: Container With Most Water (adapted for character heights)"""
        left, right = 0, len(heights) - 1
        max_area = 0
        
        while left < right:
            width = right - left
            height = min(heights[left], heights[right])
            area = width * height
            max_area = max(max_area, area)
            
            if heights[left] < heights[right]:
                left += 1
            else:
                right -= 1
        
        return max_area
    
    # ==========================================
    # 4. PARTITIONING PROBLEMS
    # ==========================================
    
    def partition_labels(self, s: str) -> List[int]:
        """LC 763: Partition Labels - Time: O(n), Space: O(1)"""
        # Record last occurrence of each character
        last_occurrence = {}
        for i, char in enumerate(s):
            last_occurrence[char] = i
        
        result = []
        start = 0
        end = 0
        
        for i, char in enumerate(s):
            end = max(end, last_occurrence[char])
            
            if i == end:
                result.append(end - start + 1)
                start = i + 1
        
        return result
    
    def min_cut_palindrome_partition(self, s: str) -> int:
        """LC 132: Palindrome Partitioning II using two pointers"""
        n = len(s)
        if n <= 1:
            return 0
        
        # Precompute palindrome table using two pointers
        is_palindrome = [[False] * n for _ in range(n)]
        
        # Every single character is a palindrome
        for i in range(n):
            is_palindrome[i][i] = True
        
        # Check for palindromes of length 2
        for i in range(n - 1):
            if s[i] == s[i + 1]:
                is_palindrome[i][i + 1] = True
        
        # Check for palindromes of length 3 and more
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j] and is_palindrome[i + 1][j - 1]:
                    is_palindrome[i][j] = True
        
        # DP for minimum cuts
        cuts = [float('inf')] * n
        
        for i in range(n):
            if is_palindrome[0][i]:
                cuts[i] = 0
            else:
                for j in range(i):
                    if is_palindrome[j + 1][i]:
                        cuts[i] = min(cuts[i], cuts[j] + 1)
        
        return cuts[n - 1]
    
    # ==========================================
    # 5. ADVANCED TWO POINTERS
    # ==========================================
    
    def longest_mountain_string(self, s: str) -> int:
        """Find longest mountain subsequence in string"""
        n = len(s)
        if n < 3:
            return 0
        
        max_length = 0
        
        for i in range(1, n - 1):
            # Check if current position can be peak
            if s[i-1] < s[i] > s[i+1]:
                # Expand left
                left = i - 1
                while left > 0 and s[left-1] < s[left]:
                    left -= 1
                
                # Expand right
                right = i + 1
                while right < n - 1 and s[right] > s[right+1]:
                    right += 1
                
                max_length = max(max_length, right - left + 1)
        
        return max_length
    
    def remove_duplicates_sorted(self, s: str) -> str:
        """Remove duplicates from sorted string - Two pointers"""
        if not s:
            return s
        
        chars = list(s)
        write = 1
        
        for read in range(1, len(chars)):
            if chars[read] != chars[write - 1]:
                chars[write] = chars[read]
                write += 1
        
        return ''.join(chars[:write])
    
    def remove_duplicates_k_times(self, s: str, k: int) -> str:
        """Remove duplicates that appear k or more times"""
        if k <= 1:
            return ""
        
        chars = list(s)
        write = 0
        
        for read in range(len(chars)):
            chars[write] = chars[read]
            write += 1
            
            # Check if last k characters are same
            if write >= k and all(chars[write-k+i] == chars[write-1] for i in range(k)):
                write -= k
        
        return ''.join(chars[:write])
    
    def merge_sorted_strings(self, s1: str, s2: str) -> str:
        """Merge two sorted strings"""
        i = j = 0
        result = []
        
        while i < len(s1) and j < len(s2):
            if s1[i] <= s2[j]:
                result.append(s1[i])
                i += 1
            else:
                result.append(s2[j])
                j += 1
        
        # Add remaining characters
        while i < len(s1):
            result.append(s1[i])
            i += 1
        
        while j < len(s2):
            result.append(s2[j])
            j += 1
        
        return ''.join(result)
    
    def is_subsequence(self, s: str, t: str) -> bool:
        """LC 392: Is Subsequence - Time: O(n), Space: O(1)"""
        i = j = 0
        
        while i < len(s) and j < len(t):
            if s[i] == t[j]:
                i += 1
            j += 1
        
        return i == len(s)

# Test Examples
def run_examples():
    stp = StringTwoPointers()
    
    print("=== TWO POINTERS EXAMPLES ===\n")
    
    # Basic operations
    print("1. BASIC OPERATIONS:")
    s1 = ["h", "e", "l", "l", "o"]
    stp.reverse_string(s1)
    print(f"Reversed string: {s1}")
    
    print(f"Reverse vowels 'hello': '{stp.reverse_vowels('hello')}'")
    print(f"Move zeros '10203': '{stp.move_zeros_to_end('10203')}'")
    
    # Palindromes
    print("\n2. PALINDROMES:")
    print(f"Is palindrome 'A man, a plan, a canal: Panama': {stp.is_palindrome('A man, a plan, a canal: Panama')}")
    print(f"Valid palindrome II 'abc': {stp.valid_palindrome_ii('abc')}")
    
    # Advanced problems
    print("\n3. ADVANCED:")
    print(f"Partition labels 'ababcbacadefegdehijhklij': {stp.partition_labels('ababcbacadefegdehijhklij')}")
    print(f"Remove duplicates 'aabbcc': '{stp.remove_duplicates_sorted('aabbcc')}'")
    print(f"Is subsequence 'ace' in 'abcde': {stp.is_subsequence('ace', 'abcde')}")
    print(f"Merge sorted 'ace' and 'bdf': '{stp.merge_sorted_strings('ace', 'bdf')}'")

if __name__ == "__main__":
    run_examples() 