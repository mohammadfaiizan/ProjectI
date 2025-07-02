"""
String Palindromes - All Variants and Techniques
==============================================

Topics: Basic palindromes, longest palindromes, palindrome partitioning
Companies: Amazon, Microsoft, Google, Facebook
Difficulty: Easy to Hard
"""

from typing import List

class StringPalindromes:
    
    # ==========================================
    # 1. BASIC PALINDROME CHECKS
    # ==========================================
    
    def is_palindrome_basic(self, s: str) -> bool:
        """Basic palindrome check - Time: O(n), Space: O(1)"""
        left, right = 0, len(s) - 1
        
        while left < right:
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1
        
        return True
    
    def is_valid_palindrome(self, s: str) -> bool:
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
                # Try removing left or right character
                return (is_palindrome_range(left + 1, right) or 
                       is_palindrome_range(left, right - 1))
            left += 1
            right -= 1
        
        return True
    
    # ==========================================
    # 2. LONGEST PALINDROMIC SUBSTRING
    # ==========================================
    
    def longest_palindrome_expand(self, s: str) -> str:
        """LC 5: Expand Around Centers - Time: O(n²), Space: O(1)"""
        if not s:
            return ""
        
        start = 0
        max_len = 1
        
        def expand_around_center(left: int, right: int) -> int:
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return right - left - 1
        
        for i in range(len(s)):
            # Odd length palindromes
            len1 = expand_around_center(i, i)
            # Even length palindromes
            len2 = expand_around_center(i, i + 1)
            
            current_max = max(len1, len2)
            if current_max > max_len:
                max_len = current_max
                start = i - (current_max - 1) // 2
        
        return s[start:start + max_len]
    
    def longest_palindrome_dp(self, s: str) -> str:
        """DP Approach - Time: O(n²), Space: O(n²)"""
        n = len(s)
        if n == 0:
            return ""
        
        dp = [[False] * n for _ in range(n)]
        start = 0
        max_len = 1
        
        # All single characters are palindromes
        for i in range(n):
            dp[i][i] = True
        
        # Check for 2-character palindromes
        for i in range(n - 1):
            if s[i] == s[i + 1]:
                dp[i][i + 1] = True
                start = i
                max_len = 2
        
        # Check for palindromes of length 3 and more
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                
                if s[i] == s[j] and dp[i + 1][j - 1]:
                    dp[i][j] = True
                    start = i
                    max_len = length
        
        return s[start:start + max_len]
    
    def manacher_algorithm(self, s: str) -> str:
        """Manacher's Algorithm - Time: O(n), Space: O(n)"""
        if not s:
            return ""
        
        # Preprocess string
        processed = '#'.join('^{}$'.format(s))
        n = len(processed)
        p = [0] * n  # Array to store palindrome lengths
        center = right = 0
        
        max_len = 0
        center_index = 0
        
        for i in range(1, n - 1):
            # Mirror of i
            mirror = 2 * center - i
            
            if i < right:
                p[i] = min(right - i, p[mirror])
            
            # Try to expand palindrome centered at i
            try:
                while processed[i + (1 + p[i])] == processed[i - (1 + p[i])]:
                    p[i] += 1
            except IndexError:
                pass
            
            # If palindrome centered at i extends past right, adjust center and right
            if i + p[i] > right:
                center, right = i, i + p[i]
            
            # Update maximum length palindrome
            if p[i] > max_len:
                max_len = p[i]
                center_index = i
        
        start = (center_index - max_len) // 2
        return s[start:start + max_len]
    
    # ==========================================
    # 3. PALINDROME PARTITIONING
    # ==========================================
    
    def partition_palindrome(self, s: str) -> List[List[str]]:
        """LC 131: Palindrome Partitioning - Time: O(n*2^n)"""
        result = []
        
        def is_palindrome(start: int, end: int) -> bool:
            while start < end:
                if s[start] != s[end]:
                    return False
                start += 1
                end -= 1
            return True
        
        def backtrack(start: int, path: List[str]):
            if start == len(s):
                result.append(path[:])
                return
            
            for end in range(start, len(s)):
                if is_palindrome(start, end):
                    path.append(s[start:end + 1])
                    backtrack(end + 1, path)
                    path.pop()
        
        backtrack(0, [])
        return result
    
    def min_cut_palindrome(self, s: str) -> int:
        """LC 132: Palindrome Partitioning II - Time: O(n²)"""
        n = len(s)
        if n <= 1:
            return 0
        
        # Precompute palindrome table
        is_palindrome = [[False] * n for _ in range(n)]
        
        for i in range(n):
            is_palindrome[i][i] = True
        
        for i in range(n - 1):
            is_palindrome[i][i + 1] = (s[i] == s[i + 1])
        
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                is_palindrome[i][j] = (s[i] == s[j] and is_palindrome[i + 1][j - 1])
        
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
    # 4. SPECIAL PALINDROME PROBLEMS
    # ==========================================
    
    def count_substrings(self, s: str) -> int:
        """LC 647: Palindromic Substrings - Time: O(n²)"""
        count = 0
        
        def expand_around_center(left: int, right: int) -> int:
            count = 0
            while left >= 0 and right < len(s) and s[left] == s[right]:
                count += 1
                left -= 1
                right += 1
            return count
        
        for i in range(len(s)):
            # Odd length palindromes
            count += expand_around_center(i, i)
            # Even length palindromes
            count += expand_around_center(i, i + 1)
        
        return count
    
    def shortest_palindrome(self, s: str) -> str:
        """LC 214: Shortest Palindrome - Time: O(n)"""
        if not s:
            return s
        
        # Use KMP to find longest palindromic prefix
        reversed_s = s[::-1]
        combined = s + "#" + reversed_s
        
        # Compute LPS array
        lps = [0] * len(combined)
        j = 0
        
        for i in range(1, len(combined)):
            while j > 0 and combined[i] != combined[j]:
                j = lps[j - 1]
            
            if combined[i] == combined[j]:
                j += 1
            
            lps[i] = j
        
        # Add minimum characters to front
        chars_to_add = len(s) - lps[-1]
        return reversed_s[:chars_to_add] + s
    
    def longest_palindrome_chars(self, s: str) -> int:
        """LC 409: Longest Palindrome from Characters"""
        from collections import Counter
        
        char_count = Counter(s)
        length = 0
        has_odd = False
        
        for count in char_count.values():
            length += count // 2 * 2  # Add even part
            if count % 2 == 1:
                has_odd = True
        
        return length + (1 if has_odd else 0)
    
    def palindrome_pairs(self, words: List[str]) -> List[List[int]]:
        """LC 336: Palindrome Pairs - Time: O(n*k²)"""
        def is_palindrome(s: str) -> bool:
            return s == s[::-1]
        
        word_dict = {word: i for i, word in enumerate(words)}
        result = []
        
        for i, word in enumerate(words):
            n = len(word)
            
            for j in range(n + 1):
                prefix = word[:j]
                suffix = word[j:]
                
                # Case 1: suffix is palindrome, find reverse of prefix
                if is_palindrome(suffix):
                    reverse_prefix = prefix[::-1]
                    if reverse_prefix in word_dict and word_dict[reverse_prefix] != i:
                        result.append([i, word_dict[reverse_prefix]])
                
                # Case 2: prefix is palindrome, find reverse of suffix
                if j != n and is_palindrome(prefix):
                    reverse_suffix = suffix[::-1]
                    if reverse_suffix in word_dict and word_dict[reverse_suffix] != i:
                        result.append([word_dict[reverse_suffix], i])
        
        return result

# Test Examples
def run_examples():
    sp = StringPalindromes()
    
    print("=== PALINDROME EXAMPLES ===\n")
    
    # Basic palindromes
    print("1. BASIC PALINDROMES:")
    print(f"Is 'racecar' palindrome: {sp.is_palindrome_basic('racecar')}")
    print(f"Valid palindrome 'A man, a plan, a canal: Panama': {sp.is_valid_palindrome('A man, a plan, a canal: Panama')}")
    print(f"Valid palindrome II 'abc': {sp.valid_palindrome_ii('abc')}")
    
    # Longest palindromes
    print("\n2. LONGEST PALINDROMES:")
    test_str = "babad"
    print(f"Longest palindrome in '{test_str}' (expand): '{sp.longest_palindrome_expand(test_str)}'")
    print(f"Longest palindrome in '{test_str}' (DP): '{sp.longest_palindrome_dp(test_str)}'")
    print(f"Longest palindrome in '{test_str}' (Manacher): '{sp.manacher_algorithm(test_str)}'")
    
    # Palindrome partitioning
    print("\n3. PALINDROME PARTITIONING:")
    partitions = sp.partition_palindrome("aab")
    print(f"Partitions of 'aab': {partitions}")
    print(f"Min cuts for 'aab': {sp.min_cut_palindrome('aab')}")
    
    # Special problems
    print("\n4. SPECIAL PROBLEMS:")
    print(f"Count palindromic substrings in 'abc': {sp.count_substrings('abc')}")
    print(f"Shortest palindrome for 'aacecaaa': '{sp.shortest_palindrome('aacecaaa')}'")
    print(f"Longest palindrome from chars 'abccccdd': {sp.longest_palindrome_chars('abccccdd')}")
    
    pairs = sp.palindrome_pairs(["abc", "cba", "ba", "ab"])
    print(f"Palindrome pairs in ['abc', 'cba', 'ba', 'ab']: {pairs}")

if __name__ == "__main__":
    run_examples() 