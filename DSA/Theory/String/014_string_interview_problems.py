"""
String Interview Problems - Company Specific Questions
=====================================================

Real interview questions from top tech companies
Companies: Google, Facebook/Meta, Amazon, Microsoft, Apple, Netflix
Difficulty: Easy to Hard
"""

from typing import List, Dict, Set, Optional
from collections import defaultdict, Counter

class StringInterviewProblems:
    
    # ==========================================
    # GOOGLE INTERVIEW PROBLEMS
    # ==========================================
    
    def decode_string(self, s: str) -> str:
        """LC 394: Decode String (Google)
        Input: s = "3[a]2[bc]"
        Output: "aaabcbc"
        """
        stack = []
        current_string = ""
        current_num = 0
        
        for char in s:
            if char.isdigit():
                current_num = current_num * 10 + int(char)
            elif char == '[':
                stack.append(current_string)
                stack.append(current_num)
                current_string = ""
                current_num = 0
            elif char == ']':
                num = stack.pop()
                prev_string = stack.pop()
                current_string = prev_string + num * current_string
            else:
                current_string += char
        
        return current_string
    
    def word_break(self, s: str, wordDict: List[str]) -> bool:
        """LC 139: Word Break (Google)"""
        word_set = set(wordDict)
        dp = [False] * (len(s) + 1)
        dp[0] = True
        
        for i in range(1, len(s) + 1):
            for j in range(i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break
        
        return dp[len(s)]
    
    def minimum_window_substring(self, s: str, t: str) -> str:
        """LC 76: Minimum Window Substring (Google)"""
        if not s or not t:
            return ""
        
        dict_t = Counter(t)
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
    
    # ==========================================
    # FACEBOOK/META INTERVIEW PROBLEMS
    # ==========================================
    
    def valid_palindrome_ii(self, s: str) -> bool:
        """LC 680: Valid Palindrome II (Facebook)"""
        def is_palindrome(left: int, right: int) -> bool:
            while left < right:
                if s[left] != s[right]:
                    return False
                left += 1
                right -= 1
            return True
        
        left, right = 0, len(s) - 1
        
        while left < right:
            if s[left] == s[right]:
                left += 1
                right -= 1
            else:
                return is_palindrome(left + 1, right) or is_palindrome(left, right - 1)
        
        return True
    
    def multiply_strings(self, num1: str, num2: str) -> str:
        """LC 43: Multiply Strings (Facebook)"""
        if num1 == "0" or num2 == "0":
            return "0"
        
        result = [0] * (len(num1) + len(num2))
        
        for i in range(len(num1) - 1, -1, -1):
            for j in range(len(num2) - 1, -1, -1):
                mul = int(num1[i]) * int(num2[j])
                p1, p2 = i + j, i + j + 1
                total = mul + result[p2]
                
                result[p2] = total % 10
                result[p1] += total // 10
        
        start = 0
        while start < len(result) and result[start] == 0:
            start += 1
        
        return ''.join(map(str, result[start:]))
    
    def add_strings(self, num1: str, num2: str) -> str:
        """LC 415: Add Strings (Facebook)"""
        i, j = len(num1) - 1, len(num2) - 1
        carry = 0
        result = []
        
        while i >= 0 or j >= 0 or carry:
            n1 = int(num1[i]) if i >= 0 else 0
            n2 = int(num2[j]) if j >= 0 else 0
            
            total = n1 + n2 + carry
            result.append(str(total % 10))
            carry = total // 10
            
            i -= 1
            j -= 1
        
        return ''.join(reversed(result))
    
    # ==========================================
    # AMAZON INTERVIEW PROBLEMS
    # ==========================================
    
    def longest_substring_without_repeating(self, s: str) -> int:
        """LC 3: Longest Substring Without Repeating Characters (Amazon)"""
        char_map = {}
        left = 0
        max_length = 0
        
        for right in range(len(s)):
            if s[right] in char_map and char_map[s[right]] >= left:
                left = char_map[s[right]] + 1
            
            char_map[s[right]] = right
            max_length = max(max_length, right - left + 1)
        
        return max_length
    
    def group_anagrams(self, strs: List[str]) -> List[List[str]]:
        """LC 49: Group Anagrams (Amazon)"""
        groups = defaultdict(list)
        
        for s in strs:
            key = ''.join(sorted(s))
            groups[key].append(s)
        
        return list(groups.values())
    
    def reverse_words_in_string(self, s: str) -> str:
        """LC 151: Reverse Words in a String (Amazon)"""
        words = s.split()
        return ' '.join(reversed(words))
    
    def compare_version_numbers(self, version1: str, version2: str) -> int:
        """LC 165: Compare Version Numbers (Amazon)"""
        v1_parts = list(map(int, version1.split('.')))
        v2_parts = list(map(int, version2.split('.')))
        
        max_len = max(len(v1_parts), len(v2_parts))
        
        for i in range(max_len):
            v1_num = v1_parts[i] if i < len(v1_parts) else 0
            v2_num = v2_parts[i] if i < len(v2_parts) else 0
            
            if v1_num < v2_num:
                return -1
            elif v1_num > v2_num:
                return 1
        
        return 0
    
    # ==========================================
    # MICROSOFT INTERVIEW PROBLEMS
    # ==========================================
    
    def find_and_replace_pattern(self, words: List[str], pattern: str) -> List[str]:
        """LC 890: Find and Replace Pattern (Microsoft)"""
        def matches(word: str, pattern: str) -> bool:
            if len(word) != len(pattern):
                return False
            
            map_wp = {}  # word to pattern
            map_pw = {}  # pattern to word
            
            for w, p in zip(word, pattern):
                if w in map_wp:
                    if map_wp[w] != p:
                        return False
                else:
                    map_wp[w] = p
                
                if p in map_pw:
                    if map_pw[p] != w:
                        return False
                else:
                    map_pw[p] = w
            
            return True
        
        return [word for word in words if matches(word, pattern)]
    
    def basic_calculator(self, s: str) -> int:
        """LC 224: Basic Calculator (Microsoft)"""
        stack = []
        number = 0
        result = 0
        sign = 1
        
        for char in s:
            if char.isdigit():
                number = number * 10 + int(char)
            elif char == '+':
                result += sign * number
                number = 0
                sign = 1
            elif char == '-':
                result += sign * number
                number = 0
                sign = -1
            elif char == '(':
                stack.append(result)
                stack.append(sign)
                result = 0
                sign = 1
            elif char == ')':
                result += sign * number
                number = 0
                result *= stack.pop()  # sign
                result += stack.pop()  # previous result
        
        return result + sign * number
    
    # ==========================================
    # APPLE INTERVIEW PROBLEMS
    # ==========================================
    
    def text_justification(self, words: List[str], maxWidth: int) -> List[str]:
        """LC 68: Text Justification (Apple)"""
        result = []
        i = 0
        
        while i < len(words):
            line = []
            line_length = 0
            
            # Pack words into current line
            while i < len(words) and line_length + len(words[i]) + len(line) <= maxWidth:
                line.append(words[i])
                line_length += len(words[i])
                i += 1
            
            # Justify the line
            if i == len(words) or len(line) == 1:
                # Last line or single word line - left justify
                justified = ' '.join(line)
                justified += ' ' * (maxWidth - len(justified))
            else:
                # Distribute spaces evenly
                total_spaces = maxWidth - line_length
                gaps = len(line) - 1
                
                if gaps == 0:
                    justified = line[0] + ' ' * total_spaces
                else:
                    spaces_per_gap = total_spaces // gaps
                    extra_spaces = total_spaces % gaps
                    
                    justified = ""
                    for j in range(len(line)):
                        justified += line[j]
                        if j < gaps:
                            justified += ' ' * spaces_per_gap
                            if j < extra_spaces:
                                justified += ' '
            
            result.append(justified)
        
        return result
    
    def string_to_integer_atoi(self, s: str) -> int:
        """LC 8: String to Integer (atoi) (Apple)"""
        s = s.lstrip()
        if not s:
            return 0
        
        sign = 1
        index = 0
        
        if s[0] in ['+', '-']:
            sign = -1 if s[0] == '-' else 1
            index = 1
        
        result = 0
        while index < len(s) and s[index].isdigit():
            result = result * 10 + int(s[index])
            index += 1
        
        result *= sign
        return max(-2**31, min(2**31 - 1, result))
    
    # ==========================================
    # NETFLIX INTERVIEW PROBLEMS
    # ==========================================
    
    def reorganize_string(self, s: str) -> str:
        """LC 767: Reorganize String (Netflix)"""
        count = Counter(s)
        max_count = max(count.values())
        
        if max_count > (len(s) + 1) // 2:
            return ""
        
        result = [''] * len(s)
        index = 0
        
        # Place most frequent character first
        for char, freq in count.most_common():
            for _ in range(freq):
                result[index] = char
                index = (index + 2) % len(s)
                if index == 0:
                    index = 1
        
        return ''.join(result)
    
    def encode_and_decode_strings(self, strs: List[str]) -> str:
        """LC 271: Encode and Decode Strings (Netflix)"""
        # Encode
        encoded = ""
        for s in strs:
            encoded += str(len(s)) + "#" + s
        
        # Decode function would be:
        def decode(encoded_str: str) -> List[str]:
            decoded = []
            i = 0
            
            while i < len(encoded_str):
                # Find length
                j = i
                while encoded_str[j] != '#':
                    j += 1
                
                length = int(encoded_str[i:j])
                decoded.append(encoded_str[j + 1:j + 1 + length])
                i = j + 1 + length
            
            return decoded
        
        return encoded
    
    # ==========================================
    # ADVANCED MIXED PROBLEMS
    # ==========================================
    
    def word_ladder(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        """LC 127: Word Ladder (Multiple companies)"""
        if endWord not in wordList:
            return 0
        
        from collections import deque
        
        queue = deque([(beginWord, 1)])
        visited = {beginWord}
        word_set = set(wordList)
        
        while queue:
            word, level = queue.popleft()
            
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    next_word = word[:i] + c + word[i+1:]
                    
                    if next_word == endWord:
                        return level + 1
                    
                    if next_word in word_set and next_word not in visited:
                        visited.add(next_word)
                        queue.append((next_word, level + 1))
        
        return 0
    
    def alien_dictionary(self, words: List[str]) -> str:
        """LC 269: Alien Dictionary (Multiple companies)"""
        graph = defaultdict(set)
        in_degree = defaultdict(int)
        
        # Initialize all characters
        for word in words:
            for char in word:
                in_degree[char] = 0
        
        # Build graph
        for i in range(len(words) - 1):
            word1, word2 = words[i], words[i + 1]
            min_len = min(len(word1), len(word2))
            
            for j in range(min_len):
                if word1[j] != word2[j]:
                    if word2[j] not in graph[word1[j]]:
                        graph[word1[j]].add(word2[j])
                        in_degree[word2[j]] += 1
                    break
            else:
                # Check if word1 is longer than word2
                if len(word1) > len(word2):
                    return ""
        
        # Topological sort
        from collections import deque
        queue = deque([char for char in in_degree if in_degree[char] == 0])
        result = []
        
        while queue:
            char = queue.popleft()
            result.append(char)
            
            for neighbor in graph[char]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return ''.join(result) if len(result) == len(in_degree) else ""

# Test Examples
def run_examples():
    sip = StringInterviewProblems()
    
    print("=== STRING INTERVIEW PROBLEMS EXAMPLES ===\n")
    
    # Google problems
    print("1. GOOGLE PROBLEMS:")
    print(f"Decode '3[a]2[bc]': {sip.decode_string('3[a]2[bc]')}")
    print(f"Word break 'leetcode': {sip.word_break('leetcode', ['leet', 'code'])}")
    print(f"Min window 'ADOBECODEBANC', 'ABC': {sip.minimum_window_substring('ADOBECODEBANC', 'ABC')}")
    
    # Facebook problems
    print("\n2. FACEBOOK PROBLEMS:")
    print(f"Valid palindrome II 'aba': {sip.valid_palindrome_ii('aba')}")
    print(f"Multiply '123' * '456': {sip.multiply_strings('123', '456')}")
    print(f"Add '123' + '456': {sip.add_strings('123', '456')}")
    
    # Amazon problems
    print("\n3. AMAZON PROBLEMS:")
    print(f"Longest substring without repeating 'abcabcbb': {sip.longest_substring_without_repeating('abcabcbb')}")
    print(f"Group anagrams ['eat','tea','tan','ate','nat','bat']: {sip.group_anagrams(['eat','tea','tan','ate','nat','bat'])}")
    print(f"Compare versions '1.01' vs '1.001': {sip.compare_version_numbers('1.01', '1.001')}")
    
    # Microsoft problems
    print("\n4. MICROSOFT PROBLEMS:")
    print(f"Find pattern match ['abc','deq','mee','aqq'], 'abb': {sip.find_and_replace_pattern(['abc','deq','mee','aqq'], 'abb')}")
    print(f"Calculator '1 + 1': {sip.basic_calculator('1 + 1')}")
    
    # Apple problems
    print("\n5. APPLE PROBLEMS:")
    print(f"String to integer '42': {sip.string_to_integer_atoi('42')}")
    
    # Netflix problems
    print("\n6. NETFLIX PROBLEMS:")
    print(f"Reorganize string 'aab': {sip.reorganize_string('aab')}")
    print(f"Encode strings ['hello','world']: {sip.encode_and_decode_strings(['hello','world'])}")
    
    # Advanced problems
    print("\n7. ADVANCED PROBLEMS:")
    print(f"Word ladder 'hit'->'cog' in ['hot','dot','dog','lot','log','cog']: {sip.word_ladder('hit', 'cog', ['hot','dot','dog','lot','log','cog'])}")

if __name__ == "__main__":
    run_examples() 