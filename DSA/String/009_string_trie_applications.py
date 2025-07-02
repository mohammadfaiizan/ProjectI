"""
String Trie Applications - Advanced Data Structure
=================================================

Topics: Trie construction, prefix/suffix problems, word search
Companies: Google, Amazon, Microsoft, Facebook
Difficulty: Medium to Hard
"""

from typing import List, Dict, Optional

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.word = None
        self.count = 0

class StringTrieApplications:
    
    def __init__(self):
        self.root = TrieNode()
    
    # ==========================================
    # 1. BASIC TRIE OPERATIONS
    # ==========================================
    
    def insert(self, word: str) -> None:
        """Insert word into trie"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.count += 1
        node.is_end_of_word = True
        node.word = word
    
    def search(self, word: str) -> bool:
        """Search for word in trie"""
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word
    
    def starts_with(self, prefix: str) -> bool:
        """Check if any word starts with prefix"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
    
    def count_words_with_prefix(self, prefix: str) -> int:
        """Count words that start with prefix"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return 0
            node = node.children[char]
        return node.count
    
    # ==========================================
    # 2. WORD SEARCH PROBLEMS
    # ==========================================
    
    def find_words_in_board(self, board: List[List[str]], words: List[str]) -> List[str]:
        """LC 212: Word Search II"""
        if not board or not board[0]:
            return []
        
        # Build trie
        root = TrieNode()
        for word in words:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end_of_word = True
            node.word = word
        
        result = []
        m, n = len(board), len(board[0])
        
        def dfs(i: int, j: int, node: TrieNode):
            if i < 0 or i >= m or j < 0 or j >= n:
                return
            
            char = board[i][j]
            if char not in node.children:
                return
            
            node = node.children[char]
            
            if node.is_end_of_word:
                result.append(node.word)
                node.is_end_of_word = False  # Avoid duplicates
            
            # Mark as visited
            board[i][j] = '#'
            
            # Explore all 4 directions
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                dfs(i + di, j + dj, node)
            
            # Restore
            board[i][j] = char
        
        for i in range(m):
            for j in range(n):
                dfs(i, j, root)
        
        return result
    
    def word_squares(self, words: List[str]) -> List[List[str]]:
        """LC 425: Word Squares"""
        if not words:
            return []
        
        n = len(words[0])
        
        # Build prefix to words mapping
        prefix_map = {}
        for word in words:
            for i in range(n + 1):
                prefix = word[:i]
                if prefix not in prefix_map:
                    prefix_map[prefix] = []
                prefix_map[prefix].append(word)
        
        result = []
        
        def backtrack(square: List[str]):
            if len(square) == n:
                result.append(square[:])
                return
            
            # Get prefix for next word
            pos = len(square)
            prefix = ''.join(square[i][pos] for i in range(pos))
            
            # Try all words with this prefix
            for word in prefix_map.get(prefix, []):
                square.append(word)
                backtrack(square)
                square.pop()
        
        backtrack([])
        return result
    
    # ==========================================
    # 3. AUTO-COMPLETE AND SUGGESTIONS
    # ==========================================
    
    def auto_complete(self, prefix: str, limit: int = 10) -> List[str]:
        """Get auto-complete suggestions"""
        node = self.root
        
        # Navigate to prefix
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        # DFS to find all words with this prefix
        suggestions = []
        
        def dfs(node: TrieNode, current_word: str):
            if len(suggestions) >= limit:
                return
            
            if node.is_end_of_word:
                suggestions.append(current_word)
            
            for char, child in node.children.items():
                dfs(child, current_word + char)
        
        dfs(node, prefix)
        return suggestions
    
    def search_suggestions_system(self, products: List[str], searchWord: str) -> List[List[str]]:
        """LC 1268: Search Suggestions System"""
        products.sort()
        result = []
        
        for i in range(len(searchWord)):
            prefix = searchWord[:i+1]
            suggestions = []
            
            for product in products:
                if product.startswith(prefix):
                    suggestions.append(product)
                    if len(suggestions) == 3:
                        break
            
            result.append(suggestions)
        
        return result
    
    # ==========================================
    # 4. PALINDROME PROBLEMS WITH TRIE
    # ==========================================
    
    def palindrome_pairs_trie(self, words: List[str]) -> List[List[int]]:
        """LC 336: Palindrome Pairs using Trie"""
        def is_palindrome(s: str) -> bool:
            return s == s[::-1]
        
        # Build trie with reversed words
        root = TrieNode()
        for i, word in enumerate(words):
            node = root
            for char in reversed(word):
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end_of_word = True
            node.word = i
        
        result = []
        
        def search_palindrome(node: TrieNode, word: str, index: int, 
                            start: int, path: List[int]):
            if node.is_end_of_word and index != node.word:
                # Check if remaining part is palindrome
                remaining = word[start:]
                if is_palindrome(remaining):
                    result.append([index, node.word])
            
            if start >= len(word):
                return
            
            char = word[start]
            if char in node.children:
                search_palindrome(node.children[char], word, index, 
                                start + 1, path + [char])
        
        for i, word in enumerate(words):
            search_palindrome(root, word, i, 0, [])
        
        return result
    
    # ==========================================
    # 5. STREAM OF CHARACTERS
    # ==========================================
    
    def stream_checker_init(self, words: List[str]):
        """LC 1032: Stream of Characters - Initialize"""
        self.trie_root = TrieNode()
        self.stream = []
        
        # Build trie with reversed words
        for word in words:
            node = self.trie_root
            for char in reversed(word):
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end_of_word = True
    
    def stream_checker_query(self, letter: str) -> bool:
        """Query for stream checker"""
        self.stream.append(letter)
        node = self.trie_root
        
        # Check stream backwards
        for i in range(len(self.stream) - 1, -1, -1):
            char = self.stream[i]
            if char not in node.children:
                break
            node = node.children[char]
            if node.is_end_of_word:
                return True
        
        return False
    
    # ==========================================
    # 6. ADVANCED TRIE APPLICATIONS
    # ==========================================
    
    def replace_words(self, dictionary: List[str], sentence: str) -> str:
        """LC 648: Replace Words"""
        # Build trie with dictionary
        root = TrieNode()
        for word in dictionary:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end_of_word = True
            node.word = word
        
        def find_root(word: str) -> str:
            node = root
            for char in word:
                if char not in node.children:
                    return word
                node = node.children[char]
                if node.is_end_of_word:
                    return node.word
            return word
        
        words = sentence.split()
        return ' '.join(find_root(word) for word in words)
    
    def longest_word_in_dictionary(self, words: List[str]) -> str:
        """LC 720: Longest Word in Dictionary"""
        # Build trie
        root = TrieNode()
        for word in words:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end_of_word = True
            node.word = word
        
        def dfs(node: TrieNode) -> str:
            if not node.is_end_of_word and node != root:
                return ""
            
            longest = node.word if node.word else ""
            
            for child in node.children.values():
                if child.is_end_of_word:
                    candidate = dfs(child)
                    if len(candidate) > len(longest) or \
                       (len(candidate) == len(longest) and candidate < longest):
                        longest = candidate
            
            return longest
        
        return dfs(root)
    
    def map_sum_pairs(self, key: str, val: int) -> None:
        """LC 677: Map Sum Pairs - Insert"""
        node = self.root
        for char in key:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.count = val
    
    def map_sum_query(self, prefix: str) -> int:
        """LC 677: Map Sum Pairs - Sum"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return 0
            node = node.children[char]
        
        def dfs(node: TrieNode) -> int:
            total = node.count if node.is_end_of_word else 0
            for child in node.children.values():
                total += dfs(child)
            return total
        
        return dfs(node)

# Test Examples
def run_examples():
    sta = StringTrieApplications()
    
    print("=== TRIE APPLICATIONS EXAMPLES ===\n")
    
    # Basic operations
    print("1. BASIC TRIE OPERATIONS:")
    words = ["apple", "app", "apricot", "banana"]
    for word in words:
        sta.insert(word)
    
    print(f"Search 'app': {sta.search('app')}")
    print(f"Starts with 'ap': {sta.starts_with('ap')}")
    print(f"Count words with prefix 'ap': {sta.count_words_with_prefix('ap')}")
    
    # Auto-complete
    print("\n2. AUTO-COMPLETE:")
    suggestions = sta.auto_complete("ap", 3)
    print(f"Auto-complete suggestions for 'ap': {suggestions}")
    
    # Search suggestions system
    products = ["mobile", "mouse", "moneypot", "monitor", "mousepad"]
    search_word = "mouse"
    result = sta.search_suggestions_system(products, search_word)
    print(f"Search suggestions for '{search_word}': {result}")
    
    # Replace words
    dictionary = ["cat", "bat", "rat"]
    sentence = "the cattle was rattled by the battery"
    replaced = sta.replace_words(dictionary, sentence)
    print(f"Replace words: '{replaced}'")
    
    # Word search in board
    board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]]
    words_to_find = ["oath","pea","eat","rain"]
    found_words = sta.find_words_in_board(board, words_to_find)
    print(f"Words found in board: {found_words}")

if __name__ == "__main__":
    run_examples() 