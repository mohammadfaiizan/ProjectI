"""
Advanced Trie Operations and Algorithms
=======================================

Topics: Complex queries, pattern matching, word games, advanced algorithms
Companies: Google, Amazon, Microsoft, Facebook, Apple, Netflix
Difficulty: Hard to Expert
Time Complexity: Varies by operation (O(m) to O(n*m))
Space Complexity: O(n*m) for trie storage plus operation-specific space
"""

from typing import List, Tuple, Optional, Dict, Any, Set, Callable
from collections import defaultdict, deque
import heapq

class AdvancedTrieOperations:
    
    def __init__(self):
        """Initialize with advanced operations tracking"""
        self.operation_count = 0
        self.algorithm_stats = {}
    
    # ==========================================
    # 1. WILDCARD AND PATTERN MATCHING
    # ==========================================
    
    def demonstrate_wildcard_matching(self) -> None:
        """
        Demonstrate wildcard pattern matching in trie
        
        Supports patterns with:
        - '.' for any single character
        - '*' for any sequence of characters
        - Multiple wildcards in same pattern
        """
        print("=== WILDCARD PATTERN MATCHING ===")
        print("Pattern support: '.' (any char), '*' (any sequence)")
        print()
        
        trie = WildcardTrie()
        
        # Build dictionary
        words = [
            "cat", "car", "card", "care", "careful", "cats", "cut", "cute",
            "dog", "dodge", "door", "down", "download", "apple", "apply", "application"
        ]
        
        print("Building dictionary:")
        for word in words:
            trie.insert(word)
        print(f"  Inserted {len(words)} words")
        print()
        
        # Test wildcard patterns
        patterns = [
            "c.t",      # cat, cut
            "c*r",      # car, care, careful, etc.
            "a**e",     # apple, etc.
            "do*",      # dog, dodge, door, down, download
            "*.e",      # care, cute, apple, etc.
            "*app*",    # apple, apply, application
            "xyz"       # no matches
        ]
        
        print("Wildcard pattern matching:")
        for pattern in patterns:
            matches = trie.find_words_matching_pattern(pattern)
            print(f"  Pattern '{pattern}': {matches}")
        print()


class WildcardTrieNode:
    """Trie node for wildcard pattern matching"""
    
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.word = None


class WildcardTrie:
    """
    Trie with wildcard pattern matching support
    
    Supports '.' for any character and '*' for any sequence
    """
    
    def __init__(self):
        self.root = WildcardTrieNode()
    
    def insert(self, word: str) -> None:
        """Insert word into trie"""
        current = self.root
        
        for char in word:
            if char not in current.children:
                current.children[char] = WildcardTrieNode()
            current = current.children[char]
        
        current.is_end_of_word = True
        current.word = word
    
    def find_words_matching_pattern(self, pattern: str) -> List[str]:
        """
        Find all words matching wildcard pattern
        
        Time: O(n * m) worst case where n is words, m is pattern length
        Space: O(h) for recursion stack
        """
        results = []
        
        def _search_pattern(node: WildcardTrieNode, pattern: str, pattern_idx: int, current_word: str):
            # Base case: reached end of pattern
            if pattern_idx == len(pattern):
                if node.is_end_of_word:
                    results.append(current_word)
                return
            
            char = pattern[pattern_idx]
            
            if char == '.':
                # '.' matches any single character
                for child_char, child_node in node.children.items():
                    _search_pattern(child_node, pattern, pattern_idx + 1, current_word + child_char)
            
            elif char == '*':
                # '*' matches any sequence (including empty)
                # Case 1: Match empty sequence (skip '*')
                _search_pattern(node, pattern, pattern_idx + 1, current_word)
                
                # Case 2: Match one or more characters
                for child_char, child_node in node.children.items():
                    _search_pattern(child_node, pattern, pattern_idx, current_word + child_char)
            
            else:
                # Regular character - must match exactly
                if char in node.children:
                    _search_pattern(node.children[char], pattern, pattern_idx + 1, current_word + char)
        
        _search_pattern(self.root, pattern, 0, "")
        return sorted(results)
    
    def find_words_with_regex(self, regex_pattern: str) -> List[str]:
        """
        Advanced regex-like pattern matching
        
        Supports:
        - [abc] for character sets
        - [a-z] for ranges
        - ? for optional character
        """
        # Simplified regex implementation
        results = []
        
        def _is_char_in_set(char: str, char_set: str) -> bool:
            """Check if character matches character set [abc] or [a-z]"""
            if '-' in char_set and len(char_set) == 3:
                # Range like [a-z]
                start, end = char_set[0], char_set[2]
                return start <= char <= end
            else:
                # Explicit set like [abc]
                return char in char_set
        
        def _search_regex(node: WildcardTrieNode, pattern: str, pattern_idx: int, current_word: str):
            if pattern_idx == len(pattern):
                if node.is_end_of_word:
                    results.append(current_word)
                return
            
            char = pattern[pattern_idx]
            
            if char == '[':
                # Find closing bracket
                close_bracket = pattern.find(']', pattern_idx)
                if close_bracket == -1:
                    return
                
                char_set = pattern[pattern_idx + 1:close_bracket]
                
                for child_char, child_node in node.children.items():
                    if _is_char_in_set(child_char, char_set):
                        _search_regex(child_node, pattern, close_bracket + 1, current_word + child_char)
            
            elif char == '?':
                # Optional character - can skip or match one
                # Skip option
                _search_regex(node, pattern, pattern_idx + 1, current_word)
                
                # Match one character option
                for child_char, child_node in node.children.items():
                    _search_regex(child_node, pattern, pattern_idx + 1, current_word + child_char)
            
            else:
                # Regular character
                if char in node.children:
                    _search_regex(node.children[char], pattern, pattern_idx + 1, current_word + char)
        
        _search_regex(self.root, regex_pattern, 0, "")
        return sorted(results)


# ==========================================
# 2. WORD GAMES AND PUZZLES
# ==========================================

class WordGameSolver:
    """
    Solves various word games using trie data structure
    
    Games supported:
    - Boggle board word finding
    - Scrabble word validation and scoring
    - Word ladder shortest path
    - Anagram detection and grouping
    """
    
    def __init__(self, dictionary: List[str]):
        self.trie = WildcardTrie()
        self.word_set = set()
        
        print("Building word game dictionary...")
        for word in dictionary:
            self.trie.insert(word.lower())
            self.word_set.add(word.lower())
        print(f"  Loaded {len(dictionary)} words")
    
    def solve_boggle(self, board: List[List[str]]) -> List[str]:
        """
        Find all valid words in Boggle board
        
        Company: Google, Facebook (word games)
        Difficulty: Hard
        Time: O(M * N * 4^L) where M,N are board dimensions, L is max word length
        Space: O(L) for recursion stack
        """
        print("=== BOGGLE SOLVER ===")
        print("Board:")
        for row in board:
            print(f"  {' '.join(row)}")
        print()
        
        rows, cols = len(board), len(board[0])
        found_words = set()
        
        def _dfs_boggle(r: int, c: int, node: WildcardTrieNode, path: str, visited: Set[Tuple[int, int]]):
            """DFS to find words starting from position (r, c)"""
            if node.is_end_of_word and len(path) >= 3:  # Minimum word length
                found_words.add(path)
            
            # Explore all 8 directions
            directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                if (0 <= nr < rows and 0 <= nc < cols and 
                    (nr, nc) not in visited):
                    
                    char = board[nr][nc].lower()
                    if char in node.children:
                        visited.add((nr, nc))
                        _dfs_boggle(nr, nc, node.children[char], path + char, visited)
                        visited.remove((nr, nc))
        
        # Try starting from each cell
        for r in range(rows):
            for c in range(cols):
                char = board[r][c].lower()
                if char in self.trie.root.children:
                    visited = {(r, c)}
                    _dfs_boggle(r, c, self.trie.root.children[char], char, visited)
        
        words_found = sorted(list(found_words))
        print(f"Found {len(words_found)} words:")
        for i, word in enumerate(words_found):
            print(f"  {i+1:2d}. {word}")
        
        return words_found
    
    def calculate_scrabble_score(self, word: str) -> int:
        """
        Calculate Scrabble score for a word
        
        Uses standard Scrabble letter values
        """
        letter_scores = {
            'a': 1, 'b': 3, 'c': 3, 'd': 2, 'e': 1, 'f': 4, 'g': 2, 'h': 4,
            'i': 1, 'j': 8, 'k': 5, 'l': 1, 'm': 3, 'n': 1, 'o': 1, 'p': 3,
            'q': 10, 'r': 1, 's': 1, 't': 1, 'u': 1, 'v': 4, 'w': 4, 'x': 8,
            'y': 4, 'z': 10
        }
        
        return sum(letter_scores.get(char.lower(), 0) for char in word)
    
    def find_high_scoring_words(self, letters: str, min_length: int = 3) -> List[Tuple[str, int]]:
        """
        Find highest-scoring valid words from given letters
        
        Useful for Scrabble strategy
        """
        print(f"=== HIGH-SCORING WORDS FROM LETTERS: '{letters}' ===")
        
        letter_count = Counter(letters.lower())
        valid_words = []
        
        def _can_form_word(word: str) -> bool:
            """Check if word can be formed from available letters"""
            word_count = Counter(word.lower())
            for char, count in word_count.items():
                if letter_count.get(char, 0) < count:
                    return False
            return True
        
        # Check all words in dictionary
        for word in self.word_set:
            if len(word) >= min_length and _can_form_word(word):
                score = self.calculate_scrabble_score(word)
                valid_words.append((word, score))
        
        # Sort by score (descending)
        valid_words.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Found {len(valid_words)} valid words:")
        for i, (word, score) in enumerate(valid_words[:10]):  # Top 10
            print(f"  {i+1:2d}. {word:<12} ({score} points)")
        
        return valid_words
    
    def word_ladder_shortest_path(self, start_word: str, end_word: str) -> List[str]:
        """
        Find shortest transformation path between two words
        
        Company: Amazon, Google
        Difficulty: Hard
        Time: O(M^2 * N) where M is word length, N is dictionary size
        Space: O(M * N)
        
        Each step changes exactly one letter to form valid word
        """
        print(f"=== WORD LADDER: '{start_word}' → '{end_word}' ===")
        
        if len(start_word) != len(end_word):
            print("Words must be same length")
            return []
        
        if start_word == end_word:
            return [start_word]
        
        start_word = start_word.lower()
        end_word = end_word.lower()
        
        if end_word not in self.word_set:
            print(f"Target word '{end_word}' not in dictionary")
            return []
        
        # BFS to find shortest path
        queue = deque([(start_word, [start_word])])
        visited = {start_word}
        
        def _get_neighbors(word: str) -> List[str]:
            """Get all valid words that differ by exactly one character"""
            neighbors = []
            for i in range(len(word)):
                for char in 'abcdefghijklmnopqrstuvwxyz':
                    if char != word[i]:
                        neighbor = word[:i] + char + word[i+1:]
                        if neighbor in self.word_set and neighbor not in visited:
                            neighbors.append(neighbor)
            return neighbors
        
        step = 0
        while queue:
            step += 1
            print(f"  Step {step}: Exploring {len(queue)} words")
            
            current_word, path = queue.popleft()
            
            for neighbor in _get_neighbors(current_word):
                new_path = path + [neighbor]
                
                if neighbor == end_word:
                    print(f"  Found path in {len(new_path)} steps: {' → '.join(new_path)}")
                    return new_path
                
                visited.add(neighbor)
                queue.append((neighbor, new_path))
        
        print("  No transformation path found")
        return []


# ==========================================
# 3. STRING ALGORITHMS WITH TRIE
# ==========================================

class StringAlgorithmsTrie:
    """
    Advanced string algorithms using trie data structure
    
    Algorithms:
    - Longest common prefix
    - String compression using trie
    - Multiple pattern matching (Aho-Corasick style)
    - Suffix-based operations
    """
    
    def __init__(self):
        self.trie = WildcardTrie()
    
    def longest_common_prefix_multiple(self, strings: List[str]) -> str:
        """
        Find longest common prefix among multiple strings using trie
        
        Company: Amazon, Microsoft
        Difficulty: Medium
        Time: O(S) where S is sum of all string lengths
        Space: O(S)
        """
        print(f"=== LONGEST COMMON PREFIX ===")
        print(f"Input strings: {strings}")
        
        if not strings:
            return ""
        
        # Build trie with all strings
        for string in strings:
            self.trie.insert(string)
        
        # Find longest path with only one child at each level
        current = self.trie.root
        prefix = ""
        
        while len(current.children) == 1:
            char = next(iter(current.children.keys()))
            child = current.children[char]
            
            # Check if all strings pass through this node
            # (simplified - in real implementation, track string count per node)
            prefix += char
            current = child
            
            # Stop if we've reached end of any string
            if current.is_end_of_word:
                break
        
        print(f"Longest common prefix: '{prefix}'")
        return prefix
    
    def compress_strings_with_trie(self, strings: List[str]) -> Dict[str, Any]:
        """
        Demonstrate string compression using trie structure
        
        Shows how trie naturally compresses common prefixes
        """
        print(f"=== STRING COMPRESSION WITH TRIE ===")
        print(f"Input strings: {strings}")
        
        # Build trie
        trie = CompressionTrie()
        total_chars = sum(len(s) for s in strings)
        
        for string in strings:
            trie.insert(string)
        
        stats = trie.get_compression_stats()
        
        print(f"Compression analysis:")
        print(f"  Original total characters: {total_chars}")
        print(f"  Trie nodes created: {stats['nodes']}")
        print(f"  Characters stored in trie: {stats['stored_chars']}")
        print(f"  Compression ratio: {stats['stored_chars'] / total_chars:.2%}")
        print(f"  Space saved: {total_chars - stats['stored_chars']} characters")
        
        return stats
    
    def multiple_pattern_search(self, text: str, patterns: List[str]) -> Dict[str, List[int]]:
        """
        Search multiple patterns simultaneously (simplified Aho-Corasick)
        
        Company: Google, Amazon (text processing)
        Difficulty: Hard
        Time: O(n + m + z) where n=text length, m=patterns total length, z=matches
        Space: O(m)
        """
        print(f"=== MULTIPLE PATTERN SEARCH ===")
        print(f"Text: '{text}'")
        print(f"Patterns: {patterns}")
        
        # Build pattern trie
        pattern_trie = WildcardTrie()
        for pattern in patterns:
            pattern_trie.insert(pattern)
        
        matches = defaultdict(list)
        
        # Search for all patterns at each position
        for i in range(len(text)):
            # Try to match patterns starting at position i
            current = pattern_trie.root
            
            for j in range(i, len(text)):
                char = text[j]
                if char not in current.children:
                    break
                
                current = current.children[char]
                
                if current.is_end_of_word:
                    pattern = text[i:j+1]
                    matches[pattern].append(i)
                    print(f"  Found '{pattern}' at position {i}")
        
        print(f"Search results:")
        for pattern, positions in matches.items():
            print(f"  '{pattern}': positions {positions}")
        
        return dict(matches)


class CompressionTrie:
    """
    Trie for analyzing string compression potential
    """
    
    def __init__(self):
        self.root = CompressionTrieNode()
        self.node_count = 1  # Count root
        self.total_chars_stored = 0
    
    def insert(self, string: str) -> None:
        """Insert string and track compression metrics"""
        current = self.root
        
        for char in string:
            if char not in current.children:
                current.children[char] = CompressionTrieNode()
                self.node_count += 1
                self.total_chars_stored += 1
            current = current.children[char]
        
        current.is_end_of_word = True
    
    def get_compression_stats(self) -> Dict[str, int]:
        """Get compression statistics"""
        return {
            'nodes': self.node_count,
            'stored_chars': self.total_chars_stored
        }


class CompressionTrieNode:
    """Node for compression analysis trie"""
    
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

from collections import Counter

def demonstrate_advanced_trie_operations():
    """Demonstrate all advanced trie operations"""
    print("=== ADVANCED TRIE OPERATIONS DEMONSTRATION ===\n")
    
    advanced_ops = AdvancedTrieOperations()
    
    # 1. Wildcard pattern matching
    advanced_ops.demonstrate_wildcard_matching()
    print("\n" + "="*60 + "\n")
    
    # 2. Word games
    print("=== WORD GAME APPLICATIONS ===")
    
    # Sample dictionary for games
    game_dictionary = [
        "cat", "car", "card", "care", "careful", "cats", "cut", "cute",
        "dog", "dodge", "door", "down", "download", "apple", "apply", "application",
        "bat", "ball", "call", "tall", "wall", "walk", "talk", "take", "make",
        "lake", "bake", "cake", "face", "race", "place", "grace"
    ]
    
    word_games = WordGameSolver(game_dictionary)
    
    # Boggle solver
    boggle_board = [
        ['c', 'a', 't'],
        ['o', 'r', 'e'],
        ['d', 'o', 'g']
    ]
    word_games.solve_boggle(boggle_board)
    print("\n" + "-"*40 + "\n")
    
    # Scrabble word finder
    available_letters = "careto"
    word_games.find_high_scoring_words(available_letters)
    print("\n" + "-"*40 + "\n")
    
    # Word ladder
    word_games.word_ladder_shortest_path("cat", "dog")
    print("\n" + "="*60 + "\n")
    
    # 3. String algorithms
    print("=== STRING ALGORITHMS WITH TRIE ===")
    
    string_algos = StringAlgorithmsTrie()
    
    # Longest common prefix
    prefix_strings = ["interspecies", "interstellar", "interstate", "intermediate"]
    string_algos.longest_common_prefix_multiple(prefix_strings)
    print("\n" + "-"*40 + "\n")
    
    # String compression
    compression_strings = ["apple", "app", "application", "apply", "banana", "band", "bandana"]
    string_algos.compress_strings_with_trie(compression_strings)
    print("\n" + "-"*40 + "\n")
    
    # Multiple pattern search
    search_text = "the quick brown fox jumps over the lazy dog"
    search_patterns = ["the", "fox", "dog", "quick", "lazy"]
    string_algos.multiple_pattern_search(search_text, search_patterns)
    print("\n" + "="*60 + "\n")
    
    # 4. Advanced regex-like patterns
    print("=== ADVANCED PATTERN MATCHING ===")
    
    regex_trie = WildcardTrie()
    regex_words = ["cat", "bat", "rat", "hat", "cap", "map", "tap", "sap", "can", "man", "ban", "fan"]
    
    for word in regex_words:
        regex_trie.insert(word)
    
    regex_patterns = [
        "[cbr]at",  # cat, bat, rat
        "[a-z]ap",  # cap, map, tap, sap
        "[cmf]an"   # can, man, fan
    ]
    
    print("Advanced regex-like pattern matching:")
    for pattern in regex_patterns:
        matches = regex_trie.find_words_with_regex(pattern)
        print(f"  Pattern '{pattern}': {matches}")


if __name__ == "__main__":
    demonstrate_advanced_trie_operations()
    
    print("\n=== ADVANCED TRIE OPERATIONS MASTERY GUIDE ===")
    
    print("\n🎯 ADVANCED OPERATION CATEGORIES:")
    print("• Pattern Matching: Wildcard, regex-like, multiple patterns")
    print("• Word Games: Boggle, Scrabble, word ladder, anagrams")
    print("• String Algorithms: LCP, compression, suffix operations")
    print("• Search Optimization: Auto-complete, fuzzy matching")
    
    print("\n📊 COMPLEXITY ANALYSIS:")
    print("• Wildcard search: O(n * m) where n=words, m=pattern length")
    print("• Boggle solving: O(M * N * 4^L) where M,N=board, L=max word")
    print("• Word ladder: O(M^2 * N) where M=word length, N=dictionary")
    print("• Multiple patterns: O(n + m + z) where z=number of matches")
    
    print("\n⚡ OPTIMIZATION STRATEGIES:")
    print("• Use efficient backtracking with early termination")
    print("• Implement iterative solutions to avoid stack overflow")
    print("• Cache results for repeated pattern matching")
    print("• Use bit manipulation for character set operations")
    print("• Implement failure functions for advanced pattern matching")
    
    print("\n🎮 REAL-WORLD APPLICATIONS:")
    print("• Auto-complete: Search suggestions, IDE completion")
    print("• Spell Check: Dictionary lookup, correction suggestions")
    print("• Text Processing: Multiple pattern search, content filtering")
    print("• Game Development: Word games, puzzle solvers")
    print("• Bioinformatics: DNA sequence matching, pattern discovery")
    
    print("\n🔧 IMPLEMENTATION TIPS:")
    print("• Handle Unicode and special characters properly")
    print("• Implement efficient memory management for large tries")
    print("• Use appropriate data structures for specific alphabets")
    print("• Add comprehensive error handling and validation")
    print("• Consider parallel processing for large-scale operations")
    
    print("\n🎓 MASTERY CHECKLIST:")
    print("• Master wildcard and regex pattern implementation")
    print("• Understand game-solving algorithms and optimizations")
    print("• Learn advanced string processing techniques")
    print("• Practice with real-world datasets and constraints")
    print("• Study failure function and finite automata theory")
