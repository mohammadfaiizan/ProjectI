"""
Game Theory String Games - Multiple Approaches
Difficulty: Hard

Game theory problems involving strings and trie structures
for competitive programming.

Problems:
1. String Nim Game
2. Word Formation Game
3. Pattern Blocking Game
4. Lexicographic Game
5. Trie Building Competition
"""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.winner = None  # For game state memoization
        self.nim_value = 0

class StringGameTheory:
    
    def __init__(self):
        self.root = TrieNode()
        self.memo = {}
    
    def string_nim_game(self, strings: List[str]) -> bool:
        """
        Determine winner of string Nim game
        Players take turns removing characters, last player wins
        Time: O(sum(|strings|))
        Space: O(sum(|strings|))
        """
        # Calculate nim value for each string
        total_nim = 0
        
        for string in strings:
            string_nim = len(string) % 2  # Simple nim value
            total_nim ^= string_nim
        
        # First player wins if total nim value is non-zero
        return total_nim != 0
    
    def word_formation_game(self, letters: List[str], dictionary: List[str]) -> bool:
        """
        Game where players form words from available letters
        Time: O(|dictionary| * max_word_length)
        Space: O(trie_size)
        """
        # Build dictionary trie
        dict_trie = TrieNode()
        
        for word in dictionary:
            node = dict_trie
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end = True
        
        # Count available letters
        letter_count = defaultdict(int)
        for letter in letters:
            letter_count[letter] += 1
        
        # Check if first player can form any word
        def can_form_word(node: TrieNode, available: Dict[str, int], depth: int = 0) -> bool:
            if node.is_end and depth > 0:
                return True
            
            for char, child in node.children.items():
                if available.get(char, 0) > 0:
                    available[char] -= 1
                    if can_form_word(child, available, depth + 1):
                        available[char] += 1
                        return True
                    available[char] += 1
            
            return False
        
        return can_form_word(dict_trie, letter_count)
    
    def pattern_blocking_game(self, text: str, patterns: List[str]) -> Dict[str, bool]:
        """
        Game where players try to complete/block patterns
        Time: O(|text| * sum(|patterns|))
        Space: O(sum(|patterns|))
        """
        # Build pattern trie
        pattern_trie = TrieNode()
        
        for pattern in patterns:
            node = pattern_trie
            for char in pattern:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end = True
        
        # Analyze each position in text
        game_states = {}
        
        for start_pos in range(len(text)):
            node = pattern_trie
            can_complete = False
            
            for end_pos in range(start_pos, len(text)):
                char = text[end_pos]
                
                if char not in node.children:
                    break
                
                node = node.children[char]
                
                if node.is_end:
                    can_complete = True
                    pattern = text[start_pos:end_pos + 1]
                    game_states[pattern] = True
        
        # Determine winning strategy
        result = {}
        for pattern in patterns:
            result[pattern] = pattern in game_states
        
        return result
    
    def lexicographic_game(self, strings: List[str]) -> Tuple[str, bool]:
        """
        Game where players choose strings to minimize/maximize lexicographic order
        Time: O(n * log n * max_length)
        Space: O(n)
        """
        # Sort strings lexicographically
        sorted_strings = sorted(strings)
        
        # First player (minimizer) chooses smallest
        # Second player (maximizer) chooses largest from remaining
        
        game_sequence = []
        remaining = sorted_strings[:]
        turn = 0  # 0 = minimizer, 1 = maximizer
        
        while remaining:
            if turn == 0:  # Minimizer's turn
                chosen = remaining.pop(0)  # Choose smallest
            else:  # Maximizer's turn
                chosen = remaining.pop()   # Choose largest
            
            game_sequence.append(chosen)
            turn = 1 - turn
        
        # Determine final lexicographic order
        final_string = ''.join(game_sequence)
        first_player_wins = final_string == ''.join(sorted(strings))
        
        return final_string, first_player_wins
    
    def trie_building_competition(self, word_sets: List[List[str]]) -> List[int]:
        """
        Competition where players build tries from word sets
        Time: O(sum(total_word_lengths))
        Space: O(max_trie_size)
        """
        scores = []
        
        for word_set in word_sets:
            # Build trie for this word set
            trie = TrieNode()
            
            for word in word_set:
                node = trie
                for char in word:
                    if char not in node.children:
                        node.children[char] = TrieNode()
                    node = node.children[char]
                node.is_end = True
            
            # Calculate score based on trie structure
            score = self._calculate_trie_score(trie)
            scores.append(score)
        
        return scores
    
    def optimal_string_game_strategy(self, game_state: str, turn: int) -> Tuple[bool, str]:
        """
        Determine optimal strategy for string manipulation game
        Time: O(game_states)
        Space: O(game_states)
        """
        state_key = (game_state, turn)
        
        if state_key in self.memo:
            return self.memo[state_key]
        
        # Base case: empty string
        if not game_state:
            # Player whose turn it is loses (no moves available)
            result = (False, "")
            self.memo[state_key] = result
            return result
        
        # Try all possible moves
        best_outcome = False
        best_move = ""
        
        # Move 1: Remove first character
        if len(game_state) > 0:
            new_state = game_state[1:]
            opponent_wins, _ = self.optimal_string_game_strategy(new_state, 1 - turn)
            
            if not opponent_wins:  # If opponent loses, current player wins
                best_outcome = True
                best_move = f"Remove first: '{game_state[0]}'"
        
        # Move 2: Remove last character
        if len(game_state) > 0:
            new_state = game_state[:-1]
            opponent_wins, _ = self.optimal_string_game_strategy(new_state, 1 - turn)
            
            if not opponent_wins and not best_outcome:
                best_outcome = True
                best_move = f"Remove last: '{game_state[-1]}'"
        
        # Move 3: Split string (if length > 1)
        if len(game_state) > 1:
            for i in range(1, len(game_state)):
                left_part = game_state[:i]
                right_part = game_state[i:]
                
                # Opponent chooses which part to play
                left_wins, _ = self.optimal_string_game_strategy(left_part, 1 - turn)
                right_wins, _ = self.optimal_string_game_strategy(right_part, 1 - turn)
                
                # Opponent will choose the part where they win
                if not left_wins and not right_wins:
                    best_outcome = True
                    best_move = f"Split at {i}: '{left_part}' | '{right_part}'"
                    break
        
        result = (best_outcome, best_move)
        self.memo[state_key] = result
        return result
    
    def _calculate_trie_score(self, node: TrieNode) -> int:
        """Calculate score for trie structure"""
        score = 0
        
        # Points for nodes
        score += 1
        
        # Bonus for word endings
        if node.is_end:
            score += 5
        
        # Bonus for branching (multiple children)
        if len(node.children) > 1:
            score += len(node.children) * 2
        
        # Recursive score for children
        for child in node.children.values():
            score += self._calculate_trie_score(child)
        
        return score


def test_string_nim():
    """Test string Nim game"""
    print("=== Testing String Nim Game ===")
    
    game = StringGameTheory()
    
    test_cases = [
        (["abc", "def"], "Two strings of equal length"),
        (["a", "bb", "ccc"], "Strings of different lengths"),
        (["hello", "world", "test"], "Three strings"),
    ]
    
    for strings, description in test_cases:
        first_player_wins = game.string_nim_game(strings)
        print(f"{description}: {strings}")
        print(f"  First player {'wins' if first_player_wins else 'loses'}")

def test_word_formation():
    """Test word formation game"""
    print("\n=== Testing Word Formation Game ===")
    
    game = StringGameTheory()
    
    letters = ['a', 'b', 'c', 'd', 'e']
    dictionary = ['abc', 'bcd', 'ace', 'xyz']
    
    print(f"Available letters: {letters}")
    print(f"Dictionary: {dictionary}")
    
    can_win = game.word_formation_game(letters, dictionary)
    print(f"First player {'can' if can_win else 'cannot'} form a word")

def test_pattern_blocking():
    """Test pattern blocking game"""
    print("\n=== Testing Pattern Blocking Game ===")
    
    game = StringGameTheory()
    
    text = "abcdefabc"
    patterns = ["abc", "def", "xyz"]
    
    print(f"Text: '{text}'")
    print(f"Patterns: {patterns}")
    
    results = game.pattern_blocking_game(text, patterns)
    
    print("Pattern completion possibilities:")
    for pattern, can_complete in results.items():
        print(f"  '{pattern}': {'Possible' if can_complete else 'Impossible'}")

def test_lexicographic_game():
    """Test lexicographic game"""
    print("\n=== Testing Lexicographic Game ===")
    
    game = StringGameTheory()
    
    strings = ["cat", "dog", "apple", "zoo"]
    
    print(f"Strings: {strings}")
    
    final_string, first_player_wins = game.lexicographic_game(strings)
    
    print(f"Game sequence result: '{final_string}'")
    print(f"First player (minimizer) {'wins' if first_player_wins else 'loses'}")

def test_trie_competition():
    """Test trie building competition"""
    print("\n=== Testing Trie Building Competition ===")
    
    game = StringGameTheory()
    
    word_sets = [
        ["cat", "car", "card"],
        ["dog", "door", "down"],
        ["apple", "app", "application"],
        ["test", "testing", "tester"]
    ]
    
    print("Word sets:")
    for i, word_set in enumerate(word_sets):
        print(f"  Player {i+1}: {word_set}")
    
    scores = game.trie_building_competition(word_sets)
    
    print("\nScores:")
    for i, score in enumerate(scores):
        print(f"  Player {i+1}: {score} points")
    
    winner = scores.index(max(scores))
    print(f"\nWinner: Player {winner + 1}")

def test_optimal_strategy():
    """Test optimal game strategy"""
    print("\n=== Testing Optimal Game Strategy ===")
    
    game = StringGameTheory()
    
    game_states = ["abc", "abcd", "hello"]
    
    for state in game_states:
        print(f"\nGame state: '{state}'")
        
        can_win, best_move = game.optimal_string_game_strategy(state, 0)
        
        print(f"First player {'wins' if can_win else 'loses'}")
        if best_move:
            print(f"Best move: {best_move}")

if __name__ == "__main__":
    test_string_nim()
    test_word_formation()
    test_pattern_blocking()
    test_lexicographic_game()
    test_trie_competition()
    test_optimal_strategy()
