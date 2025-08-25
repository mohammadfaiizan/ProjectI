"""
Word Ladder with Trie - Multiple Approaches
Difficulty: Medium

Given two words (beginWord and endWord), and a dictionary's word list, find all 
transformation sequences from beginWord to endWord, where:
1. Only one letter can be changed at a time
2. Each transformed word must exist in the word list
3. Use trie data structure to optimize the solution

This combines the classic Word Ladder problem with trie optimizations for:
- Efficient neighbor finding
- Word validation
- Path reconstruction
- Memory optimization

Related Problems:
- LeetCode 127: Word Ladder (shortest path)
- LeetCode 126: Word Ladder II (all shortest paths)
"""

from typing import List, Set, Dict, Optional, Tuple
from collections import deque, defaultdict
import string
import time

class TrieNode:
    """Trie node for word ladder optimization"""
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.word = ""

class WordLadderTrie:
    
    def __init__(self, word_list: List[str]):
        """
        Initialize trie with word list.
        
        Time: O(N * L) where N=number of words, L=word length
        Space: O(N * L)
        """
        self.root = TrieNode()
        self.word_set = set(word_list)
        self.word_length = len(word_list[0]) if word_list else 0
        
        # Build trie
        for word in word_list:
            self._insert(word)
    
    def _insert(self, word: str) -> None:
        """Insert word into trie"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True
        node.word = word
    
    def find_shortest_path_length(self, begin_word: str, end_word: str) -> int:
        """
        Approach 1: BFS with Trie for Neighbor Generation
        
        Use BFS to find shortest transformation sequence length.
        Trie helps generate valid neighbors efficiently.
        
        Time: O(N * L^2) where N=words, L=word length
        Space: O(N * L) for trie + O(N) for BFS
        """
        if end_word not in self.word_set:
            return 0
        
        if begin_word == end_word:
            return 1
        
        queue = deque([(begin_word, 1)])
        visited = {begin_word}
        
        while queue:
            current_word, length = queue.popleft()
            
            # Generate all valid neighbors using trie
            neighbors = self._get_trie_neighbors(current_word)
            
            for neighbor in neighbors:
                if neighbor == end_word:
                    return length + 1
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, length + 1))
        
        return 0
    
    def _get_trie_neighbors(self, word: str) -> List[str]:
        """Get all valid neighbors using trie traversal"""
        neighbors = []
        
        for i in range(len(word)):
            # Try replacing character at position i
            for char in string.ascii_lowercase:
                if char != word[i]:
                    new_word = word[:i] + char + word[i+1:]
                    if self._is_valid_word_trie(new_word):
                        neighbors.append(new_word)
        
        return neighbors
    
    def _is_valid_word_trie(self, word: str) -> bool:
        """Check if word exists using trie traversal"""
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_word
    
    def find_all_shortest_paths(self, begin_word: str, end_word: str) -> List[List[str]]:
        """
        Approach 2: Bidirectional BFS with Trie
        
        Use bidirectional BFS to find all shortest transformation paths.
        
        Time: O(N * L^2) with better constants
        Space: O(N * L)
        """
        if end_word not in self.word_set:
            return []
        
        if begin_word == end_word:
            return [[begin_word]]
        
        # Add begin_word to word set if not present
        extended_word_set = self.word_set | {begin_word}
        
        # Build adjacency graph using trie
        graph = self._build_adjacency_graph(extended_word_set)
        
        # Bidirectional BFS
        return self._bidirectional_bfs(begin_word, end_word, graph)
    
    def _build_adjacency_graph(self, word_set: Set[str]) -> Dict[str, List[str]]:
        """Build adjacency graph with trie optimization"""
        graph = defaultdict(list)
        
        # Group words by pattern (e.g., "h*t", "*ot", "ho*")
        pattern_dict = defaultdict(list)
        
        for word in word_set:
            for i in range(len(word)):
                pattern = word[:i] + '*' + word[i+1:]
                pattern_dict[pattern].append(word)
        
        # Build adjacency list
        for word in word_set:
            for i in range(len(word)):
                pattern = word[:i] + '*' + word[i+1:]
                for neighbor in pattern_dict[pattern]:
                    if neighbor != word:
                        graph[word].append(neighbor)
        
        return graph
    
    def _bidirectional_bfs(self, begin_word: str, end_word: str, 
                          graph: Dict[str, List[str]]) -> List[List[str]]:
        """Bidirectional BFS to find all shortest paths"""
        # Forward and backward queues
        forward = {begin_word: [[begin_word]]}
        backward = {end_word: [[end_word]]}
        
        direction = 1  # 1 for forward, -1 for backward
        result = []
        
        while forward and backward and not result:
            # Choose smaller queue to expand
            if len(forward) > len(backward):
                forward, backward = backward, forward
                direction *= -1
            
            # Expand current level
            current_level = forward
            forward = {}
            
            for word in current_level:
                for neighbor in graph[word]:
                    if neighbor in backward:
                        # Found connection
                        for path1 in current_level[word]:
                            for path2 in backward[neighbor]:
                                if direction == 1:
                                    result.append(path1 + path2[::-1])
                                else:
                                    result.append(path2 + path1[::-1])
                    
                    if neighbor not in forward:
                        forward[neighbor] = []
                    
                    for path in current_level[word]:
                        forward[neighbor].append(path + [neighbor])
        
        return result
    
    def find_shortest_path_optimized(self, begin_word: str, end_word: str) -> List[str]:
        """
        Approach 3: A* Search with Trie Heuristic
        
        Use A* search with edit distance heuristic and trie optimization.
        
        Time: O(N * L^2) with better practical performance
        Space: O(N * L)
        """
        if end_word not in self.word_set:
            return []
        
        if begin_word == end_word:
            return [begin_word]
        
        import heapq
        
        def heuristic(word: str) -> int:
            """Edit distance heuristic to end_word"""
            return sum(c1 != c2 for c1, c2 in zip(word, end_word))
        
        # Priority queue: (f_score, g_score, word, path)
        pq = [(heuristic(begin_word), 0, begin_word, [begin_word])]
        visited = {begin_word: 0}
        
        while pq:
            f_score, g_score, current_word, path = heapq.heappop(pq)
            
            if current_word == end_word:
                return path
            
            # Skip if we've found a better path to this word
            if current_word in visited and visited[current_word] < g_score:
                continue
            
            neighbors = self._get_trie_neighbors(current_word)
            
            for neighbor in neighbors:
                new_g_score = g_score + 1
                
                if neighbor not in visited or new_g_score < visited[neighbor]:
                    visited[neighbor] = new_g_score
                    f_score = new_g_score + heuristic(neighbor)
                    heapq.heappush(pq, (f_score, new_g_score, neighbor, path + [neighbor]))
        
        return []
    
    def find_paths_with_length_limit(self, begin_word: str, end_word: str, 
                                   max_length: int) -> List[List[str]]:
        """
        Approach 4: DFS with Trie and Length Limit
        
        Find all paths within given length limit using DFS.
        
        Time: O(exponential) but pruned by length limit
        Space: O(L * max_paths)
        """
        if end_word not in self.word_set:
            return []
        
        if begin_word == end_word:
            return [[begin_word]]
        
        result = []
        visited = set()
        
        def dfs(current_word: str, path: List[str]) -> None:
            if len(path) > max_length:
                return
            
            if current_word == end_word:
                result.append(path[:])
                return
            
            if current_word in visited:
                return
            
            visited.add(current_word)
            
            neighbors = self._get_trie_neighbors(current_word)
            for neighbor in neighbors:
                if neighbor not in visited:
                    path.append(neighbor)
                    dfs(neighbor, path)
                    path.pop()
            
            visited.remove(current_word)
        
        dfs(begin_word, [begin_word])
        return result
    
    def get_transformation_tree(self, begin_word: str) -> Dict[str, List[str]]:
        """
        Approach 5: Build Transformation Tree with Trie
        
        Build complete transformation tree from begin_word using trie.
        
        Time: O(N * L^2)
        Space: O(N * L)
        """
        if begin_word not in self.word_set:
            # Add temporarily
            temp_added = True
            self.word_set.add(begin_word)
            self._insert(begin_word)
        else:
            temp_added = False
        
        tree = defaultdict(list)
        queue = deque([begin_word])
        visited = {begin_word}
        
        while queue:
            current_word = queue.popleft()
            neighbors = self._get_trie_neighbors(current_word)
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    tree[current_word].append(neighbor)
                    queue.append(neighbor)
        
        # Clean up if we added begin_word temporarily
        if temp_added:
            self.word_set.remove(begin_word)
        
        return dict(tree)
    
    def find_paths_with_pattern(self, begin_word: str, end_word: str, 
                               pattern: str) -> List[List[str]]:
        """
        Approach 6: Pattern-Constrained Path Finding
        
        Find paths that satisfy additional pattern constraints.
        Pattern: '*' = any char, others must match exactly
        
        Time: O(exponential) with pattern pruning
        Space: O(max_paths * L)
        """
        if end_word not in self.word_set:
            return []
        
        if len(pattern) != len(begin_word):
            return []
        
        def matches_pattern(word: str, pos: int) -> bool:
            """Check if word at given position matches pattern"""
            if pos >= len(pattern):
                return True
            
            if pattern[pos] == '*':
                return True
            
            return pos < len(word) and word[pos] == pattern[pos]
        
        result = []
        
        def dfs_with_pattern(current_word: str, path: List[str], 
                           visited: Set[str]) -> None:
            if current_word == end_word:
                result.append(path[:])
                return
            
            neighbors = self._get_trie_neighbors(current_word)
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    # Check pattern constraint for next position
                    next_pos = len(path)
                    if matches_pattern(neighbor, next_pos):
                        visited.add(neighbor)
                        path.append(neighbor)
                        dfs_with_pattern(neighbor, path, visited)
                        path.pop()
                        visited.remove(neighbor)
        
        if matches_pattern(begin_word, 0):
            visited = {begin_word}
            dfs_with_pattern(begin_word, [begin_word], visited)
        
        return result


def test_basic_functionality():
    """Test basic word ladder functionality"""
    print("=== Testing Basic Word Ladder with Trie ===")
    
    test_cases = [
        # Classic examples
        {
            "word_list": ["hot","dot","dog","lot","log","cog"],
            "begin": "hit",
            "end": "cog",
            "expected_length": 5
        },
        
        # No path exists
        {
            "word_list": ["hot","dot","dog","lot","log"],
            "begin": "hit", 
            "end": "cog",
            "expected_length": 0
        },
        
        # Single step
        {
            "word_list": ["cat", "bat"],
            "begin": "cat",
            "end": "bat",
            "expected_length": 2
        },
        
        # Same word
        {
            "word_list": ["word"],
            "begin": "word",
            "end": "word", 
            "expected_length": 1
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"Word list: {test_case['word_list']}")
        print(f"Begin: '{test_case['begin']}', End: '{test_case['end']}'")
        print(f"Expected length: {test_case['expected_length']}")
        
        try:
            ladder = WordLadderTrie(test_case['word_list'])
            
            # Test shortest path length
            length = ladder.find_shortest_path_length(test_case['begin'], test_case['end'])
            length_status = "✓" if length == test_case['expected_length'] else "✗"
            print(f"  Shortest path length: {length} {length_status}")
            
            # Test shortest path
            path = ladder.find_shortest_path_optimized(test_case['begin'], test_case['end'])
            if path:
                path_length = len(path)
                path_status = "✓" if path_length == test_case['expected_length'] else "✗"
                print(f"  Shortest path: {path} (length: {path_length}) {path_status}")
            else:
                print(f"  Shortest path: None")
            
            # Test all shortest paths
            all_paths = ladder.find_all_shortest_paths(test_case['begin'], test_case['end'])
            print(f"  All shortest paths: {len(all_paths)} found")
            for j, p in enumerate(all_paths[:3]):  # Show first 3
                print(f"    Path {j+1}: {p}")
            if len(all_paths) > 3:
                print(f"    ... and {len(all_paths) - 3} more")
        
        except Exception as e:
            print(f"  Error: {e}")


def demonstrate_trie_optimization():
    """Demonstrate trie optimization benefits"""
    print("\n=== Trie Optimization Demo ===")
    
    word_list = ["hot", "dot", "dog", "lot", "log", "cog", "hit", "hat", "cat", "bat"]
    begin_word = "hit"
    
    print(f"Word list: {word_list}")
    print(f"Starting word: '{begin_word}'")
    
    ladder = WordLadderTrie(word_list)
    
    print(f"\nTrie-based neighbor finding for '{begin_word}':")
    
    # Show trie traversal for neighbor finding
    neighbors = ladder._get_trie_neighbors(begin_word)
    print(f"Valid neighbors: {neighbors}")
    
    # Compare with brute force approach
    print(f"\nComparison with brute force:")
    
    def brute_force_neighbors(word: str, word_set: Set[str]) -> List[str]:
        neighbors = []
        for i in range(len(word)):
            for char in string.ascii_lowercase:
                if char != word[i]:
                    new_word = word[:i] + char + word[i+1:]
                    if new_word in word_set:
                        neighbors.append(new_word)
        return neighbors
    
    brute_neighbors = brute_force_neighbors(begin_word, set(word_list))
    print(f"Brute force neighbors: {brute_neighbors}")
    
    print(f"\nBoth methods find same neighbors: {set(neighbors) == set(brute_neighbors)}")
    
    # Show trie structure benefits
    print(f"\nTrie structure benefits:")
    print(f"  • Efficient prefix-based lookup")
    print(f"  • Memory sharing for common prefixes")
    print(f"  • O(L) word validation vs O(L) hash lookup")
    print(f"  • Enables advanced pattern matching")


def demonstrate_bidirectional_search():
    """Demonstrate bidirectional BFS approach"""
    print("\n=== Bidirectional BFS Demo ===")
    
    word_list = ["hot", "dot", "dog", "lot", "log", "cog"]
    begin_word = "hit"
    end_word = "cog"
    
    print(f"Finding all shortest paths from '{begin_word}' to '{end_word}'")
    print(f"Word list: {word_list}")
    
    ladder = WordLadderTrie(word_list)
    
    # Build adjacency graph
    extended_set = set(word_list) | {begin_word}
    graph = ladder._build_adjacency_graph(extended_set)
    
    print(f"\nAdjacency graph:")
    for word, neighbors in graph.items():
        print(f"  {word}: {neighbors}")
    
    # Find all shortest paths
    all_paths = ladder.find_all_shortest_paths(begin_word, end_word)
    
    print(f"\nAll shortest paths ({len(all_paths)} found):")
    for i, path in enumerate(all_paths):
        print(f"  Path {i+1}: {' -> '.join(path)}")
        
        # Verify path validity
        valid = True
        for j in range(len(path) - 1):
            word1, word2 = path[j], path[j+1]
            diff_count = sum(c1 != c2 for c1, c2 in zip(word1, word2))
            if diff_count != 1:
                valid = False
                break
        
        print(f"    Valid: {valid}")


def demonstrate_astar_search():
    """Demonstrate A* search approach"""
    print("\n=== A* Search Demo ===")
    
    word_list = ["hot", "dot", "dog", "lot", "log", "cog", "hog", "bog"]
    begin_word = "hit"
    end_word = "cog"
    
    print(f"A* search from '{begin_word}' to '{end_word}'")
    print(f"Word list: {word_list}")
    
    ladder = WordLadderTrie(word_list)
    
    # Show heuristic calculation
    def heuristic(word: str, target: str) -> int:
        return sum(c1 != c2 for c1, c2 in zip(word, target))
    
    print(f"\nHeuristic values (edit distance to '{end_word}'):")
    for word in [begin_word] + word_list:
        h_val = heuristic(word, end_word)
        print(f"  {word}: {h_val}")
    
    # Find optimal path
    optimal_path = ladder.find_shortest_path_optimized(begin_word, end_word)
    
    print(f"\nA* optimal path: {optimal_path}")
    
    if optimal_path:
        print(f"Path analysis:")
        for i in range(len(optimal_path)):
            word = optimal_path[i]
            h_val = heuristic(word, end_word)
            g_val = i
            f_val = g_val + h_val
            
            print(f"  Step {i}: '{word}' (g={g_val}, h={h_val}, f={f_val})")


def demonstrate_pattern_constraints():
    """Demonstrate pattern-constrained path finding"""
    print("\n=== Pattern-Constrained Paths Demo ===")
    
    word_list = ["cat", "bat", "bet", "bit", "bot", "cot", "cut"]
    begin_word = "cat"
    end_word = "cut"
    
    print(f"Finding paths from '{begin_word}' to '{end_word}'")
    print(f"Word list: {word_list}")
    
    ladder = WordLadderTrie(word_list)
    
    # Test different patterns
    patterns = [
        "***",  # No constraints
        "c**",  # Must start with 'c'
        "*a*",  # Must have 'a' in middle
        "**t",  # Must end with 't'
        "c*t",  # Must start with 'c' and end with 't'
    ]
    
    for pattern in patterns:
        print(f"\nPattern '{pattern}':")
        
        paths = ladder.find_paths_with_pattern(begin_word, end_word, pattern)
        
        if paths:
            print(f"  Found {len(paths)} valid paths:")
            for i, path in enumerate(paths):
                print(f"    Path {i+1}: {' -> '.join(path)}")
                
                # Verify pattern compliance
                compliant = True
                for j, word in enumerate(path):
                    for k, (p_char, w_char) in enumerate(zip(pattern, word)):
                        if p_char != '*' and p_char != w_char:
                            compliant = False
                            break
                    if not compliant:
                        break
                
                print(f"      Pattern compliant: {compliant}")
        else:
            print(f"  No paths found matching pattern")


def benchmark_approaches():
    """Benchmark different approaches"""
    print("\n=== Benchmarking Approaches ===")
    
    import random
    
    # Generate test data
    def generate_word_list(size: int, word_length: int) -> List[str]:
        words = set()
        while len(words) < size:
            word = ''.join(random.choices(string.ascii_lowercase[:6], k=word_length))
            words.add(word)
        return list(words)
    
    test_scenarios = [
        ("Small", generate_word_list(50, 4)),
        ("Medium", generate_word_list(200, 5)),
        ("Large", generate_word_list(500, 6)),
    ]
    
    approaches = [
        ("BFS with Trie", "find_shortest_path_length"),
        ("A* Search", "find_shortest_path_optimized"),
        ("Bidirectional BFS", "find_all_shortest_paths"),
    ]
    
    for scenario_name, word_list in test_scenarios:
        # Pick random start and end words
        begin_word = random.choice(word_list)
        end_word = random.choice(word_list)
        
        print(f"\n--- {scenario_name} Dataset ---")
        print(f"Words: {len(word_list)}, Length: {len(word_list[0]) if word_list else 0}")
        print(f"Path: '{begin_word}' -> '{end_word}'")
        
        ladder = WordLadderTrie(word_list)
        
        for approach_name, method_name in approaches:
            start_time = time.time()
            
            try:
                method = getattr(ladder, method_name)
                result = method(begin_word, end_word)
                end_time = time.time()
                
                execution_time = (end_time - start_time) * 1000
                
                if isinstance(result, int):
                    print(f"  {approach_name:18}: length {result:3} in {execution_time:6.2f}ms")
                elif isinstance(result, list) and result and isinstance(result[0], str):
                    print(f"  {approach_name:18}: path length {len(result):3} in {execution_time:6.2f}ms")
                elif isinstance(result, list):
                    print(f"  {approach_name:18}: {len(result):3} paths in {execution_time:6.2f}ms")
                else:
                    print(f"  {approach_name:18}: result in {execution_time:6.2f}ms")
            
            except Exception as e:
                print(f"  {approach_name:18}: Error - {str(e)[:30]}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Spell correction with word suggestions
    print("1. Spell Correction Chain:")
    
    dictionary = ["cat", "bat", "hat", "hit", "hot", "pot", "pit", "bit", "bot", "cot"]
    misspelled = "cit"  # Not in dictionary
    target = "cat"
    
    # Find correction path
    ladder = WordLadderTrie(dictionary)
    
    # Add misspelled word temporarily
    extended_dict = dictionary + [misspelled]
    temp_ladder = WordLadderTrie(extended_dict)
    
    correction_path = temp_ladder.find_shortest_path_optimized(misspelled, target)
    
    print(f"   Dictionary: {dictionary}")
    print(f"   Misspelled: '{misspelled}' -> Target: '{target}'")
    print(f"   Correction path: {correction_path}")
    print(f"   Suggestions: {' -> '.join(correction_path) if correction_path else 'No path found'}")
    
    # Application 2: Gene mutation analysis
    print(f"\n2. Gene Mutation Sequence:")
    
    gene_sequences = ["ATAT", "ACAT", "ACAR", "AGAR", "AGAT", "GGAT"]
    start_gene = "ATAT"
    target_gene = "GGAT"
    
    gene_ladder = WordLadderTrie(gene_sequences)
    mutation_path = gene_ladder.find_shortest_path_optimized(start_gene, target_gene)
    
    print(f"   Gene sequences: {gene_sequences}")
    print(f"   Start: {start_gene} -> Target: {target_gene}")
    print(f"   Mutation path: {mutation_path}")
    
    if mutation_path:
        print(f"   Mutation steps:")
        for i in range(len(mutation_path) - 1):
            current = mutation_path[i]
            next_gene = mutation_path[i + 1]
            
            # Find mutation position
            for j, (c1, c2) in enumerate(zip(current, next_gene)):
                if c1 != c2:
                    print(f"     Step {i+1}: {current} -> {next_gene} (position {j}: {c1}->{c2})")
                    break
    
    # Application 3: Password transformation
    print(f"\n3. Secure Password Transformation:")
    
    password_patterns = ["pass", "word", "code", "safe", "lock", "key1", "key2"]
    old_password = "pass"
    new_password = "key1"
    
    password_ladder = WordLadderTrie(password_patterns)
    transformation_steps = password_ladder.find_shortest_path_optimized(old_password, new_password)
    
    print(f"   Password patterns: {password_patterns}")
    print(f"   Transform: '{old_password}' -> '{new_password}'")
    print(f"   Steps: {transformation_steps}")
    
    if transformation_steps:
        print(f"   Security consideration: {len(transformation_steps) - 1} intermediate passwords")
    
    # Application 4: Word game solver
    print(f"\n4. Word Game Optimization:")
    
    game_words = ["word", "work", "fork", "form", "worm", "warm", "ward", "hard"]
    start_word = "word"
    end_word = "hard"
    
    game_ladder = WordLadderTrie(game_words)
    
    # Find all possible paths within length limit
    all_game_paths = game_ladder.find_paths_with_length_limit(start_word, end_word, 6)
    
    print(f"   Game dictionary: {game_words}")
    print(f"   Challenge: '{start_word}' -> '{end_word}' (max 6 steps)")
    print(f"   Solutions found: {len(all_game_paths)}")
    
    for i, path in enumerate(all_game_paths[:3]):
        score = len(path) - 1  # Steps taken
        print(f"     Solution {i+1}: {' -> '.join(path)} (score: {score})")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    edge_cases = [
        # Empty word list
        {
            "word_list": [],
            "begin": "cat",
            "end": "bat",
            "description": "Empty word list"
        },
        
        # Single word
        {
            "word_list": ["word"],
            "begin": "word",
            "end": "word",
            "description": "Single word, same start/end"
        },
        
        # No path possible
        {
            "word_list": ["abc", "def"],
            "begin": "abc",
            "end": "def",
            "description": "No transformation path possible"
        },
        
        # Different word lengths
        {
            "word_list": ["cat", "cats"],
            "begin": "cat",
            "end": "cats",
            "description": "Different word lengths"
        },
        
        # Very short words
        {
            "word_list": ["a", "b", "c"],
            "begin": "a",
            "end": "c",
            "description": "Single character words"
        },
        
        # Target not in dictionary
        {
            "word_list": ["cat", "bat"],
            "begin": "cat",
            "end": "rat",
            "description": "End word not in dictionary"
        },
    ]
    
    for case in edge_cases:
        print(f"\n{case['description']}:")
        print(f"  Word list: {case['word_list']}")
        print(f"  Begin: '{case['begin']}', End: '{case['end']}'")
        
        try:
            if case['word_list']:  # Skip if empty
                ladder = WordLadderTrie(case['word_list'])
                length = ladder.find_shortest_path_length(case['begin'], case['end'])
                path = ladder.find_shortest_path_optimized(case['begin'], case['end'])
                
                print(f"  Shortest path length: {length}")
                print(f"  Shortest path: {path}")
            else:
                print(f"  Skipped (empty word list)")
        
        except Exception as e:
            print(f"  Error: {e}")


def analyze_complexity():
    """Analyze time and space complexity"""
    print("\n=== Complexity Analysis ===")
    
    complexity_analysis = [
        ("BFS with Trie Neighbors",
         "Time: O(N * L^2 * 26) for neighbor generation + O(N) BFS",
         "Space: O(N * L) for trie + O(N) for BFS queue"),
        
        ("Bidirectional BFS",
         "Time: O(N * L^2) with better constants due to meeting in middle",
         "Space: O(N * L) for graph + O(N) for forward/backward sets"),
        
        ("A* Search with Heuristic",
         "Time: O(N * L^2) with better practical performance",
         "Space: O(N * L) for trie + O(N) for priority queue"),
        
        ("DFS with Length Limit",
         "Time: O(exponential) but bounded by max_length",
         "Space: O(max_length) for recursion + O(paths) for results"),
        
        ("Pattern-Constrained Search",
         "Time: O(exponential) with pattern pruning",
         "Space: O(max_path_length * num_paths)"),
        
        ("Transformation Tree Building",
         "Time: O(N * L^2) for complete reachability analysis",
         "Space: O(N^2) worst case for storing all relationships"),
    ]
    
    print("Approach Analysis:")
    for approach, time_complexity, space_complexity in complexity_analysis:
        print(f"\n{approach}:")
        print(f"  {time_complexity}")
        print(f"  {space_complexity}")
    
    print(f"\nKey Variables:")
    print(f"  • N = number of words in dictionary")
    print(f"  • L = length of each word")
    print(f"  • 26 = alphabet size for neighbor generation")
    
    print(f"\nTrie Optimization Benefits:")
    print(f"  • Faster word validation: O(L) vs O(L) hash lookup")
    print(f"  • Memory efficiency through prefix sharing")
    print(f"  • Enables pattern-based search optimizations")
    print(f"  • Better cache locality for related words")
    
    print(f"\nPractical Considerations:")
    print(f"  • Dictionary size vs word length trade-offs")
    print(f"  • Memory usage for large dictionaries")
    print(f"  • Heuristic effectiveness in A* search")
    print(f"  • Bidirectional search reduces search space significantly")
    
    print(f"\nRecommendations:")
    print(f"  • Use BFS with Trie for single shortest path")
    print(f"  • Use Bidirectional BFS for all shortest paths")
    print(f"  • Use A* for optimal single path with good heuristic")
    print(f"  • Use DFS with limits for exploring multiple solutions")


if __name__ == "__main__":
    test_basic_functionality()
    demonstrate_trie_optimization()
    demonstrate_bidirectional_search()
    demonstrate_astar_search()
    demonstrate_pattern_constraints()
    benchmark_approaches()
    demonstrate_real_world_applications()
    test_edge_cases()
    analyze_complexity()

"""
Word Ladder with Trie demonstrates comprehensive graph traversal approaches enhanced with trie optimization:

1. BFS with Trie Neighbors - Use trie for efficient neighbor generation and validation
2. Bidirectional BFS - Meet-in-the-middle approach for finding all shortest paths
3. A* Search with Heuristic - Use edit distance heuristic for optimal pathfinding
4. DFS with Length Limit - Explore all paths within specified length constraints
5. Transformation Tree Building - Build complete reachability graph using trie
6. Pattern-Constrained Search - Find paths satisfying additional pattern requirements

Key concepts:
- Graph traversal algorithms (BFS, DFS, A*) enhanced with trie structures
- Bidirectional search for optimal performance
- Heuristic-guided search for better practical performance
- Pattern matching and constraint satisfaction in path finding
- Trie optimization for word validation and neighbor generation

Real-world applications:
- Spell correction with suggestion chains
- Gene mutation sequence analysis
- Secure password transformation protocols
- Word game optimization and puzzle solving

Each approach demonstrates different strategies for combining graph algorithms
with trie data structures for efficient word transformation problems.
"""
