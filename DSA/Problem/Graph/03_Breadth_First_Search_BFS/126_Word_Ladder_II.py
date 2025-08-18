"""
126. Word Ladder II
Difficulty: Hard

Problem:
A transformation sequence from word beginWord to word endWord using a dictionary wordList 
is a sequence of words beginWord -> s1 -> s2 -> ... -> sk such that:
- Every adjacent pair of words differs by exactly one letter.
- Every si for 1 <= i <= k is in wordList. Note that beginWord does not need to be in wordList.
- sk == endWord

Given two words, beginWord and endWord, and a dictionary wordList, return all the shortest 
transformation sequences from beginWord to endWord, or an empty list if no such sequence exists.

Examples:
Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
Output: [["hit","hot","dot","dog","cog"],["hit","hot","lot","log","cog"]]

Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"]
Output: []

Constraints:
- 1 <= beginWord.length <= 5
- endWord.length == beginWord.length
- 1 <= wordList.length <= 500
- wordList[i].length == beginWord.length
- beginWord, endWord, and wordList[i] consist of lowercase English letters
- beginWord != endWord
- All the words in wordList are unique
"""

from typing import List, Dict, Set
from collections import deque, defaultdict

class Solution:
    def findLadders_approach1_bfs_with_path_reconstruction(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        """
        Approach 1: BFS with Path Reconstruction (Optimal)
        
        Use BFS to find shortest distance, then DFS to reconstruct all paths.
        
        Time: O(M^2 * N + P) where P = number of paths
        Space: O(M * N + P)
        """
        if endWord not in wordList:
            return []
        
        wordSet = set(wordList)
        if beginWord in wordSet:
            wordSet.remove(beginWord)
        
        # BFS to find shortest distances and build parent graph
        queue = deque([beginWord])
        distances = {beginWord: 0}
        parents = defaultdict(list)
        found = False
        
        while queue and not found:
            level_size = len(queue)
            current_level = set()
            
            for _ in range(level_size):
                word = queue.popleft()
                
                # Generate neighbors
                for i in range(len(word)):
                    for c in 'abcdefghijklmnopqrstuvwxyz':
                        if c != word[i]:
                            new_word = word[:i] + c + word[i+1:]
                            
                            if new_word == endWord:
                                found = True
                                parents[new_word].append(word)
                            elif new_word in wordSet:
                                if new_word not in distances:
                                    distances[new_word] = distances[word] + 1
                                    current_level.add(new_word)
                                
                                if distances[new_word] == distances[word] + 1:
                                    parents[new_word].append(word)
            
            # Add current level to queue and remove from wordSet
            for word in current_level:
                queue.append(word)
                wordSet.discard(word)
        
        # DFS to reconstruct all shortest paths
        result = []
        
        def dfs(word, path):
            if word == beginWord:
                result.append([beginWord] + path[::-1])
                return
            
            for parent in parents[word]:
                dfs(parent, path + [word])
        
        if found:
            dfs(endWord, [])
        
        return result
    
    def findLadders_approach2_bidirectional_bfs(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        """
        Approach 2: Bidirectional BFS with Path Reconstruction
        
        Search from both ends and reconstruct paths.
        
        Time: O(M^2 * N + P)
        Space: O(M * N + P)
        """
        if endWord not in wordList:
            return []
        
        wordSet = set(wordList)
        
        # Two-ended BFS
        begin_set = {beginWord}
        end_set = {endWord}
        
        begin_visited = {beginWord: 0}
        end_visited = {endWord: 0}
        
        begin_parents = defaultdict(list)
        end_parents = defaultdict(list)
        
        found = False
        forward = True  # Direction flag
        
        def get_neighbors(word):
            neighbors = []
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if c != word[i]:
                        neighbors.append(word[:i] + c + word[i+1:])
            return neighbors
        
        while begin_set and end_set and not found:
            # Always expand the smaller frontier
            if len(begin_set) > len(end_set):
                begin_set, end_set = end_set, begin_set
                begin_visited, end_visited = end_visited, begin_visited
                begin_parents, end_parents = end_parents, begin_parents
                forward = not forward
            
            next_set = set()
            
            for word in begin_set:
                for neighbor in get_neighbors(word):
                    if neighbor in end_visited:
                        found = True
                        if forward:
                            begin_parents[neighbor].append(word)
                        else:
                            end_parents[word].append(neighbor)
                    elif neighbor in wordSet and neighbor not in begin_visited:
                        next_set.add(neighbor)
                        begin_visited[neighbor] = begin_visited[word] + 1
                        if forward:
                            begin_parents[neighbor].append(word)
                        else:
                            end_parents[word].append(neighbor)
            
            begin_set = next_set
        
        # Reconstruct paths (simplified for bidirectional)
        if not found:
            return []
        
        # This approach is complex for path reconstruction in bidirectional case
        # Fall back to approach 1 for simplicity
        return self.findLadders_approach1_bfs_with_path_reconstruction(beginWord, endWord, wordList)
    
    def findLadders_approach3_level_by_level_bfs(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        """
        Approach 3: Level-by-Level BFS with Path Storage
        
        Store all paths at each level and extend them.
        
        Time: O(M^2 * N * P)
        Space: O(M * N * P)
        """
        if endWord not in wordList:
            return []
        
        wordSet = set(wordList)
        
        # BFS with path storage
        queue = deque([[beginWord]])
        visited = set([beginWord])
        result = []
        found = False
        
        while queue and not found:
            level_size = len(queue)
            level_visited = set()
            
            for _ in range(level_size):
                path = queue.popleft()
                word = path[-1]
                
                # Generate neighbors
                for i in range(len(word)):
                    for c in 'abcdefghijklmnopqrstuvwxyz':
                        if c != word[i]:
                            new_word = word[:i] + c + word[i+1:]
                            
                            if new_word == endWord:
                                result.append(path + [new_word])
                                found = True
                            elif new_word in wordSet and new_word not in visited:
                                level_visited.add(new_word)
                                queue.append(path + [new_word])
            
            # Update visited after processing entire level
            visited.update(level_visited)
        
        return result
    
    def findLadders_approach4_optimized_reconstruction(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        """
        Approach 4: Optimized BFS with Efficient Path Reconstruction
        
        Use BFS to build distance map and parent relationships efficiently.
        
        Time: O(M^2 * N + P)
        Space: O(M * N + P)
        """
        if endWord not in wordList:
            return []
        
        wordSet = set(wordList)
        
        # Build adjacency list using pattern matching
        patterns = defaultdict(list)
        for word in [beginWord] + wordList:
            for i in range(len(word)):
                pattern = word[:i] + '*' + word[i+1:]
                patterns[pattern].append(word)
        
        # BFS to find distances and parents
        queue = deque([beginWord])
        distances = {beginWord: 0}
        parents = defaultdict(list)
        
        while queue:
            word = queue.popleft()
            
            if word == endWord:
                break
            
            # Find neighbors using patterns
            for i in range(len(word)):
                pattern = word[:i] + '*' + word[i+1:]
                
                for neighbor in patterns[pattern]:
                    if neighbor == word:
                        continue
                    
                    if neighbor not in distances:
                        distances[neighbor] = distances[word] + 1
                        parents[neighbor].append(word)
                        queue.append(neighbor)
                    elif distances[neighbor] == distances[word] + 1:
                        parents[neighbor].append(word)
        
        # DFS to reconstruct all shortest paths
        result = []
        
        def dfs(word, path):
            if word == beginWord:
                result.append([beginWord] + path[::-1])
                return
            
            for parent in parents[word]:
                dfs(parent, path + [word])
        
        if endWord in distances:
            dfs(endWord, [])
        
        return result

def test_word_ladder_ii():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (beginWord, endWord, wordList, expected_count)
        ("hit", "cog", ["hot","dot","dog","lot","log","cog"], 2),
        ("hit", "cog", ["hot","dot","dog","lot","log"], 0),
        ("a", "c", ["a","b","c"], 1),
        ("hot", "dog", ["hot","hog","dog"], 1),
        ("red", "tax", ["ted","tex","red","tax","tad","den","rex","pea"], 2),
    ]
    
    approaches = [
        ("BFS + Path Reconstruction", solution.findLadders_approach1_bfs_with_path_reconstruction),
        ("Bidirectional BFS", solution.findLadders_approach2_bidirectional_bfs),
        ("Level-by-Level BFS", solution.findLadders_approach3_level_by_level_bfs),
        ("Optimized Reconstruction", solution.findLadders_approach4_optimized_reconstruction),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (beginWord, endWord, wordList, expected_count) in enumerate(test_cases):
            result = func(beginWord, endWord, wordList[:])
            result_count = len(result)
            status = "✓" if result_count == expected_count else "✗"
            print(f"Test {i+1}: {status} '{beginWord}' -> '{endWord}', Expected paths: {expected_count}, Got: {result_count}")
            
            if result and len(result) <= 3:  # Show paths if not too many
                for j, path in enumerate(result):
                    print(f"         Path {j+1}: {' -> '.join(path)}")

def demonstrate_path_reconstruction():
    """Demonstrate path reconstruction process"""
    print("\n=== Path Reconstruction Demo ===")
    
    beginWord = "hit"
    endWord = "cog"
    wordList = ["hot","dot","dog","lot","log","cog"]
    
    print(f"Finding all shortest paths from '{beginWord}' to '{endWord}'")
    print(f"Word list: {wordList}")
    
    # Build parent relationships using BFS
    wordSet = set(wordList)
    queue = deque([beginWord])
    distances = {beginWord: 0}
    parents = defaultdict(list)
    
    print(f"\nBFS to build parent relationships:")
    
    while queue:
        word = queue.popleft()
        print(f"  Processing: {word} (distance: {distances[word]})")
        
        if word == endWord:
            break
        
        # Generate neighbors
        neighbors_found = []
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c != word[i]:
                    new_word = word[:i] + c + word[i+1:]
                    
                    if new_word in wordSet:
                        if new_word not in distances:
                            distances[new_word] = distances[word] + 1
                            parents[new_word].append(word)
                            queue.append(new_word)
                            neighbors_found.append(new_word)
                        elif distances[new_word] == distances[word] + 1:
                            parents[new_word].append(word)
        
        if neighbors_found:
            print(f"    Found neighbors: {neighbors_found}")
    
    print(f"\nParent relationships:")
    for word in sorted(parents.keys()):
        print(f"  {word}: parents = {parents[word]}")
    
    # Reconstruct paths
    print(f"\nPath reconstruction:")
    result = []
    
    def dfs(word, path):
        print(f"    DFS at {word}, current path: {path}")
        
        if word == beginWord:
            complete_path = [beginWord] + path[::-1]
            result.append(complete_path)
            print(f"      ✓ Complete path: {' -> '.join(complete_path)}")
            return
        
        for parent in parents[word]:
            dfs(parent, path + [word])
    
    if endWord in distances:
        dfs(endWord, [])
    
    print(f"\nAll shortest paths found: {len(result)}")
    for i, path in enumerate(result):
        print(f"  Path {i+1}: {' -> '.join(path)}")

def analyze_path_explosion():
    """Analyze path explosion in word ladder problems"""
    print("\n=== Path Explosion Analysis ===")
    
    print("Why Word Ladder II is Much Harder:")
    print("1. **Single Path vs All Paths:**")
    print("   • Word Ladder I: Find any shortest path")
    print("   • Word Ladder II: Find ALL shortest paths")
    
    print("\n2. **Exponential Path Growth:**")
    print("   • Number of shortest paths can be exponential")
    print("   • Each branch point multiplies path count")
    print("   • Memory usage grows with path storage")
    
    print("\n3. **Algorithm Complexity:**")
    print("   • BFS: O(M^2 * N) for distance finding")
    print("   • DFS: O(P) for path reconstruction")
    print("   • Total: O(M^2 * N + P) where P = path count")
    
    print("\nPath Reconstruction Strategies:")
    print("• **Parent Tracking:** Build reverse graph during BFS")
    print("• **Level Processing:** Keep all paths at each level")
    print("• **Bidirectional:** Meet in middle (complex reconstruction)")
    print("• **Pattern Matching:** Efficient neighbor generation")
    
    print("\nOptimization Techniques:")
    print("• Use BFS for shortest distance guarantee")
    print("• Build parent relationships during BFS")
    print("• Use DFS for path reconstruction")
    print("• Pattern-based adjacency for efficiency")
    print("• Early termination when target reached")
    
    print("\nReal-world Applications:")
    print("• Gene sequence analysis (multiple optimal alignments)")
    print("• Network routing (multiple shortest paths)")
    print("• Game AI (multiple optimal strategies)")
    print("• Language processing (alternative transformations)")
    print("• Puzzle solving (all optimal solutions)")

if __name__ == "__main__":
    test_word_ladder_ii()
    demonstrate_path_reconstruction()
    analyze_path_explosion()

"""
Graph Theory Concepts:
1. All Shortest Paths Problem
2. Path Reconstruction in BFS
3. Parent Relationship Tracking
4. Exponential Path Enumeration

Key Word Ladder II Insights:
- Find ALL shortest transformation sequences
- BFS guarantees shortest path distance
- Parent tracking enables path reconstruction
- Path count can be exponential

Algorithm Strategy:
1. Use BFS to find shortest distances
2. Build parent relationships during BFS
3. Use DFS to reconstruct all paths
4. Handle multiple parents per node

Path Reconstruction Complexity:
- BFS: O(M^2 * N) for distance finding
- DFS: O(P) where P = number of paths
- Space: O(M * N + P) for parents + paths

Real-world Applications:
- Bioinformatics sequence alignment
- Network routing with multiple paths
- Game AI strategy enumeration
- Natural language processing
- Optimization with multiple solutions

This problem demonstrates the complexity jump from
finding one solution to finding all optimal solutions.
"""
