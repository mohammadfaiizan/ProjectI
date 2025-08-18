"""
127. Word Ladder
Difficulty: Hard

Problem:
A transformation sequence from word beginWord to word endWord using a dictionary wordList 
is a sequence of words beginWord -> s1 -> s2 -> ... -> sk such that:
- Every adjacent pair of words differs by exactly one letter.
- Every si for 1 <= i <= k is in wordList. Note that beginWord does not need to be in wordList.
- sk == endWord

Given two words, beginWord and endWord, and a dictionary wordList, return the length of the 
shortest transformation sequence from beginWord to endWord, or 0 if no such sequence exists.

Examples:
Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
Output: 5

Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"]
Output: 0

Constraints:
- 1 <= beginWord.length <= 10
- endWord.length == beginWord.length
- 1 <= wordList.length <= 5000
- wordList[i].length == beginWord.length
- beginWord, endWord, and wordList[i] consist of lowercase English letters
- beginWord != endWord
- All the words in wordList are unique
"""

from typing import List, Set
from collections import deque, defaultdict

class Solution:
    def ladderLength_approach1_standard_bfs(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        """
        Approach 1: Standard BFS (Optimal)
        
        Use BFS to find shortest transformation sequence.
        Each level represents one transformation step.
        
        Time: O(M^2 * N) where M = word length, N = wordList size
        Space: O(M^2 * N) for adjacency construction + O(N) for BFS
        """
        if endWord not in wordList:
            return 0
        
        wordSet = set(wordList)
        if beginWord in wordSet:
            wordSet.remove(beginWord)
        
        queue = deque([(beginWord, 1)])  # (word, steps)
        
        while queue:
            word, steps = queue.popleft()
            
            if word == endWord:
                return steps
            
            # Generate all possible one-character transformations
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if c != word[i]:
                        new_word = word[:i] + c + word[i+1:]
                        
                        if new_word in wordSet:
                            if new_word == endWord:
                                return steps + 1
                            
                            wordSet.remove(new_word)  # Mark as visited
                            queue.append((new_word, steps + 1))
        
        return 0
    
    def ladderLength_approach2_bidirectional_bfs(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        """
        Approach 2: Bidirectional BFS (Optimized)
        
        Search from both ends simultaneously to reduce search space.
        
        Time: O(M^2 * N)
        Space: O(M^2 * N)
        """
        if endWord not in wordList:
            return 0
        
        wordSet = set(wordList)
        
        # Two sets for bidirectional search
        begin_set = {beginWord}
        end_set = {endWord}
        visited = set()
        
        steps = 1
        
        while begin_set and end_set:
            # Always expand the smaller set
            if len(begin_set) > len(end_set):
                begin_set, end_set = end_set, begin_set
            
            temp_set = set()
            
            for word in begin_set:
                for i in range(len(word)):
                    for c in 'abcdefghijklmnopqrstuvwxyz':
                        if c != word[i]:
                            new_word = word[:i] + c + word[i+1:]
                            
                            if new_word in end_set:
                                return steps + 1
                            
                            if new_word in wordSet and new_word not in visited:
                                visited.add(new_word)
                                temp_set.add(new_word)
            
            begin_set = temp_set
            steps += 1
        
        return 0
    
    def ladderLength_approach3_precomputed_adjacency(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        """
        Approach 3: Precomputed Adjacency List
        
        Build adjacency list first, then use standard BFS.
        
        Time: O(M^2 * N) for building + O(M * N) for BFS
        Space: O(M^2 * N)
        """
        if endWord not in wordList:
            return 0
        
        # Include beginWord in the word list for adjacency building
        if beginWord not in wordList:
            wordList.append(beginWord)
        
        # Build adjacency list using pattern matching
        adjacency = defaultdict(list)
        
        for word in wordList:
            for i in range(len(word)):
                pattern = word[:i] + '*' + word[i+1:]
                adjacency[pattern].append(word)
        
        # BFS
        queue = deque([(beginWord, 1)])
        visited = {beginWord}
        
        while queue:
            word, steps = queue.popleft()
            
            if word == endWord:
                return steps
            
            # Find all adjacent words using patterns
            for i in range(len(word)):
                pattern = word[:i] + '*' + word[i+1:]
                
                for neighbor in adjacency[pattern]:
                    if neighbor not in visited:
                        if neighbor == endWord:
                            return steps + 1
                        
                        visited.add(neighbor)
                        queue.append((neighbor, steps + 1))
        
        return 0
    
    def ladderLength_approach4_optimized_bidirectional(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        """
        Approach 4: Optimized Bidirectional with Pattern Matching
        
        Combine bidirectional search with efficient adjacency computation.
        
        Time: O(M^2 * N)
        Space: O(M^2 * N)
        """
        if endWord not in wordList:
            return 0
        
        # Build pattern-based adjacency
        if beginWord not in wordList:
            wordList.append(beginWord)
        
        adjacency = defaultdict(list)
        for word in wordList:
            for i in range(len(word)):
                pattern = word[:i] + '*' + word[i+1:]
                adjacency[pattern].append(word)
        
        def get_neighbors(word):
            """Get all neighbors of a word"""
            neighbors = []
            for i in range(len(word)):
                pattern = word[:i] + '*' + word[i+1:]
                neighbors.extend(adjacency[pattern])
            return neighbors
        
        # Bidirectional BFS
        begin_queue = deque([beginWord])
        end_queue = deque([endWord])
        
        begin_visited = {beginWord: 1}
        end_visited = {endWord: 1}
        
        while begin_queue or end_queue:
            # Expand begin side
            if begin_queue:
                for _ in range(len(begin_queue)):
                    word = begin_queue.popleft()
                    
                    for neighbor in get_neighbors(word):
                        if neighbor in end_visited:
                            return begin_visited[word] + end_visited[neighbor]
                        
                        if neighbor not in begin_visited:
                            begin_visited[neighbor] = begin_visited[word] + 1
                            begin_queue.append(neighbor)
            
            # Expand end side
            if end_queue:
                for _ in range(len(end_queue)):
                    word = end_queue.popleft()
                    
                    for neighbor in get_neighbors(word):
                        if neighbor in begin_visited:
                            return end_visited[word] + begin_visited[neighbor]
                        
                        if neighbor not in end_visited:
                            end_visited[neighbor] = end_visited[word] + 1
                            end_queue.append(neighbor)
        
        return 0

def test_word_ladder():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (beginWord, endWord, wordList, expected)
        ("hit", "cog", ["hot","dot","dog","lot","log","cog"], 5),
        ("hit", "cog", ["hot","dot","dog","lot","log"], 0),
        ("a", "c", ["a","b","c"], 2),
        ("hot", "dog", ["hot","dog"], 0),  # No intermediate transformation
        ("hot", "dog", ["hot","hog","dog"], 3),
        ("game", "thee", ["hate","tree","thee","game","hare","care","core"], 0),
        ("qa", "sq", ["si","go","se","cm","so","ph","mt","db","mb","sb","kr","ln","tm","le","av","sm","ar","ci","ca","br","ti","ba","to","ra","fa","yo","ow","sn","ya","cr","po","fe","ho","ma","re","or","rn","au","ur","rh","sr","tc","lt","lo","as","fr","nb","yb","if","pb","ge","th","pm","rb","sh","co","ga","li","ha","hz","no","bi","di","hi","qa","pi","os","uh","wm","an","me","mo","na","la","st","er","sc","ne","mn","mi","am","ex","pt","io","be","fm","ta","tb","ni","mr","pa","he","lr","sq","ye"], 5),
    ]
    
    approaches = [
        ("Standard BFS", solution.ladderLength_approach1_standard_bfs),
        ("Bidirectional BFS", solution.ladderLength_approach2_bidirectional_bfs),
        ("Precomputed Adjacency", solution.ladderLength_approach3_precomputed_adjacency),
        ("Optimized Bidirectional", solution.ladderLength_approach4_optimized_bidirectional),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (beginWord, endWord, wordList, expected) in enumerate(test_cases):
            result = func(beginWord, endWord, wordList[:])  # Pass copy
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} '{beginWord}' -> '{endWord}', Expected: {expected}, Got: {result}")

def demonstrate_word_transformation():
    """Demonstrate word transformation process"""
    print("\n=== Word Transformation Demo ===")
    
    beginWord = "hit"
    endWord = "cog"
    wordList = ["hot","dot","dog","lot","log","cog"]
    
    print(f"Transform '{beginWord}' -> '{endWord}'")
    print(f"Word list: {wordList}")
    
    # BFS with path tracking
    if endWord not in wordList:
        print("End word not in word list!")
        return
    
    wordSet = set(wordList)
    queue = deque([(beginWord, 1, [beginWord])])  # (word, steps, path)
    visited = {beginWord}
    
    print(f"\nBFS exploration:")
    
    while queue:
        word, steps, path = queue.popleft()
        
        print(f"  Step {steps}: '{word}', Path: {' -> '.join(path)}")
        
        if word == endWord:
            print(f"  🎯 Found target! Transformation length: {steps}")
            print(f"  Complete path: {' -> '.join(path)}")
            break
        
        # Generate neighbors
        neighbors = []
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c != word[i]:
                    new_word = word[:i] + c + word[i+1:]
                    if new_word in wordSet and new_word not in visited:
                        neighbors.append(new_word)
                        visited.add(new_word)
                        queue.append((new_word, steps + 1, path + [new_word]))
        
        if neighbors:
            print(f"    Added neighbors: {neighbors}")

def analyze_word_ladder_optimizations():
    """Analyze different optimization techniques"""
    print("\n=== Word Ladder Optimizations ===")
    
    print("1. Standard BFS:")
    print("   ✅ Simple and intuitive")
    print("   ✅ Guaranteed shortest path")
    print("   ❌ May explore large search space")
    print("   • Time: O(M^2 * N), Space: O(M * N)")
    
    print("\n2. Bidirectional BFS:")
    print("   ✅ Reduces search space significantly")
    print("   ✅ Especially effective for long paths")
    print("   ❌ More complex implementation")
    print("   • Time: O(M^2 * N), Space: O(M * N)")
    print("   • Practical speedup: ~2x faster")
    
    print("\n3. Pattern-based Adjacency:")
    print("   ✅ Precomputes neighbors efficiently")
    print("   ✅ Avoids repeated character substitutions")
    print("   ❌ Higher space complexity")
    print("   • Time: O(M^2 * N), Space: O(M^2 * N)")
    
    print("\n4. Optimization Techniques:")
    print("   • Early termination when target found")
    print("   • Remove visited words to avoid cycles")
    print("   • Use sets for O(1) lookup operations")
    print("   • Always expand smaller frontier in bidirectional search")
    
    print("\nWhen to use each approach:")
    print("• Standard BFS: When simplicity is priority")
    print("• Bidirectional BFS: When word lists are large")
    print("• Pattern-based: When multiple queries on same word list")
    print("• Hybrid: Combine bidirectional + pattern for best performance")

def visualize_transformation_graph():
    """Visualize the word transformation as a graph"""
    print("\n=== Transformation Graph Visualization ===")
    
    words = ["hit", "hot", "dot", "dog", "cog"]
    
    print("Words:", words)
    print("\nAdjacency relationships (words differing by 1 character):")
    
    def differs_by_one(w1, w2):
        if len(w1) != len(w2):
            return False
        diff_count = sum(1 for i in range(len(w1)) if w1[i] != w2[i])
        return diff_count == 1
    
    # Build and display adjacency
    adjacency = defaultdict(list)
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            if i != j and differs_by_one(word1, word2):
                adjacency[word1].append(word2)
    
    for word in words:
        print(f"  {word} -> {adjacency[word]}")
    
    print(f"\nShortest path from 'hit' to 'cog':")
    print("  hit -> hot -> dot -> dog -> cog")
    print("  Length: 5 (including start and end)")

if __name__ == "__main__":
    test_word_ladder()
    demonstrate_word_transformation()
    analyze_word_ladder_optimizations()
    visualize_transformation_graph()

"""
Graph Theory Concepts:
1. Shortest Path in Unweighted Graph
2. Word Transformation as Graph Problem
3. Bidirectional BFS Optimization
4. Pattern-based Adjacency Construction

Key Algorithm Insights:
- Model words as graph nodes, one-character differences as edges
- BFS guarantees shortest transformation sequence
- Bidirectional search reduces search space exponentially
- Pattern matching optimizes neighbor generation

Optimization Techniques:
- Bidirectional BFS: Search from both ends
- Pattern-based adjacency: Precompute using wildcards
- Early termination: Stop when target found
- Set operations: O(1) lookups and visited tracking

Real-world Applications:
- Natural language processing (word similarity)
- Spell checkers and auto-correction
- Game AI (word puzzle solvers)
- Bioinformatics (DNA sequence alignment)
- Edit distance and string similarity
- Machine learning (word embeddings)

This problem demonstrates BFS for shortest path finding
in implicitly defined graphs with optimization techniques.
"""
