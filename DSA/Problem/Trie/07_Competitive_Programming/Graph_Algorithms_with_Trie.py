"""
Graph Algorithms with Trie - Multiple Approaches
Difficulty: Hard

Graph algorithms that utilize trie data structures for optimization
in competitive programming scenarios.

Problems:
1. Shortest Path with String Constraints
2. Graph Traversal with Pattern Matching
3. String-based Graph Coloring
4. Path Compression using Trie
5. Network Flow with String Labels
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import deque, defaultdict
import heapq

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.node_id = -1

class GraphTrieAlgorithms:
    
    def __init__(self):
        self.root = TrieNode()
    
    def shortest_path_string_constraints(self, graph: Dict[int, List[Tuple[int, int]]], 
                                       start: int, end: int, 
                                       node_labels: Dict[int, str],
                                       required_pattern: str) -> Tuple[int, List[int]]:
        """
        Find shortest path where concatenated node labels contain required pattern
        Time: O(V * E * log V + pattern_matching)
        Space: O(V + E)
        """
        # Build trie for pattern matching
        def build_pattern_trie(pattern: str) -> TrieNode:
            root = TrieNode()
            node = root
            for char in pattern:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end = True
            return root
        
        pattern_trie = build_pattern_trie(required_pattern)
        
        # Dijkstra with string state
        # State: (distance, node, trie_position, path)
        pq = [(0, start, pattern_trie, [start])]
        visited = set()
        
        while pq:
            dist, node, trie_pos, path = heapq.heappop(pq)
            
            state = (node, id(trie_pos))
            if state in visited:
                continue
            visited.add(state)
            
            # Check if we reached end with complete pattern
            if node == end and trie_pos.is_end:
                return dist, path
            
            # Explore neighbors
            for neighbor, weight in graph.get(node, []):
                new_dist = dist + weight
                new_path = path + [neighbor]
                
                # Update trie position based on neighbor's label
                neighbor_label = node_labels.get(neighbor, "")
                new_trie_pos = trie_pos
                
                for char in neighbor_label:
                    if char in new_trie_pos.children:
                        new_trie_pos = new_trie_pos.children[char]
                    else:
                        new_trie_pos = pattern_trie  # Reset if no match
                        break
                
                new_state = (neighbor, id(new_trie_pos))
                if new_state not in visited:
                    heapq.heappush(pq, (new_dist, neighbor, new_trie_pos, new_path))
        
        return -1, []  # No path found
    
    def graph_traversal_pattern_matching(self, graph: Dict[int, List[int]], 
                                       node_labels: Dict[int, str],
                                       patterns: List[str]) -> Dict[str, List[List[int]]]:
        """
        Find all paths that match given patterns during traversal
        Time: O(V + E) * paths * pattern_length
        Space: O(V + patterns)
        """
        # Build pattern tries
        pattern_tries = {}
        for pattern in patterns:
            root = TrieNode()
            node = root
            for char in pattern:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end = True
            pattern_tries[pattern] = root
        
        result = {pattern: [] for pattern in patterns}
        
        def dfs(node: int, path: List[int], trie_states: Dict[str, TrieNode]):
            """DFS with pattern matching state"""
            current_label = node_labels.get(node, "")
            
            # Update trie states for each pattern
            new_trie_states = {}
            for pattern, trie_pos in trie_states.items():
                new_pos = trie_pos
                
                for char in current_label:
                    if char in new_pos.children:
                        new_pos = new_pos.children[char]
                    else:
                        new_pos = pattern_tries[pattern]  # Reset
                        break
                
                new_trie_states[pattern] = new_pos
                
                # Check if pattern is complete
                if new_pos.is_end:
                    result[pattern].append(path[:])
            
            # Continue DFS
            for neighbor in graph.get(node, []):
                if neighbor not in path:  # Avoid cycles
                    dfs(neighbor, path + [neighbor], new_trie_states)
        
        # Start DFS from each node
        for start_node in graph:
            initial_states = {pattern: trie for pattern, trie in pattern_tries.items()}
            dfs(start_node, [start_node], initial_states)
        
        return result
    
    def string_based_graph_coloring(self, graph: Dict[int, List[int]], 
                                  node_strings: Dict[int, str],
                                  color_patterns: List[str]) -> Dict[int, int]:
        """
        Color graph where adjacent nodes can't have strings matching same pattern
        Time: O(V * patterns + E * patterns)
        Space: O(V + patterns)
        """
        # Build pattern recognition
        def matches_pattern(string: str, pattern: str) -> bool:
            """Check if string matches pattern (simplified)"""
            return pattern in string
        
        # Determine which patterns each node matches
        node_pattern_sets = {}
        for node, string in node_strings.items():
            patterns_matched = set()
            for i, pattern in enumerate(color_patterns):
                if matches_pattern(string, pattern):
                    patterns_matched.add(i)
            node_pattern_sets[node] = patterns_matched
        
        # Graph coloring with pattern constraints
        coloring = {}
        
        def can_color(node: int, color: int) -> bool:
            """Check if node can be colored with given color"""
            node_patterns = node_pattern_sets.get(node, set())
            
            for neighbor in graph.get(node, []):
                if neighbor in coloring and coloring[neighbor] == color:
                    neighbor_patterns = node_pattern_sets.get(neighbor, set())
                    # Check if they share any patterns
                    if node_patterns & neighbor_patterns:
                        return False
            return True
        
        # Greedy coloring
        nodes = sorted(graph.keys(), key=lambda x: len(graph.get(x, [])), reverse=True)
        
        for node in nodes:
            for color in range(len(color_patterns) + 1):
                if can_color(node, color):
                    coloring[node] = color
                    break
        
        return coloring
    
    def path_compression_trie(self, paths: List[List[str]]) -> Dict[str, any]:
        """
        Compress multiple paths using trie structure
        Time: O(total_path_length)
        Space: O(compressed_size)
        """
        # Build path trie
        path_trie = TrieNode()
        path_count = 0
        
        for path in paths:
            node = path_trie
            for step in path:
                if step not in node.children:
                    node.children[step] = TrieNode()
                node = node.children[step]
            node.is_end = True
            node.node_id = path_count
            path_count += 1
        
        # Calculate compression statistics
        def calculate_trie_stats(node: TrieNode, depth: int = 0) -> Dict[str, int]:
            stats = {'nodes': 1, 'max_depth': depth, 'leaves': 0}
            
            if node.is_end:
                stats['leaves'] = 1
            
            for child in node.children.values():
                child_stats = calculate_trie_stats(child, depth + 1)
                stats['nodes'] += child_stats['nodes']
                stats['max_depth'] = max(stats['max_depth'], child_stats['max_depth'])
                stats['leaves'] += child_stats['leaves']
            
            return stats
        
        trie_stats = calculate_trie_stats(path_trie)
        original_size = sum(len(path) for path in paths)
        
        return {
            'trie_root': path_trie,
            'original_size': original_size,
            'compressed_nodes': trie_stats['nodes'],
            'compression_ratio': original_size / trie_stats['nodes'] if trie_stats['nodes'] > 0 else 0,
            'max_depth': trie_stats['max_depth'],
            'unique_paths': trie_stats['leaves']
        }


def test_shortest_path_constraints():
    """Test shortest path with string constraints"""
    print("=== Testing Shortest Path with String Constraints ===")
    
    algo = GraphTrieAlgorithms()
    
    # Graph: node -> [(neighbor, weight)]
    graph = {
        0: [(1, 2), (2, 3)],
        1: [(3, 1)],
        2: [(3, 1), (4, 2)],
        3: [(4, 1)],
        4: []
    }
    
    # Node labels
    node_labels = {
        0: "start",
        1: "mid",
        2: "alt",
        3: "end",
        4: "goal"
    }
    
    required_pattern = "goal"
    
    print(f"Graph: {graph}")
    print(f"Node labels: {node_labels}")
    print(f"Required pattern: '{required_pattern}'")
    
    distance, path = algo.shortest_path_string_constraints(graph, 0, 4, node_labels, required_pattern)
    
    print(f"Shortest path distance: {distance}")
    print(f"Path: {path}")

def test_pattern_matching_traversal():
    """Test graph traversal with pattern matching"""
    print("\n=== Testing Pattern Matching Traversal ===")
    
    algo = GraphTrieAlgorithms()
    
    graph = {
        0: [1, 2],
        1: [3],
        2: [3, 4],
        3: [4],
        4: []
    }
    
    node_labels = {
        0: "A",
        1: "B", 
        2: "A",
        3: "B",
        4: "C"
    }
    
    patterns = ["AB", "ABC", "BAC"]
    
    print(f"Graph: {graph}")
    print(f"Node labels: {node_labels}")
    print(f"Patterns: {patterns}")
    
    matching_paths = algo.graph_traversal_pattern_matching(graph, node_labels, patterns)
    
    for pattern, paths in matching_paths.items():
        print(f"Pattern '{pattern}': {len(paths)} matching paths")
        for path in paths[:3]:  # Show first 3 paths
            print(f"  {path}")

def test_path_compression():
    """Test path compression using trie"""
    print("\n=== Testing Path Compression ===")
    
    algo = GraphTrieAlgorithms()
    
    paths = [
        ["start", "A", "B", "end"],
        ["start", "A", "C", "end"],
        ["start", "B", "C", "end"],
        ["start", "A", "B", "C", "end"],
        ["other", "X", "Y", "end"]
    ]
    
    print(f"Original paths: {paths}")
    
    compression_info = algo.path_compression_trie(paths)
    
    print("Compression results:")
    for key, value in compression_info.items():
        if key != 'trie_root':
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

if __name__ == "__main__":
    test_shortest_path_constraints()
    test_pattern_matching_traversal()
    test_path_compression()
