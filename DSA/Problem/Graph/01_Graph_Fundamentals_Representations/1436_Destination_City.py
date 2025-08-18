"""
1436. Destination City
Difficulty: Easy

Problem:
You are given the array paths, where paths[i] = [cityAi, cityBi] means there exists 
a direct path going from cityAi to cityBi. Return the destination city, that is, 
the city without any path going from it to any other city.

It is guaranteed that the graph of paths forms a line without any loop, 
therefore, there will be exactly one destination city.

Examples:
Input: paths = [["London","New York"],["New York","Lima"],["Lima","Sao Paulo"]]
Output: "Sao Paulo"

Input: paths = [["B","C"],["D","B"],["C","A"]]
Output: "A"

Input: paths = [["A","Z"]]
Output: "Z"

Constraints:
- 1 <= paths.length <= 100
- paths[i].length == 2
- 1 <= cityAi.length, cityBi.length <= 10
- cityAi != cityBi
- All pairs (cityAi, cityBi) are distinct
- All cities are distinct
"""

from typing import List
from collections import defaultdict

class Solution:
    def destCity_approach1_out_degree(self, paths: List[List[str]]) -> str:
        """
        Approach 1: Out-degree analysis
        
        The destination city has out-degree 0 (no outgoing edges).
        All other cities have out-degree 1 (exactly one outgoing edge).
        
        Time: O(N) where N = number of paths
        Space: O(N) for storing cities
        """
        # Track all cities and their out-degrees
        all_cities = set()
        cities_with_outgoing = set()
        
        for from_city, to_city in paths:
            all_cities.add(from_city)
            all_cities.add(to_city)
            cities_with_outgoing.add(from_city)
        
        # Find city that's in all_cities but not in cities_with_outgoing
        for city in all_cities:
            if city not in cities_with_outgoing:
                return city
        
        return ""  # Should never reach here for valid input
    
    def destCity_approach2_set_difference(self, paths: List[List[str]]) -> str:
        """
        Approach 2: Set operations
        
        Destination = (All cities) - (Cities with outgoing paths)
        
        Time: O(N)
        Space: O(N)
        """
        all_cities = set()
        source_cities = set()
        
        for from_city, to_city in paths:
            all_cities.add(from_city)
            all_cities.add(to_city)
            source_cities.add(from_city)
        
        # Destination is the difference
        destination = all_cities - source_cities
        return list(destination)[0]
    
    def destCity_approach3_dictionary_lookup(self, paths: List[List[str]]) -> str:
        """
        Approach 3: Dictionary-based lookup
        
        Build a mapping from source to destination.
        The destination city won't appear as a key.
        
        Time: O(N)
        Space: O(N)
        """
        path_map = {}
        destinations = set()
        
        for from_city, to_city in paths:
            path_map[from_city] = to_city
            destinations.add(to_city)
        
        # Find destination city that's not a source
        for city in destinations:
            if city not in path_map:
                return city
        
        return ""
    
    def destCity_approach4_follow_path(self, paths: List[List[str]]) -> str:
        """
        Approach 4: Follow the path to the end
        
        Since it's guaranteed to be a line, we can start from any city
        and follow the path until we reach the end.
        
        Time: O(N)
        Space: O(N)
        """
        # Build adjacency map
        adj = {}
        for from_city, to_city in paths:
            adj[from_city] = to_city
        
        # Start from the first city and follow the path
        current = paths[0][0]
        while current in adj:
            current = adj[current]
        
        return current
    
    def destCity_approach5_graph_traversal(self, paths: List[List[str]]) -> str:
        """
        Approach 5: Explicit graph traversal with in/out degree
        
        More verbose but shows complete graph analysis.
        
        Time: O(N)
        Space: O(N)
        """
        # Build complete graph representation
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        all_cities = set()
        
        for from_city, to_city in paths:
            graph[from_city].append(to_city)
            out_degree[from_city] += 1
            in_degree[to_city] += 1
            all_cities.add(from_city)
            all_cities.add(to_city)
        
        # Find city with out_degree = 0
        for city in all_cities:
            if out_degree[city] == 0:
                return city
        
        return ""

def test_destination_city():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (paths, expected)
        ([["London","New York"],["New York","Lima"],["Lima","Sao Paulo"]], "Sao Paulo"),
        ([["B","C"],["D","B"],["C","A"]], "A"),
        ([["A","Z"]], "Z"),
        ([["pYyNGfBYbm","wxAscRuzOl"],["kzwEQHfwce","pYyNGfBYbm"]], "wxAscRuzOl"),
        ([["qMTSlfgZlC","ePvzZaqrkk"],["xKhZXfuBeC","qMTSlfgZlC"]], "ePvzZaqrkk"),
    ]
    
    approaches = [
        ("Out-degree Analysis", solution.destCity_approach1_out_degree),
        ("Set Difference", solution.destCity_approach2_set_difference),
        ("Dictionary Lookup", solution.destCity_approach3_dictionary_lookup),
        ("Follow Path", solution.destCity_approach4_follow_path),
        ("Graph Traversal", solution.destCity_approach5_graph_traversal),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (paths, expected) in enumerate(test_cases):
            result = func(paths)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} Input: {paths}")
            print(f"         Expected: {expected}, Got: {result}")

def demonstrate_path_structure():
    """Demonstrate the linear path structure"""
    print("\n=== Path Structure Analysis ===")
    
    paths = [["London","New York"],["New York","Lima"],["Lima","Sao Paulo"]]
    print(f"Paths: {paths}")
    
    # Build adjacency representation
    adj = {}
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    all_cities = set()
    
    for from_city, to_city in paths:
        adj[from_city] = to_city
        out_degree[from_city] += 1
        in_degree[to_city] += 1
        all_cities.add(from_city)
        all_cities.add(to_city)
    
    print(f"\nAll cities: {sorted(all_cities)}")
    print(f"Adjacency map: {adj}")
    
    print(f"\nDegree analysis:")
    for city in sorted(all_cities):
        print(f"  {city}: in_degree={in_degree[city]}, out_degree={out_degree[city]}")
    
    # Identify special nodes
    sources = [city for city in all_cities if in_degree[city] == 0]
    destinations = [city for city in all_cities if out_degree[city] == 0]
    intermediate = [city for city in all_cities if in_degree[city] == 1 and out_degree[city] == 1]
    
    print(f"\nNode classification:")
    print(f"  Source nodes (in=0): {sources}")
    print(f"  Destination nodes (out=0): {destinations}")
    print(f"  Intermediate nodes (in=1, out=1): {intermediate}")
    
    # Trace the complete path
    print(f"\nComplete path:")
    if sources:
        current = sources[0]
        path_sequence = [current]
        while current in adj:
            current = adj[current]
            path_sequence.append(current)
        print(f"  {' -> '.join(path_sequence)}")

def analyze_problem_constraints():
    """Analyze what the problem constraints guarantee"""
    print("\n=== Problem Constraints Analysis ===")
    
    print("Given constraints guarantee:")
    print("1. The graph forms a LINE (not a cycle or tree)")
    print("2. No loops exist")
    print("3. Each city appears in at most 2 paths (once as source, once as destination)")
    print("4. Exactly one destination city exists")
    print("5. Graph is connected as a single path")
    
    print("\nThis means:")
    print("- Graph has exactly one source (in-degree 0)")
    print("- Graph has exactly one destination (out-degree 0)")
    print("- All other nodes are intermediate (in-degree 1, out-degree 1)")
    print("- Total nodes = paths + 1")
    print("- Graph structure: source -> ... -> destination")

if __name__ == "__main__":
    test_destination_city()
    demonstrate_path_structure()
    analyze_problem_constraints()

"""
Graph Theory Concepts:
1. Linear Graph Structure (Path Graph)
2. In-degree and Out-degree Analysis
3. Source and Sink Nodes
4. Graph Traversal on Simple Paths

Key Properties:
- This represents a "Path Graph" in graph theory
- Exactly one source node (in-degree = 0)
- Exactly one sink/destination node (out-degree = 0)
- All intermediate nodes have in-degree = out-degree = 1
- Total edges = total nodes - 1

Optimization Insights:
- All approaches are O(N) time complexity
- Set operations provide clean, readable solutions
- Following the path is intuitive but not always most efficient
- Out-degree analysis is the most direct approach

Real-world Applications:
- Flight itinerary planning (layover sequence)
- Package delivery routes
- Assembly line processes
- Data pipeline stages
- Network routing paths
"""
