"""
Geometry String Intersection - Multiple Approaches
Difficulty: Hard

Geometric problems involving strings and trie structures
for competitive programming.

Problems:
1. 2D Grid Path Encoding
2. String Coordinate Mapping
3. Geometric Hash Functions
4. Spatial String Queries
5. Path Intersection Detection
"""

from typing import List, Dict, Tuple, Set
from collections import defaultdict
import math

class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __repr__(self):
        return f"({self.x}, {self.y})"

class TrieNode:
    def __init__(self):
        self.children = {}
        self.paths = []
        self.coordinates = None

class GeometryStringIntersection:
    
    def __init__(self):
        self.direction_map = {
            'U': (0, 1), 'D': (0, -1),
            'L': (-1, 0), 'R': (1, 0),
            'N': (0, 1), 'S': (0, -1),
            'E': (1, 0), 'W': (-1, 0)
        }
    
    def encode_2d_path(self, path_string: str, start: Point = None) -> List[Point]:
        """
        Encode 2D path from direction string
        Time: O(|path|)
        Space: O(|path|)
        """
        if start is None:
            start = Point(0, 0)
        
        points = [start]
        current = Point(start.x, start.y)
        
        for direction in path_string:
            if direction in self.direction_map:
                dx, dy = self.direction_map[direction]
                current = Point(current.x + dx, current.y + dy)
                points.append(current)
        
        return points
    
    def string_coordinate_mapping(self, strings: List[str]) -> Dict[str, Point]:
        """
        Map strings to 2D coordinates using hash functions
        Time: O(n * m) where n=strings, m=avg_length
        Space: O(n)
        """
        mapping = {}
        
        for s in strings:
            # Simple hash to coordinates
            hash_val = hash(s)
            x = hash_val % 1000
            y = (hash_val // 1000) % 1000
            
            # Ensure positive coordinates
            x = abs(x)
            y = abs(y)
            
            mapping[s] = Point(x, y)
        
        return mapping
    
    def geometric_hash_spatial_query(self, points: List[Point], 
                                   query_rect: Tuple[Point, Point]) -> List[Point]:
        """
        Spatial query using geometric hashing
        Time: O(n) where n=points
        Space: O(n)
        """
        min_point, max_point = query_rect
        result = []
        
        for point in points:
            if (min_point.x <= point.x <= max_point.x and
                min_point.y <= point.y <= max_point.y):
                result.append(point)
        
        return result
    
    def path_intersection_detection(self, paths: List[str]) -> List[Tuple[int, int, Point]]:
        """
        Detect intersections between encoded paths
        Time: O(total_path_length^2)
        Space: O(total_path_length)
        """
        # Encode all paths
        encoded_paths = []
        for path_string in paths:
            points = self.encode_2d_path(path_string)
            encoded_paths.append(points)
        
        intersections = []
        
        # Check all pairs of paths
        for i in range(len(encoded_paths)):
            for j in range(i + 1, len(encoded_paths)):
                path1 = encoded_paths[i]
                path2 = encoded_paths[j]
                
                # Find intersection points
                points1 = set(path1)
                points2 = set(path2)
                
                common_points = points1 & points2
                
                for point in common_points:
                    intersections.append((i, j, point))
        
        return intersections
    
    def build_spatial_trie(self, path_strings: List[str]) -> TrieNode:
        """
        Build trie with spatial information
        Time: O(sum(|paths|))
        Space: O(trie_size)
        """
        root = TrieNode()
        
        for path_idx, path_string in enumerate(path_strings):
            node = root
            points = self.encode_2d_path(path_string)
            
            # Insert path into trie character by character
            for i, direction in enumerate(path_string):
                if direction not in node.children:
                    node.children[direction] = TrieNode()
                
                node = node.children[direction]
                
                # Store spatial information
                if i < len(points):
                    node.coordinates = points[i]
                
                node.paths.append(path_idx)
        
        return root
    
    def range_query_paths(self, trie_root: TrieNode, 
                         query_region: Tuple[Point, Point]) -> Set[int]:
        """
        Find all paths that pass through query region
        Time: O(trie_size)
        Space: O(result_size)
        """
        min_point, max_point = query_region
        matching_paths = set()
        
        def dfs(node: TrieNode):
            if node.coordinates:
                point = node.coordinates
                if (min_point.x <= point.x <= max_point.x and
                    min_point.y <= point.y <= max_point.y):
                    matching_paths.update(node.paths)
            
            for child in node.children.values():
                dfs(child)
        
        dfs(trie_root)
        return matching_paths
    
    def closest_string_pairs(self, string_coordinates: Dict[str, Point]) -> List[Tuple[str, str, float]]:
        """
        Find closest pairs of strings based on coordinate distance
        Time: O(n^2)
        Space: O(n)
        """
        strings = list(string_coordinates.keys())
        closest_pairs = []
        
        for i in range(len(strings)):
            for j in range(i + 1, len(strings)):
                str1, str2 = strings[i], strings[j]
                point1 = string_coordinates[str1]
                point2 = string_coordinates[str2]
                
                # Calculate Euclidean distance
                distance = math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
                closest_pairs.append((str1, str2, distance))
        
        # Sort by distance
        closest_pairs.sort(key=lambda x: x[2])
        
        return closest_pairs


def test_path_encoding():
    """Test 2D path encoding"""
    print("=== Testing 2D Path Encoding ===")
    
    geo = GeometryStringIntersection()
    
    path_strings = ["UURR", "RDRD", "ULDR"]
    
    print("Path encodings:")
    for path in path_strings:
        points = geo.encode_2d_path(path)
        print(f"'{path}': {points}")

def test_coordinate_mapping():
    """Test string to coordinate mapping"""
    print("\n=== Testing String Coordinate Mapping ===")
    
    geo = GeometryStringIntersection()
    
    strings = ["hello", "world", "test", "example"]
    
    print(f"Strings: {strings}")
    
    mapping = geo.string_coordinate_mapping(strings)
    
    print("Coordinate mapping:")
    for string, point in mapping.items():
        print(f"'{string}': {point}")

def test_path_intersections():
    """Test path intersection detection"""
    print("\n=== Testing Path Intersection Detection ===")
    
    geo = GeometryStringIntersection()
    
    paths = ["RRUU", "UURR", "RDLU"]
    
    print(f"Paths: {paths}")
    
    intersections = geo.path_intersection_detection(paths)
    
    print("Intersections found:")
    for path1_idx, path2_idx, point in intersections:
        print(f"Paths {path1_idx} and {path2_idx} intersect at {point}")

def test_spatial_queries():
    """Test spatial queries"""
    print("\n=== Testing Spatial Queries ===")
    
    geo = GeometryStringIntersection()
    
    # Build spatial trie
    paths = ["RRUU", "DDLL", "URDR"]
    trie_root = geo.build_spatial_trie(paths)
    
    print(f"Built spatial trie for paths: {paths}")
    
    # Query region
    query_region = (Point(-1, -1), Point(2, 2))
    print(f"Query region: {query_region[0]} to {query_region[1]}")
    
    matching_paths = geo.range_query_paths(trie_root, query_region)
    print(f"Paths in query region: {matching_paths}")

def test_closest_pairs():
    """Test closest string pairs"""
    print("\n=== Testing Closest String Pairs ===")
    
    geo = GeometryStringIntersection()
    
    strings = ["abc", "def", "ghi", "xyz"]
    coordinates = geo.string_coordinate_mapping(strings)
    
    print("String coordinates:")
    for string, point in coordinates.items():
        print(f"'{string}': {point}")
    
    closest_pairs = geo.closest_string_pairs(coordinates)
    
    print("\nClosest pairs (top 3):")
    for i, (str1, str2, distance) in enumerate(closest_pairs[:3]):
        print(f"{i+1}. '{str1}' and '{str2}': distance = {distance:.2f}")

if __name__ == "__main__":
    test_path_encoding()
    test_coordinate_mapping()
    test_path_intersections()
    test_spatial_queries()
    test_closest_pairs()
