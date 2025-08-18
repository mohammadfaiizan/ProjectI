"""
733. Flood Fill
Difficulty: Easy

Problem:
An image is represented by an m x n integer grid image where image[i][j] represents 
the pixel value at the position (i, j).

You are also given three integers sr, sc, and color. You should perform a flood fill 
on the image starting from the pixel image[sr][sc].

To perform a flood fill, consider the starting pixel, plus any pixels connected 
4-directionally to the starting pixel of the same color as the starting pixel, plus 
any pixels connected 4-directionally to those pixels (also with the same color), and so on. 
Replace the color of all of the aforementioned pixels with color.

Return the modified image after performing the flood fill.

Examples:
Input: image = [[1,1,1],[1,1,0],[1,0,1]], sr = 1, sc = 1, color = 2
Output: [[2,2,2],[2,2,0],[2,0,1]]

Input: image = [[0,0,0],[0,0,0]], sr = 0, sc = 0, color = 0
Output: [[0,0,0],[0,0,0]]

Constraints:
- m == image.length
- n == image[i].length
- 1 <= m, n <= 50
- 0 <= image[i][j], color < 2^16
- 0 <= sr < m
- 0 <= sc < n
"""

from typing import List
from collections import deque

class Solution:
    def floodFill_approach1_dfs_recursive(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        """
        Approach 1: DFS with Recursion (Classic Flood Fill)
        
        Start from (sr, sc) and recursively fill all connected pixels
        of the same original color with the new color.
        
        Time: O(M*N) - worst case visit all pixels
        Space: O(M*N) - recursion stack depth
        """
        if not image or not image[0]:
            return image
        
        original_color = image[sr][sc]
        
        # Edge case: if new color is same as original, no change needed
        if original_color == color:
            return image
        
        m, n = len(image), len(image[0])
        
        def dfs(i, j):
            # Boundary check and color check
            if (i < 0 or i >= m or j < 0 or j >= n or 
                image[i][j] != original_color):
                return
            
            # Fill current pixel
            image[i][j] = color
            
            # Recursively fill 4 directions
            dfs(i + 1, j)  # Down
            dfs(i - 1, j)  # Up
            dfs(i, j + 1)  # Right
            dfs(i, j - 1)  # Left
        
        dfs(sr, sc)
        return image
    
    def floodFill_approach2_dfs_iterative(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        """
        Approach 2: DFS with Iteration (Stack-based)
        
        Use explicit stack to avoid recursion depth issues.
        
        Time: O(M*N)
        Space: O(M*N) - stack size
        """
        if not image or not image[0]:
            return image
        
        original_color = image[sr][sc]
        
        if original_color == color:
            return image
        
        m, n = len(image), len(image[0])
        stack = [(sr, sc)]
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        while stack:
            i, j = stack.pop()
            
            # Check bounds and color
            if (i < 0 or i >= m or j < 0 or j >= n or 
                image[i][j] != original_color):
                continue
            
            # Fill current pixel
            image[i][j] = color
            
            # Add neighbors to stack
            for di, dj in directions:
                stack.append((i + di, j + dj))
        
        return image
    
    def floodFill_approach3_bfs(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        """
        Approach 3: BFS (Level-order traversal)
        
        Use BFS to fill pixels level by level.
        
        Time: O(M*N)
        Space: O(min(M,N)) - queue size
        """
        if not image or not image[0]:
            return image
        
        original_color = image[sr][sc]
        
        if original_color == color:
            return image
        
        m, n = len(image), len(image[0])
        queue = deque([(sr, sc)])
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        # Mark starting pixel
        image[sr][sc] = color
        
        while queue:
            i, j = queue.popleft()
            
            # Check all 4 neighbors
            for di, dj in directions:
                ni, nj = i + di, j + dj
                
                if (0 <= ni < m and 0 <= nj < n and 
                    image[ni][nj] == original_color):
                    image[ni][nj] = color
                    queue.append((ni, nj))
        
        return image
    
    def floodFill_approach4_non_destructive(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        """
        Approach 4: Non-destructive (creates new image)
        
        Create a copy of the image and perform flood fill on copy.
        
        Time: O(M*N)
        Space: O(M*N) - new image + recursion stack
        """
        if not image or not image[0]:
            return image
        
        # Create deep copy
        result = [row[:] for row in image]
        original_color = result[sr][sc]
        
        if original_color == color:
            return result
        
        m, n = len(result), len(result[0])
        
        def dfs(i, j):
            if (i < 0 or i >= m or j < 0 or j >= n or 
                result[i][j] != original_color):
                return
            
            result[i][j] = color
            
            dfs(i + 1, j)
            dfs(i - 1, j)
            dfs(i, j + 1)
            dfs(i, j - 1)
        
        dfs(sr, sc)
        return result
    
    def floodFill_approach5_visited_tracking(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        """
        Approach 5: Explicit visited tracking
        
        Use visited set to track processed pixels.
        Useful when we can't modify original colors during processing.
        
        Time: O(M*N)
        Space: O(M*N) - visited set
        """
        if not image or not image[0]:
            return image
        
        original_color = image[sr][sc]
        
        if original_color == color:
            return image
        
        m, n = len(image), len(image[0])
        visited = set()
        
        def dfs(i, j):
            if ((i, j) in visited or i < 0 or i >= m or 
                j < 0 or j >= n or image[i][j] != original_color):
                return
            
            visited.add((i, j))
            image[i][j] = color
            
            dfs(i + 1, j)
            dfs(i - 1, j)
            dfs(i, j + 1)
            dfs(i, j - 1)
        
        dfs(sr, sc)
        return image

def test_flood_fill():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (image, sr, sc, color, expected)
        ([[1,1,1],[1,1,0],[1,0,1]], 1, 1, 2, [[2,2,2],[2,2,0],[2,0,1]]),
        ([[0,0,0],[0,0,0]], 0, 0, 0, [[0,0,0],[0,0,0]]),  # Same color
        ([[0,0,0],[0,1,1]], 1, 1, 1, [[0,0,0],[0,1,1]]),  # Same color
        ([[1]], 0, 0, 2, [[2]]),  # Single pixel
        ([[1,2,3],[4,5,6]], 0, 0, 9, [[9,2,3],[4,5,6]]),  # Single pixel change
        ([[0,0,0],[0,0,0],[0,0,0]], 1, 1, 2, [[2,2,2],[2,2,2],[2,2,2]]),  # All same color
    ]
    
    approaches = [
        ("DFS Recursive", solution.floodFill_approach1_dfs_recursive),
        ("DFS Iterative", solution.floodFill_approach2_dfs_iterative),
        ("BFS", solution.floodFill_approach3_bfs),
        ("Non-destructive", solution.floodFill_approach4_non_destructive),
        ("Visited Tracking", solution.floodFill_approach5_visited_tracking),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (image, sr, sc, color, expected) in enumerate(test_cases):
            # Create deep copy for testing
            test_image = [row[:] for row in image]
            result = func(test_image, sr, sc, color)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status}")
            print(f"         Input: {image} at ({sr},{sc}) -> {color}")
            print(f"         Expected: {expected}")
            print(f"         Got: {result}")

def demonstrate_flood_fill_process():
    """Demonstrate step-by-step flood fill process"""
    print("\n=== Flood Fill Process Demo ===")
    
    image = [
        [1, 1, 1],
        [1, 1, 0],
        [1, 0, 1]
    ]
    sr, sc, color = 1, 1, 2
    
    print(f"Original image:")
    for i, row in enumerate(image):
        print(f"  Row {i}: {row}")
    
    print(f"\nStarting flood fill from ({sr}, {sc}) with color {color}")
    print(f"Original color at start: {image[sr][sc]}")
    
    # Manual step-by-step process
    original_color = image[sr][sc]
    visited = set()
    steps = []
    
    def dfs_trace(i, j, step):
        if ((i, j) in visited or i < 0 or i >= 3 or 
            j < 0 or j >= 3 or image[i][j] != original_color):
            return step
        
        visited.add((i, j))
        old_color = image[i][j]
        image[i][j] = color
        steps.append((step, i, j, old_color, color))
        step += 1
        
        # Explore 4 directions
        step = dfs_trace(i + 1, j, step)
        step = dfs_trace(i - 1, j, step)
        step = dfs_trace(i, j + 1, step)
        step = dfs_trace(i, j - 1, step)
        
        return step
    
    dfs_trace(sr, sc, 1)
    
    print(f"\nFlood fill steps:")
    for step, i, j, old, new in steps:
        print(f"  Step {step}: Fill ({i},{j}) from {old} to {new}")
    
    print(f"\nFinal image:")
    for i, row in enumerate(image):
        print(f"  Row {i}: {row}")

def visualize_flood_fill():
    """Create visual representation of flood fill"""
    print("\n=== Flood Fill Visualization ===")
    
    # Create a more interesting pattern
    image = [
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 1, 1, 2],
        [3, 3, 1, 2]
    ]
    sr, sc, color = 0, 0, 9
    
    print("Original image with emoji representation:")
    emoji_map = {1: "🟦", 2: "🟨", 3: "🟩", 9: "🟥"}
    
    for i, row in enumerate(image):
        display_row = [emoji_map.get(cell, "⬜") for cell in row]
        print(f"  Row {i}: {' '.join(display_row)} {row}")
    
    print(f"\nFlood filling from ({sr}, {sc}) with color {color}")
    
    # Perform flood fill
    original_color = image[sr][sc]
    if original_color != color:
        def dfs(i, j):
            if (i < 0 or i >= 4 or j < 0 or j >= 4 or 
                image[i][j] != original_color):
                return
            
            image[i][j] = color
            dfs(i + 1, j)
            dfs(i - 1, j)
            dfs(i, j + 1)
            dfs(i, j - 1)
        
        dfs(sr, sc)
    
    print("\nAfter flood fill:")
    for i, row in enumerate(image):
        display_row = [emoji_map.get(cell, "⬜") for cell in row]
        print(f"  Row {i}: {' '.join(display_row)} {row}")

def analyze_flood_fill_applications():
    """Analyze real-world applications of flood fill"""
    print("\n=== Flood Fill Applications ===")
    
    applications = [
        ("Paint Programs", "Fill enclosed areas with color", "Photoshop bucket fill"),
        ("Image Processing", "Region growing, segmentation", "Medical image analysis"),
        ("Game Development", "Territory marking, pathfinding", "Strategy games, map generation"),
        ("Computer Graphics", "Texture mapping, rendering", "3D graphics, mesh processing"),
        ("GIS Systems", "Land classification, region analysis", "Geographic data processing"),
        ("Circuit Design", "Component region identification", "PCB layout tools"),
        ("Maze Solving", "Path finding, dead-end detection", "Robotics, AI pathfinding"),
        ("Data Visualization", "Interactive region highlighting", "Dashboard interfaces"),
    ]
    
    print(f"{'Domain':<18} {'Application':<35} {'Examples'}")
    print("-" * 80)
    
    for domain, application, examples in applications:
        print(f"{domain:<18} {application:<35} {examples}")
    
    print(f"\nAlgorithm Characteristics:")
    print(f"- Simple to implement and understand")
    print(f"- Natural recursive structure")
    print(f"- Can be adapted for various connectivity rules")
    print(f"- Foundation for many computer graphics algorithms")

def compare_traversal_methods():
    """Compare DFS vs BFS for flood fill"""
    print("\n=== DFS vs BFS Comparison ===")
    
    comparison = [
        ("Aspect", "DFS", "BFS"),
        ("Memory Usage", "O(depth) recursion", "O(width) queue"),
        ("Implementation", "Simple recursive", "Queue management"),
        ("Stack Overflow Risk", "Yes, for deep regions", "No"),
        ("Fill Order", "Depth-first", "Level-by-level"),
        ("Cache Performance", "Better locality", "Scattered access"),
        ("Practical Use", "Most common", "Special cases"),
    ]
    
    for aspect, dfs_char, bfs_char in comparison:
        if aspect == "Aspect":
            print(f"{aspect:<20} {dfs_char:<20} {bfs_char}")
            print("-" * 60)
        else:
            print(f"{aspect:<20} {dfs_char:<20} {bfs_char}")
    
    print(f"\nRecommendation:")
    print(f"- Use DFS for most flood fill applications")
    print(f"- Use BFS when you need level-by-level processing")
    print(f"- Use iterative DFS for very large regions")

if __name__ == "__main__":
    test_flood_fill()
    demonstrate_flood_fill_process()
    visualize_flood_fill()
    analyze_flood_fill_applications()
    compare_traversal_methods()

"""
Graph Theory Concepts:
1. Connected Component Modification
2. Color/Value Propagation in Graphs
3. 4-directional Grid Connectivity
4. Region Growing Algorithms

Key Flood Fill Concepts:
- Starting point and propagation rules
- Boundary conditions and termination
- Color matching and replacement
- Connected region processing

Algorithm Variants:
- DFS: Natural recursive implementation
- BFS: Level-by-level filling
- Iterative: Stack-based to avoid recursion limits
- Non-destructive: Preserves original image

Optimization Considerations:
- Early termination for same color
- Memory usage (recursion vs iteration)
- Cache locality and performance
- Boundary checking efficiency

Real-world Applications:
- Paint bucket tool in graphics software
- Region growing in image segmentation
- Territory marking in games
- Interactive selection tools
- Maze solving and pathfinding

This is the classic flood fill algorithm - foundation for many graphics
and image processing applications!
"""
