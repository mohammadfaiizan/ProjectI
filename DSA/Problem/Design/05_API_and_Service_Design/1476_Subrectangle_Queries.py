"""
1476. Subrectangle Queries - Multiple Approaches
Difficulty: Medium

Implement the class SubrectangleQueries which receives a rectangle as a 2D array of integers and supports two methods:

1. updateSubrectangle(int row1, int col1, int row2, int col2, int newValue): Updates all values with newValue in the subrectangle whose upper left coordinate is (row1,col1) and bottom right coordinate is (row2,col2).

2. getValue(int row, int col): Returns the current value of the coordinate (row,col) from the rectangle.
"""

from typing import List, Tuple
import copy

class SubrectangleQueriesNaive:
    """
    Approach 1: Naive Direct Update
    
    Directly update all cells in the subrectangle.
    
    Time Complexity:
    - updateSubrectangle: O((row2-row1+1) * (col2-col1+1))
    - getValue: O(1)
    
    Space Complexity: O(rows * cols)
    """
    
    def __init__(self, rectangle: List[List[int]]):
        self.rectangle = [row[:] for row in rectangle]  # Deep copy
    
    def updateSubrectangle(self, row1: int, col1: int, row2: int, col2: int, newValue: int) -> None:
        for r in range(row1, row2 + 1):
            for c in range(col1, col2 + 1):
                self.rectangle[r][c] = newValue
    
    def getValue(self, row: int, col: int) -> int:
        return self.rectangle[row][col]

class SubrectangleQueriesHistoryBased:
    """
    Approach 2: History-Based Updates
    
    Store update operations and apply them when querying.
    
    Time Complexity:
    - updateSubrectangle: O(1)
    - getValue: O(u) where u is number of updates
    
    Space Complexity: O(rows * cols + updates)
    """
    
    def __init__(self, rectangle: List[List[int]]):
        self.original = [row[:] for row in rectangle]
        self.updates = []  # List of (row1, col1, row2, col2, newValue)
    
    def updateSubrectangle(self, row1: int, col1: int, row2: int, col2: int, newValue: int) -> None:
        self.updates.append((row1, col1, row2, col2, newValue))
    
    def getValue(self, row: int, col: int) -> int:
        # Start with original value
        value = self.original[row][col]
        
        # Apply updates in chronological order
        for r1, c1, r2, c2, new_val in self.updates:
            if r1 <= row <= r2 and c1 <= col <= c2:
                value = new_val
        
        return value

class SubrectangleQueriesLazy:
    """
    Approach 3: Lazy Propagation with Segment Tree Concept
    
    Use lazy propagation idea for efficient range updates.
    
    Time Complexity:
    - updateSubrectangle: O(1)
    - getValue: O(log n) in theory, O(u) in this implementation
    
    Space Complexity: O(rows * cols + updates)
    """
    
    def __init__(self, rectangle: List[List[int]]):
        self.original = [row[:] for row in rectangle]
        self.lazy_updates = []  # Store pending updates
        self.rows = len(rectangle)
        self.cols = len(rectangle[0]) if rectangle else 0
    
    def updateSubrectangle(self, row1: int, col1: int, row2: int, col2: int, newValue: int) -> None:
        # Add to lazy updates
        self.lazy_updates.append((row1, col1, row2, col2, newValue))
    
    def getValue(self, row: int, col: int) -> int:
        # Apply all relevant lazy updates
        value = self.original[row][col]
        
        for r1, c1, r2, c2, new_val in self.lazy_updates:
            if r1 <= row <= r2 and c1 <= col <= c2:
                value = new_val
        
        return value
    
    def _flush_updates(self) -> None:
        """Apply all lazy updates to the original matrix"""
        for r1, c1, r2, c2, new_val in self.lazy_updates:
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    self.original[r][c] = new_val
        
        self.lazy_updates.clear()

class SubrectangleQueriesAdvanced:
    """
    Approach 4: Advanced with Features and Analytics
    
    Enhanced version with statistics and additional functionality.
    
    Time Complexity:
    - updateSubrectangle: O(1) for lazy, O(area) for immediate
    - getValue: O(u) where u is updates affecting the cell
    
    Space Complexity: O(rows * cols + updates + analytics)
    """
    
    def __init__(self, rectangle: List[List[int]]):
        self.original = [row[:] for row in rectangle]
        self.current = [row[:] for row in rectangle]  # Current state
        self.updates = []  # History of updates
        
        # Analytics
        self.rows = len(rectangle)
        self.cols = len(rectangle[0]) if rectangle else 0
        self.update_count = 0
        self.query_count = 0
        self.cell_access_count = {}  # Track cell access frequency
        self.update_areas = []  # Track update area sizes
        
        # Features
        self.lazy_mode = True  # Toggle between lazy and immediate updates
        self.update_history = []  # Detailed update history with timestamps
        
    def updateSubrectangle(self, row1: int, col1: int, row2: int, col2: int, newValue: int) -> None:
        import time
        
        self.update_count += 1
        area = (row2 - row1 + 1) * (col2 - col1 + 1)
        self.update_areas.append(area)
        
        # Store update with metadata
        update_info = {
            'row1': row1, 'col1': col1, 'row2': row2, 'col2': col2,
            'newValue': newValue, 'timestamp': time.time(),
            'update_id': self.update_count, 'area': area
        }
        
        self.updates.append((row1, col1, row2, col2, newValue))
        self.update_history.append(update_info)
        
        if not self.lazy_mode:
            # Immediate update
            for r in range(row1, row2 + 1):
                for c in range(col1, col2 + 1):
                    self.current[r][c] = newValue
    
    def getValue(self, row: int, col: int) -> int:
        self.query_count += 1
        
        # Track cell access
        cell_key = (row, col)
        self.cell_access_count[cell_key] = self.cell_access_count.get(cell_key, 0) + 1
        
        if self.lazy_mode:
            # Apply updates on-demand
            value = self.original[row][col]
            
            for r1, c1, r2, c2, new_val in self.updates:
                if r1 <= row <= r2 and c1 <= col <= c2:
                    value = new_val
            
            return value
        else:
            # Return current state
            return self.current[row][col]
    
    def getStatistics(self) -> dict:
        """Get comprehensive statistics"""
        avg_update_area = sum(self.update_areas) / max(1, len(self.update_areas))
        most_accessed_cell = max(self.cell_access_count.items(), 
                               key=lambda x: x[1], default=((0, 0), 0))
        
        return {
            'total_updates': self.update_count,
            'total_queries': self.query_count,
            'average_update_area': avg_update_area,
            'max_update_area': max(self.update_areas, default=0),
            'most_accessed_cell': most_accessed_cell,
            'matrix_size': (self.rows, self.cols),
            'lazy_mode': self.lazy_mode
        }
    
    def getUpdateHistory(self, limit: int = 10) -> List[dict]:
        """Get recent update history"""
        return self.update_history[-limit:]
    
    def getCellHistory(self, row: int, col: int) -> List[dict]:
        """Get history of updates affecting a specific cell"""
        cell_history = []
        
        for update in self.update_history:
            if (update['row1'] <= row <= update['row2'] and 
                update['col1'] <= col <= update['col2']):
                cell_history.append(update)
        
        return cell_history
    
    def setLazyMode(self, lazy: bool) -> None:
        """Toggle between lazy and immediate update modes"""
        if self.lazy_mode and not lazy:
            # Switching from lazy to immediate - apply all pending updates
            self._flush_all_updates()
        
        self.lazy_mode = lazy
    
    def _flush_all_updates(self) -> None:
        """Apply all pending updates to current matrix"""
        for r1, c1, r2, c2, new_val in self.updates:
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    self.current[r][c] = new_val
    
    def getCurrentMatrix(self) -> List[List[int]]:
        """Get current state of the matrix"""
        if self.lazy_mode:
            # Apply all updates to get current state
            result = [row[:] for row in self.original]
            
            for r1, c1, r2, c2, new_val in self.updates:
                for r in range(r1, r2 + 1):
                    for c in range(c1, c2 + 1):
                        result[r][c] = new_val
            
            return result
        else:
            return [row[:] for row in self.current]

class SubrectangleQueriesOptimized:
    """
    Approach 5: Optimized for Query-Heavy Workloads
    
    Optimize for scenarios with many queries relative to updates.
    
    Time Complexity:
    - updateSubrectangle: O(1) to O(area) depending on strategy
    - getValue: O(1) to O(u) depending on strategy
    
    Space Complexity: O(rows * cols + optimizations)
    """
    
    def __init__(self, rectangle: List[List[int]]):
        self.original = [row[:] for row in rectangle]
        self.updates = []
        
        # Optimization parameters
        self.max_lazy_updates = 50  # Threshold for applying updates
        self.query_count_since_flush = 0
        self.total_queries = 0
        
    def updateSubrectangle(self, row1: int, col1: int, row2: int, col2: int, newValue: int) -> None:
        self.updates.append((row1, col1, row2, col2, newValue))
        
        # Apply updates if we have too many pending
        if len(self.updates) > self.max_lazy_updates:
            self._apply_updates()
    
    def getValue(self, row: int, col: int) -> int:
        self.total_queries += 1
        self.query_count_since_flush += 1
        
        # Consider applying updates if we've had many queries
        if self.query_count_since_flush > 100 and len(self.updates) > 10:
            self._apply_updates()
        
        # Get value with current updates
        value = self.original[row][col]
        
        for r1, c1, r2, c2, new_val in self.updates:
            if r1 <= row <= r2 and c1 <= col <= c2:
                value = new_val
        
        return value
    
    def _apply_updates(self) -> None:
        """Apply all pending updates to the matrix"""
        for r1, c1, r2, c2, new_val in self.updates:
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    self.original[r][c] = new_val
        
        self.updates.clear()
        self.query_count_since_flush = 0


def test_subrectangle_queries_basic():
    """Test basic subrectangle queries functionality"""
    print("=== Testing Basic Subrectangle Queries Functionality ===")
    
    implementations = [
        ("Naive", SubrectangleQueriesNaive),
        ("History-Based", SubrectangleQueriesHistoryBased),
        ("Lazy", SubrectangleQueriesLazy),
        ("Advanced", SubrectangleQueriesAdvanced),
        ("Optimized", SubrectangleQueriesOptimized)
    ]
    
    # Test matrix
    test_matrix = [
        [1, 2, 1],
        [4, 3, 4],
        [3, 2, 1]
    ]
    
    for name, QueryClass in implementations:
        print(f"\n{name}:")
        
        queries = QueryClass(test_matrix)
        
        # Test initial values
        print(f"  Initial getValue(0, 2): {queries.getValue(0, 2)}")
        
        # Update subrectangle
        queries.updateSubrectangle(0, 0, 2, 2, 5)
        print(f"  After updateSubrectangle(0, 0, 2, 2, 5):")
        
        # Test values after update
        test_positions = [(0, 2), (1, 1), (2, 1)]
        for row, col in test_positions:
            value = queries.getValue(row, col)
            print(f"    getValue({row}, {col}): {value}")

def test_subrectangle_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Subrectangle Edge Cases ===")
    
    # Single cell matrix
    print("Single cell matrix:")
    single_cell = [[42]]
    queries = SubrectangleQueriesAdvanced(single_cell)
    
    print(f"  Initial value: {queries.getValue(0, 0)}")
    queries.updateSubrectangle(0, 0, 0, 0, 100)
    print(f"  After update: {queries.getValue(0, 0)}")
    
    # Large matrix with small updates
    print(f"\nLarge matrix with small updates:")
    large_matrix = [[i * 10 + j for j in range(10)] for i in range(10)]
    large_queries = SubrectangleQueriesLazy(large_matrix)
    
    # Small update in corner
    large_queries.updateSubrectangle(0, 0, 1, 1, 999)
    
    # Test boundary values
    boundary_tests = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)]
    for row, col in boundary_tests:
        value = large_queries.getValue(row, col)
        print(f"    ({row}, {col}): {value}")
    
    # Multiple overlapping updates
    print(f"\nOverlapping updates:")
    overlap_matrix = [[0] * 3 for _ in range(3)]
    overlap_queries = SubrectangleQueriesHistoryBased(overlap_matrix)
    
    # Apply overlapping updates
    overlap_queries.updateSubrectangle(0, 0, 2, 2, 1)  # Fill with 1
    overlap_queries.updateSubrectangle(1, 1, 1, 1, 5)  # Center to 5
    overlap_queries.updateSubrectangle(0, 0, 1, 2, 3)  # Top rows to 3
    
    print(f"  Final matrix state:")
    for r in range(3):
        row_values = [overlap_queries.getValue(r, c) for c in range(3)]
        print(f"    Row {r}: {row_values}")

def test_advanced_features():
    """Test advanced features"""
    print("\n=== Testing Advanced Features ===")
    
    matrix = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    
    queries = SubrectangleQueriesAdvanced(matrix)
    
    # Multiple updates
    queries.updateSubrectangle(0, 0, 1, 1, 2)
    queries.updateSubrectangle(1, 1, 2, 2, 3)
    queries.updateSubrectangle(0, 0, 0, 2, 4)
    
    # Some queries
    queries.getValue(0, 0)
    queries.getValue(1, 1)
    queries.getValue(2, 2)
    queries.getValue(0, 1)  # Access same cells multiple times
    queries.getValue(0, 1)
    
    # Get statistics
    stats = queries.getStatistics()
    print("Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Get update history
    history = queries.getUpdateHistory(5)
    print(f"\nRecent update history:")
    for i, update in enumerate(history):
        print(f"  Update {i+1}: ({update['row1']},{update['col1']}) to ({update['row2']},{update['col2']}) = {update['newValue']}")
    
    # Get cell-specific history
    cell_history = queries.getCellHistory(0, 0)
    print(f"\nHistory for cell (0,0):")
    for update in cell_history:
        print(f"  Update {update['update_id']}: value = {update['newValue']}, area = {update['area']}")
    
    # Test mode switching
    print(f"\nTesting lazy mode switching:")
    print(f"  Current lazy mode: {queries.lazy_mode}")
    
    queries.setLazyMode(False)
    print(f"  Switched to immediate mode")
    
    queries.updateSubrectangle(2, 0, 2, 2, 9)
    value = queries.getValue(2, 1)
    print(f"  getValue(2, 1) in immediate mode: {value}")

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: Image processing
    print("Application 1: Image Processing - Region Filling")
    
    # Represent image as matrix of pixel values
    image = [
        [100, 100, 100, 200, 200],
        [100, 150, 150, 200, 200],
        [100, 150, 150, 200, 255],
        [50,  50,  50,  255, 255],
        [50,  50,  50,  255, 255]
    ]
    
    image_processor = SubrectangleQueriesAdvanced(image)
    
    print("  Original image (pixel values):")
    for row in image:
        print(f"    {row}")
    
    # Fill regions
    print(f"\n  Applying region fills:")
    
    # Fill top-left region with red (value 255)
    image_processor.updateSubrectangle(0, 0, 2, 2, 255)
    print(f"    Filled region (0,0) to (2,2) with red (255)")
    
    # Fill bottom-right corner with blue (value 0)
    image_processor.updateSubrectangle(3, 3, 4, 4, 0)
    print(f"    Filled region (3,3) to (4,4) with blue (0)")
    
    # Show result
    current_image = image_processor.getCurrentMatrix()
    print(f"\n  Processed image:")
    for row in current_image:
        print(f"    {row}")
    
    # Application 2: Game map editing
    print(f"\nApplication 2: Game Map Terrain Editing")
    
    # Terrain types: 0=water, 1=grass, 2=mountain, 3=desert
    game_map = [
        [1, 1, 1, 1, 1],
        [1, 2, 2, 1, 1],
        [1, 2, 2, 1, 1],
        [0, 0, 1, 1, 3],
        [0, 0, 1, 3, 3]
    ]
    
    terrain_editor = SubrectangleQueriesOptimized(game_map)
    
    terrain_names = {0: "Water", 1: "Grass", 2: "Mountain", 3: "Desert"}
    
    print("  Original terrain map:")
    for r, row in enumerate(game_map):
        terrain_row = [terrain_names[cell] for cell in row]
        print(f"    Row {r}: {terrain_row}")
    
    # Edit terrain
    print(f"\n  Terrain modifications:")
    
    # Convert mountain area to grass
    terrain_editor.updateSubrectangle(1, 1, 2, 2, 1)
    print(f"    Converted mountain area to grass")
    
    # Create water channel
    terrain_editor.updateSubrectangle(0, 2, 4, 2, 0)
    print(f"    Created water channel through middle")
    
    # Show specific locations
    test_locations = [(1, 1), (2, 2), (0, 2), (3, 2)]
    print(f"\n  Checking specific locations:")
    for r, c in test_locations:
        terrain_type = terrain_editor.getValue(r, c)
        terrain_name = terrain_names[terrain_type]
        print(f"    Location ({r},{c}): {terrain_name}")
    
    # Application 3: Spreadsheet region formatting
    print(f"\nApplication 3: Spreadsheet Cell Formatting")
    
    # Format codes: 0=normal, 1=bold, 2=italic, 3=highlighted
    spreadsheet = [[0] * 6 for _ in range(4)]
    formatter = SubrectangleQueriesAdvanced(spreadsheet)
    
    format_names = {0: "Normal", 1: "Bold", 2: "Italic", 3: "Highlighted"}
    
    print("  Applying cell formatting:")
    
    # Header row formatting
    formatter.updateSubrectangle(0, 0, 0, 5, 1)  # Bold headers
    print(f"    Applied bold formatting to header row")
    
    # Highlight important section
    formatter.updateSubrectangle(1, 1, 2, 3, 3)  # Highlight data area
    print(f"    Highlighted data section")
    
    # Italicize summary row
    formatter.updateSubrectangle(3, 0, 3, 5, 2)  # Italic summary
    print(f"    Applied italic to summary row")
    
    # Show formatting grid
    print(f"\n  Current formatting:")
    current_formatting = formatter.getCurrentMatrix()
    for r, row in enumerate(current_formatting):
        format_row = [format_names[cell] for cell in row]
        print(f"    Row {r}: {format_row}")
    
    # Show statistics
    stats = formatter.getStatistics()
    print(f"\n  Formatting statistics:")
    print(f"    Total format operations: {stats['total_updates']}")
    print(f"    Average operation area: {stats['average_update_area']:.1f} cells")

def test_performance():
    """Test performance with different update patterns"""
    print("\n=== Testing Performance ===")
    
    import time
    
    # Create large matrix
    size = 100
    large_matrix = [[i * size + j for j in range(size)] for i in range(size)]
    
    implementations = [
        ("Naive", SubrectangleQueriesNaive),
        ("History-Based", SubrectangleQueriesHistoryBased),
        ("Optimized", SubrectangleQueriesOptimized)
    ]
    
    update_patterns = [
        ("Small updates", [(0, 0, 1, 1), (5, 5, 6, 6), (10, 10, 11, 11)]),
        ("Medium updates", [(0, 0, 9, 9), (20, 20, 29, 29), (50, 50, 59, 59)]),
        ("Large updates", [(0, 0, 49, 49), (25, 25, 74, 74)])
    ]
    
    for pattern_name, updates in update_patterns:
        print(f"\n{pattern_name}:")
        
        for impl_name, QueryClass in implementations:
            queries = QueryClass(large_matrix)
            
            # Time updates
            start_time = time.time()
            
            for r1, c1, r2, c2 in updates:
                queries.updateSubrectangle(r1, c1, r2, c2, 999)
            
            update_time = (time.time() - start_time) * 1000
            
            # Time queries
            start_time = time.time()
            
            # Test random queries
            import random
            for _ in range(100):
                r = random.randint(0, size - 1)
                c = random.randint(0, size - 1)
                queries.getValue(r, c)
            
            query_time = (time.time() - start_time) * 1000
            
            print(f"    {impl_name}: Updates {update_time:.2f}ms, Queries {query_time:.2f}ms")

def stress_test_subrectangle():
    """Stress test with many operations"""
    print("\n=== Stress Testing Subrectangle Queries ===")
    
    import time
    import random
    
    # Medium-sized matrix for stress test
    size = 50
    matrix = [[0] * size for _ in range(size)]
    
    queries = SubrectangleQueriesHistoryBased(matrix)
    
    print(f"Stress test on {size}x{size} matrix:")
    
    # Many small updates
    num_updates = 1000
    num_queries = 2000
    
    start_time = time.time()
    
    # Random updates
    for _ in range(num_updates):
        r1 = random.randint(0, size - 5)
        c1 = random.randint(0, size - 5)
        r2 = r1 + random.randint(0, 4)
        c2 = c1 + random.randint(0, 4)
        value = random.randint(1, 100)
        
        queries.updateSubrectangle(r1, c1, r2, c2, value)
    
    update_time = time.time() - start_time
    
    # Random queries
    start_time = time.time()
    
    for _ in range(num_queries):
        r = random.randint(0, size - 1)
        c = random.randint(0, size - 1)
        queries.getValue(r, c)
    
    query_time = time.time() - start_time
    
    print(f"  {num_updates} updates: {update_time:.2f}s ({update_time*1000/num_updates:.3f}ms each)")
    print(f"  {num_queries} queries: {query_time:.2f}s ({query_time*1000/num_queries:.3f}ms each)")

def test_memory_efficiency():
    """Test memory efficiency of different approaches"""
    print("\n=== Testing Memory Efficiency ===")
    
    # Test with many updates
    matrix = [[1] * 10 for _ in range(10)]
    
    approaches = [
        ("Naive (immediate update)", SubrectangleQueriesNaive),
        ("History-based (lazy)", SubrectangleQueriesHistoryBased),
        ("Advanced (configurable)", SubrectangleQueriesAdvanced)
    ]
    
    num_updates = 100
    
    for name, QueryClass in approaches:
        queries = QueryClass(matrix)
        
        # Apply many updates
        for i in range(num_updates):
            r1, c1 = i % 8, i % 8
            r2, c2 = (i % 8) + 1, (i % 8) + 1
            queries.updateSubrectangle(r1, c1, r2, c2, i)
        
        # Estimate memory usage
        if hasattr(queries, 'updates'):
            update_memory = len(queries.updates) * 5  # 5 values per update
            approach_type = "Lazy (stores updates)"
        else:
            update_memory = 0
            approach_type = "Immediate (no update storage)"
        
        matrix_memory = 10 * 10  # Original matrix
        total_memory = matrix_memory + update_memory
        
        print(f"  {name}: ~{total_memory} units")
        print(f"    Matrix: {matrix_memory}, Updates: {update_memory}")
        print(f"    Type: {approach_type}")

def benchmark_query_patterns():
    """Benchmark different query patterns"""
    print("\n=== Benchmarking Query Patterns ===")
    
    import time
    
    matrix = [[i * 20 + j for j in range(20)] for i in range(20)]
    queries = SubrectangleQueriesAdvanced(matrix)
    
    # Apply some updates first
    queries.updateSubrectangle(0, 0, 5, 5, 100)
    queries.updateSubrectangle(10, 10, 15, 15, 200)
    queries.updateSubrectangle(5, 5, 10, 10, 150)
    
    query_patterns = [
        ("Random access", lambda: (random.randint(0, 19), random.randint(0, 19))),
        ("Sequential access", None),  # Special case
        ("Localized access", lambda: (random.randint(0, 5), random.randint(0, 5))),
        ("Sparse access", lambda: (random.randint(0, 19) * random.choice([0, 1]), 
                                 random.randint(0, 19) * random.choice([0, 1])))
    ]
    
    num_queries = 1000
    
    for pattern_name, coord_gen in query_patterns:
        start_time = time.time()
        
        if pattern_name == "Sequential access":
            # Sequential pattern
            for i in range(20):
                for j in range(min(20, num_queries - i * 20)):
                    queries.getValue(i, j)
        else:
            # Pattern-based access
            import random
            random.seed(42)  # Consistent results
            
            for _ in range(num_queries):
                r, c = coord_gen()
                queries.getValue(r, c)
        
        elapsed = (time.time() - start_time) * 1000
        
        print(f"  {pattern_name}: {elapsed:.2f}ms for {num_queries} queries")
        print(f"    Average: {elapsed/num_queries:.3f}ms per query")

if __name__ == "__main__":
    test_subrectangle_queries_basic()
    test_subrectangle_edge_cases()
    test_advanced_features()
    demonstrate_applications()
    test_performance()
    stress_test_subrectangle()
    test_memory_efficiency()
    benchmark_query_patterns()

"""
Subrectangle Queries Design demonstrates key concepts:

Core Approaches:
1. Naive - Direct update of all cells in the subrectangle
2. History-Based - Store update operations and apply during queries
3. Lazy - Delay updates until query time with lazy propagation
4. Advanced - Enhanced with analytics, mode switching, and features
5. Optimized - Adaptive strategy based on query/update patterns

Key Design Principles:
- Range update vs point query optimization trade-offs
- Lazy evaluation for efficient batch operations
- Memory vs time complexity considerations
- Update history tracking for audit and rollback

Performance Characteristics:
- Naive: O(area) updates, O(1) queries
- History-Based: O(1) updates, O(u) queries where u is update count
- Lazy: O(1) updates, O(u) queries with potential optimizations
- Optimized: Adaptive performance based on usage patterns

Real-world Applications:
- Image processing and region filling operations
- Game development for terrain and map editing
- Spreadsheet applications for cell formatting
- Graphics programming for rectangular selections
- Database systems for range update operations
- Geographic Information Systems (GIS) for area updates

The history-based approach provides the best balance
for most use cases, offering O(1) updates while
maintaining reasonable query performance for typical
update-to-query ratios.
"""
