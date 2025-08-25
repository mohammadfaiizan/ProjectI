"""
631. Design Excel Sum Formula - Multiple Approaches
Difficulty: Hard

Design the basic function of Excel and implement the function of the sum formula.

Implement the Excel class:
- Excel(int height, char width) Initializes the object with the height and the width of the sheet. The sheet is an integer matrix matrix[0...height-1][A...width] where A corresponds to 0, B corresponds to 1, ..., Z corresponds to 25.
- void set(int row, char column, int val) Changes the value at matrix[row][column] to be val.
- int get(int row, char column) Returns the value at matrix[row][column].
- int sum(int row, char column, List<String> numbers) Sets the value at matrix[row][column] to be the sum of cells represented by numbers and returns the sum.
"""

from typing import List, Dict, Set, Tuple
from collections import defaultdict, deque

class ExcelBasic:
    """
    Approach 1: Basic Implementation with Manual Recalculation
    
    Simple implementation that recalculates formulas manually.
    
    Time Complexity:
    - set/get: O(1)
    - sum: O(n) where n is cells in formula
    
    Space Complexity: O(rows * cols + formulas)
    """
    
    def __init__(self, height: int, width: str):
        self.height = height
        self.width = ord(width.upper()) - ord('A') + 1
        
        # Initialize matrix
        self.matrix = [[0] * self.width for _ in range(height)]
        
        # Store formulas for recalculation
        self.formulas = {}  # (row, col) -> list of cell references
    
    def _col_to_index(self, column: str) -> int:
        """Convert column letter to index"""
        return ord(column.upper()) - ord('A')
    
    def _parse_cell_ref(self, cell_ref: str) -> Tuple[int, int]:
        """Parse cell reference like 'A1' to (row, col)"""
        col_str = ""
        row_str = ""
        
        for char in cell_ref:
            if char.isalpha():
                col_str += char
            else:
                row_str += char
        
        row = int(row_str) - 1  # Convert to 0-based
        col = self._col_to_index(col_str)
        
        return row, col
    
    def _parse_range(self, range_ref: str) -> List[Tuple[int, int]]:
        """Parse range like 'A1:C3' to list of (row, col) tuples"""
        if ':' not in range_ref:
            # Single cell
            return [self._parse_cell_ref(range_ref)]
        
        start_cell, end_cell = range_ref.split(':')
        start_row, start_col = self._parse_cell_ref(start_cell)
        end_row, end_col = self._parse_cell_ref(end_cell)
        
        cells = []
        for r in range(start_row, end_row + 1):
            for c in range(start_col, end_col + 1):
                cells.append((r, c))
        
        return cells
    
    def set(self, row: int, column: str, val: int) -> None:
        col_idx = self._col_to_index(column)
        self.matrix[row - 1][col_idx] = val
        
        # Remove any formula for this cell
        if (row - 1, col_idx) in self.formulas:
            del self.formulas[(row - 1, col_idx)]
    
    def get(self, row: int, column: str) -> int:
        col_idx = self._col_to_index(column)
        return self.matrix[row - 1][col_idx]
    
    def sum(self, row: int, column: str, numbers: List[str]) -> int:
        col_idx = self._col_to_index(column)
        target_row = row - 1
        
        # Parse all cell references
        all_cells = []
        for number in numbers:
            all_cells.extend(self._parse_range(number))
        
        # Calculate sum
        total = 0
        for r, c in all_cells:
            if 0 <= r < self.height and 0 <= c < self.width:
                total += self.matrix[r][c]
        
        # Store formula and result
        self.formulas[(target_row, col_idx)] = all_cells
        self.matrix[target_row][col_idx] = total
        
        return total

class ExcelDependencyGraph:
    """
    Approach 2: Dependency Graph with Auto-Recalculation
    
    Track dependencies and automatically recalculate affected cells.
    
    Time Complexity:
    - set: O(d) where d is dependent cells
    - get: O(1)
    - sum: O(n + d) where n is formula cells, d is dependents
    
    Space Complexity: O(rows * cols + dependencies)
    """
    
    def __init__(self, height: int, width: str):
        self.height = height
        self.width = ord(width.upper()) - ord('A') + 1
        
        self.matrix = [[0] * self.width for _ in range(height)]
        
        # Dependency tracking
        self.formulas = {}  # (row, col) -> list of referenced cells
        self.dependents = defaultdict(set)  # (row, col) -> set of cells that depend on it
    
    def _col_to_index(self, column: str) -> int:
        return ord(column.upper()) - ord('A')
    
    def _parse_cell_ref(self, cell_ref: str) -> Tuple[int, int]:
        col_str = ""
        row_str = ""
        
        for char in cell_ref:
            if char.isalpha():
                col_str += char
            else:
                row_str += char
        
        row = int(row_str) - 1
        col = self._col_to_index(col_str)
        
        return row, col
    
    def _parse_range(self, range_ref: str) -> List[Tuple[int, int]]:
        if ':' not in range_ref:
            return [self._parse_cell_ref(range_ref)]
        
        start_cell, end_cell = range_ref.split(':')
        start_row, start_col = self._parse_cell_ref(start_cell)
        end_row, end_col = self._parse_cell_ref(end_cell)
        
        cells = []
        for r in range(start_row, end_row + 1):
            for c in range(start_col, end_col + 1):
                cells.append((r, c))
        
        return cells
    
    def _calculate_formula(self, row: int, col: int) -> int:
        """Calculate value for cell with formula"""
        if (row, col) not in self.formulas:
            return self.matrix[row][col]
        
        total = 0
        for r, c in self.formulas[(row, col)]:
            if 0 <= r < self.height and 0 <= c < self.width:
                total += self.matrix[r][c]
        
        return total
    
    def _recalculate_dependents(self, row: int, col: int) -> None:
        """Recalculate all cells that depend on this cell"""
        if (row, col) not in self.dependents:
            return
        
        # Use BFS to recalculate dependents
        queue = deque([(row, col)])
        visited = set()
        
        while queue:
            curr_row, curr_col = queue.popleft()
            
            if (curr_row, curr_col) in visited:
                continue
            
            visited.add((curr_row, curr_col))
            
            # Recalculate dependents
            for dep_row, dep_col in self.dependents[(curr_row, curr_col)]:
                if (dep_row, dep_col) in self.formulas:
                    new_value = self._calculate_formula(dep_row, dep_col)
                    self.matrix[dep_row][dep_col] = new_value
                    
                    # Add this cell's dependents to queue
                    queue.append((dep_row, dep_col))
    
    def set(self, row: int, column: str, val: int) -> None:
        col_idx = self._col_to_index(column)
        target_row = row - 1
        
        # Remove existing dependencies if this cell had a formula
        if (target_row, col_idx) in self.formulas:
            for ref_row, ref_col in self.formulas[(target_row, col_idx)]:
                self.dependents[(ref_row, ref_col)].discard((target_row, col_idx))
            
            del self.formulas[(target_row, col_idx)]
        
        # Set new value
        self.matrix[target_row][col_idx] = val
        
        # Recalculate dependents
        self._recalculate_dependents(target_row, col_idx)
    
    def get(self, row: int, column: str) -> int:
        col_idx = self._col_to_index(column)
        return self.matrix[row - 1][col_idx]
    
    def sum(self, row: int, column: str, numbers: List[str]) -> int:
        col_idx = self._col_to_index(column)
        target_row = row - 1
        
        # Remove existing dependencies
        if (target_row, col_idx) in self.formulas:
            for ref_row, ref_col in self.formulas[(target_row, col_idx)]:
                self.dependents[(ref_row, ref_col)].discard((target_row, col_idx))
        
        # Parse new formula
        all_cells = []
        for number in numbers:
            all_cells.extend(self._parse_range(number))
        
        # Add new dependencies
        for ref_row, ref_col in all_cells:
            self.dependents[(ref_row, ref_col)].add((target_row, col_idx))
        
        # Store formula
        self.formulas[(target_row, col_idx)] = all_cells
        
        # Calculate and store result
        total = self._calculate_formula(target_row, col_idx)
        self.matrix[target_row][col_idx] = total
        
        # Recalculate dependents
        self._recalculate_dependents(target_row, col_idx)
        
        return total

class ExcelAdvanced:
    """
    Approach 3: Advanced with Multiple Formula Types and Features
    
    Enhanced Excel with various formula types and advanced features.
    
    Time Complexity: Varies by operation
    Space Complexity: O(rows * cols + formulas + dependencies)
    """
    
    def __init__(self, height: int, width: str):
        self.height = height
        self.width = ord(width.upper()) - ord('A') + 1
        
        self.matrix = [[0] * self.width for _ in range(height)]
        
        # Formula and dependency tracking
        self.formulas = {}  # (row, col) -> formula info
        self.dependents = defaultdict(set)
        
        # Analytics
        self.operation_count = {'set': 0, 'get': 0, 'sum': 0}
        self.cell_access_count = defaultdict(int)
        
        # Features
        self.cell_history = defaultdict(list)  # Track value changes
        self.formula_cache = {}  # Cache formula results
    
    def _col_to_index(self, column: str) -> int:
        return ord(column.upper()) - ord('A')
    
    def _index_to_col(self, index: int) -> str:
        return chr(ord('A') + index)
    
    def _parse_cell_ref(self, cell_ref: str) -> Tuple[int, int]:
        col_str = ""
        row_str = ""
        
        for char in cell_ref:
            if char.isalpha():
                col_str += char
            else:
                row_str += char
        
        row = int(row_str) - 1
        col = self._col_to_index(col_str)
        
        return row, col
    
    def _parse_range(self, range_ref: str) -> List[Tuple[int, int]]:
        if ':' not in range_ref:
            return [self._parse_cell_ref(range_ref)]
        
        start_cell, end_cell = range_ref.split(':')
        start_row, start_col = self._parse_cell_ref(start_cell)
        end_row, end_col = self._parse_cell_ref(end_cell)
        
        cells = []
        for r in range(start_row, end_row + 1):
            for c in range(start_col, end_col + 1):
                cells.append((r, c))
        
        return cells
    
    def _calculate_formula(self, row: int, col: int) -> int:
        """Calculate value for cell with formula"""
        if (row, col) not in self.formulas:
            return self.matrix[row][col]
        
        formula_info = self.formulas[(row, col)]
        formula_type = formula_info['type']
        
        if formula_type == 'SUM':
            total = 0
            for r, c in formula_info['references']:
                if 0 <= r < self.height and 0 <= c < self.width:
                    total += self.matrix[r][c]
            return total
        
        return self.matrix[row][col]
    
    def _recalculate_dependents(self, row: int, col: int) -> None:
        """Recalculate all dependent cells"""
        if (row, col) not in self.dependents:
            return
        
        # Clear cache for affected cells
        for dep_row, dep_col in self.dependents[(row, col)]:
            self.formula_cache.pop((dep_row, dep_col), None)
        
        # Topological sort for proper recalculation order
        visited = set()
        stack = []
        
        def dfs(r: int, c: int):
            if (r, c) in visited:
                return
            
            visited.add((r, c))
            
            for dep_r, dep_c in self.dependents.get((r, c), []):
                dfs(dep_r, dep_c)
            
            stack.append((r, c))
        
        dfs(row, col)
        
        # Recalculate in reverse topological order
        for r, c in reversed(stack):
            if (r, c) in self.formulas:
                new_value = self._calculate_formula(r, c)
                old_value = self.matrix[r][c]
                
                if new_value != old_value:
                    self.matrix[r][c] = new_value
                    self.cell_history[(r, c)].append((new_value, 'auto_recalc'))
    
    def set(self, row: int, column: str, val: int) -> None:
        self.operation_count['set'] += 1
        
        col_idx = self._col_to_index(column)
        target_row = row - 1
        
        # Track access
        self.cell_access_count[(target_row, col_idx)] += 1
        
        # Remove existing formula
        if (target_row, col_idx) in self.formulas:
            formula_info = self.formulas[(target_row, col_idx)]
            
            for ref_row, ref_col in formula_info['references']:
                self.dependents[(ref_row, ref_col)].discard((target_row, col_idx))
            
            del self.formulas[(target_row, col_idx)]
        
        # Store old value for history
        old_value = self.matrix[target_row][col_idx]
        
        # Set new value
        self.matrix[target_row][col_idx] = val
        
        # Record history
        self.cell_history[(target_row, col_idx)].append((val, 'set'))
        
        # Clear cache
        self.formula_cache.pop((target_row, col_idx), None)
        
        # Recalculate dependents
        self._recalculate_dependents(target_row, col_idx)
    
    def get(self, row: int, column: str) -> int:
        self.operation_count['get'] += 1
        
        col_idx = self._col_to_index(column)
        target_row = row - 1
        
        self.cell_access_count[(target_row, col_idx)] += 1
        
        return self.matrix[target_row][col_idx]
    
    def sum(self, row: int, column: str, numbers: List[str]) -> int:
        self.operation_count['sum'] += 1
        
        col_idx = self._col_to_index(column)
        target_row = row - 1
        
        # Remove existing dependencies
        if (target_row, col_idx) in self.formulas:
            formula_info = self.formulas[(target_row, col_idx)]
            
            for ref_row, ref_col in formula_info['references']:
                self.dependents[(ref_row, ref_col)].discard((target_row, col_idx))
        
        # Parse new formula
        all_cells = []
        for number in numbers:
            all_cells.extend(self._parse_range(number))
        
        # Add new dependencies
        for ref_row, ref_col in all_cells:
            self.dependents[(ref_row, ref_col)].add((target_row, col_idx))
        
        # Store formula
        self.formulas[(target_row, col_idx)] = {
            'type': 'SUM',
            'references': all_cells,
            'formula_text': f"SUM({','.join(numbers)})"
        }
        
        # Calculate result
        total = self._calculate_formula(target_row, col_idx)
        self.matrix[target_row][col_idx] = total
        
        # Record history
        self.cell_history[(target_row, col_idx)].append((total, 'sum_formula'))
        
        # Cache result
        self.formula_cache[(target_row, col_idx)] = total
        
        # Recalculate dependents
        self._recalculate_dependents(target_row, col_idx)
        
        return total
    
    def get_statistics(self) -> dict:
        """Get Excel statistics"""
        total_formulas = len(self.formulas)
        total_dependencies = sum(len(deps) for deps in self.dependents.values())
        
        return {
            'operation_count': self.operation_count.copy(),
            'total_formulas': total_formulas,
            'total_dependencies': total_dependencies,
            'cache_size': len(self.formula_cache),
            'most_accessed_cell': max(self.cell_access_count.items(), 
                                    key=lambda x: x[1], default=((0, 0), 0))
        }
    
    def get_cell_info(self, row: int, column: str) -> dict:
        """Get detailed cell information"""
        col_idx = self._col_to_index(column)
        target_row = row - 1
        
        info = {
            'value': self.matrix[target_row][col_idx],
            'access_count': self.cell_access_count[(target_row, col_idx)],
            'has_formula': (target_row, col_idx) in self.formulas,
            'dependents_count': len(self.dependents[(target_row, col_idx)]),
            'history_length': len(self.cell_history[(target_row, col_idx)])
        }
        
        if (target_row, col_idx) in self.formulas:
            info['formula'] = self.formulas[(target_row, col_idx)]
        
        return info


def test_excel_basic():
    """Test basic Excel functionality"""
    print("=== Testing Basic Excel Functionality ===")
    
    implementations = [
        ("Basic", ExcelBasic),
        ("Dependency Graph", ExcelDependencyGraph),
        ("Advanced", ExcelAdvanced)
    ]
    
    for name, ExcelClass in implementations:
        print(f"\n{name}:")
        
        excel = ExcelClass(3, 'C')
        
        # Test basic operations
        excel.set(1, 'A', 3)
        print(f"  set(1, 'A', 3)")
        
        excel.set(2, 'B', 4)
        print(f"  set(2, 'B', 4)")
        
        result = excel.sum(3, 'C', ['A1', 'B2'])
        print(f"  sum(3, 'C', ['A1', 'B2']): {result}")
        
        value = excel.get(3, 'C')
        print(f"  get(3, 'C'): {value}")
        
        # Test dependency update
        excel.set(1, 'A', 5)
        print(f"  set(1, 'A', 5)")
        
        updated_value = excel.get(3, 'C')
        print(f"  get(3, 'C') after update: {updated_value}")

def test_excel_advanced_formulas():
    """Test advanced formula functionality"""
    print("\n=== Testing Advanced Formula Functionality ===")
    
    excel = ExcelAdvanced(5, 'E')
    
    # Set up test data
    test_data = [
        (1, 'A', 10),
        (1, 'B', 20),
        (1, 'C', 30),
        (2, 'A', 5),
        (2, 'B', 15),
        (2, 'C', 25)
    ]
    
    print("Setting up test data:")
    for row, col, val in test_data:
        excel.set(row, col, val)
        print(f"  {col}{row} = {val}")
    
    # Test range sum
    print(f"\nTesting range formulas:")
    
    # Sum entire row
    row_sum = excel.sum(3, 'A', ['A1:C1'])
    print(f"  SUM(A1:C1) = {row_sum}")
    
    # Sum entire column
    col_sum = excel.sum(3, 'B', ['A1:A2'])
    print(f"  SUM(A1:A2) = {col_sum}")
    
    # Sum mixed cells
    mixed_sum = excel.sum(3, 'C', ['A1', 'B2', 'C1'])
    print(f"  SUM(A1,B2,C1) = {mixed_sum}")
    
    # Test cascading updates
    print(f"\nTesting cascading updates:")
    
    print(f"  Before: A1={excel.get(1, 'A')}, SUM(A1:C1)={excel.get(3, 'A')}")
    
    excel.set(1, 'A', 100)  # This should update the sum
    
    print(f"  After A1=100: A1={excel.get(1, 'A')}, SUM(A1:C1)={excel.get(3, 'A')}")

def test_excel_dependencies():
    """Test dependency tracking"""
    print("\n=== Testing Dependency Tracking ===")
    
    excel = ExcelDependencyGraph(4, 'D')
    
    # Create dependency chain: A1 -> B1 -> C1
    excel.set(1, 'A', 10)
    excel.sum(1, 'B', ['A1'])  # B1 = SUM(A1)
    excel.sum(1, 'C', ['B1'])  # C1 = SUM(B1)
    excel.sum(1, 'D', ['A1', 'B1', 'C1'])  # D1 = SUM(A1,B1,C1)
    
    print("Created dependency chain: A1 -> B1 -> C1 -> D1")
    print(f"  A1: {excel.get(1, 'A')}")
    print(f"  B1: {excel.get(1, 'B')}")
    print(f"  C1: {excel.get(1, 'C')}")
    print(f"  D1: {excel.get(1, 'D')}")
    
    # Update A1 and see cascade
    print(f"\nUpdating A1 from 10 to 50:")
    excel.set(1, 'A', 50)
    
    print(f"  A1: {excel.get(1, 'A')}")
    print(f"  B1: {excel.get(1, 'B')}")
    print(f"  C1: {excel.get(1, 'C')}")
    print(f"  D1: {excel.get(1, 'D')}")

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: Financial spreadsheet
    print("Application 1: Financial Budget Spreadsheet")
    
    budget = ExcelAdvanced(10, 'F')
    
    # Income section
    budget.set(1, 'A', 5000)  # Salary
    budget.set(1, 'B', 1000)  # Freelance
    budget.sum(1, 'C', ['A1:B1'])  # Total Income
    
    # Expenses section
    budget.set(3, 'A', 1200)  # Rent
    budget.set(3, 'B', 400)   # Groceries
    budget.set(3, 'C', 300)   # Utilities
    budget.sum(3, 'D', ['A3:C3'])  # Total Expenses
    
    # Net income
    budget.sum(5, 'A', ['C1', 'D3'])  # This would need subtraction in real Excel
    
    print("  Budget setup:")
    print(f"    Salary: {budget.get(1, 'A')}")
    print(f"    Freelance: {budget.get(1, 'B')}")
    print(f"    Total Income: {budget.get(1, 'C')}")
    print(f"    Total Expenses: {budget.get(3, 'D')}")
    
    # Simulate salary increase
    print(f"\n  Simulating 10% salary increase:")
    new_salary = int(budget.get(1, 'A') * 1.1)
    budget.set(1, 'A', new_salary)
    
    print(f"    New Salary: {budget.get(1, 'A')}")
    print(f"    New Total Income: {budget.get(1, 'C')}")
    
    # Application 2: Sales tracking
    print(f"\nApplication 2: Sales Tracking Dashboard")
    
    sales = ExcelAdvanced(8, 'E')
    
    # Monthly sales data
    months = ['Jan', 'Feb', 'Mar', 'Apr']
    monthly_sales = [10000, 12000, 15000, 18000]
    
    print("  Monthly sales data:")
    for i, (month, amount) in enumerate(zip(months, monthly_sales)):
        row = i + 1
        sales.set(row, 'A', amount)
        print(f"    {month}: {amount}")
    
    # Calculate Q1 total
    q1_total = sales.sum(5, 'A', ['A1:A4'])
    print(f"  Q1 Total: {q1_total}")
    
    # Calculate average
    # Note: Real Excel would have AVERAGE function
    avg_cell = sales.sum(6, 'A', ['A1:A4'])  # This gives sum, would need division
    print(f"  Sum for average calculation: {avg_cell}")
    
    # Application 3: Inventory management
    print(f"\nApplication 3: Inventory Management")
    
    inventory = ExcelAdvanced(6, 'D')
    
    # Initial stock
    products = [
        ('Widget A', 100, 25),    # Product, Quantity, Price
        ('Widget B', 75, 40),
        ('Widget C', 50, 60)
    ]
    
    print("  Initial inventory:")
    for i, (product, qty, price) in enumerate(products):
        row = i + 1
        inventory.set(row, 'A', qty)     # Quantity
        inventory.set(row, 'B', price)   # Price
        
        # Value = Qty * Price (simulated)
        value = qty * price
        inventory.set(row, 'C', value)
        
        print(f"    {product}: {qty} units @ ${price} = ${value}")
    
    # Total inventory value
    total_value = inventory.sum(4, 'C', ['C1:C3'])
    print(f"  Total Inventory Value: ${total_value}")
    
    # Simulate stock change
    print(f"\n  Simulating stock change for Widget A:")
    inventory.set(1, 'A', 80)  # Reduce quantity
    
    # Update value calculation (in real Excel this would be automatic with formula)
    new_qty = inventory.get(1, 'A')
    price = inventory.get(1, 'B')
    new_value = new_qty * price
    inventory.set(1, 'C', new_value)
    
    # Recalculate total
    new_total = inventory.sum(4, 'C', ['C1:C3'])
    print(f"    New quantity: {new_qty}")
    print(f"    New total value: ${new_total}")

def test_performance():
    """Test performance with larger spreadsheets"""
    print("\n=== Testing Performance ===")
    
    import time
    
    # Create larger spreadsheet
    excel = ExcelDependencyGraph(20, 'T')
    
    print("Performance test with 20x20 spreadsheet:")
    
    # Fill with data
    start_time = time.time()
    
    for row in range(1, 21):
        for col_idx in range(20):
            col = chr(ord('A') + col_idx)
            value = row * 20 + col_idx + 1
            excel.set(row, col, value)
    
    fill_time = (time.time() - start_time) * 1000
    
    # Create formulas
    start_time = time.time()
    
    # Row sums
    for row in range(1, 11):  # First 10 rows
        excel.sum(row, 'U', [f'A{row}:T{row}'])  # Sum entire row
    
    formula_time = (time.time() - start_time) * 1000
    
    # Update values and measure cascade
    start_time = time.time()
    
    # Update first column
    for row in range(1, 6):
        excel.set(row, 'A', 1000)
    
    update_time = (time.time() - start_time) * 1000
    
    print(f"  Fill 400 cells: {fill_time:.2f}ms")
    print(f"  Create 10 formulas: {formula_time:.2f}ms")
    print(f"  Update 5 cells with cascade: {update_time:.2f}ms")

def test_advanced_features():
    """Test advanced Excel features"""
    print("\n=== Testing Advanced Features ===")
    
    excel = ExcelAdvanced(5, 'E')
    
    # Create test scenario
    excel.set(1, 'A', 10)
    excel.set(1, 'B', 20)
    excel.sum(1, 'C', ['A1', 'B1'])
    
    excel.set(2, 'A', 30)
    excel.sum(2, 'B', ['A1:A2'])
    
    # Get statistics
    stats = excel.get_statistics()
    print("Excel statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Get cell information
    cell_info = excel.get_cell_info(1, 'C')
    print(f"\nCell C1 information:")
    for key, value in cell_info.items():
        print(f"  {key}: {value}")
    
    # Test more operations to build history
    excel.set(1, 'A', 15)
    excel.set(1, 'A', 25)
    
    updated_info = excel.get_cell_info(1, 'A')
    print(f"\nCell A1 after updates:")
    for key, value in updated_info.items():
        print(f"  {key}: {value}")

def stress_test_excel():
    """Stress test Excel with many dependencies"""
    print("\n=== Stress Testing Excel ===")
    
    import time
    
    excel = ExcelBasic(50, 'Z')  # Large spreadsheet
    
    # Create pyramid of dependencies
    num_levels = 10
    
    print(f"Creating dependency pyramid with {num_levels} levels:")
    
    start_time = time.time()
    
    # Level 1: Base values
    for i in range(num_levels):
        excel.set(1, chr(ord('A') + i), (i + 1) * 10)
    
    # Each level sums previous level
    for level in range(2, num_levels + 1):
        for i in range(num_levels - level + 1):
            col = chr(ord('A') + i)
            prev_cols = [chr(ord('A') + j) + str(level - 1) for j in range(i, i + 2)]
            excel.sum(level, col, prev_cols)
    
    create_time = (time.time() - start_time) * 1000
    
    # Update base value and measure cascade
    start_time = time.time()
    
    excel.set(1, 'A', 1000)  # This should cascade through pyramid
    
    cascade_time = (time.time() - start_time) * 1000
    
    print(f"  Creation time: {create_time:.2f}ms")
    print(f"  Cascade update time: {cascade_time:.2f}ms")
    
    # Check final value at top
    top_value = excel.get(num_levels, 'A')
    print(f"  Final value at top of pyramid: {top_value}")

if __name__ == "__main__":
    test_excel_basic()
    test_excel_advanced_formulas()
    test_excel_dependencies()
    demonstrate_applications()
    test_performance()
    test_advanced_features()
    stress_test_excel()

"""
Excel Sum Formula Design demonstrates key concepts:

Core Approaches:
1. Basic - Simple manual recalculation when formulas change
2. Dependency Graph - Automatic dependency tracking and recalculation
3. Advanced - Enhanced with analytics, caching, and additional features

Key Design Principles:
- Spreadsheet cell management and formula evaluation
- Dependency graph construction and topological ordering
- Automatic recalculation of dependent cells
- Efficient range parsing and cell reference handling

Performance Characteristics:
- Basic: O(n) recalculation per update
- Dependency Graph: O(d) where d is dependent cells
- Advanced: Additional overhead for features and caching

Real-world Applications:
- Financial modeling and budgeting spreadsheets
- Sales tracking and business analytics dashboards
- Inventory management and supply chain tracking
- Scientific data analysis and calculations
- Project planning and resource allocation
- Educational tools for mathematical modeling

The dependency graph approach provides the most robust
foundation for a production spreadsheet application,
enabling automatic recalculation while maintaining
good performance characteristics.
"""
