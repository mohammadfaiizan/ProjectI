"""
Tree Applications and Real-World Use Cases
==========================================

Topics: Database indexing, file systems, decision trees, syntax trees
Companies: Google, Amazon, Microsoft, Oracle, MongoDB, file system companies
Difficulty: Medium to Hard
Time Complexity: Varies by application (O(log n) to O(n))
Space Complexity: O(n) for tree storage plus application-specific space
"""

from typing import List, Optional, Dict, Any, Tuple, Union, Callable
from collections import defaultdict, deque
import math
import random

class TreeApplications:
    
    def __init__(self):
        """Initialize with application tracking"""
        self.application_count = 0
        self.performance_metrics = {}
    
    # ==========================================
    # 1. DATABASE INDEXING SYSTEMS
    # ==========================================
    
    def explain_database_indexing(self) -> None:
        """
        Explain how trees are used in database indexing systems
        """
        print("=== DATABASE INDEXING WITH TREES ===")
        print("Trees are fundamental to database performance and scalability")
        print()
        print("PRIMARY INDEX STRUCTURES:")
        print("• B+ Trees: Most common for relational databases")
        print("• B Trees: Traditional balanced trees for disk-based storage")
        print("• LSM Trees: Log-structured merge trees for write-heavy workloads")
        print("• R Trees: Spatial indexing for geographic data")
        print("• Hash Index: Hash tables for equality lookups")
        print()
        print("B+ TREE ADVANTAGES IN DATABASES:")
        print("• Sequential leaf access for range queries")
        print("• High fanout reduces I/O operations")
        print("• All data in leaves → consistent performance")
        print("• Supports both point and range queries efficiently")
        print("• Self-balancing maintains performance under updates")
        print()
        print("INDEX TYPES:")
        print("• Primary Index: Clustered, data sorted by key")
        print("• Secondary Index: Non-clustered, points to primary keys")
        print("• Composite Index: Multiple columns as key")
        print("• Partial Index: Subset of rows based on condition")
        print("• Functional Index: Based on expression/function result")
        print()
        print("QUERY OPTIMIZATION:")
        print("• Range queries: O(log n + k) where k is result size")
        print("• Point queries: O(log n) for single record lookup")
        print("• Join operations: Index nested loop joins")
        print("• Sort operations: Index provides sorted order")
        print("• Group by: Index enables efficient grouping")
    
    def demonstrate_database_index(self) -> None:
        """
        Demonstrate a simplified database index using B+ tree
        """
        print("=== DATABASE INDEX DEMONSTRATION ===")
        print("Simulating database table with B+ tree index")
        print()
        
        # Create sample employee database
        db_index = DatabaseIndex()
        
        # Sample employee records
        employees = [
            (101, "Alice Johnson", "Engineering", 95000),
            (105, "Bob Smith", "Marketing", 75000),
            (102, "Carol Davis", "Engineering", 88000),
            (108, "David Wilson", "Sales", 65000),
            (103, "Eve Brown", "Engineering", 92000),
            (107, "Frank Miller", "Marketing", 78000),
            (104, "Grace Lee", "Sales", 70000),
            (106, "Henry Taylor", "Engineering", 85000)
        ]
        
        print("Inserting employee records:")
        for emp_id, name, dept, salary in employees:
            db_index.insert_record(emp_id, {
                "name": name,
                "department": dept,
                "salary": salary
            })
            print(f"  Inserted: {emp_id} - {name}")
        
        print(f"\nIndex structure after insertions:")
        db_index.display_index()
        
        # Demonstrate different query types
        print("\nDatabase query demonstrations:")
        
        # Point query
        print("\n1. Point query: SELECT * FROM employees WHERE emp_id = 105")
        result = db_index.point_query(105)
        if result:
            print(f"   Result: {result}")
        else:
            print("   No record found")
        
        # Range query
        print("\n2. Range query: SELECT * FROM employees WHERE emp_id BETWEEN 103 AND 107")
        results = db_index.range_query(103, 107)
        print(f"   Found {len(results)} records:")
        for emp_id, record in results:
            print(f"     {emp_id}: {record['name']} - {record['department']}")
        
        # Statistics
        print(f"\n3. Index statistics:")
        stats = db_index.get_statistics()
        for key, value in stats.items():
            print(f"   {key}: {value}")


class DatabaseIndex:
    """
    Simplified database index using B+ tree structure
    
    Demonstrates how databases use trees for efficient data access
    """
    
    def __init__(self, order: int = 3):
        self.order = order  # B+ tree order
        self.root = None
        self.record_count = 0
        self.leaf_nodes = []  # For range queries
        self.io_operations = 0
    
    def insert_record(self, key: int, record: Dict[str, Any]) -> None:
        """Insert a record with given key"""
        # Simplified insertion - in real systems, this would be more complex
        self.record_count += 1
        self.io_operations += 1  # Simulate disk I/O
        
        # Store in sorted list for demonstration
        if not hasattr(self, 'records'):
            self.records = []
        
        # Insert in sorted order
        inserted = False
        for i, (stored_key, stored_record) in enumerate(self.records):
            if key < stored_key:
                self.records.insert(i, (key, record))
                inserted = True
                break
        
        if not inserted:
            self.records.append((key, record))
    
    def point_query(self, key: int) -> Optional[Dict[str, Any]]:
        """Perform point query for specific key"""
        self.io_operations += math.log2(self.record_count) if self.record_count > 0 else 1
        
        # Binary search simulation
        left, right = 0, len(self.records) - 1
        
        while left <= right:
            mid = (left + right) // 2
            mid_key, mid_record = self.records[mid]
            
            if mid_key == key:
                return mid_record
            elif mid_key < key:
                left = mid + 1
            else:
                right = mid - 1
        
        return None
    
    def range_query(self, start_key: int, end_key: int) -> List[Tuple[int, Dict[str, Any]]]:
        """Perform range query"""
        self.io_operations += math.log2(self.record_count) if self.record_count > 0 else 1
        
        results = []
        for key, record in self.records:
            if start_key <= key <= end_key:
                results.append((key, record))
        
        # In real B+ tree, this would traverse linked leaves
        self.io_operations += len(results) // 10  # Simulate page reads
        
        return results
    
    def display_index(self) -> None:
        """Display index structure"""
        print("    Index entries (key -> record):")
        for key, record in self.records:
            print(f"      {key} -> {record['name']} ({record['department']})")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            "total_records": self.record_count,
            "estimated_height": math.ceil(math.log(self.record_count, self.order)) if self.record_count > 0 else 0,
            "io_operations": self.io_operations,
            "space_utilization": "75-100%"  # Typical for B+ trees
        }


# ==========================================
# 2. FILE SYSTEM STRUCTURES
# ==========================================

class FileSystemTree:
    """
    File system implementation using tree structures
    
    Demonstrates hierarchical organization and directory traversal
    """
    
    def explain_file_system_trees(self) -> None:
        """Explain how file systems use tree structures"""
        print("=== FILE SYSTEM TREE STRUCTURES ===")
        print("File systems organize data in hierarchical tree structures")
        print()
        print("DIRECTORY TREE STRUCTURE:")
        print("• Root directory at the top (/)")
        print("• Directories are internal nodes")
        print("• Files are leaf nodes (typically)")
        print("• Path represents route from root to file/directory")
        print("• Tree depth determines maximum path length")
        print()
        print("FILE SYSTEM OPERATIONS:")
        print("• Create: Add new node to tree")
        print("• Delete: Remove node and handle children")
        print("• Move: Change parent-child relationships")
        print("• Copy: Duplicate subtree structure")
        print("• Search: Traverse tree to find files")
        print()
        print("MODERN FILE SYSTEM FEATURES:")
        print("• B-trees for directory indexing (ext4, NTFS)")
        print("• Copy-on-write trees (Btrfs, ZFS)")
        print("• Merkle trees for integrity verification")
        print("• Log-structured trees for write optimization")
        print()
        print("PERFORMANCE CONSIDERATIONS:")
        print("• Directory depth affects access time")
        print("• Balanced trees prevent deep nesting")
        print("• Caching frequently accessed directories")
        print("• Parallel access to different subtrees")
    
    def demonstrate_file_system(self) -> None:
        """Demonstrate file system operations"""
        print("=== FILE SYSTEM DEMONSTRATION ===")
        print("Creating a sample file system structure")
        print()
        
        fs = FileSystem()
        
        # Create directory structure
        directories = [
            "/home",
            "/home/user",
            "/home/user/documents",
            "/home/user/pictures",
            "/var",
            "/var/log",
            "/usr",
            "/usr/bin",
            "/usr/lib"
        ]
        
        print("Creating directories:")
        for directory in directories:
            fs.create_directory(directory)
            print(f"  Created: {directory}")
        
        # Create files
        files = [
            "/home/user/documents/report.txt",
            "/home/user/documents/notes.md",
            "/home/user/pictures/photo1.jpg",
            "/home/user/pictures/photo2.jpg",
            "/var/log/system.log",
            "/usr/bin/python",
            "/usr/lib/library.so"
        ]
        
        print("\nCreating files:")
        for file_path in files:
            fs.create_file(file_path, f"Content of {file_path}")
            print(f"  Created: {file_path}")
        
        print("\nFile system structure:")
        fs.display_tree()
        
        # Demonstrate operations
        print("\nFile system operations:")
        
        # List directory
        print("\n1. List directory: /home/user")
        contents = fs.list_directory("/home/user")
        for item in contents:
            print(f"   {item}")
        
        # Find files
        print("\n2. Find all .jpg files:")
        jpg_files = fs.find_files_by_extension(".jpg")
        for file_path in jpg_files:
            print(f"   {file_path}")
        
        # Get file info
        print("\n3. File information: /home/user/documents/report.txt")
        info = fs.get_file_info("/home/user/documents/report.txt")
        if info:
            print(f"   Size: {info['size']} bytes")
            print(f"   Type: {info['type']}")
        
        # Calculate directory size
        print("\n4. Directory size: /home/user")
        size = fs.calculate_directory_size("/home/user")
        print(f"   Total size: {size} bytes")


class FileSystemNode:
    """Node in file system tree"""
    
    def __init__(self, name: str, is_directory: bool = False, content: str = ""):
        self.name = name
        self.is_directory = is_directory
        self.content = content
        self.children: Dict[str, 'FileSystemNode'] = {}
        self.parent: Optional['FileSystemNode'] = None
        self.size = len(content) if not is_directory else 0
    
    def __str__(self):
        return f"{'DIR' if self.is_directory else 'FILE'}: {self.name}"


class FileSystem:
    """
    Simple file system implementation using tree structure
    """
    
    def __init__(self):
        self.root = FileSystemNode("/", True)
        self.current_directory = self.root
    
    def create_directory(self, path: str) -> bool:
        """Create directory at given path"""
        parent, name = self._get_parent_and_name(path)
        if parent and name not in parent.children:
            new_dir = FileSystemNode(name, True)
            new_dir.parent = parent
            parent.children[name] = new_dir
            return True
        return False
    
    def create_file(self, path: str, content: str = "") -> bool:
        """Create file at given path"""
        parent, name = self._get_parent_and_name(path)
        if parent and name not in parent.children:
            new_file = FileSystemNode(name, False, content)
            new_file.parent = parent
            parent.children[name] = new_file
            return True
        return False
    
    def list_directory(self, path: str) -> List[str]:
        """List contents of directory"""
        node = self._navigate_to_path(path)
        if node and node.is_directory:
            return list(node.children.keys())
        return []
    
    def find_files_by_extension(self, extension: str) -> List[str]:
        """Find all files with given extension"""
        results = []
        
        def dfs(node, current_path):
            if not node.is_directory and node.name.endswith(extension):
                results.append(current_path + "/" + node.name)
            
            for child_name, child_node in node.children.items():
                child_path = current_path + "/" + child_name if current_path != "/" else "/" + child_name
                dfs(child_node, child_path)
        
        dfs(self.root, "")
        return results
    
    def get_file_info(self, path: str) -> Optional[Dict[str, Any]]:
        """Get file information"""
        node = self._navigate_to_path(path)
        if node:
            return {
                "name": node.name,
                "type": "directory" if node.is_directory else "file",
                "size": node.size
            }
        return None
    
    def calculate_directory_size(self, path: str) -> int:
        """Calculate total size of directory"""
        node = self._navigate_to_path(path)
        if not node or not node.is_directory:
            return 0
        
        def calculate_size(node):
            total = node.size
            for child in node.children.values():
                total += calculate_size(child)
            return total
        
        return calculate_size(node)
    
    def _navigate_to_path(self, path: str) -> Optional[FileSystemNode]:
        """Navigate to given path"""
        if path == "/":
            return self.root
        
        parts = [p for p in path.split("/") if p]
        current = self.root
        
        for part in parts:
            if part in current.children:
                current = current.children[part]
            else:
                return None
        
        return current
    
    def _get_parent_and_name(self, path: str) -> Tuple[Optional[FileSystemNode], str]:
        """Get parent directory and name from path"""
        if path == "/":
            return None, "/"
        
        parts = [p for p in path.split("/") if p]
        name = parts[-1]
        parent_path = "/" + "/".join(parts[:-1]) if len(parts) > 1 else "/"
        
        parent = self._navigate_to_path(parent_path)
        return parent, name
    
    def display_tree(self) -> None:
        """Display file system tree structure"""
        def display_recursive(node, prefix="", is_last=True):
            print(f"  {prefix}{'└── ' if is_last else '├── '}{node.name}" + 
                  (f" ({node.size} bytes)" if not node.is_directory else ""))
            
            children = list(node.children.values())
            for i, child in enumerate(children):
                is_last_child = (i == len(children) - 1)
                new_prefix = prefix + ("    " if is_last else "│   ")
                display_recursive(child, new_prefix, is_last_child)
        
        display_recursive(self.root)


# ==========================================
# 3. DECISION TREES AND ML APPLICATIONS
# ==========================================

class DecisionTreeApplications:
    """
    Decision trees for machine learning and classification
    
    Demonstrates how trees are used in AI/ML applications
    """
    
    def explain_decision_trees(self) -> None:
        """Explain decision trees in machine learning"""
        print("=== DECISION TREES IN MACHINE LEARNING ===")
        print("Trees for classification and regression problems")
        print()
        print("DECISION TREE STRUCTURE:")
        print("• Internal nodes: Feature tests/conditions")
        print("• Edges: Test outcomes (yes/no, <value, >=value)")
        print("• Leaves: Predictions/classifications")
        print("• Root to leaf: Decision path for input")
        print()
        print("TREE CONSTRUCTION ALGORITHMS:")
        print("• ID3: Information gain (entropy reduction)")
        print("• C4.5: Gain ratio (handles continuous features)")
        print("• CART: Gini impurity or MSE minimization")
        print("• Random Forest: Ensemble of decision trees")
        print()
        print("SPLITTING CRITERIA:")
        print("• Information Gain: Entropy reduction")
        print("• Gini Impurity: Probability of misclassification")
        print("• Chi-square: Statistical significance")
        print("• Variance Reduction: For regression problems")
        print()
        print("ADVANTAGES:")
        print("• Interpretable and explainable")
        print("• Handles both numerical and categorical data")
        print("• No assumptions about data distribution")
        print("• Feature selection built-in")
        print()
        print("DISADVANTAGES:")
        print("• Prone to overfitting")
        print("• Unstable (small data changes → different trees)")
        print("• Biased toward features with more levels")
        print("• Cannot capture linear relationships well")
    
    def demonstrate_decision_tree(self) -> None:
        """Demonstrate decision tree construction and usage"""
        print("=== DECISION TREE DEMONSTRATION ===")
        print("Building decision tree for loan approval")
        print()
        
        # Sample loan application data
        data = [
            # (income, credit_score, age, has_job, loan_approved)
            (45000, 720, 25, True, True),
            (55000, 680, 35, True, True),
            (35000, 640, 28, False, False),
            (75000, 780, 45, True, True),
            (25000, 600, 22, False, False),
            (65000, 720, 38, True, True),
            (40000, 650, 30, True, False),
            (80000, 800, 50, True, True),
            (30000, 580, 26, False, False),
            (60000, 700, 40, True, True)
        ]
        
        print("Training data (income, credit_score, age, has_job, approved):")
        for row in data:
            print(f"  {row}")
        
        # Build decision tree
        tree_builder = DecisionTreeBuilder()
        decision_tree = tree_builder.build_tree(data)
        
        print("\nBuilt decision tree:")
        tree_builder.display_tree(decision_tree)
        
        # Test predictions
        test_cases = [
            (50000, 700, 30, True),   # Should approve
            (30000, 600, 25, False),  # Should reject
            (70000, 750, 35, True),   # Should approve
            (35000, 620, 28, True)    # Uncertain
        ]
        
        print("\nPredictions for test cases:")
        for income, credit, age, job in test_cases:
            prediction = tree_builder.predict(decision_tree, (income, credit, age, job))
            status = "APPROVED" if prediction else "REJECTED"
            print(f"  Income: ${income}, Credit: {credit}, Age: {age}, Job: {job} → {status}")


class DecisionNode:
    """Node in decision tree"""
    
    def __init__(self, feature=None, threshold=None, value=None):
        self.feature = feature      # Feature index for splitting
        self.threshold = threshold  # Threshold value for splitting
        self.value = value         # Prediction value (for leaf nodes)
        self.left = None           # Left child (condition false)
        self.right = None          # Right child (condition true)
        self.is_leaf = value is not None


class DecisionTreeBuilder:
    """
    Simple decision tree builder using information gain
    """
    
    def __init__(self, max_depth=5, min_samples=2):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.feature_names = ["income", "credit_score", "age", "has_job"]
    
    def build_tree(self, data: List[Tuple]) -> DecisionNode:
        """Build decision tree from training data"""
        print("Building decision tree...")
        return self._build_recursive(data, depth=0)
    
    def _build_recursive(self, data: List[Tuple], depth: int) -> DecisionNode:
        """Recursively build decision tree"""
        # Extract labels (last column)
        labels = [row[-1] for row in data]
        
        # Check stopping criteria
        if (depth >= self.max_depth or 
            len(data) < self.min_samples or 
            len(set(labels)) == 1):
            
            # Create leaf node with majority vote
            prediction = max(set(labels), key=labels.count)
            print(f"  Created leaf at depth {depth}: prediction = {prediction}")
            return DecisionNode(value=prediction)
        
        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split(data)
        
        if best_gain == 0:
            # No improvement possible
            prediction = max(set(labels), key=labels.count)
            return DecisionNode(value=prediction)
        
        print(f"  Split at depth {depth}: {self.feature_names[best_feature]} <= {best_threshold}")
        
        # Split data
        left_data = [row for row in data if row[best_feature] <= best_threshold]
        right_data = [row for row in data if row[best_feature] > best_threshold]
        
        # Create internal node
        node = DecisionNode(feature=best_feature, threshold=best_threshold)
        node.left = self._build_recursive(left_data, depth + 1)
        node.right = self._build_recursive(right_data, depth + 1)
        
        return node
    
    def _find_best_split(self, data: List[Tuple]) -> Tuple[int, float, float]:
        """Find best feature and threshold for splitting"""
        best_gain = 0
        best_feature = 0
        best_threshold = 0
        
        current_entropy = self._calculate_entropy([row[-1] for row in data])
        
        # Try each feature
        for feature_idx in range(len(data[0]) - 1):  # Exclude label column
            # Try different thresholds
            values = sorted(set(row[feature_idx] for row in data))
            
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2
                
                # Split data
                left = [row for row in data if row[feature_idx] <= threshold]
                right = [row for row in data if row[feature_idx] > threshold]
                
                if len(left) == 0 or len(right) == 0:
                    continue
                
                # Calculate information gain
                left_entropy = self._calculate_entropy([row[-1] for row in left])
                right_entropy = self._calculate_entropy([row[-1] for row in right])
                
                weighted_entropy = (len(left) * left_entropy + len(right) * right_entropy) / len(data)
                gain = current_entropy - weighted_entropy
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _calculate_entropy(self, labels: List[bool]) -> float:
        """Calculate entropy of label distribution"""
        if not labels:
            return 0
        
        true_count = sum(labels)
        false_count = len(labels) - true_count
        total = len(labels)
        
        if true_count == 0 or false_count == 0:
            return 0
        
        p_true = true_count / total
        p_false = false_count / total
        
        return -(p_true * math.log2(p_true) + p_false * math.log2(p_false))
    
    def predict(self, tree: DecisionNode, sample: Tuple) -> bool:
        """Make prediction for a single sample"""
        current = tree
        
        while not current.is_leaf:
            if sample[current.feature] <= current.threshold:
                current = current.left
            else:
                current = current.right
        
        return current.value
    
    def display_tree(self, node: DecisionNode, depth: int = 0) -> None:
        """Display decision tree structure"""
        indent = "  " * depth
        
        if node.is_leaf:
            prediction = "APPROVE" if node.value else "REJECT"
            print(f"{indent}Predict: {prediction}")
        else:
            feature_name = self.feature_names[node.feature]
            print(f"{indent}If {feature_name} <= {node.threshold}:")
            self.display_tree(node.left, depth + 1)
            print(f"{indent}Else:")
            self.display_tree(node.right, depth + 1)


# ==========================================
# 4. COMPILER SYNTAX TREES
# ==========================================

class SyntaxTreeApplications:
    """
    Abstract Syntax Trees (AST) in compilers and interpreters
    """
    
    def explain_syntax_trees(self) -> None:
        """Explain syntax trees in compiler design"""
        print("=== ABSTRACT SYNTAX TREES (AST) ===")
        print("Trees representing program structure in compilers")
        print()
        print("AST COMPONENTS:")
        print("• Internal nodes: Operators, control structures")
        print("• Leaves: Operands, identifiers, literals")
        print("• Structure: Reflects precedence and associativity")
        print("• No syntactic details: parentheses, semicolons eliminated")
        print()
        print("COMPILER PHASES USING AST:")
        print("• Parsing: Convert tokens to AST")
        print("• Semantic analysis: Type checking, scope resolution")
        print("• Optimization: Tree transformations")
        print("• Code generation: Traverse AST to emit code")
        print()
        print("TREE TRAVERSAL FOR CODE GENERATION:")
        print("• In-order: For infix expressions")
        print("• Post-order: For stack-based evaluation")
        print("• Pre-order: For prefix operations")
        print("• Level-order: For certain optimizations")
        print()
        print("APPLICATIONS:")
        print("• Programming language compilers")
        print("• Interpreters and virtual machines")
        print("• Code analysis and refactoring tools")
        print("• Mathematical expression evaluators")
    
    def demonstrate_expression_tree(self) -> None:
        """Demonstrate expression tree for mathematical expressions"""
        print("=== EXPRESSION TREE DEMONSTRATION ===")
        print("Building and evaluating expression trees")
        print()
        
        # Build expression tree for: ((2 + 3) * 4) - (8 / 2)
        expression = "((2 + 3) * 4) - (8 / 2)"
        print(f"Expression: {expression}")
        
        builder = ExpressionTreeBuilder()
        tree = builder.build_from_infix(expression)
        
        print("\nExpression tree structure:")
        builder.display_tree(tree)
        
        print("\nDifferent traversals:")
        print(f"In-order (infix): {builder.inorder_traversal(tree)}")
        print(f"Pre-order (prefix): {builder.preorder_traversal(tree)}")
        print(f"Post-order (postfix): {builder.postorder_traversal(tree)}")
        
        print(f"\nEvaluation result: {builder.evaluate(tree)}")
        
        # Demonstrate tree transformations
        print("\nExpression tree optimizations:")
        optimized = builder.optimize_tree(tree)
        print("After constant folding:")
        builder.display_tree(optimized)
        print(f"Optimized result: {builder.evaluate(optimized)}")


class ExpressionNode:
    """Node in expression tree"""
    
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
        self.is_operator = value in "+-*/"
    
    def __str__(self):
        return str(self.value)


class ExpressionTreeBuilder:
    """
    Expression tree builder and evaluator
    """
    
    def __init__(self):
        self.precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    
    def build_from_infix(self, expression: str) -> ExpressionNode:
        """Build expression tree from infix notation"""
        # Simple recursive descent parser
        tokens = self._tokenize(expression)
        self.tokens = tokens
        self.pos = 0
        return self._parse_expression()
    
    def _tokenize(self, expression: str) -> List[str]:
        """Tokenize expression string"""
        tokens = []
        i = 0
        while i < len(expression):
            if expression[i].isspace():
                i += 1
            elif expression[i].isdigit():
                num = ""
                while i < len(expression) and expression[i].isdigit():
                    num += expression[i]
                    i += 1
                tokens.append(num)
            else:
                tokens.append(expression[i])
                i += 1
        return tokens
    
    def _parse_expression(self) -> ExpressionNode:
        """Parse expression with precedence"""
        left = self._parse_term()
        
        while (self.pos < len(self.tokens) and 
               self.tokens[self.pos] in "+-"):
            operator = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_term()
            left = ExpressionNode(operator, left, right)
        
        return left
    
    def _parse_term(self) -> ExpressionNode:
        """Parse multiplication and division"""
        left = self._parse_factor()
        
        while (self.pos < len(self.tokens) and 
               self.tokens[self.pos] in "*/"):
            operator = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_factor()
            left = ExpressionNode(operator, left, right)
        
        return left
    
    def _parse_factor(self) -> ExpressionNode:
        """Parse factors (numbers and parenthesized expressions)"""
        if self.pos >= len(self.tokens):
            return None
        
        token = self.tokens[self.pos]
        
        if token == "(":
            self.pos += 1
            node = self._parse_expression()
            if self.pos < len(self.tokens) and self.tokens[self.pos] == ")":
                self.pos += 1
            return node
        elif token.isdigit():
            self.pos += 1
            return ExpressionNode(int(token))
        
        return None
    
    def evaluate(self, node: ExpressionNode) -> float:
        """Evaluate expression tree"""
        if not node:
            return 0
        
        if not node.is_operator:
            return float(node.value)
        
        left_val = self.evaluate(node.left)
        right_val = self.evaluate(node.right)
        
        if node.value == '+':
            return left_val + right_val
        elif node.value == '-':
            return left_val - right_val
        elif node.value == '*':
            return left_val * right_val
        elif node.value == '/':
            return left_val / right_val if right_val != 0 else float('inf')
        
        return 0
    
    def inorder_traversal(self, node: ExpressionNode) -> str:
        """In-order traversal (infix notation)"""
        if not node:
            return ""
        
        if not node.is_operator:
            return str(node.value)
        
        left = self.inorder_traversal(node.left)
        right = self.inorder_traversal(node.right)
        return f"({left} {node.value} {right})"
    
    def preorder_traversal(self, node: ExpressionNode) -> str:
        """Pre-order traversal (prefix notation)"""
        if not node:
            return ""
        
        if not node.is_operator:
            return str(node.value)
        
        left = self.preorder_traversal(node.left)
        right = self.preorder_traversal(node.right)
        return f"{node.value} {left} {right}"
    
    def postorder_traversal(self, node: ExpressionNode) -> str:
        """Post-order traversal (postfix notation)"""
        if not node:
            return ""
        
        if not node.is_operator:
            return str(node.value)
        
        left = self.postorder_traversal(node.left)
        right = self.postorder_traversal(node.right)
        return f"{left} {right} {node.value}"
    
    def optimize_tree(self, node: ExpressionNode) -> ExpressionNode:
        """Perform constant folding optimization"""
        if not node or not node.is_operator:
            return node
        
        # Recursively optimize children
        left = self.optimize_tree(node.left)
        right = self.optimize_tree(node.right)
        
        # If both children are constants, fold them
        if (left and right and 
            not left.is_operator and not right.is_operator):
            
            temp_node = ExpressionNode(node.value, left, right)
            result = self.evaluate(temp_node)
            return ExpressionNode(result)
        
        return ExpressionNode(node.value, left, right)
    
    def display_tree(self, node: ExpressionNode, depth: int = 0) -> None:
        """Display expression tree structure"""
        if not node:
            return
        
        indent = "  " * depth
        print(f"{indent}{node.value}")
        
        if node.left or node.right:
            if node.left:
                self.display_tree(node.left, depth + 1)
            else:
                print(f"{'  ' * (depth + 1)}None")
            
            if node.right:
                self.display_tree(node.right, depth + 1)
            else:
                print(f"{'  ' * (depth + 1)}None")


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_tree_applications():
    """Demonstrate all tree applications"""
    print("=== TREE APPLICATIONS DEMONSTRATION ===\n")
    
    applications = TreeApplications()
    
    # 1. Database indexing
    applications.explain_database_indexing()
    print("\n" + "="*60 + "\n")
    
    applications.demonstrate_database_index()
    print("\n" + "="*60 + "\n")
    
    # 2. File systems
    file_system = FileSystemTree()
    file_system.explain_file_system_trees()
    print("\n" + "="*60 + "\n")
    
    file_system.demonstrate_file_system()
    print("\n" + "="*60 + "\n")
    
    # 3. Decision trees
    decision_trees = DecisionTreeApplications()
    decision_trees.explain_decision_trees()
    print("\n" + "="*60 + "\n")
    
    decision_trees.demonstrate_decision_tree()
    print("\n" + "="*60 + "\n")
    
    # 4. Syntax trees
    syntax_trees = SyntaxTreeApplications()
    syntax_trees.explain_syntax_trees()
    print("\n" + "="*60 + "\n")
    
    syntax_trees.demonstrate_expression_tree()


if __name__ == "__main__":
    demonstrate_tree_applications()
    
    print("\n=== TREE APPLICATIONS MASTERY GUIDE ===")
    
    print("\n🎯 APPLICATION DOMAINS:")
    print("• Database Systems: B+ trees for indexing and query optimization")
    print("• File Systems: Hierarchical organization and directory structures")
    print("• Machine Learning: Decision trees for classification and regression")
    print("• Compilers: Abstract syntax trees for program representation")
    print("• Networks: Routing trees and spanning tree protocols")
    
    print("\n📊 PERFORMANCE IMPACT:")
    print("• Database queries: O(log n) vs O(n) without indexes")
    print("• File operations: O(depth) vs O(n) linear search")
    print("• ML predictions: O(depth) vs O(n*features) linear models")
    print("• Code compilation: O(n) vs O(n²) without proper parsing")
    
    print("\n⚡ OPTIMIZATION STRATEGIES:")
    print("• Choose appropriate tree type for workload")
    print("• Balance tree structures to minimize height")
    print("• Use caching for frequently accessed nodes")
    print("• Implement parallel processing for independent subtrees")
    print("• Apply domain-specific optimizations")
    
    print("\n🔧 IMPLEMENTATION CONSIDERATIONS:")
    print("• Handle concurrent access in multi-user systems")
    print("• Implement efficient serialization for persistence")
    print("• Plan for crash recovery and data integrity")
    print("• Design for horizontal scaling when needed")
    print("• Consider memory vs disk storage trade-offs")
    
    print("\n🏆 REAL-WORLD IMPACT:")
    print("• Database performance: 100x+ speedup with proper indexing")
    print("• File system efficiency: Instant navigation vs linear search")
    print("• ML interpretability: Human-readable decision paths")
    print("• Compiler optimization: Efficient code generation")
    print("• System reliability: Structured error handling and recovery")
    
    print("\n🎓 LEARNING PROGRESSION:")
    print("1. Master basic tree operations and properties")
    print("2. Understand specific domain requirements")
    print("3. Learn application-specific optimizations")
    print("4. Study real-world implementations")
    print("5. Practice with large-scale systems")
    
    print("\n💡 SUCCESS TIPS:")
    print("• Understand the problem domain deeply")
    print("• Choose the right tree structure for the use case")
    print("• Consider both theoretical and practical constraints")
    print("• Study existing successful implementations")
    print("• Test with realistic data sizes and workloads")
