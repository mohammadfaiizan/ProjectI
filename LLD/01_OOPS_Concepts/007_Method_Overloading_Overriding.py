"""
METHOD OVERLOADING AND OVERRIDING - Method Design Techniques
============================================================

Problem Statement:
Demonstrate method overloading and overriding concepts:
- Method overriding in inheritance hierarchies
- Method overloading simulation in Python
- Dynamic method dispatch
- Super() method usage patterns
- Method signature design principles

Learning Objectives:
- Master method overriding techniques
- Simulate method overloading effectively
- Understand dynamic dispatch mechanisms
- Use super() method appropriately
- Design flexible method interfaces
"""

from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any, Optional, overload
from functools import singledispatch, wraps
from datetime import datetime
import inspect


# Base class for method overriding demonstration
class Shape(ABC):
    """Abstract base class for shapes."""
    
    def __init__(self, name: str):
        self.name = name
        self.created_at = datetime.now()
    
    @abstractmethod
    def calculate_area(self) -> float:
        """Calculate area - must be overridden."""
        pass
    
    @abstractmethod
    def calculate_perimeter(self) -> float:
        """Calculate perimeter - must be overridden."""
        pass
    
    def get_info(self) -> str:
        """Get shape information - can be overridden."""
        return f"{self.name}: Area={self.calculate_area():.2f}, Perimeter={self.calculate_perimeter():.2f}"
    
    def display(self) -> str:
        """Display shape - template method."""
        return f"Displaying {self.name}"
    
    def __str__(self) -> str:
        return f"{self.name} shape"


# Method overriding examples
class Rectangle(Shape):
    """Rectangle class demonstrating method overriding."""
    
    def __init__(self, width: float, height: float):
        super().__init__("Rectangle")
        self.width = width
        self.height = height
    
    def calculate_area(self) -> float:
        """Override abstract method."""
        return self.width * self.height
    
    def calculate_perimeter(self) -> float:
        """Override abstract method."""
        return 2 * (self.width + self.height)
    
    def get_info(self) -> str:
        """Override parent method with additional information."""
        base_info = super().get_info()  # Call parent method
        return f"{base_info}, Dimensions: {self.width}x{self.height}"
    
    def display(self) -> str:
        """Override display method."""
        parent_display = super().display()
        return f"{parent_display} with dimensions {self.width}x{self.height}"


class Circle(Shape):
    """Circle class demonstrating method overriding."""
    
    def __init__(self, radius: float):
        super().__init__("Circle")
        self.radius = radius
    
    def calculate_area(self) -> float:
        """Override abstract method."""
        import math
        return math.pi * self.radius ** 2
    
    def calculate_perimeter(self) -> float:
        """Override abstract method."""
        import math
        return 2 * math.pi * self.radius
    
    def get_info(self) -> str:
        """Override parent method."""
        base_info = super().get_info()
        return f"{base_info}, Radius: {self.radius}"
    
    def display(self) -> str:
        """Override display method."""
        parent_display = super().display()
        return f"{parent_display} with radius {self.radius}"


class Triangle(Shape):
    """Triangle class demonstrating method overriding."""
    
    def __init__(self, side_a: float, side_b: float, side_c: float):
        super().__init__("Triangle")
        self.side_a = side_a
        self.side_b = side_b
        self.side_c = side_c
    
    def calculate_area(self) -> float:
        """Override abstract method using Heron's formula."""
        s = (self.side_a + self.side_b + self.side_c) / 2
        import math
        return math.sqrt(s * (s - self.side_a) * (s - self.side_b) * (s - self.side_c))
    
    def calculate_perimeter(self) -> float:
        """Override abstract method."""
        return self.side_a + self.side_b + self.side_c
    
    def get_info(self) -> str:
        """Override parent method."""
        base_info = super().get_info()
        return f"{base_info}, Sides: {self.side_a}, {self.side_b}, {self.side_c}"
    
    def is_valid_triangle(self) -> bool:
        """Additional method specific to triangle."""
        return (self.side_a + self.side_b > self.side_c and
                self.side_a + self.side_c > self.side_b and
                self.side_b + self.side_c > self.side_a)


# Method overloading simulation using default parameters
class Calculator:
    """Calculator demonstrating method overloading simulation."""
    
    def add(self, a: Union[int, float], b: Union[int, float] = 0, c: Union[int, float] = 0) -> Union[int, float]:
        """
        Add method with optional parameters (simulates overloading).
        add(5) -> 5
        add(5, 3) -> 8
        add(5, 3, 2) -> 10
        """
        return a + b + c
    
    def multiply(self, *args: Union[int, float]) -> Union[int, float]:
        """
        Multiply method with variable arguments.
        multiply(5) -> 5
        multiply(5, 3) -> 15
        multiply(5, 3, 2) -> 30
        """
        if not args:
            return 0
        
        result = 1
        for arg in args:
            result *= arg
        return result
    
    def power(self, base: Union[int, float], exponent: Union[int, float] = 2) -> Union[int, float]:
        """
        Power method with default exponent.
        power(5) -> 25 (5^2)
        power(5, 3) -> 125 (5^3)
        """
        return base ** exponent
    
    def divide(self, dividend: Union[int, float], divisor: Union[int, float] = 1) -> Union[int, float]:
        """
        Division method with default divisor.
        divide(10) -> 10.0 (10/1)
        divide(10, 2) -> 5.0 (10/2)
        """
        if divisor == 0:
            raise ValueError("Cannot divide by zero")
        return dividend / divisor


# Method overloading using type hints and overload decorator
class MathOperations:
    """Math operations with type-based overloading."""
    
    @overload
    def process(self, value: int) -> str:
        """Process integer value."""
        ...
    
    @overload
    def process(self, value: float) -> str:
        """Process float value."""
        ...
    
    @overload
    def process(self, value: str) -> str:
        """Process string value."""
        ...
    
    @overload
    def process(self, value: List[Union[int, float]]) -> str:
        """Process list of numbers."""
        ...
    
    def process(self, value: Union[int, float, str, List[Union[int, float]]]) -> str:
        """
        Actual implementation that handles all types.
        Type checker will use the overload signatures above.
        """
        if isinstance(value, int):
            return f"Processing integer: {value} -> {value * 2}"
        elif isinstance(value, float):
            return f"Processing float: {value:.2f} -> {value * 1.5:.2f}"
        elif isinstance(value, str):
            return f"Processing string: '{value}' -> '{value.upper()}'"
        elif isinstance(value, list):
            total = sum(value)
            return f"Processing list of {len(value)} numbers -> sum: {total}"
        else:
            return f"Unknown type: {type(value)}"


# Single dispatch for method overloading based on first argument type
@singledispatch
def format_data(data):
    """Generic data formatting function."""
    return f"Unknown data type: {type(data)}"

@format_data.register
def _(data: int):
    """Format integer data."""
    return f"Integer: {data:,}"

@format_data.register
def _(data: float):
    """Format float data."""
    return f"Float: {data:.3f}"

@format_data.register
def _(data: str):
    """Format string data."""
    return f"String: '{data}' (length: {len(data)})"

@format_data.register
def _(data: list):
    """Format list data."""
    return f"List: [{', '.join(str(item) for item in data)}] (count: {len(data)})"

@format_data.register
def _(data: dict):
    """Format dictionary data."""
    return f"Dict: {data} (keys: {len(data)})"


# Advanced method overriding with hooks
class DatabaseConnection(ABC):
    """Abstract database connection class."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.is_connected = False
        self.transaction_count = 0
    
    def connect(self) -> bool:
        """Template method for connection."""
        print("Starting connection process...")
        
        # Hook for pre-connection setup
        self._pre_connect_hook()
        
        # Actual connection (to be overridden)
        success = self._do_connect()
        
        if success:
            self.is_connected = True
            # Hook for post-connection setup
            self._post_connect_hook()
            print("Connection established successfully")
        else:
            print("Connection failed")
        
        return success
    
    def disconnect(self) -> bool:
        """Template method for disconnection."""
        if not self.is_connected:
            return True
        
        print("Starting disconnection process...")
        
        # Hook for pre-disconnection cleanup
        self._pre_disconnect_hook()
        
        # Actual disconnection (to be overridden)
        success = self._do_disconnect()
        
        if success:
            self.is_connected = False
            # Hook for post-disconnection cleanup
            self._post_disconnect_hook()
            print("Disconnection completed")
        
        return success
    
    @abstractmethod
    def _do_connect(self) -> bool:
        """Actual connection implementation (must override)."""
        pass
    
    @abstractmethod
    def _do_disconnect(self) -> bool:
        """Actual disconnection implementation (must override)."""
        pass
    
    def _pre_connect_hook(self) -> None:
        """Hook called before connection (can override)."""
        pass
    
    def _post_connect_hook(self) -> None:
        """Hook called after successful connection (can override)."""
        pass
    
    def _pre_disconnect_hook(self) -> None:
        """Hook called before disconnection (can override)."""
        pass
    
    def _post_disconnect_hook(self) -> None:
        """Hook called after disconnection (can override)."""
        pass
    
    def execute_query(self, query: str) -> str:
        """Execute query (can be overridden)."""
        if not self.is_connected:
            return "Error: Not connected to database"
        
        self.transaction_count += 1
        return f"Executing query: {query}"


class MySQLConnection(DatabaseConnection):
    """MySQL database connection implementation."""
    
    def __init__(self, host: str, database: str, username: str, password: str):
        connection_string = f"mysql://{username}@{host}/{database}"
        super().__init__(connection_string)
        self.host = host
        self.database = database
        self.username = username
        self.password = password
    
    def _do_connect(self) -> bool:
        """Override: MySQL-specific connection."""
        print(f"Connecting to MySQL server at {self.host}")
        print(f"Authenticating user {self.username}")
        print(f"Selecting database {self.database}")
        # Simulate connection
        return True
    
    def _do_disconnect(self) -> bool:
        """Override: MySQL-specific disconnection."""
        print("Closing MySQL connection")
        return True
    
    def _pre_connect_hook(self) -> None:
        """Override: MySQL pre-connection setup."""
        print("Setting MySQL connection parameters")
        print("Configuring character set to UTF-8")
    
    def _post_connect_hook(self) -> None:
        """Override: MySQL post-connection setup."""
        print("Setting MySQL session variables")
        print("Enabling MySQL query logging")
    
    def execute_query(self, query: str) -> str:
        """Override: MySQL-specific query execution."""
        base_result = super().execute_query(query)
        if "Error" in base_result:
            return base_result
        
        return f"MySQL: {base_result} (Transaction #{self.transaction_count})"


class PostgreSQLConnection(DatabaseConnection):
    """PostgreSQL database connection implementation."""
    
    def __init__(self, host: str, database: str, username: str, password: str, port: int = 5432):
        connection_string = f"postgresql://{username}@{host}:{port}/{database}"
        super().__init__(connection_string)
        self.host = host
        self.database = database
        self.username = username
        self.password = password
        self.port = port
    
    def _do_connect(self) -> bool:
        """Override: PostgreSQL-specific connection."""
        print(f"Connecting to PostgreSQL server at {self.host}:{self.port}")
        print(f"Authenticating user {self.username}")
        print(f"Connecting to database {self.database}")
        return True
    
    def _do_disconnect(self) -> bool:
        """Override: PostgreSQL-specific disconnection."""
        print("Closing PostgreSQL connection")
        return True
    
    def _post_connect_hook(self) -> None:
        """Override: PostgreSQL post-connection setup."""
        print("Setting PostgreSQL search path")
        print("Configuring PostgreSQL timezone")
    
    def execute_query(self, query: str) -> str:
        """Override: PostgreSQL-specific query execution."""
        base_result = super().execute_query(query)
        if "Error" in base_result:
            return base_result
        
        return f"PostgreSQL: {base_result} (PID: {hash(self) % 10000})"


# Method chaining with overriding
class QueryBuilder:
    """Query builder demonstrating method chaining and overriding."""
    
    def __init__(self):
        self.query_parts = []
        self.parameters = {}
    
    def select(self, *columns: str) -> 'QueryBuilder':
        """Add SELECT clause."""
        if columns:
            self.query_parts.append(f"SELECT {', '.join(columns)}")
        else:
            self.query_parts.append("SELECT *")
        return self
    
    def from_table(self, table: str) -> 'QueryBuilder':
        """Add FROM clause."""
        self.query_parts.append(f"FROM {table}")
        return self
    
    def where(self, condition: str, **params) -> 'QueryBuilder':
        """Add WHERE clause."""
        self.query_parts.append(f"WHERE {condition}")
        self.parameters.update(params)
        return self
    
    def order_by(self, column: str, direction: str = "ASC") -> 'QueryBuilder':
        """Add ORDER BY clause."""
        self.query_parts.append(f"ORDER BY {column} {direction}")
        return self
    
    def limit(self, count: int) -> 'QueryBuilder':
        """Add LIMIT clause."""
        self.query_parts.append(f"LIMIT {count}")
        return self
    
    def build(self) -> str:
        """Build the final query."""
        return " ".join(self.query_parts)
    
    def __str__(self) -> str:
        return self.build()


class MySQLQueryBuilder(QueryBuilder):
    """MySQL-specific query builder."""
    
    def limit(self, count: int, offset: int = 0) -> 'MySQLQueryBuilder':
        """Override: MySQL LIMIT with OFFSET."""
        if offset > 0:
            self.query_parts.append(f"LIMIT {offset}, {count}")
        else:
            self.query_parts.append(f"LIMIT {count}")
        return self
    
    def use_index(self, index_name: str) -> 'MySQLQueryBuilder':
        """MySQL-specific USE INDEX hint."""
        # Insert after FROM clause
        for i, part in enumerate(self.query_parts):
            if part.startswith("FROM"):
                self.query_parts.insert(i + 1, f"USE INDEX ({index_name})")
                break
        return self


def demonstrate_method_overloading_overriding():
    """
    Demonstrate method overloading and overriding concepts.
    """
    print("=== METHOD OVERLOADING AND OVERRIDING DEMONSTRATION ===\n")
    
    # 1. Method Overriding with Inheritance
    print("1. Method Overriding with Inheritance:")
    
    shapes = [
        Rectangle(5.0, 3.0),
        Circle(4.0),
        Triangle(3.0, 4.0, 5.0)
    ]
    
    for shape in shapes:
        print(f"Shape: {shape}")
        print(f"  Info: {shape.get_info()}")
        print(f"  Display: {shape.display()}")
        
        # Triangle-specific method
        if isinstance(shape, Triangle):
            print(f"  Valid triangle: {shape.is_valid_triangle()}")
        print()
    
    # 2. Method Overloading Simulation
    print("2. Method Overloading Simulation:")
    
    calc = Calculator()
    
    print(f"add(5): {calc.add(5)}")
    print(f"add(5, 3): {calc.add(5, 3)}")
    print(f"add(5, 3, 2): {calc.add(5, 3, 2)}")
    
    print(f"multiply(5): {calc.multiply(5)}")
    print(f"multiply(5, 3): {calc.multiply(5, 3)}")
    print(f"multiply(5, 3, 2): {calc.multiply(5, 3, 2)}")
    
    print(f"power(5): {calc.power(5)}")
    print(f"power(5, 3): {calc.power(5, 3)}")
    
    print(f"divide(10): {calc.divide(10)}")
    print(f"divide(10, 2): {calc.divide(10, 2)}")
    print()
    
    # 3. Type-based Method Overloading
    print("3. Type-based Method Overloading:")
    
    math_ops = MathOperations()
    
    test_values = [42, 3.14159, "hello world", [1, 2, 3, 4, 5]]
    
    for value in test_values:
        result = math_ops.process(value)
        print(f"  {result}")
    print()
    
    # 4. Single Dispatch Method Overloading
    print("4. Single Dispatch Method Overloading:")
    
    test_data = [
        123456,
        3.14159265,
        "Python Programming",
        [1, 2, 3, 4, 5],
        {"name": "John", "age": 30, "city": "New York"}
    ]
    
    for data in test_data:
        formatted = format_data(data)
        print(f"  {formatted}")
    print()
    
    # 5. Advanced Method Overriding with Hooks
    print("5. Advanced Method Overriding with Database Connections:")
    
    # MySQL connection
    mysql_conn = MySQLConnection("localhost", "myapp", "user", "password")
    print("MySQL Connection:")
    mysql_conn.connect()
    print(f"  {mysql_conn.execute_query('SELECT * FROM users')}")
    mysql_conn.disconnect()
    print()
    
    # PostgreSQL connection
    postgres_conn = PostgreSQLConnection("localhost", "myapp", "user", "password", 5432)
    print("PostgreSQL Connection:")
    postgres_conn.connect()
    print(f"  {postgres_conn.execute_query('SELECT * FROM products')}")
    postgres_conn.disconnect()
    print()
    
    # 6. Method Chaining with Overriding
    print("6. Method Chaining with Query Builders:")
    
    # Standard query builder
    standard_query = (QueryBuilder()
                     .select("name", "email", "age")
                     .from_table("users")
                     .where("age > :min_age", min_age=18)
                     .order_by("name")
                     .limit(10))
    
    print(f"Standard Query: {standard_query}")
    
    # MySQL-specific query builder
    mysql_query = (MySQLQueryBuilder()
                  .select("id", "title", "content")
                  .from_table("articles")
                  .use_index("idx_title")
                  .where("published = :published", published=True)
                  .order_by("created_at", "DESC")
                  .limit(5, 10))  # MySQL-specific LIMIT with offset
    
    print(f"MySQL Query: {mysql_query}")
    print()
    
    # 7. Dynamic Method Dispatch
    print("7. Dynamic Method Dispatch:")
    
    def process_shapes(shape_list: List[Shape]) -> None:
        """Process shapes using dynamic dispatch."""
        for shape in shape_list:
            # Method calls are resolved at runtime based on actual object type
            print(f"Processing {type(shape).__name__}:")
            print(f"  Area: {shape.calculate_area():.2f}")
            print(f"  Perimeter: {shape.calculate_perimeter():.2f}")
            print(f"  Info: {shape.get_info()}")
    
    mixed_shapes = [
        Rectangle(4.0, 6.0),
        Circle(3.5),
        Triangle(6.0, 8.0, 10.0)
    ]
    
    process_shapes(mixed_shapes)
    print()
    
    # 8. Method Resolution Order (MRO)
    print("8. Method Resolution Order:")
    
    class A:
        def method(self):
            return "A"
    
    class B(A):
        def method(self):
            return "B -> " + super().method()
    
    class C(A):
        def method(self):
            return "C -> " + super().method()
    
    class D(B, C):
        def method(self):
            return "D -> " + super().method()
    
    d = D()
    print(f"D.method(): {d.method()}")
    print(f"MRO: {[cls.__name__ for cls in D.__mro__]}")
    print()
    
    print("=== METHOD OVERLOADING AND OVERRIDING DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_method_overloading_overriding()
