"""
Python Encapsulation: Private, Protected, and Public Access Control Patterns
Implementation-focused with minimal comments, maximum functionality coverage
"""

import weakref
from typing import Any, Dict, List, Optional, Union

# Basic encapsulation conventions
class BasicEncapsulation:
    def __init__(self, name, value):
        self.public_attr = name          # Public: accessible everywhere
        self._protected_attr = value     # Protected: internal use, subclass access
        self.__private_attr = "secret"   # Private: name mangling applied
    
    def public_method(self):
        """Public method - accessible everywhere"""
        return f"Public: {self.public_attr}"
    
    def _protected_method(self):
        """Protected method - intended for internal/subclass use"""
        return f"Protected: {self._protected_attr}"
    
    def __private_method(self):
        """Private method - name mangling applied"""
        return f"Private: {self.__private_attr}"
    
    def access_all_attributes(self):
        """Public method that can access all attributes"""
        return {
            "public": self.public_attr,
            "protected": self._protected_attr,
            "private": self.__private_attr,
            "private_method": self.__private_method()
        }

def basic_encapsulation_demo():
    obj = BasicEncapsulation("test", 42)
    
    # Public access
    public_value = obj.public_attr
    public_method_result = obj.public_method()
    
    # Protected access (discouraged but possible)
    protected_value = obj._protected_attr
    protected_method_result = obj._protected_method()
    
    # Private access attempts
    try:
        # This will fail - attribute doesn't exist with this name
        private_direct = obj.__private_attr
        private_access_failed = False
    except AttributeError:
        private_access_failed = True
    
    # Name mangling - actual private attribute name
    mangled_name = f"_BasicEncapsulation__private_attr"
    private_via_mangling = getattr(obj, mangled_name, "Not found")
    
    # Inspect all attributes
    all_attributes = [attr for attr in dir(obj) if not attr.startswith('_BasicEncapsulation__')]
    mangled_attributes = [attr for attr in dir(obj) if attr.startswith('_BasicEncapsulation__')]
    
    return {
        "public_access": {"value": public_value, "method": public_method_result},
        "protected_access": {"value": protected_value, "method": protected_method_result},
        "private_access_failed": private_access_failed,
        "private_via_mangling": private_via_mangling,
        "all_access": obj.access_all_attributes(),
        "attributes": {"normal": all_attributes[:5], "mangled": mangled_attributes},
        "name_mangling_example": mangled_name
    }

# Property-based encapsulation
class TemperatureController:
    def __init__(self, initial_temp=20):
        self._temperature = initial_temp
        self._min_temp = -50
        self._max_temp = 100
        self._history = []
        self._access_count = 0
    
    @property
    def temperature(self):
        """Public interface for temperature with access tracking"""
        self._access_count += 1
        return self._temperature
    
    @temperature.setter
    def temperature(self, value):
        """Controlled temperature setting with validation"""
        if not isinstance(value, (int, float)):
            raise TypeError("Temperature must be a number")
        
        if value < self._min_temp or value > self._max_temp:
            raise ValueError(f"Temperature must be between {self._min_temp} and {self._max_temp}")
        
        old_temp = self._temperature
        self._temperature = value
        self._history.append({"from": old_temp, "to": value})
    
    @property
    def temperature_range(self):
        """Read-only property exposing valid range"""
        return {"min": self._min_temp, "max": self._max_temp}
    
    @property
    def statistics(self):
        """Read-only computed property"""
        return {
            "current": self._temperature,
            "access_count": self._access_count,
            "changes": len(self._history),
            "history": self._history[-5:]  # Last 5 changes
        }
    
    def _validate_range(self, min_temp, max_temp):
        """Protected method for internal validation"""
        if min_temp >= max_temp:
            raise ValueError("Minimum temperature must be less than maximum")
        return True
    
    def update_range(self, min_temp, max_temp):
        """Public method using protected validation"""
        self._validate_range(min_temp, max_temp)
        self._min_temp = min_temp
        self._max_temp = max_temp
        
        # Ensure current temperature is still valid
        if self._temperature < min_temp:
            self.temperature = min_temp
        elif self._temperature > max_temp:
            self.temperature = max_temp

def property_encapsulation_demo():
    controller = TemperatureController(25)
    
    # Normal temperature operations
    initial_temp = controller.temperature
    controller.temperature = 30
    new_temp = controller.temperature
    
    # Test validation
    try:
        controller.temperature = 150  # Should fail
        validation_works = False
    except ValueError:
        validation_works = True
    
    # Multiple accesses to test tracking
    temp1 = controller.temperature
    temp2 = controller.temperature
    temp3 = controller.temperature
    
    # Range operations
    initial_range = controller.temperature_range
    controller.update_range(-20, 80)
    new_range = controller.temperature_range
    
    # Statistics
    stats = controller.statistics
    
    return {
        "temperature_control": {
            "initial": initial_temp,
            "after_change": new_temp,
            "validation_works": validation_works
        },
        "access_tracking": {
            "access_count": stats["access_count"],
            "multiple_reads": [temp1, temp2, temp3]
        },
        "range_management": {
            "initial_range": initial_range,
            "new_range": new_range
        },
        "statistics": stats
    }

# Advanced encapsulation with descriptors
class ValidatedAttribute:
    def __init__(self, validator=None, default=None):
        self.validator = validator
        self.default = default
        self.name = None
    
    def __set_name__(self, owner, name):
        self.name = f"_validated_{name}"
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.name, self.default)
    
    def __set__(self, obj, value):
        if self.validator:
            self.validator(value)
        setattr(obj, self.name, value)
    
    def __delete__(self, obj):
        if hasattr(obj, self.name):
            delattr(obj, self.name)

def positive_number_validator(value):
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError("Value must be a positive number")

def string_validator(value):
    if not isinstance(value, str) or len(value) < 1:
        raise ValueError("Value must be a non-empty string")

class Product:
    name = ValidatedAttribute(string_validator, "Unknown")
    price = ValidatedAttribute(positive_number_validator, 0.0)
    quantity = ValidatedAttribute(positive_number_validator, 1)
    
    def __init__(self, name, price, quantity=1):
        self.name = name
        self.price = price
        self.quantity = quantity
        self._created_at = id(self)
    
    @property
    def total_value(self):
        """Computed property using validated attributes"""
        return self.price * self.quantity
    
    def __repr__(self):
        return f"Product('{self.name}', ${self.price}, qty={self.quantity})"

def descriptor_encapsulation_demo():
    # Create product with valid data
    product = Product("Laptop", 999.99, 5)
    
    valid_creation = {
        "name": product.name,
        "price": product.price,
        "quantity": product.quantity,
        "total_value": product.total_value
    }
    
    # Test validation
    validation_tests = {}
    
    # Valid updates
    product.price = 1199.99
    validation_tests["valid_price_update"] = product.price
    
    # Invalid updates
    try:
        product.price = -100
        validation_tests["negative_price"] = "Failed to validate"
    except ValueError:
        validation_tests["negative_price"] = "Validation worked"
    
    try:
        product.name = ""
        validation_tests["empty_name"] = "Failed to validate"
    except ValueError:
        validation_tests["empty_name"] = "Validation worked"
    
    try:
        product.quantity = 0
        validation_tests["zero_quantity"] = "Failed to validate"
    except ValueError:
        validation_tests["zero_quantity"] = "Validation worked"
    
    return {
        "valid_creation": valid_creation,
        "validation_tests": validation_tests,
        "final_product": repr(product)
    }

# Access control with permissions
class PermissionLevel:
    READ = 1
    WRITE = 2
    DELETE = 4
    ADMIN = 7  # READ + WRITE + DELETE

class SecureData:
    def __init__(self, data, owner_permissions=PermissionLevel.ADMIN):
        self._data = data
        self._owner_permissions = owner_permissions
        self._access_log = []
        self._current_user_permissions = PermissionLevel.READ  # Default
    
    def _log_access(self, operation, success, user_permissions):
        """Private method for logging access attempts"""
        self._access_log.append({
            "operation": operation,
            "success": success,
            "permissions": user_permissions,
            "timestamp": id(self)
        })
    
    def _check_permission(self, required_permission):
        """Private permission check"""
        return bool(self._current_user_permissions & required_permission)
    
    def set_user_permissions(self, permissions):
        """Public method to set current user permissions"""
        if self._check_permission(PermissionLevel.ADMIN):
            self._current_user_permissions = permissions
            self._log_access("permission_change", True, permissions)
            return True
        else:
            self._log_access("permission_change", False, self._current_user_permissions)
            raise PermissionError("Admin access required to change permissions")
    
    def read_data(self):
        """Read operation with permission check"""
        if self._check_permission(PermissionLevel.READ):
            self._log_access("read", True, self._current_user_permissions)
            return self._data.copy() if isinstance(self._data, dict) else self._data
        else:
            self._log_access("read", False, self._current_user_permissions)
            raise PermissionError("Read permission required")
    
    def write_data(self, new_data):
        """Write operation with permission check"""
        if self._check_permission(PermissionLevel.WRITE):
            old_data = self._data
            self._data = new_data
            self._log_access("write", True, self._current_user_permissions)
            return {"old": old_data, "new": new_data}
        else:
            self._log_access("write", False, self._current_user_permissions)
            raise PermissionError("Write permission required")
    
    def delete_data(self):
        """Delete operation with permission check"""
        if self._check_permission(PermissionLevel.DELETE):
            deleted_data = self._data
            self._data = None
            self._log_access("delete", True, self._current_user_permissions)
            return deleted_data
        else:
            self._log_access("delete", False, self._current_user_permissions)
            raise PermissionError("Delete permission required")
    
    def get_access_log(self):
        """Read-only access to log (requires read permission)"""
        if self._check_permission(PermissionLevel.READ):
            return self._access_log.copy()
        else:
            raise PermissionError("Read permission required for access log")

def permission_based_encapsulation_demo():
    secure_data = SecureData({"secret": "classified", "public": "everyone"})
    
    # Admin operations (default permission)
    admin_results = {
        "initial_read": secure_data.read_data(),
        "permission_change": secure_data.set_user_permissions(PermissionLevel.READ),
    }
    
    # Read-only user operations
    readonly_results = {}
    try:
        readonly_results["read_success"] = secure_data.read_data()
    except PermissionError:
        readonly_results["read_success"] = "Permission denied"
    
    try:
        secure_data.write_data({"new": "data"})
        readonly_results["write_blocked"] = False
    except PermissionError:
        readonly_results["write_blocked"] = True
    
    try:
        secure_data.delete_data()
        readonly_results["delete_blocked"] = False
    except PermissionError:
        readonly_results["delete_blocked"] = True
    
    # Change to write permissions
    secure_data.set_user_permissions(PermissionLevel.ADMIN)  # Need admin to change permissions
    secure_data.set_user_permissions(PermissionLevel.READ | PermissionLevel.WRITE)
    
    write_results = {
        "write_success": secure_data.write_data({"updated": "data"}),
    }
    
    try:
        secure_data.delete_data()
        write_results["delete_still_blocked"] = False
    except PermissionError:
        write_results["delete_still_blocked"] = True
    
    # Get access log
    access_log = secure_data.get_access_log()
    
    return {
        "admin_operations": admin_results,
        "readonly_operations": readonly_results,
        "write_operations": write_results,
        "access_log_entries": len(access_log),
        "recent_log": access_log[-3:] if len(access_log) >= 3 else access_log
    }

# Information hiding with interfaces
class PaymentProcessor:
    """Abstract interface for payment processing"""
    
    def process_payment(self, amount, method):
        raise NotImplementedError("Subclasses must implement process_payment")
    
    def validate_payment(self, amount, method):
        raise NotImplementedError("Subclasses must implement validate_payment")

class CreditCardProcessor(PaymentProcessor):
    def __init__(self):
        self.__encryption_key = "secret_key_12345"
        self._transaction_log = []
        self._processing_fees = {"credit": 0.029, "debit": 0.015}
    
    def __encrypt_card_data(self, card_number):
        """Private method for card encryption"""
        # Simulate encryption
        return f"encrypted_{card_number[-4:]}"
    
    def __calculate_fee(self, amount, card_type):
        """Private fee calculation"""
        rate = self._processing_fees.get(card_type, 0.03)
        return amount * rate
    
    def _log_transaction(self, amount, method, status):
        """Protected method for transaction logging"""
        self._transaction_log.append({
            "amount": amount,
            "method": method,
            "status": status,
            "id": len(self._transaction_log) + 1
        })
    
    def validate_payment(self, amount, method):
        """Public interface method"""
        if amount <= 0:
            return False, "Amount must be positive"
        
        if method not in ["credit", "debit"]:
            return False, "Invalid payment method"
        
        if amount > 10000:
            return False, "Amount exceeds daily limit"
        
        return True, "Payment validated"
    
    def process_payment(self, amount, method, card_number="1234567890123456"):
        """Public interface method"""
        is_valid, message = self.validate_payment(amount, method)
        
        if not is_valid:
            self._log_transaction(amount, method, f"Failed: {message}")
            return {"success": False, "message": message}
        
        # Process payment using private methods
        encrypted_card = self.__encrypt_card_data(card_number)
        fee = self.__calculate_fee(amount, method)
        
        # Simulate processing
        processed_amount = amount + fee
        
        self._log_transaction(amount, method, "Success")
        
        return {
            "success": True,
            "amount": amount,
            "fee": fee,
            "total": processed_amount,
            "card": encrypted_card,
            "transaction_id": len(self._transaction_log)
        }
    
    def get_transaction_summary(self):
        """Public method exposing controlled transaction data"""
        return {
            "total_transactions": len(self._transaction_log),
            "successful_count": len([t for t in self._transaction_log if t["status"] == "Success"]),
            "recent_transactions": self._transaction_log[-3:]
        }

def interface_hiding_demo():
    processor = CreditCardProcessor()
    
    # Valid payment processing
    payment1 = processor.process_payment(100.00, "credit")
    payment2 = processor.process_payment(50.00, "debit")
    
    # Invalid payment attempts
    payment3 = processor.process_payment(-10.00, "credit")  # Negative amount
    payment4 = processor.process_payment(15000.00, "credit")  # Exceeds limit
    payment5 = processor.process_payment(100.00, "crypto")  # Invalid method
    
    # Try to access private methods (should fail)
    private_access_attempts = {}
    
    try:
        # This should fail - private method not accessible
        processor.__encrypt_card_data("1234567890123456")
        private_access_attempts["encrypt_direct"] = "Accessible"
    except AttributeError:
        private_access_attempts["encrypt_direct"] = "Properly hidden"
    
    # Access via name mangling (possible but discouraged)
    mangled_method = getattr(processor, "_CreditCardProcessor__encrypt_card_data", None)
    private_access_attempts["encrypt_mangled"] = "Found via mangling" if mangled_method else "Not found"
    
    # Public interface works
    summary = processor.get_transaction_summary()
    
    return {
        "successful_payments": [payment1, payment2],
        "failed_payments": [payment3, payment4, payment5],
        "private_access": private_access_attempts,
        "transaction_summary": summary,
        "encapsulation_demo": "Internal implementation hidden, public interface clean"
    }

# Getter/setter vs property comparison
class TraditionalGetterSetter:
    def __init__(self, value):
        self._value = value
        self._access_count = 0
    
    def get_value(self):
        """Traditional getter method"""
        self._access_count += 1
        return self._value
    
    def set_value(self, value):
        """Traditional setter method"""
        if value < 0:
            raise ValueError("Value cannot be negative")
        self._value = value
    
    def get_access_count(self):
        return self._access_count

class PropertyBased:
    def __init__(self, value):
        self._value = value
        self._access_count = 0
    
    @property
    def value(self):
        """Property getter"""
        self._access_count += 1
        return self._value
    
    @value.setter
    def value(self, value):
        """Property setter"""
        if value < 0:
            raise ValueError("Value cannot be negative")
        self._value = value
    
    @property
    def access_count(self):
        """Read-only property"""
        return self._access_count

def getter_setter_comparison_demo():
    # Traditional approach
    traditional = TraditionalGetterSetter(10)
    trad_value1 = traditional.get_value()
    traditional.set_value(20)
    trad_value2 = traditional.get_value()
    trad_access_count = traditional.get_access_count()
    
    # Property approach
    property_obj = PropertyBased(10)
    prop_value1 = property_obj.value
    property_obj.value = 20
    prop_value2 = property_obj.value
    prop_access_count = property_obj.access_count
    
    # Validation test
    try:
        traditional.set_value(-5)
        trad_validation = False
    except ValueError:
        trad_validation = True
    
    try:
        property_obj.value = -5
        prop_validation = False
    except ValueError:
        prop_validation = True
    
    return {
        "traditional_approach": {
            "values": [trad_value1, trad_value2],
            "access_count": trad_access_count,
            "validation_works": trad_validation,
            "syntax": "obj.get_value() / obj.set_value(x)"
        },
        "property_approach": {
            "values": [prop_value1, prop_value2],
            "access_count": prop_access_count,
            "validation_works": prop_validation,
            "syntax": "obj.value / obj.value = x"
        },
        "recommendation": "Use properties for cleaner, more Pythonic interface"
    }

# Real-world encapsulation example: Database connection
class DatabaseConnection:
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, connection_string="default"):
        if not self._initialized:
            self.__connection_string = connection_string
            self._connection_pool = []
            self._active_connections = 0
            self._max_connections = 10
            self.__is_connected = False
            self._query_log = []
            DatabaseConnection._initialized = True
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
    
    def __del__(self):
        if hasattr(self, "_DatabaseConnection__is_connected") and self.__is_connected:
            self.disconnect()
    
    def _validate_query(self, query):
        """Protected method for query validation"""
        dangerous_keywords = ["DROP", "DELETE FROM", "TRUNCATE"]
        query_upper = query.upper()
        
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                return False, f"Dangerous operation detected: {keyword}"
        
        return True, "Query validated"
    
    def _log_query(self, query, success, result_count=0):
        """Protected method for query logging"""
        self._query_log.append({
            "query": query[:50] + "..." if len(query) > 50 else query,
            "success": success,
            "result_count": result_count,
            "connection_id": id(self)
        })
    
    def connect(self):
        """Public method to establish connection"""
        if not self.__is_connected:
            if self._active_connections < self._max_connections:
                self.__is_connected = True
                self._active_connections += 1
                return True
            else:
                raise ConnectionError("Maximum connections exceeded")
        return True
    
    def disconnect(self):
        """Public method to close connection"""
        if self.__is_connected:
            self.__is_connected = False
            self._active_connections = max(0, self._active_connections - 1)
    
    @property
    def is_connected(self):
        """Read-only property for connection status"""
        return self.__is_connected
    
    @property
    def connection_info(self):
        """Read-only property with connection details"""
        return {
            "active_connections": self._active_connections,
            "max_connections": self._max_connections,
            "is_connected": self.__is_connected,
            "query_count": len(self._query_log)
        }
    
    def execute_query(self, query):
        """Public method for query execution"""
        if not self.__is_connected:
            raise ConnectionError("Not connected to database")
        
        is_valid, validation_message = self._validate_query(query)
        if not is_valid:
            self._log_query(query, False)
            raise ValueError(validation_message)
        
        # Simulate query execution
        result_count = hash(query) % 100  # Simulate variable result count
        self._log_query(query, True, result_count)
        
        return {
            "query": query,
            "result_count": result_count,
            "success": True
        }
    
    def get_query_statistics(self):
        """Public method for query analytics"""
        if not self._query_log:
            return {"total": 0, "successful": 0, "failed": 0}
        
        total = len(self._query_log)
        successful = len([q for q in self._query_log if q["success"]])
        failed = total - successful
        
        return {
            "total": total,
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / total * 100) if total > 0 else 0,
            "recent_queries": self._query_log[-3:]
        }

def real_world_encapsulation_demo():
    # Test singleton pattern
    db1 = DatabaseConnection("connection1")
    db2 = DatabaseConnection("connection2")
    singleton_test = db1 is db2
    
    # Test context manager
    with DatabaseConnection() as db:
        query_results = []
        
        # Execute valid queries
        query_results.append(db.execute_query("SELECT * FROM users"))
        query_results.append(db.execute_query("INSERT INTO logs VALUES ('test')"))
        
        # Try dangerous query
        try:
            db.execute_query("DROP TABLE users")
            dangerous_blocked = False
        except ValueError:
            dangerous_blocked = True
        
        connection_info = db.connection_info
        query_stats = db.get_query_statistics()
    
    # Test private attribute access
    try:
        connection_string = db._DatabaseConnection__connection_string
        private_accessible = True
    except AttributeError:
        private_accessible = False
    
    return {
        "singleton_pattern": singleton_test,
        "query_results": query_results,
        "dangerous_query_blocked": dangerous_blocked,
        "connection_info": connection_info,
        "query_statistics": query_stats,
        "private_access": private_accessible,
        "encapsulation_features": [
            "Singleton pattern for connection management",
            "Protected validation methods",
            "Private connection details",
            "Public interface for operations",
            "Property-based status access"
        ]
    }

# Comprehensive testing
def run_all_encapsulation_demos():
    """Execute all encapsulation demonstrations"""
    demo_functions = [
        ('basic_encapsulation', basic_encapsulation_demo),
        ('property_encapsulation', property_encapsulation_demo),
        ('descriptor_encapsulation', descriptor_encapsulation_demo),
        ('permission_based', permission_based_encapsulation_demo),
        ('interface_hiding', interface_hiding_demo),
        ('getter_setter_comparison', getter_setter_comparison_demo),
        ('real_world_example', real_world_encapsulation_demo)
    ]
    
    results = {}
    for name, func in demo_functions:
        try:
            result = func()
            results[name] = result
        except Exception as e:
            results[name] = {'error': str(e)}
    
    return results

if __name__ == "__main__":
    print("=== Python Encapsulation and Access Control Demo ===")
    
    # Run all demonstrations
    all_results = run_all_encapsulation_demos()
    
    for category, data in all_results.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        
        if 'error' in data:
            print(f"  Error: {data['error']}")
            continue
        
        # Display results
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict) and len(value) > 3:
                    print(f"  {key}: {dict(list(value.items())[:3])}... (truncated)")
                elif isinstance(value, list) and len(value) > 5:
                    print(f"  {key}: {value[:5]}... (showing first 5)")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  Result: {data}")
    
    print("\n=== ENCAPSULATION PRINCIPLES ===")
    
    principles = {
        "Public Attributes": "No underscore prefix - accessible everywhere",
        "Protected Attributes": "Single underscore (_) - internal use, subclass access",
        "Private Attributes": "Double underscore (__) - name mangling, class-only access",
        "Properties": "Pythonic way to control attribute access with validation",
        "Descriptors": "Advanced attribute control with reusable validation logic",
        "Information Hiding": "Expose only necessary interface, hide implementation details",
        "Access Control": "Permission-based systems for sensitive operations"
    }
    
    for principle, description in principles.items():
        print(f"  {principle}: {description}")
    
    print("\n=== BEST PRACTICES ===")
    
    best_practices = [
        "Use public attributes for simple data without validation needs",
        "Use properties for attributes requiring validation or computation",
        "Mark internal methods with single underscore (_) prefix",
        "Use double underscore (__) sparingly for true privacy needs",
        "Prefer properties over getter/setter methods for Pythonic code",
        "Implement descriptors for reusable validation patterns",
        "Design clear public interfaces that hide implementation complexity",
        "Document access levels and intended usage patterns",
        "Use context managers for resource management",
        "Implement permission systems for security-critical applications"
    ]
    
    for practice in best_practices:
        print(f"  • {practice}")
    
    print("\n=== COMMON PATTERNS ===")
    
    patterns = [
        "Property-based validation with controlled access",
        "Descriptor classes for reusable attribute behavior",
        "Permission-based access control systems",
        "Interface segregation with public/private method separation",
        "Singleton pattern with private instance management",
        "Context managers for automatic resource cleanup",
        "Logging and monitoring of sensitive operations"
    ]
    
    for pattern in patterns:
        print(f"  • {pattern}")
    
    print("\n=== Encapsulation and Access Control Complete! ===")
    print("  Information hiding and access control patterns mastered")
