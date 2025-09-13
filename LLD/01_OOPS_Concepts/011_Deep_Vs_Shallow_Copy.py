"""
DEEP VS SHALLOW COPY - Object Copying Mechanisms
================================================

Problem Statement:
Demonstrate object copying concepts:
- Shallow copy vs deep copy differences
- When to use each copying method
- Custom copy behavior implementation
- Memory implications of copying
- Copy patterns in object design

Learning Objectives:
- Understand shallow vs deep copy mechanisms
- Implement custom copy methods
- Choose appropriate copying strategies
- Handle mutable object copying correctly
- Design copy-safe classes
"""

import copy
from typing import List, Dict, Any, Optional
from datetime import datetime


class Person:
    """
    Person class demonstrating basic copying behavior.
    """
    
    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email
        self.created_at = datetime.now()
    
    def __str__(self) -> str:
        return f"Person(name='{self.name}', age={self.age}, email='{self.email}')"
    
    def __repr__(self) -> str:
        return f"Person('{self.name}', {self.age}, '{self.email}')"


class Address:
    """
    Address class for composition examples.
    """
    
    def __init__(self, street: str, city: str, country: str, postal_code: str):
        self.street = street
        self.city = city
        self.country = country
        self.postal_code = postal_code
    
    def __str__(self) -> str:
        return f"{self.street}, {self.city}, {self.country} {self.postal_code}"
    
    def __repr__(self) -> str:
        return f"Address('{self.street}', '{self.city}', '{self.country}', '{self.postal_code}')"


class Employee:
    """
    Employee class demonstrating shallow vs deep copy with nested objects.
    """
    
    def __init__(self, person: Person, employee_id: str, department: str, salary: float):
        self.person = person  # Nested object
        self.employee_id = employee_id
        self.department = department
        self.salary = salary
        self.address = None  # Will be set later
        self.skills = []  # Mutable list
        self.projects = {}  # Mutable dictionary
        self.performance_ratings = []  # List of dictionaries
    
    def set_address(self, address: Address) -> None:
        """Set employee address."""
        self.address = address
    
    def add_skill(self, skill: str) -> None:
        """Add skill to employee."""
        if skill not in self.skills:
            self.skills.append(skill)
    
    def add_project(self, project_name: str, role: str, start_date: str) -> None:
        """Add project to employee."""
        self.projects[project_name] = {
            'role': role,
            'start_date': start_date,
            'status': 'active'
        }
    
    def add_performance_rating(self, year: int, rating: float, comments: str) -> None:
        """Add performance rating."""
        self.performance_ratings.append({
            'year': year,
            'rating': rating,
            'comments': comments,
            'reviewed_at': datetime.now()
        })
    
    def __str__(self) -> str:
        return f"Employee(id='{self.employee_id}', name='{self.person.name}', dept='{self.department}')"
    
    def __repr__(self) -> str:
        return f"Employee({self.person!r}, '{self.employee_id}', '{self.department}', {self.salary})"


class CustomCopyEmployee:
    """
    Employee class with custom copy behavior.
    """
    
    def __init__(self, name: str, employee_id: str, department: str):
        self.name = name
        self.employee_id = employee_id
        self.department = department
        self.sensitive_data = {"ssn": "123-45-6789", "bank_account": "9876543210"}
        self.work_history = []
        self.created_at = datetime.now()
    
    def add_work_history(self, company: str, position: str, duration: str) -> None:
        """Add work history entry."""
        self.work_history.append({
            'company': company,
            'position': position,
            'duration': duration
        })
    
    def __copy__(self):
        """
        Implement shallow copy behavior.
        Creates new instance but shares mutable objects.
        """
        print(f"Custom shallow copy called for {self.name}")
        
        # Create new instance
        new_employee = CustomCopyEmployee(self.name, self.employee_id, self.department)
        
        # Copy attributes (shallow copy of mutable objects)
        new_employee.sensitive_data = self.sensitive_data  # Shared reference
        new_employee.work_history = self.work_history      # Shared reference
        new_employee.created_at = self.created_at
        
        return new_employee
    
    def __deepcopy__(self, memo):
        """
        Implement deep copy behavior.
        Creates new instance with deep copies of all mutable objects.
        """
        print(f"Custom deep copy called for {self.name}")
        
        # Create new instance
        new_employee = CustomCopyEmployee(self.name, self.employee_id, self.department)
        
        # Deep copy mutable attributes
        new_employee.sensitive_data = copy.deepcopy(self.sensitive_data, memo)
        new_employee.work_history = copy.deepcopy(self.work_history, memo)
        new_employee.created_at = self.created_at  # datetime is immutable
        
        return new_employee
    
    def __str__(self) -> str:
        return f"CustomCopyEmployee(name='{self.name}', id='{self.employee_id}')"


class ImmutablePerson:
    """
    Immutable person class demonstrating copy behavior with immutable objects.
    """
    
    def __init__(self, name: str, age: int, email: str):
        # Use __slots__ to prevent attribute addition
        self._name = name
        self._age = age
        self._email = email
        self._created_at = datetime.now()
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def age(self) -> int:
        return self._age
    
    @property
    def email(self) -> str:
        return self._email
    
    @property
    def created_at(self) -> datetime:
        return self._created_at
    
    def with_age(self, new_age: int) -> 'ImmutablePerson':
        """Return new instance with updated age (immutable pattern)."""
        return ImmutablePerson(self._name, new_age, self._email)
    
    def with_email(self, new_email: str) -> 'ImmutablePerson':
        """Return new instance with updated email."""
        return ImmutablePerson(self._name, self._age, new_email)
    
    def __str__(self) -> str:
        return f"ImmutablePerson(name='{self._name}', age={self._age})"
    
    def __repr__(self) -> str:
        return f"ImmutablePerson('{self._name}', {self._age}, '{self._email}')"


class CopyTracker:
    """
    Class that tracks copying operations for demonstration.
    """
    
    copy_count = 0
    deepcopy_count = 0
    
    def __init__(self, name: str, data: List[Any]):
        self.name = name
        self.data = data
        self.id = id(self)
    
    def __copy__(self):
        """Track shallow copies."""
        CopyTracker.copy_count += 1
        print(f"Shallow copy #{CopyTracker.copy_count} of {self.name}")
        
        new_tracker = CopyTracker(self.name, self.data)  # Shares data reference
        return new_tracker
    
    def __deepcopy__(self, memo):
        """Track deep copies."""
        CopyTracker.deepcopy_count += 1
        print(f"Deep copy #{CopyTracker.deepcopy_count} of {self.name}")
        
        new_tracker = CopyTracker(self.name, copy.deepcopy(self.data, memo))
        return new_tracker
    
    @classmethod
    def reset_counters(cls):
        """Reset copy counters."""
        cls.copy_count = 0
        cls.deepcopy_count = 0
    
    def __str__(self) -> str:
        return f"CopyTracker(name='{self.name}', id={self.id}, data_id={id(self.data)})"


def demonstrate_assignment_vs_copy():
    """Demonstrate the difference between assignment and copying."""
    print("=== ASSIGNMENT VS COPY ===")
    
    # Original object
    original_person = Person("Alice", 30, "alice@example.com")
    print(f"Original: {original_person} (id: {id(original_person)})")
    
    # Assignment (same object)
    assigned_person = original_person
    print(f"Assigned: {assigned_person} (id: {id(assigned_person)})")
    print(f"Same object: {original_person is assigned_person}")
    
    # Modify assigned object
    assigned_person.age = 31
    print(f"After modifying assigned object:")
    print(f"  Original: {original_person}")
    print(f"  Assigned: {assigned_person}")
    print()


def demonstrate_shallow_copy():
    """Demonstrate shallow copy behavior."""
    print("=== SHALLOW COPY DEMONSTRATION ===")
    
    # Create employee with nested objects
    person = Person("Bob", 25, "bob@example.com")
    employee = Employee(person, "EMP001", "Engineering", 75000.0)
    
    # Add address
    address = Address("123 Main St", "New York", "USA", "10001")
    employee.set_address(address)
    
    # Add skills and projects
    employee.add_skill("Python")
    employee.add_skill("JavaScript")
    employee.add_project("Project Alpha", "Developer", "2023-01-01")
    employee.add_performance_rating(2023, 4.5, "Excellent performance")
    
    print(f"Original employee: {employee}")
    print(f"  Person: {employee.person} (id: {id(employee.person)})")
    print(f"  Address: {employee.address} (id: {id(employee.address)})")
    print(f"  Skills: {employee.skills} (id: {id(employee.skills)})")
    print(f"  Projects: {employee.projects} (id: {id(employee.projects)})")
    
    # Shallow copy
    shallow_copy_employee = copy.copy(employee)
    print(f"\nShallow copy: {shallow_copy_employee}")
    print(f"  Person: {shallow_copy_employee.person} (id: {id(shallow_copy_employee.person)})")
    print(f"  Address: {shallow_copy_employee.address} (id: {id(shallow_copy_employee.address)})")
    print(f"  Skills: {shallow_copy_employee.skills} (id: {id(shallow_copy_employee.skills)})")
    print(f"  Projects: {shallow_copy_employee.projects} (id: {id(shallow_copy_employee.projects)})")
    
    # Check if objects are the same
    print(f"\nObject identity checks:")
    print(f"  Same employee object: {employee is shallow_copy_employee}")
    print(f"  Same person object: {employee.person is shallow_copy_employee.person}")
    print(f"  Same address object: {employee.address is shallow_copy_employee.address}")
    print(f"  Same skills list: {employee.skills is shallow_copy_employee.skills}")
    print(f"  Same projects dict: {employee.projects is shallow_copy_employee.projects}")
    
    # Modify nested objects
    print(f"\nModifying nested objects:")
    shallow_copy_employee.person.name = "Bob Smith"  # Affects original
    shallow_copy_employee.skills.append("React")     # Affects original
    shallow_copy_employee.projects["Project Beta"] = {"role": "Lead", "start_date": "2023-06-01", "status": "active"}
    
    print(f"After modification:")
    print(f"  Original person name: {employee.person.name}")
    print(f"  Original skills: {employee.skills}")
    print(f"  Original projects: {list(employee.projects.keys())}")
    print()


def demonstrate_deep_copy():
    """Demonstrate deep copy behavior."""
    print("=== DEEP COPY DEMONSTRATION ===")
    
    # Create employee with nested objects
    person = Person("Carol", 28, "carol@example.com")
    employee = Employee(person, "EMP002", "Marketing", 65000.0)
    
    # Add address
    address = Address("456 Oak Ave", "San Francisco", "USA", "94102")
    employee.set_address(address)
    
    # Add skills and projects
    employee.add_skill("Marketing")
    employee.add_skill("Analytics")
    employee.add_project("Campaign X", "Manager", "2023-02-01")
    employee.add_performance_rating(2023, 4.2, "Good performance")
    
    print(f"Original employee: {employee}")
    print(f"  Person: {employee.person} (id: {id(employee.person)})")
    print(f"  Address: {employee.address} (id: {id(employee.address)})")
    print(f"  Skills: {employee.skills} (id: {id(employee.skills)})")
    print(f"  Projects: {employee.projects} (id: {id(employee.projects)})")
    
    # Deep copy
    deep_copy_employee = copy.deepcopy(employee)
    print(f"\nDeep copy: {deep_copy_employee}")
    print(f"  Person: {deep_copy_employee.person} (id: {id(deep_copy_employee.person)})")
    print(f"  Address: {deep_copy_employee.address} (id: {id(deep_copy_employee.address)})")
    print(f"  Skills: {deep_copy_employee.skills} (id: {id(deep_copy_employee.skills)})")
    print(f"  Projects: {deep_copy_employee.projects} (id: {id(deep_copy_employee.projects)})")
    
    # Check if objects are different
    print(f"\nObject identity checks:")
    print(f"  Same employee object: {employee is deep_copy_employee}")
    print(f"  Same person object: {employee.person is deep_copy_employee.person}")
    print(f"  Same address object: {employee.address is deep_copy_employee.address}")
    print(f"  Same skills list: {employee.skills is deep_copy_employee.skills}")
    print(f"  Same projects dict: {employee.projects is deep_copy_employee.projects}")
    
    # Modify nested objects
    print(f"\nModifying nested objects:")
    deep_copy_employee.person.name = "Carol Johnson"  # Does NOT affect original
    deep_copy_employee.skills.append("SEO")           # Does NOT affect original
    deep_copy_employee.projects["Campaign Y"] = {"role": "Lead", "start_date": "2023-07-01", "status": "active"}
    
    print(f"After modification:")
    print(f"  Original person name: {employee.person.name}")
    print(f"  Deep copy person name: {deep_copy_employee.person.name}")
    print(f"  Original skills: {employee.skills}")
    print(f"  Deep copy skills: {deep_copy_employee.skills}")
    print(f"  Original projects: {list(employee.projects.keys())}")
    print(f"  Deep copy projects: {list(deep_copy_employee.projects.keys())}")
    print()


def demonstrate_custom_copy_methods():
    """Demonstrate custom copy method implementation."""
    print("=== CUSTOM COPY METHODS DEMONSTRATION ===")
    
    # Create employee with custom copy behavior
    employee = CustomCopyEmployee("David", "EMP003", "Sales")
    employee.add_work_history("TechCorp", "Junior Developer", "2020-2022")
    employee.add_work_history("StartupInc", "Senior Developer", "2022-2023")
    
    print(f"Original employee: {employee}")
    print(f"  Sensitive data: {employee.sensitive_data} (id: {id(employee.sensitive_data)})")
    print(f"  Work history: {employee.work_history} (id: {id(employee.work_history)})")
    
    # Custom shallow copy
    shallow_copy = copy.copy(employee)
    print(f"\nCustom shallow copy: {shallow_copy}")
    print(f"  Sensitive data: {shallow_copy.sensitive_data} (id: {id(shallow_copy.sensitive_data)})")
    print(f"  Work history: {shallow_copy.work_history} (id: {id(shallow_copy.work_history)})")
    print(f"  Shares sensitive data: {employee.sensitive_data is shallow_copy.sensitive_data}")
    print(f"  Shares work history: {employee.work_history is shallow_copy.work_history}")
    
    # Custom deep copy
    deep_copy = copy.deepcopy(employee)
    print(f"\nCustom deep copy: {deep_copy}")
    print(f"  Sensitive data: {deep_copy.sensitive_data} (id: {id(deep_copy.sensitive_data)})")
    print(f"  Work history: {deep_copy.work_history} (id: {id(deep_copy.work_history)})")
    print(f"  Shares sensitive data: {employee.sensitive_data is deep_copy.sensitive_data}")
    print(f"  Shares work history: {employee.work_history is deep_copy.work_history}")
    
    # Modify data
    shallow_copy.sensitive_data["ssn"] = "999-88-7777"  # Affects original
    deep_copy.work_history.append({"company": "NewCorp", "position": "Manager", "duration": "2023-present"})  # Does NOT affect original
    
    print(f"\nAfter modifications:")
    print(f"  Original SSN: {employee.sensitive_data['ssn']}")
    print(f"  Shallow copy SSN: {shallow_copy.sensitive_data['ssn']}")
    print(f"  Original work history count: {len(employee.work_history)}")
    print(f"  Deep copy work history count: {len(deep_copy.work_history)}")
    print()


def demonstrate_copy_tracking():
    """Demonstrate copy operation tracking."""
    print("=== COPY TRACKING DEMONSTRATION ===")
    
    CopyTracker.reset_counters()
    
    # Create tracker with mutable data
    data = [1, 2, [3, 4], {"key": "value"}]
    tracker = CopyTracker("Tracker1", data)
    
    print(f"Original tracker: {tracker}")
    
    # Perform multiple copies
    shallow1 = copy.copy(tracker)
    shallow2 = copy.copy(tracker)
    
    deep1 = copy.deepcopy(tracker)
    deep2 = copy.deepcopy(tracker)
    
    print(f"\nCopy statistics:")
    print(f"  Shallow copies: {CopyTracker.copy_count}")
    print(f"  Deep copies: {CopyTracker.deepcopy_count}")
    
    # Show data sharing
    print(f"\nData sharing analysis:")
    print(f"  Original data id: {id(tracker.data)}")
    print(f"  Shallow1 data id: {id(shallow1.data)} (same: {tracker.data is shallow1.data})")
    print(f"  Shallow2 data id: {id(shallow2.data)} (same: {tracker.data is shallow2.data})")
    print(f"  Deep1 data id: {id(deep1.data)} (same: {tracker.data is deep1.data})")
    print(f"  Deep2 data id: {id(deep2.data)} (same: {tracker.data is deep2.data})")
    print()


def demonstrate_immutable_objects():
    """Demonstrate copying with immutable objects."""
    print("=== IMMUTABLE OBJECTS DEMONSTRATION ===")
    
    # Create immutable person
    person = ImmutablePerson("Eve", 32, "eve@example.com")
    print(f"Original person: {person} (id: {id(person)})")
    
    # Shallow copy of immutable object
    shallow_copy = copy.copy(person)
    print(f"Shallow copy: {shallow_copy} (id: {id(shallow_copy)})")
    print(f"Same object: {person is shallow_copy}")
    
    # Deep copy of immutable object
    deep_copy = copy.deepcopy(person)
    print(f"Deep copy: {deep_copy} (id: {id(deep_copy)})")
    print(f"Same object: {person is deep_copy}")
    
    # Create new instances using immutable pattern
    older_person = person.with_age(33)
    new_email_person = person.with_email("eve.smith@example.com")
    
    print(f"\nImmutable updates:")
    print(f"  Original: {person}")
    print(f"  Older: {older_person} (id: {id(older_person)})")
    print(f"  New email: {new_email_person} (id: {id(new_email_person)})")
    print()


def demonstrate_copy_performance():
    """Demonstrate performance implications of copying."""
    print("=== COPY PERFORMANCE DEMONSTRATION ===")
    
    import time
    
    # Create large data structure
    large_data = {
        'numbers': list(range(10000)),
        'nested': [{'id': i, 'data': list(range(100))} for i in range(100)],
        'strings': [f"string_{i}" for i in range(1000)]
    }
    
    print(f"Large data structure created with {len(large_data)} top-level keys")
    
    # Time shallow copy
    start_time = time.time()
    shallow_copy = copy.copy(large_data)
    shallow_time = time.time() - start_time
    
    # Time deep copy
    start_time = time.time()
    deep_copy = copy.deepcopy(large_data)
    deep_time = time.time() - start_time
    
    print(f"\nPerformance comparison:")
    print(f"  Shallow copy time: {shallow_time:.6f} seconds")
    print(f"  Deep copy time: {deep_time:.6f} seconds")
    print(f"  Deep copy is {deep_time/shallow_time:.1f}x slower")
    
    # Memory usage (approximate)
    print(f"\nMemory sharing:")
    print(f"  Original numbers id: {id(large_data['numbers'])}")
    print(f"  Shallow copy numbers id: {id(shallow_copy['numbers'])} (shared: {large_data['numbers'] is shallow_copy['numbers']})")
    print(f"  Deep copy numbers id: {id(deep_copy['numbers'])} (shared: {large_data['numbers'] is deep_copy['numbers']})")
    print()


def demonstrate_deep_vs_shallow_copy():
    """
    Main demonstration function for deep vs shallow copy concepts.
    """
    print("=== DEEP VS SHALLOW COPY DEMONSTRATION ===\n")
    
    # 1. Assignment vs Copy
    demonstrate_assignment_vs_copy()
    
    # 2. Shallow Copy
    demonstrate_shallow_copy()
    
    # 3. Deep Copy
    demonstrate_deep_copy()
    
    # 4. Custom Copy Methods
    demonstrate_custom_copy_methods()
    
    # 5. Copy Tracking
    demonstrate_copy_tracking()
    
    # 6. Immutable Objects
    demonstrate_immutable_objects()
    
    # 7. Performance Implications
    demonstrate_copy_performance()
    
    # 8. Best Practices Summary
    print("=== COPY BEST PRACTICES ===")
    print("✓ Use shallow copy when you need a new container but can share contents")
    print("✓ Use deep copy when you need completely independent objects")
    print("✓ Implement custom __copy__ and __deepcopy__ for special behavior")
    print("✓ Consider immutable objects to avoid copying issues")
    print("✓ Be aware of performance implications of deep copying large structures")
    print("✓ Use copy.copy() for shallow copy, copy.deepcopy() for deep copy")
    print("✓ Test copy behavior with mutable nested objects")
    print("✓ Document copying expectations in class documentation")
    print()
    
    print("=== DEEP VS SHALLOW COPY DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_deep_vs_shallow_copy()
