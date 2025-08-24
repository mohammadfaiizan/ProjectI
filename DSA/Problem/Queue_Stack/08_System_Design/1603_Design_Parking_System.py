"""
1603. Design Parking System - Multiple Approaches
Difficulty: Easy

Design a parking system for a parking lot. The parking lot has three kinds of parking spaces: big, medium, and small, with a fixed number of slots for each size.

Implement the ParkingSystem class:
- ParkingSystem(int big, int medium, int small) Initializes object of the ParkingSystem class. The number of slots for each parking space are given as part of the constructor.
- bool addCar(int carType) Checks whether there is a parking space of carType for the car that wants to get into the parking lot. carType can be of three kinds: big, medium, or small, which are represented by 1, 2, and 3 respectively. A car can only park in a parking space of its carType. If there is no space available, return false, else park the car in that size space and return true.
"""

from typing import List, Dict
from collections import defaultdict

class ParkingSystemArray:
    """
    Approach 1: Array-based Implementation (Optimal)
    
    Use array to track available spaces for each car type.
    
    Time: O(1) for addCar, Space: O(1)
    """
    
    def __init__(self, big: int, medium: int, small: int):
        # Index 0 unused, 1=big, 2=medium, 3=small
        self.spaces = [0, big, medium, small]
    
    def addCar(self, carType: int) -> bool:
        """Add car to parking lot"""
        if self.spaces[carType] > 0:
            self.spaces[carType] -= 1
            return True
        return False


class ParkingSystemDict:
    """
    Approach 2: Dictionary-based Implementation
    
    Use dictionary to map car types to available spaces.
    
    Time: O(1) for addCar, Space: O(1)
    """
    
    def __init__(self, big: int, medium: int, small: int):
        self.spaces = {1: big, 2: medium, 3: small}
    
    def addCar(self, carType: int) -> bool:
        """Add car to parking lot"""
        if self.spaces[carType] > 0:
            self.spaces[carType] -= 1
            return True
        return False


class ParkingSystemDetailed:
    """
    Approach 3: Detailed Tracking Implementation
    
    Track detailed information about parked cars.
    
    Time: O(1) for addCar, Space: O(total_capacity)
    """
    
    def __init__(self, big: int, medium: int, small: int):
        self.capacity = {1: big, 2: medium, 3: small}
        self.available = {1: big, 2: medium, 3: small}
        self.parked_cars = {1: [], 2: [], 3: []}  # Track car IDs
        self.car_counter = 0
    
    def addCar(self, carType: int) -> bool:
        """Add car to parking lot with detailed tracking"""
        if self.available[carType] > 0:
            self.available[carType] -= 1
            self.car_counter += 1
            self.parked_cars[carType].append(self.car_counter)
            return True
        return False
    
    def removeCar(self, carType: int) -> bool:
        """Remove car from parking lot"""
        if self.parked_cars[carType]:
            self.parked_cars[carType].pop()
            self.available[carType] += 1
            return True
        return False
    
    def getStatus(self) -> Dict:
        """Get current parking status"""
        return {
            "capacity": self.capacity.copy(),
            "available": self.available.copy(),
            "occupied": {k: self.capacity[k] - self.available[k] for k in [1, 2, 3]},
            "utilization": {k: (self.capacity[k] - self.available[k]) / self.capacity[k] * 100 
                           if self.capacity[k] > 0 else 0 for k in [1, 2, 3]}
        }


class ParkingSystemQueue:
    """
    Approach 4: Queue-based Implementation
    
    Use queues to manage parking spaces as resources.
    
    Time: O(1) for addCar, Space: O(total_capacity)
    """
    
    def __init__(self, big: int, medium: int, small: int):
        from collections import deque
        
        # Create queues with available slot IDs
        self.big_slots = deque(range(1, big + 1))
        self.medium_slots = deque(range(1, medium + 1))
        self.small_slots = deque(range(1, small + 1))
        
        self.slot_queues = {
            1: self.big_slots,
            2: self.medium_slots,
            3: self.small_slots
        }
    
    def addCar(self, carType: int) -> bool:
        """Add car to parking lot using queue"""
        queue = self.slot_queues[carType]
        
        if queue:
            slot_id = queue.popleft()  # Assign specific slot
            return True
        
        return False
    
    def getAvailableSlots(self, carType: int) -> int:
        """Get number of available slots"""
        return len(self.slot_queues[carType])


class ParkingSystemBitset:
    """
    Approach 5: Bitset Implementation
    
    Use bitset to track occupied slots efficiently.
    
    Time: O(1) for addCar, Space: O(total_capacity)
    """
    
    def __init__(self, big: int, medium: int, small: int):
        self.capacity = [0, big, medium, small]
        self.occupied = [0, 0, 0, 0]  # Bitset for each type
        self.max_bits = [0, big, medium, small]
    
    def addCar(self, carType: int) -> bool:
        """Add car using bitset tracking"""
        # Find first available slot
        for slot in range(self.capacity[carType]):
            if not (self.occupied[carType] & (1 << slot)):
                # Mark slot as occupied
                self.occupied[carType] |= (1 << slot)
                return True
        
        return False
    
    def removeCar(self, carType: int, slot: int) -> bool:
        """Remove car from specific slot"""
        if slot < self.capacity[carType] and (self.occupied[carType] & (1 << slot)):
            self.occupied[carType] &= ~(1 << slot)
            return True
        return False
    
    def getOccupiedCount(self, carType: int) -> int:
        """Get number of occupied slots"""
        return bin(self.occupied[carType]).count('1')


def test_parking_system_implementations():
    """Test parking system implementations"""
    
    implementations = [
        ("Array", ParkingSystemArray),
        ("Dictionary", ParkingSystemDict),
        ("Detailed", ParkingSystemDetailed),
        ("Queue", ParkingSystemQueue),
        ("Bitset", ParkingSystemBitset),
    ]
    
    test_cases = [
        {
            "init": (1, 1, 0),
            "operations": ["addCar", "addCar", "addCar", "addCar"],
            "values": [1, 2, 3, 1],
            "expected": [True, True, False, False],
            "description": "Example 1"
        },
        {
            "init": (2, 2, 2),
            "operations": ["addCar", "addCar", "addCar", "addCar", "addCar", "addCar", "addCar"],
            "values": [1, 1, 2, 2, 3, 3, 1],
            "expected": [True, True, True, True, True, True, False],
            "description": "Fill all spaces"
        },
        {
            "init": (0, 0, 1),
            "operations": ["addCar", "addCar", "addCar"],
            "values": [1, 2, 3],
            "expected": [False, False, True],
            "description": "Only small spaces"
        },
    ]
    
    print("=== Testing Parking System Implementations ===")
    
    for impl_name, impl_class in implementations:
        print(f"\n--- {impl_name} Implementation ---")
        
        for test_case in test_cases:
            try:
                big, medium, small = test_case["init"]
                parking = impl_class(big, medium, small)
                results = []
                
                for i, op in enumerate(test_case["operations"]):
                    if op == "addCar":
                        result = parking.addCar(test_case["values"][i])
                        results.append(result)
                
                expected = test_case["expected"]
                status = "✓" if results == expected else "✗"
                
                print(f"  {test_case['description']:15} | {status} | {results}")
                if results != expected:
                    print(f"    Expected: {expected}")
                
            except Exception as e:
                print(f"  {test_case['description']:15} | ERROR: {str(e)[:40]}")


def demonstrate_parking_system():
    """Demonstrate parking system step by step"""
    print("\n=== Parking System Step-by-Step Demo ===")
    
    parking = ParkingSystemDetailed(2, 1, 1)  # 2 big, 1 medium, 1 small
    
    car_arrivals = [
        (1, "Big SUV"),
        (2, "Medium sedan"),
        (3, "Small compact"),
        (1, "Another big car"),
        (1, "Third big car"),  # Should fail
        (2, "Another sedan"),  # Should fail
    ]
    
    print("Strategy: Track available spaces and detailed parking information")
    print(f"Parking lot: 2 big, 1 medium, 1 small spaces")
    
    def print_status():
        """Print current parking status"""
        status = parking.getStatus()
        print(f"  Status: Big={status['available'][1]}/{status['capacity'][1]}, "
              f"Medium={status['available'][2]}/{status['capacity'][2]}, "
              f"Small={status['available'][3]}/{status['capacity'][3]}")
        print(f"  Utilization: Big={status['utilization'][1]:.1f}%, "
              f"Medium={status['utilization'][2]:.1f}%, "
              f"Small={status['utilization'][3]:.1f}%")
    
    print(f"\nInitial state:")
    print_status()
    
    for i, (car_type, description) in enumerate(car_arrivals):
        print(f"\nStep {i+1}: {description} (type {car_type}) arrives")
        
        success = parking.addCar(car_type)
        
        if success:
            print(f"  ✓ Car parked successfully")
        else:
            print(f"  ✗ No space available")
        
        print_status()


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Shopping mall parking
    print("1. Shopping Mall Parking Management:")
    mall_parking = ParkingSystemDetailed(50, 100, 200)  # Big, Medium, Small
    
    # Simulate peak hours
    peak_arrivals = [
        (1, 15),  # 15 big cars (SUVs, trucks)
        (2, 45),  # 45 medium cars (sedans)
        (3, 80),  # 80 small cars (compacts)
    ]
    
    print("  Peak hour simulation:")
    for car_type, count in peak_arrivals:
        type_names = {1: "Big", 2: "Medium", 3: "Small"}
        successful = 0
        
        for _ in range(count):
            if mall_parking.addCar(car_type):
                successful += 1
        
        print(f"    {type_names[car_type]} cars: {successful}/{count} parked")
    
    status = mall_parking.getStatus()
    print(f"  Final utilization:")
    for car_type in [1, 2, 3]:
        type_names = {1: "Big", 2: "Medium", 3: "Small"}
        util = status['utilization'][car_type]
        print(f"    {type_names[car_type]}: {util:.1f}% full")
    
    # Application 2: Airport parking
    print(f"\n2. Airport Long-term Parking:")
    airport_parking = ParkingSystemQueue(20, 50, 30)  # Premium, Standard, Economy
    
    # Simulate different parking preferences
    bookings = [
        (1, "Premium parking", 5),
        (2, "Standard parking", 25),
        (3, "Economy parking", 30),
        (2, "More standard", 30),  # Some will fail
    ]
    
    print("  Parking reservations:")
    for car_type, description, count in bookings:
        successful = 0
        
        for _ in range(count):
            if airport_parking.addCar(car_type):
                successful += 1
        
        print(f"    {description}: {successful}/{count} spaces allocated")
        
        # Show remaining availability
        remaining = airport_parking.getAvailableSlots(car_type)
        print(f"      Remaining slots: {remaining}")
    
    # Application 3: Residential parking
    print(f"\n3. Residential Complex Parking:")
    residential = ParkingSystemArray(10, 20, 15)  # Visitor, Resident, Compact
    
    # Simulate daily parking patterns
    daily_pattern = [
        ("Morning rush", [(2, 15), (3, 10)]),  # Residents leaving
        ("Afternoon visitors", [(1, 8), (2, 5)]),  # Visitors arriving
        ("Evening return", [(2, 20), (3, 15)]),  # Residents returning
    ]
    
    print("  Daily parking pattern simulation:")
    for time_period, arrivals in daily_pattern:
        print(f"    {time_period}:")
        
        for car_type, count in arrivals:
            type_names = {1: "Visitor", 2: "Resident", 3: "Compact"}
            successful = 0
            
            for _ in range(count):
                if residential.addCar(car_type):
                    successful += 1
            
            print(f"      {type_names[car_type]}: {successful}/{count} parked")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Array", "O(1)", "O(1)", "Simple array access"),
        ("Dictionary", "O(1)", "O(1)", "Hash table lookup"),
        ("Detailed", "O(1)", "O(capacity)", "Additional tracking overhead"),
        ("Queue", "O(1)", "O(capacity)", "Queue operations"),
        ("Bitset", "O(capacity)", "O(capacity)", "Bit manipulation for slot finding"),
    ]
    
    print(f"{'Approach':<15} | {'AddCar':<8} | {'Space':<12} | {'Notes'}")
    print("-" * 55)
    
    for approach, addcar, space, notes in approaches:
        print(f"{approach:<15} | {addcar:<8} | {space:<12} | {notes}")
    
    print(f"\nArray and Dictionary approaches are optimal for basic functionality")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    edge_cases = [
        ("Zero capacity", ParkingSystemArray(0, 0, 0), [(1, False), (2, False), (3, False)]),
        ("Single space each", ParkingSystemArray(1, 1, 1), [(1, True), (1, False), (2, True), (3, True)]),
        ("Only big spaces", ParkingSystemArray(2, 0, 0), [(1, True), (2, False), (3, False), (1, True)]),
        ("Large capacity", ParkingSystemArray(1000, 1000, 1000), [(1, True), (2, True), (3, True)]),
    ]
    
    for description, parking_system, test_ops in edge_cases:
        print(f"\n{description}:")
        
        for car_type, expected in test_ops:
            try:
                result = parking_system.addCar(car_type)
                status = "✓" if result == expected else "✗"
                print(f"  addCar({car_type}): {status} -> {result}")
            except Exception as e:
                print(f"  addCar({car_type}): ERROR -> {str(e)[:30]}")


def demonstrate_advanced_features():
    """Demonstrate advanced parking system features"""
    print("\n=== Advanced Parking System Features ===")
    
    # Advanced parking system with removal capability
    advanced_parking = ParkingSystemDetailed(2, 2, 2)
    
    print("Advanced features demonstration:")
    
    # Park some cars
    cars = [(1, "SUV-1"), (2, "Sedan-1"), (3, "Compact-1"), (1, "SUV-2")]
    
    for car_type, name in cars:
        success = advanced_parking.addCar(car_type)
        print(f"  Park {name} (type {car_type}): {'Success' if success else 'Failed'}")
    
    # Show status
    status = advanced_parking.getStatus()
    print(f"\n  Current status:")
    for car_type in [1, 2, 3]:
        occupied = status['occupied'][car_type]
        capacity = status['capacity'][car_type]
        print(f"    Type {car_type}: {occupied}/{capacity} occupied")
    
    # Remove a car
    print(f"\n  Remove a big car:")
    removed = advanced_parking.removeCar(1)
    print(f"    Removal: {'Success' if removed else 'Failed'}")
    
    # Try to park another car
    print(f"\n  Try to park another big car:")
    success = advanced_parking.addCar(1)
    print(f"    Parking: {'Success' if success else 'Failed'}")
    
    # Final status
    final_status = advanced_parking.getStatus()
    print(f"\n  Final utilization:")
    for car_type in [1, 2, 3]:
        util = final_status['utilization'][car_type]
        print(f"    Type {car_type}: {util:.1f}%")


if __name__ == "__main__":
    test_parking_system_implementations()
    demonstrate_parking_system()
    demonstrate_real_world_applications()
    analyze_time_complexity()
    test_edge_cases()
    demonstrate_advanced_features()

"""
Design Parking System demonstrates system design for resource management
with multiple implementation approaches for efficient space allocation
and tracking in parking lot management systems.
"""
