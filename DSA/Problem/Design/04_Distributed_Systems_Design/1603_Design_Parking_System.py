"""
1603. Design Parking System - Multiple Approaches
Difficulty: Easy

Design a parking system for a parking lot. The parking lot has three kinds of parking spaces: big, medium, and small, with a fixed number of slots for each size.

Implement the ParkingSystem class:
- ParkingSystem(int big, int medium, int small) Initializes object of the ParkingSystem class. The number of slots for each parking space are given as part of the constructor.
- bool addCar(int carType) Checks whether there is a parking space of carType for the car that wants to get into the parking lot. carType can be of three kinds: big, medium, or small, which are represented by 1, 2, and 3 respectively. A car can only park in a parking space of its carType. If there is no space available, return false, else park the car in that size space and return true.
"""

from typing import Dict, List, Optional
from collections import defaultdict
from enum import Enum
import threading

class CarType(Enum):
    BIG = 1
    MEDIUM = 2
    SMALL = 3

class ParkingSystemSimple:
    """
    Approach 1: Simple Array-based Implementation
    
    Use simple counters for each parking space type.
    
    Time Complexity:
    - __init__: O(1)
    - addCar: O(1)
    
    Space Complexity: O(1)
    """
    
    def __init__(self, big: int, medium: int, small: int):
        self.spaces = [0, big, medium, small]  # Index 0 unused, 1=big, 2=medium, 3=small
    
    def addCar(self, carType: int) -> bool:
        if carType < 1 or carType > 3 or self.spaces[carType] <= 0:
            return False
        
        self.spaces[carType] -= 1
        return True

class ParkingSystemDetailed:
    """
    Approach 2: Detailed Implementation with Car Tracking
    
    Track individual cars and their parking slots.
    
    Time Complexity:
    - __init__: O(1)
    - addCar: O(1)
    
    Space Complexity: O(total_parked_cars)
    """
    
    def __init__(self, big: int, medium: int, small: int):
        self.capacity = {1: big, 2: medium, 3: small}
        self.occupied = {1: 0, 2: medium, 3: 0}
        self.parked_cars = []  # List of (car_id, car_type, timestamp)
        self.car_counter = 0
        
    def addCar(self, carType: int) -> bool:
        if carType not in self.capacity:
            return False
        
        if self.occupied[carType] >= self.capacity[carType]:
            return False
        
        # Park the car
        self.car_counter += 1
        import time
        timestamp = time.time()
        
        self.parked_cars.append((self.car_counter, carType, timestamp))
        self.occupied[carType] += 1
        
        return True
    
    def removeCar(self, carId: int) -> bool:
        """Remove a parked car by ID"""
        for i, (car_id, car_type, _) in enumerate(self.parked_cars):
            if car_id == carId:
                self.parked_cars.pop(i)
                self.occupied[car_type] -= 1
                return True
        return False
    
    def getStatus(self) -> dict:
        """Get current parking status"""
        return {
            'capacity': self.capacity.copy(),
            'occupied': self.occupied.copy(),
            'available': {k: self.capacity[k] - self.occupied[k] for k in self.capacity},
            'total_parked': len(self.parked_cars)
        }

class ParkingSystemAdvanced:
    """
    Approach 3: Advanced with Features and Analytics
    
    Enhanced parking system with statistics and additional features.
    
    Time Complexity:
    - __init__: O(1)
    - addCar: O(1)
    
    Space Complexity: O(total_operations)
    """
    
    def __init__(self, big: int, medium: int, small: int):
        self.capacity = {1: big, 2: medium, 3: small}
        self.occupied = {1: 0, 2: 0, 3: 0}
        
        # Car tracking
        self.parked_cars = {}  # car_id -> (car_type, timestamp, slot_id)
        self.car_counter = 0
        
        # Slot management
        self.available_slots = {
            1: set(range(big)),
            2: set(range(medium)),
            3: set(range(small))
        }
        
        # Statistics
        self.total_requests = 0
        self.successful_parkings = 0
        self.rejected_requests = 0
        self.car_type_stats = {1: 0, 2: 0, 3: 0}
        self.peak_occupancy = {1: 0, 2: 0, 3: 0}
        
        # History
        self.parking_history = []  # (timestamp, action, car_type, success)
    
    def addCar(self, carType: int) -> bool:
        import time
        timestamp = time.time()
        
        self.total_requests += 1
        
        if carType not in self.capacity:
            self._record_action(timestamp, "park", carType, False)
            self.rejected_requests += 1
            return False
        
        if not self.available_slots[carType]:
            self._record_action(timestamp, "park", carType, False)
            self.rejected_requests += 1
            return False
        
        # Assign slot
        slot_id = self.available_slots[carType].pop()
        self.car_counter += 1
        
        # Park the car
        self.parked_cars[self.car_counter] = (carType, timestamp, slot_id)
        self.occupied[carType] += 1
        self.successful_parkings += 1
        self.car_type_stats[carType] += 1
        
        # Update peak occupancy
        self.peak_occupancy[carType] = max(self.peak_occupancy[carType], self.occupied[carType])
        
        self._record_action(timestamp, "park", carType, True)
        
        return True
    
    def removeCar(self, carId: int) -> bool:
        """Remove a parked car"""
        import time
        timestamp = time.time()
        
        if carId not in self.parked_cars:
            return False
        
        car_type, park_timestamp, slot_id = self.parked_cars[carId]
        
        # Remove car
        del self.parked_cars[carId]
        self.occupied[car_type] -= 1
        self.available_slots[car_type].add(slot_id)
        
        self._record_action(timestamp, "remove", car_type, True)
        
        return True
    
    def _record_action(self, timestamp: float, action: str, car_type: int, success: bool) -> None:
        """Record parking action in history"""
        self.parking_history.append((timestamp, action, car_type, success))
        
        # Keep only recent history to manage memory
        if len(self.parking_history) > 1000:
            self.parking_history = self.parking_history[-500:]
    
    def getStatistics(self) -> dict:
        """Get comprehensive parking statistics"""
        utilization = {}
        for car_type in self.capacity:
            if self.capacity[car_type] > 0:
                utilization[car_type] = self.occupied[car_type] / self.capacity[car_type]
            else:
                utilization[car_type] = 0.0
        
        success_rate = self.successful_parkings / max(1, self.total_requests)
        
        return {
            'total_requests': self.total_requests,
            'successful_parkings': self.successful_parkings,
            'rejected_requests': self.rejected_requests,
            'success_rate': success_rate,
            'current_occupancy': self.occupied.copy(),
            'capacity': self.capacity.copy(),
            'utilization': utilization,
            'peak_occupancy': self.peak_occupancy.copy(),
            'car_type_distribution': self.car_type_stats.copy()
        }
    
    def getAvailableSlots(self, carType: int) -> List[int]:
        """Get list of available slot IDs for a car type"""
        if carType in self.available_slots:
            return sorted(list(self.available_slots[carType]))
        return []
    
    def findCarLocation(self, carId: int) -> Optional[dict]:
        """Find the location of a parked car"""
        if carId in self.parked_cars:
            car_type, timestamp, slot_id = self.parked_cars[carId]
            return {
                'car_id': carId,
                'car_type': car_type,
                'slot_id': slot_id,
                'parked_time': timestamp,
                'duration': time.time() - timestamp
            }
        return None
    
    def getParkingHistory(self, limit: int = 10) -> List[dict]:
        """Get recent parking history"""
        recent_history = self.parking_history[-limit:]
        
        history_formatted = []
        for timestamp, action, car_type, success in recent_history:
            history_formatted.append({
                'timestamp': timestamp,
                'action': action,
                'car_type': car_type,
                'success': success
            })
        
        return history_formatted

class ParkingSystemConcurrent:
    """
    Approach 4: Thread-Safe Implementation
    
    Parking system designed for concurrent access.
    
    Time Complexity:
    - __init__: O(1)
    - addCar: O(1) + lock overhead
    
    Space Complexity: O(total_parked_cars)
    """
    
    def __init__(self, big: int, medium: int, small: int):
        self.capacity = {1: big, 2: medium, 3: small}
        self.occupied = {1: 0, 2: 0, 3: 0}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Parked cars tracking
        self.parked_cars = {}
        self.car_counter = 0
        
        # Statistics
        self.total_requests = 0
        self.successful_parkings = 0
    
    def addCar(self, carType: int) -> bool:
        with self.lock:
            self.total_requests += 1
            
            if carType not in self.capacity:
                return False
            
            if self.occupied[carType] >= self.capacity[carType]:
                return False
            
            # Park the car
            self.car_counter += 1
            import time
            self.parked_cars[self.car_counter] = (carType, time.time())
            self.occupied[carType] += 1
            self.successful_parkings += 1
            
            return True
    
    def removeCar(self, carId: int) -> bool:
        """Thread-safe car removal"""
        with self.lock:
            if carId not in self.parked_cars:
                return False
            
            car_type, _ = self.parked_cars[carId]
            del self.parked_cars[carId]
            self.occupied[car_type] -= 1
            
            return True
    
    def getStatus(self) -> dict:
        """Thread-safe status retrieval"""
        with self.lock:
            return {
                'capacity': self.capacity.copy(),
                'occupied': self.occupied.copy(),
                'total_requests': self.total_requests,
                'successful_parkings': self.successful_parkings,
                'current_cars': len(self.parked_cars)
            }

class ParkingSystemDistributed:
    """
    Approach 5: Distributed System Design
    
    Parking system designed for distributed environments.
    
    Time Complexity:
    - __init__: O(1)
    - addCar: O(1) + coordination overhead
    
    Space Complexity: O(total_parked_cars + distributed_state)
    """
    
    def __init__(self, big: int, medium: int, small: int, node_id: str = "node1"):
        self.node_id = node_id
        self.capacity = {1: big, 2: medium, 3: small}
        self.local_occupied = {1: 0, 2: 0, 3: 0}
        
        # Distributed state tracking
        self.global_state = {
            'total_occupied': {1: 0, 2: 0, 3: 0},
            'node_states': {node_id: self.local_occupied.copy()}
        }
        
        # Local car tracking
        self.local_cars = {}
        self.car_counter = 0
        
        # Consistency and coordination
        self.state_version = 0
        self.pending_operations = []
        
        # Statistics
        self.coordination_requests = 0
        self.state_synchronizations = 0
    
    def addCar(self, carType: int) -> bool:
        """Add car with distributed coordination"""
        self.coordination_requests += 1
        
        if carType not in self.capacity:
            return False
        
        # Check global availability (simulated)
        global_occupied = self.global_state['total_occupied'][carType]
        
        if global_occupied >= self.capacity[carType]:
            return False
        
        # Simulate distributed coordination
        if self._coordinate_parking_request(carType):
            # Park locally
            self.car_counter += 1
            import time
            self.local_cars[self.car_counter] = (carType, time.time())
            self.local_occupied[carType] += 1
            
            # Update global state
            self._update_global_state(carType, 1)
            
            return True
        
        return False
    
    def _coordinate_parking_request(self, carType: int) -> bool:
        """Simulate coordination with other nodes"""
        # In a real system, this would involve:
        # 1. Distributed consensus (Raft, PBFT, etc.)
        # 2. Atomic reservation of parking space
        # 3. Conflict resolution for concurrent requests
        
        # Simulate coordination delay
        import time
        coordination_delay = 0.001  # 1ms simulated network delay
        time.sleep(coordination_delay)
        
        # Simplified coordination - check if space is still available
        return self.global_state['total_occupied'][carType] < self.capacity[carType]
    
    def _update_global_state(self, carType: int, change: int) -> None:
        """Update global state after local change"""
        self.global_state['total_occupied'][carType] += change
        self.global_state['node_states'][self.node_id][carType] += change
        self.state_version += 1
    
    def synchronize_with_peer(self, peer_state: dict) -> None:
        """Synchronize state with peer node"""
        self.state_synchronizations += 1
        
        # Merge peer state into global state
        for node_id, node_state in peer_state.get('node_states', {}).items():
            if node_id != self.node_id:
                self.global_state['node_states'][node_id] = node_state
        
        # Recalculate total occupied
        total_occupied = {1: 0, 2: 0, 3: 0}
        for node_state in self.global_state['node_states'].values():
            for car_type in total_occupied:
                total_occupied[car_type] += node_state.get(car_type, 0)
        
        self.global_state['total_occupied'] = total_occupied
        self.state_version += 1
    
    def get_distributed_status(self) -> dict:
        """Get status including distributed information"""
        return {
            'node_id': self.node_id,
            'local_state': {
                'capacity': self.capacity.copy(),
                'occupied': self.local_occupied.copy(),
                'cars': len(self.local_cars)
            },
            'global_state': self.global_state.copy(),
            'coordination_stats': {
                'coordination_requests': self.coordination_requests,
                'state_synchronizations': self.state_synchronizations,
                'state_version': self.state_version
            }
        }


def test_parking_system_basic():
    """Test basic parking system functionality"""
    print("=== Testing Basic Parking System Functionality ===")
    
    implementations = [
        ("Simple", ParkingSystemSimple),
        ("Detailed", ParkingSystemDetailed),
        ("Advanced", ParkingSystemAdvanced),
        ("Concurrent", ParkingSystemConcurrent),
        ("Distributed", lambda b, m, s: ParkingSystemDistributed(b, m, s, "test_node"))
    ]
    
    for name, ParkingSystemClass in implementations:
        print(f"\n{name}:")
        
        parking = ParkingSystemClass(1, 1, 0)  # 1 big, 1 medium, 0 small
        
        # Test sequence from problem
        test_cases = [
            (1, True),   # Park big car
            (2, True),   # Park medium car
            (3, False),  # Try to park small car (no space)
            (1, False)   # Try to park another big car (no space)
        ]
        
        for carType, expected in test_cases:
            result = parking.addCar(carType)
            status = "✓" if result == expected else "✗"
            print(f"  addCar({carType}): {result} {status}")

def test_parking_system_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Parking System Edge Cases ===")
    
    # Test zero capacity
    print("Zero capacity:")
    parking = ParkingSystemSimple(0, 0, 0)
    
    for carType in [1, 2, 3]:
        result = parking.addCar(carType)
        print(f"  addCar({carType}) with zero capacity: {result}")
    
    # Test invalid car types
    print(f"\nInvalid car types:")
    parking = ParkingSystemAdvanced(1, 1, 1)
    
    invalid_types = [0, 4, -1, 10]
    for carType in invalid_types:
        result = parking.addCar(carType)
        print(f"  addCar({carType}) invalid type: {result}")
    
    # Test capacity limits
    print(f"\nCapacity limits:")
    parking = ParkingSystemDetailed(2, 1, 3)
    
    # Fill each type to capacity
    car_types = [1, 1, 2, 3, 3, 3]  # 2 big, 1 medium, 3 small
    
    for i, carType in enumerate(car_types):
        result = parking.addCar(carType)
        print(f"  Car {i+1} (type {carType}): {result}")
    
    # Try to exceed capacity
    for carType in [1, 2, 3]:
        result = parking.addCar(carType)
        print(f"  Excess car (type {carType}): {result}")
    
    status = parking.getStatus()
    print(f"  Final status: {status}")

def test_advanced_features():
    """Test advanced parking system features"""
    print("\n=== Testing Advanced Features ===")
    
    parking = ParkingSystemAdvanced(3, 2, 4)
    
    # Park several cars
    cars_to_park = [1, 2, 1, 3, 3, 2, 1, 3]
    parked_car_ids = []
    
    print("Parking cars:")
    for i, carType in enumerate(cars_to_park):
        success = parking.addCar(carType)
        print(f"  Car {i+1} (type {carType}): {success}")
        if success:
            parked_car_ids.append(parking.car_counter)
    
    # Get statistics
    stats = parking.getStatistics()
    print(f"\nStatistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}: {value}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Test car location finding
    print(f"\nCar locations:")
    for car_id in parked_car_ids[:3]:  # Check first 3 cars
        location = parking.findCarLocation(car_id)
        if location:
            print(f"  Car {car_id}: Type {location['car_type']}, Slot {location['slot_id']}")
    
    # Test available slots
    print(f"\nAvailable slots:")
    for carType in [1, 2, 3]:
        available = parking.getAvailableSlots(carType)
        print(f"  Type {carType}: {available}")
    
    # Remove some cars
    print(f"\nRemoving cars:")
    for car_id in parked_car_ids[:2]:
        success = parking.removeCar(car_id)
        print(f"  Remove car {car_id}: {success}")
    
    # Check updated statistics
    updated_stats = parking.getStatistics()
    print(f"\nUpdated occupancy: {updated_stats['current_occupancy']}")

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: Shopping mall parking
    print("Application 1: Shopping Mall Parking System")
    
    mall_parking = ParkingSystemAdvanced(50, 100, 200)  # Large parking structure
    
    # Simulate shopping mall traffic
    import random
    random.seed(42)  # For reproducible results
    
    # Simulate morning rush
    print("  Morning rush (8 AM - 10 AM):")
    
    morning_cars = []
    for hour in range(8, 10):
        cars_this_hour = 30 + random.randint(-10, 20)  # Variable traffic
        
        parked_count = 0
        rejected_count = 0
        
        for _ in range(cars_this_hour):
            # Random car type distribution (more small cars)
            car_type = random.choices([1, 2, 3], weights=[1, 2, 4])[0]
            
            if mall_parking.addCar(car_type):
                parked_count += 1
                morning_cars.append(mall_parking.car_counter)
            else:
                rejected_count += 1
        
        print(f"    {hour}:00 - Parked: {parked_count}, Rejected: {rejected_count}")
    
    # Check status
    morning_stats = mall_parking.getStatistics()
    print(f"  Morning status:")
    print(f"    Total parked: {morning_stats['successful_parkings']}")
    print(f"    Occupancy: {morning_stats['current_occupancy']}")
    print(f"    Utilization: {morning_stats['utilization']}")
    
    # Application 2: Airport parking reservation
    print(f"\nApplication 2: Airport Parking Reservation System")
    
    airport_parking = ParkingSystemConcurrent(200, 500, 1000)
    
    # Simulate reservation system
    import threading
    import time
    
    def simulate_reservations(parking_system, thread_id, reservations):
        """Simulate concurrent reservation requests"""
        random.seed(thread_id)
        
        for _ in range(reservations):
            car_type = random.choice([1, 2, 3])
            success = parking_system.addCar(car_type)
            
            # Small delay to simulate processing time
            time.sleep(0.001)
    
    print("  Simulating concurrent reservations:")
    
    # Start multiple reservation threads
    threads = []
    reservations_per_thread = 100
    num_threads = 5
    
    start_time = time.time()
    
    for i in range(num_threads):
        thread = threading.Thread(
            target=simulate_reservations,
            args=(airport_parking, i, reservations_per_thread)
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all threads
    for thread in threads:
        thread.join()
    
    elapsed_time = time.time() - start_time
    total_reservations = num_threads * reservations_per_thread
    
    final_status = airport_parking.getStatus()
    print(f"    Processed {total_reservations} reservations in {elapsed_time:.2f}s")
    print(f"    Successful parkings: {final_status['successful_parkings']}")
    print(f"    Success rate: {final_status['successful_parkings'] / total_reservations:.1%}")
    
    # Application 3: Smart city parking network
    print(f"\nApplication 3: Smart City Parking Network")
    
    # Create multiple parking nodes
    nodes = [
        ParkingSystemDistributed(20, 30, 50, "downtown"),
        ParkingSystemDistributed(40, 60, 80, "mall"),
        ParkingSystemDistributed(10, 15, 25, "airport")
    ]
    
    # Simulate distributed parking requests
    print("  Parking across distributed nodes:")
    
    requests = [
        ("downtown", 1), ("mall", 2), ("airport", 3),
        ("downtown", 2), ("mall", 1), ("airport", 2),
        ("downtown", 3), ("mall", 3), ("airport", 1)
    ]
    
    for node_name, car_type in requests:
        # Find the node
        node = next(n for n in nodes if n.node_id == node_name)
        
        success = node.addCar(car_type)
        print(f"    {node_name} - Car type {car_type}: {success}")
        
        # Simulate state synchronization
        if success:
            # Share state with other nodes
            for other_node in nodes:
                if other_node != node:
                    other_node.synchronize_with_peer(node.get_distributed_status())
    
    # Show distributed status
    print(f"\n  Network status:")
    for node in nodes:
        status = node.get_distributed_status()
        local_state = status['local_state']
        print(f"    {node.node_id}: Occupied {local_state['occupied']}, Cars {local_state['cars']}")

def test_concurrent_access():
    """Test concurrent access patterns"""
    print("\n=== Testing Concurrent Access ===")
    
    import threading
    import time
    import random
    
    parking = ParkingSystemConcurrent(100, 100, 100)
    
    # Test high concurrency
    num_threads = 10
    operations_per_thread = 200
    
    def parking_operations(thread_id: int, results: list):
        """Perform parking operations in a thread"""
        random.seed(thread_id)
        
        start_time = time.time()
        successful_parks = 0
        failed_parks = 0
        
        for _ in range(operations_per_thread):
            car_type = random.choice([1, 2, 3])
            
            if parking.addCar(car_type):
                successful_parks += 1
            else:
                failed_parks += 1
            
            # Small random delay
            time.sleep(random.uniform(0.0001, 0.001))
        
        elapsed = time.time() - start_time
        results.append({
            'thread_id': thread_id,
            'successful_parks': successful_parks,
            'failed_parks': failed_parks,
            'elapsed_time': elapsed
        })
    
    print(f"Testing {num_threads} concurrent threads...")
    
    # Start threads
    threads = []
    results = []
    
    overall_start = time.time()
    
    for i in range(num_threads):
        thread = threading.Thread(target=parking_operations, args=(i, results))
        threads.append(thread)
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    overall_time = time.time() - overall_start
    
    # Analyze results
    total_attempts = num_threads * operations_per_thread
    total_successful = sum(r['successful_parks'] for r in results)
    total_failed = sum(r['failed_parks'] for r in results)
    
    print(f"  Results:")
    print(f"    Total time: {overall_time:.2f}s")
    print(f"    Total attempts: {total_attempts}")
    print(f"    Successful: {total_successful}")
    print(f"    Failed: {total_failed}")
    print(f"    Success rate: {total_successful / total_attempts:.1%}")
    print(f"    Throughput: {total_attempts / overall_time:.0f} ops/sec")
    
    # Final parking status
    final_status = parking.getStatus()
    print(f"    Final occupancy: {final_status['occupied']}")

def stress_test_parking_system():
    """Stress test parking system"""
    print("\n=== Stress Testing Parking System ===")
    
    import time
    import random
    
    # Large parking system
    parking = ParkingSystemAdvanced(1000, 2000, 3000)
    
    # Stress test parameters
    num_operations = 50000
    
    print(f"Stress test: {num_operations} operations")
    
    start_time = time.time()
    
    operations = []
    parked_cars = []
    
    # Mix of parking and removing cars
    for i in range(num_operations):
        if i < num_operations * 0.7:  # 70% parking operations
            car_type = random.choice([1, 2, 3])
            success = parking.addCar(car_type)
            
            if success:
                parked_cars.append(parking.car_counter)
            
            operations.append(('park', car_type, success))
        
        else:  # 30% removal operations
            if parked_cars:
                car_to_remove = random.choice(parked_cars)
                success = parking.removeCar(car_to_remove)
                
                if success:
                    parked_cars.remove(car_to_remove)
                
                operations.append(('remove', car_to_remove, success))
    
    elapsed = time.time() - start_time
    
    # Get final statistics
    final_stats = parking.getStatistics()
    
    park_ops = sum(1 for op in operations if op[0] == 'park')
    remove_ops = sum(1 for op in operations if op[0] == 'remove')
    successful_parks = sum(1 for op in operations if op[0] == 'park' and op[2])
    successful_removes = sum(1 for op in operations if op[0] == 'remove' and op[2])
    
    print(f"  Completed in {elapsed:.2f}s")
    print(f"  Park operations: {park_ops} (successful: {successful_parks})")
    print(f"  Remove operations: {remove_ops} (successful: {successful_removes})")
    print(f"  Throughput: {len(operations) / elapsed:.0f} ops/sec")
    print(f"  Final occupancy: {final_stats['current_occupancy']}")
    print(f"  Peak occupancy: {final_stats['peak_occupancy']}")

def benchmark_implementations():
    """Benchmark different implementations"""
    print("\n=== Benchmarking Implementations ===")
    
    import time
    
    implementations = [
        ("Simple", ParkingSystemSimple),
        ("Detailed", ParkingSystemDetailed),
        ("Advanced", ParkingSystemAdvanced),
        ("Concurrent", ParkingSystemConcurrent)
    ]
    
    # Test parameters
    capacity = (500, 500, 500)
    num_operations = 10000
    
    for name, ParkingSystemClass in implementations:
        parking = ParkingSystemClass(*capacity)
        
        # Benchmark parking operations
        start_time = time.time()
        
        import random
        random.seed(42)  # Consistent test data
        
        for _ in range(num_operations):
            car_type = random.choice([1, 2, 3])
            parking.addCar(car_type)
        
        elapsed = (time.time() - start_time) * 1000
        
        print(f"  {name}: {elapsed:.2f}ms for {num_operations} operations")
        print(f"    Average: {elapsed/num_operations:.4f}ms per operation")

def test_memory_efficiency():
    """Test memory efficiency"""
    print("\n=== Testing Memory Efficiency ===")
    
    implementations = [
        ("Simple", ParkingSystemSimple),
        ("Detailed", ParkingSystemDetailed),
        ("Advanced", ParkingSystemAdvanced)
    ]
    
    capacity = (1000, 1000, 1000)
    cars_to_park = 1500  # Park many cars
    
    for name, ParkingSystemClass in implementations:
        parking = ParkingSystemClass(*capacity)
        
        # Park many cars
        parked_count = 0
        for i in range(cars_to_park):
            car_type = (i % 3) + 1
            if parking.addCar(car_type):
                parked_count += 1
        
        # Estimate memory usage
        if hasattr(parking, 'parked_cars'):
            if isinstance(parking.parked_cars, dict):
                memory_estimate = len(parking.parked_cars) * 3  # Rough estimate
            else:
                memory_estimate = len(parking.parked_cars) * 3
        else:
            memory_estimate = 3  # Just the basic counters
        
        print(f"  {name}:")
        print(f"    Parked cars: {parked_count}")
        print(f"    Estimated memory: ~{memory_estimate} units")
        
        if hasattr(parking, 'getStatistics'):
            stats = parking.getStatistics()
            print(f"    Success rate: {stats.get('success_rate', 0):.1%}")

if __name__ == "__main__":
    test_parking_system_basic()
    test_parking_system_edge_cases()
    test_advanced_features()
    demonstrate_applications()
    test_concurrent_access()
    stress_test_parking_system()
    benchmark_implementations()
    test_memory_efficiency()

"""
Parking System Design demonstrates key concepts:

Core Approaches:
1. Simple - Basic counter-based implementation
2. Detailed - Car tracking with additional metadata
3. Advanced - Full-featured with statistics and analytics
4. Concurrent - Thread-safe for multi-user environments
5. Distributed - Designed for distributed parking networks

Key Design Principles:
- Resource allocation and capacity management
- Real-time availability tracking
- Concurrent access handling for multi-user systems
- Extensibility for additional features and analytics

Performance Characteristics:
- Simple: O(1) operations, minimal memory
- Detailed: O(1) operations with car tracking overhead
- Advanced: O(1) operations with statistics collection
- Concurrent: O(1) operations plus synchronization overhead

Real-world Applications:
- Shopping mall parking management
- Airport parking reservation systems
- Smart city parking networks
- Hospital and university parking systems
- Event venue parking coordination
- Residential complex parking allocation

The simple approach is sufficient for basic requirements,
while advanced implementations provide comprehensive
features needed for enterprise parking management systems.
"""
