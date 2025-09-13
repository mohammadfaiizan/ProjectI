"""
ELEVATOR SYSTEM DESIGN - Complete System Design
===============================================

Problem Statement:
Design a comprehensive elevator control system that handles:
- Multiple elevators in a building
- Efficient request scheduling and optimization
- Different elevator types (passenger, freight, express)
- Emergency handling and safety protocols
- Load balancing and energy optimization
- Maintenance mode and diagnostics
- Access control and security features
- Real-time monitoring and analytics
- Peak hour traffic management
- Destination dispatch systems

Requirements:
- Support buildings with multiple floors and elevators
- Implement efficient scheduling algorithms (SCAN, LOOK, etc.)
- Handle simultaneous requests from multiple floors
- Manage elevator capacity and weight limits
- Implement emergency protocols (fire, power outage)
- Support VIP access and restricted floors
- Provide real-time status monitoring
- Optimize for minimal wait time and energy consumption
- Handle maintenance scheduling
- Support different building configurations

Design Patterns Used:
- State: Elevator operational states
- Strategy: Scheduling algorithms
- Observer: Status monitoring
- Command: Elevator operations
- Factory: Elevator creation
- Singleton: Building controller
- Chain of Responsibility: Request handling
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
from enum import Enum
import uuid
import threading
import time
from dataclasses import dataclass, field
import heapq
from collections import deque
import math


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class ElevatorState(Enum):
    IDLE = "idle"
    MOVING_UP = "moving_up"
    MOVING_DOWN = "moving_down"
    LOADING = "loading"
    UNLOADING = "unloading"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"
    OUT_OF_SERVICE = "out_of_service"


class ElevatorType(Enum):
    PASSENGER = "passenger"
    FREIGHT = "freight"
    EXPRESS = "express"
    SERVICE = "service"


class RequestType(Enum):
    HALL_CALL = "hall_call"
    CAR_CALL = "car_call"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"


class Direction(Enum):
    UP = "up"
    DOWN = "down"
    NONE = "none"


class Priority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    EMERGENCY = 4


@dataclass
class ElevatorRequest:
    """Elevator request data."""
    request_id: str
    floor: int
    direction: Direction
    request_type: RequestType
    priority: Priority
    timestamp: datetime
    passenger_count: int = 1
    weight: float = 0.0
    access_level: int = 0  # Security access level
    
    def __post_init__(self):
        if not self.request_id:
            self.request_id = str(uuid.uuid4())
    
    def __lt__(self, other):
        """For priority queue ordering."""
        return (self.priority.value, self.timestamp) < (other.priority.value, other.timestamp)


@dataclass
class ElevatorStatus:
    """Current elevator status."""
    elevator_id: str
    current_floor: int
    state: ElevatorState
    direction: Direction
    target_floors: List[int]
    passenger_count: int
    current_weight: float
    last_maintenance: datetime
    total_trips: int
    total_distance: int
    energy_consumed: float


# ============================================================================
# SCHEDULING STRATEGIES
# ============================================================================

class SchedulingStrategy(ABC):
    """Abstract scheduling strategy."""
    
    @abstractmethod
    def schedule_request(self, request: ElevatorRequest, elevators: List['Elevator']) -> Optional['Elevator']:
        """Schedule a request to the best elevator."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        pass


class FCFSSchedulingStrategy(SchedulingStrategy):
    """First Come First Serve scheduling."""
    
    def schedule_request(self, request: ElevatorRequest, elevators: List['Elevator']) -> Optional['Elevator']:
        """Schedule request to first available elevator."""
        available_elevators = [e for e in elevators if e.can_handle_request(request)]
        
        if not available_elevators:
            return None
        
        # Return first available elevator
        return available_elevators[0]
    
    def get_strategy_name(self) -> str:
        return "First Come First Serve"


class SCANSchedulingStrategy(SchedulingStrategy):
    """SCAN (Elevator) scheduling algorithm."""
    
    def schedule_request(self, request: ElevatorRequest, elevators: List['Elevator']) -> Optional['Elevator']:
        """Schedule request using SCAN algorithm."""
        available_elevators = [e for e in elevators if e.can_handle_request(request)]
        
        if not available_elevators:
            return None
        
        best_elevator = None
        min_cost = float('inf')
        
        for elevator in available_elevators:
            cost = self._calculate_cost(request, elevator)
            if cost < min_cost:
                min_cost = cost
                best_elevator = elevator
        
        return best_elevator
    
    def _calculate_cost(self, request: ElevatorRequest, elevator: 'Elevator') -> float:
        """Calculate cost of assigning request to elevator."""
        # Distance cost
        distance_cost = abs(elevator.current_floor - request.floor)
        
        # Direction cost (prefer elevators moving in same direction)
        direction_cost = 0
        if elevator.direction != Direction.NONE and elevator.direction != request.direction:
            direction_cost = 10
        
        # Load cost (prefer less loaded elevators)
        load_factor = elevator.current_weight / elevator.max_weight
        load_cost = load_factor * 5
        
        # Queue cost (prefer elevators with fewer pending requests)
        queue_cost = len(elevator.target_floors) * 2
        
        return distance_cost + direction_cost + load_cost + queue_cost
    
    def get_strategy_name(self) -> str:
        return "SCAN Algorithm"


class LOOKSchedulingStrategy(SchedulingStrategy):
    """LOOK scheduling algorithm (optimized SCAN)."""
    
    def schedule_request(self, request: ElevatorRequest, elevators: List['Elevator']) -> Optional['Elevator']:
        """Schedule request using LOOK algorithm."""
        available_elevators = [e for e in elevators if e.can_handle_request(request)]
        
        if not available_elevators:
            return None
        
        # Find elevator that will reach the request floor soonest
        best_elevator = None
        min_time = float('inf')
        
        for elevator in available_elevators:
            estimated_time = self._estimate_arrival_time(request, elevator)
            if estimated_time < min_time:
                min_time = estimated_time
                best_elevator = elevator
        
        return best_elevator
    
    def _estimate_arrival_time(self, request: ElevatorRequest, elevator: 'Elevator') -> float:
        """Estimate time for elevator to reach request floor."""
        if elevator.state == ElevatorState.IDLE:
            return abs(elevator.current_floor - request.floor) * elevator.floor_travel_time
        
        # Calculate time based on current queue and direction
        time_estimate = 0
        current_floor = elevator.current_floor
        
        # Add time for current targets in same direction
        if elevator.direction == Direction.UP:
            floors_to_visit = [f for f in elevator.target_floors if f > current_floor]
            if request.floor > current_floor:
                floors_to_visit.append(request.floor)
        elif elevator.direction == Direction.DOWN:
            floors_to_visit = [f for f in elevator.target_floors if f < current_floor]
            if request.floor < current_floor:
                floors_to_visit.append(request.floor)
        else:
            floors_to_visit = [request.floor]
        
        floors_to_visit.sort()
        
        # Calculate travel time
        for floor in floors_to_visit:
            time_estimate += abs(current_floor - floor) * elevator.floor_travel_time
            time_estimate += elevator.door_operation_time  # Stop time
            current_floor = floor
            
            if floor == request.floor:
                break
        
        return time_estimate
    
    def get_strategy_name(self) -> str:
        return "LOOK Algorithm"


class DestinationDispatchStrategy(SchedulingStrategy):
    """Destination dispatch scheduling for modern elevators."""
    
    def schedule_request(self, request: ElevatorRequest, elevators: List['Elevator']) -> Optional['Elevator']:
        """Schedule request using destination dispatch algorithm."""
        available_elevators = [e for e in elevators if e.can_handle_request(request)]
        
        if not available_elevators:
            return None
        
        # Group passengers by destination and optimize elevator assignment
        best_elevator = None
        min_total_time = float('inf')
        
        for elevator in available_elevators:
            total_time = self._calculate_total_journey_time(request, elevator)
            if total_time < min_total_time:
                min_total_time = total_time
                best_elevator = elevator
        
        return best_elevator
    
    def _calculate_total_journey_time(self, request: ElevatorRequest, elevator: 'Elevator') -> float:
        """Calculate total journey time including all passengers."""
        # Simplified calculation - in real system would be more complex
        base_time = abs(elevator.current_floor - request.floor) * elevator.floor_travel_time
        queue_delay = len(elevator.target_floors) * (elevator.floor_travel_time + elevator.door_operation_time)
        
        return base_time + queue_delay
    
    def get_strategy_name(self) -> str:
        return "Destination Dispatch"


# ============================================================================
# ELEVATOR CLASS
# ============================================================================

class Elevator:
    """Individual elevator with state management."""
    
    def __init__(self, elevator_id: str, elevator_type: ElevatorType, 
                 max_floor: int, min_floor: int = 1):
        self.elevator_id = elevator_id
        self.elevator_type = elevator_type
        self.max_floor = max_floor
        self.min_floor = min_floor
        
        # Current state
        self.current_floor = 1
        self.state = ElevatorState.IDLE
        self.direction = Direction.NONE
        self.target_floors: List[int] = []
        
        # Capacity and load
        self.max_passengers = self._get_max_passengers()
        self.max_weight = self._get_max_weight()
        self.current_passengers = 0
        self.current_weight = 0.0
        
        # Performance characteristics
        self.floor_travel_time = 2.0  # seconds per floor
        self.door_operation_time = 3.0  # seconds for door open/close
        self.acceleration_time = 1.0  # seconds to reach full speed
        
        # Maintenance and statistics
        self.last_maintenance = datetime.now() - timedelta(days=30)
        self.maintenance_interval = timedelta(days=90)
        self.total_trips = 0
        self.total_distance = 0
        self.energy_consumed = 0.0
        
        # Security and access
        self.access_levels: Set[int] = {0}  # Default public access
        self.restricted_floors: Set[int] = set()
        
        # Threading
        self._lock = threading.Lock()
        self._moving = False
        self._stop_requested = False
        
        # Observers
        self.observers: List['ElevatorObserver'] = []
        
        print(f"🛗 Elevator {elevator_id} ({elevator_type.value}) initialized")
    
    def _get_max_passengers(self) -> int:
        """Get maximum passengers based on elevator type."""
        capacities = {
            ElevatorType.PASSENGER: 12,
            ElevatorType.FREIGHT: 4,
            ElevatorType.EXPRESS: 16,
            ElevatorType.SERVICE: 6
        }
        return capacities.get(self.elevator_type, 12)
    
    def _get_max_weight(self) -> float:
        """Get maximum weight based on elevator type."""
        weights = {
            ElevatorType.PASSENGER: 1000.0,  # kg
            ElevatorType.FREIGHT: 3000.0,
            ElevatorType.EXPRESS: 1200.0,
            ElevatorType.SERVICE: 800.0
        }
        return weights.get(self.elevator_type, 1000.0)
    
    def add_observer(self, observer: 'ElevatorObserver') -> None:
        """Add status observer."""
        self.observers.append(observer)
    
    def remove_observer(self, observer: 'ElevatorObserver') -> None:
        """Remove status observer."""
        if observer in self.observers:
            self.observers.remove(observer)
    
    def notify_observers(self) -> None:
        """Notify all observers of status change."""
        status = self.get_status()
        for observer in self.observers:
            observer.on_elevator_status_changed(status)
    
    def can_handle_request(self, request: ElevatorRequest) -> bool:
        """Check if elevator can handle the request."""
        with self._lock:
            # Check if elevator is operational
            if self.state in [ElevatorState.MAINTENANCE, ElevatorState.OUT_OF_SERVICE]:
                return False
            
            # Check floor range
            if request.floor < self.min_floor or request.floor > self.max_floor:
                return False
            
            # Check access level
            if request.access_level not in self.access_levels:
                return False
            
            # Check restricted floors
            if request.floor in self.restricted_floors:
                return False
            
            # Check capacity (for car calls)
            if request.request_type == RequestType.CAR_CALL:
                if (self.current_passengers + request.passenger_count > self.max_passengers or
                    self.current_weight + request.weight > self.max_weight):
                    return False
            
            # Check elevator type compatibility
            if (self.elevator_type == ElevatorType.FREIGHT and 
                request.request_type == RequestType.HALL_CALL and 
                request.weight == 0):
                return False  # Freight elevators typically don't handle passenger hall calls
            
            return True
    
    def add_request(self, request: ElevatorRequest) -> bool:
        """Add a request to the elevator's queue."""
        with self._lock:
            if not self.can_handle_request(request):
                return False
            
            # Add target floor if not already in queue
            if request.floor not in self.target_floors:
                self.target_floors.append(request.floor)
                self._sort_target_floors()
            
            # Update passenger count and weight for car calls
            if request.request_type == RequestType.CAR_CALL:
                self.current_passengers += request.passenger_count
                self.current_weight += request.weight
            
            # Start moving if idle
            if self.state == ElevatorState.IDLE:
                self._start_movement()
            
            self.notify_observers()
            return True
    
    def _sort_target_floors(self) -> None:
        """Sort target floors based on current direction."""
        if not self.target_floors:
            return
        
        if self.direction == Direction.UP:
            # Sort floors above current floor in ascending order,
            # then floors below in descending order
            above = [f for f in self.target_floors if f > self.current_floor]
            below = [f for f in self.target_floors if f < self.current_floor]
            above.sort()
            below.sort(reverse=True)
            self.target_floors = above + below
        elif self.direction == Direction.DOWN:
            # Sort floors below current floor in descending order,
            # then floors above in ascending order
            below = [f for f in self.target_floors if f < self.current_floor]
            above = [f for f in self.target_floors if f > self.current_floor]
            below.sort(reverse=True)
            above.sort()
            self.target_floors = below + above
        else:
            # If no direction, sort by distance from current floor
            self.target_floors.sort(key=lambda f: abs(f - self.current_floor))
    
    def _start_movement(self) -> None:
        """Start elevator movement."""
        if not self.target_floors:
            return
        
        target_floor = self.target_floors[0]
        
        if target_floor > self.current_floor:
            self.direction = Direction.UP
            self.state = ElevatorState.MOVING_UP
        elif target_floor < self.current_floor:
            self.direction = Direction.DOWN
            self.state = ElevatorState.MOVING_DOWN
        else:
            # Already at target floor
            self._arrive_at_floor()
            return
        
        # Start movement in separate thread
        threading.Thread(target=self._move_to_floor, args=(target_floor,), daemon=True).start()
    
    def _move_to_floor(self, target_floor: int) -> None:
        """Move elevator to target floor."""
        with self._lock:
            self._moving = True
        
        while self.current_floor != target_floor and not self._stop_requested:
            # Simulate movement
            time.sleep(self.floor_travel_time)
            
            with self._lock:
                if self.direction == Direction.UP:
                    self.current_floor += 1
                elif self.direction == Direction.DOWN:
                    self.current_floor -= 1
                
                # Update statistics
                self.total_distance += 1
                self.energy_consumed += self._calculate_energy_consumption()
                
                self.notify_observers()
            
            # Check if we've reached any target floor
            if self.current_floor in self.target_floors:
                self._arrive_at_floor()
                break
        
        with self._lock:
            self._moving = False
    
    def _arrive_at_floor(self) -> None:
        """Handle arrival at a floor."""
        with self._lock:
            if self.current_floor in self.target_floors:
                self.target_floors.remove(self.current_floor)
                self.total_trips += 1
                
                # Simulate door operations
                self.state = ElevatorState.LOADING
                self.notify_observers()
                
                # Simulate loading/unloading time
                threading.Thread(target=self._handle_passenger_exchange, daemon=True).start()
    
    def _handle_passenger_exchange(self) -> None:
        """Handle passenger loading/unloading."""
        time.sleep(self.door_operation_time)
        
        with self._lock:
            # Simulate passenger exchange (simplified)
            # In real system, this would be more complex
            
            if self.target_floors:
                self._sort_target_floors()
                self._start_movement()
            else:
                # No more targets, become idle
                self.state = ElevatorState.IDLE
                self.direction = Direction.NONE
                self.current_passengers = 0
                self.current_weight = 0.0
            
            self.notify_observers()
    
    def _calculate_energy_consumption(self) -> float:
        """Calculate energy consumption per floor."""
        # Simplified energy calculation
        base_consumption = 5.0  # kWh per floor
        load_factor = self.current_weight / self.max_weight
        return base_consumption * (1 + load_factor * 0.5)
    
    def emergency_stop(self) -> None:
        """Emergency stop the elevator."""
        with self._lock:
            self.state = ElevatorState.EMERGENCY
            self._stop_requested = True
            self.target_floors.clear()
            self.notify_observers()
    
    def set_maintenance_mode(self, maintenance: bool = True) -> None:
        """Set elevator maintenance mode."""
        with self._lock:
            if maintenance:
                self.state = ElevatorState.MAINTENANCE
                self.target_floors.clear()
                self.last_maintenance = datetime.now()
            else:
                self.state = ElevatorState.IDLE
            
            self.notify_observers()
    
    def needs_maintenance(self) -> bool:
        """Check if elevator needs maintenance."""
        return datetime.now() - self.last_maintenance > self.maintenance_interval
    
    def add_access_level(self, level: int) -> None:
        """Add access level to elevator."""
        self.access_levels.add(level)
    
    def remove_access_level(self, level: int) -> None:
        """Remove access level from elevator."""
        self.access_levels.discard(level)
    
    def add_restricted_floor(self, floor: int) -> None:
        """Add restricted floor."""
        self.restricted_floors.add(floor)
    
    def remove_restricted_floor(self, floor: int) -> None:
        """Remove restricted floor."""
        self.restricted_floors.discard(floor)
    
    def get_status(self) -> ElevatorStatus:
        """Get current elevator status."""
        return ElevatorStatus(
            elevator_id=self.elevator_id,
            current_floor=self.current_floor,
            state=self.state,
            direction=self.direction,
            target_floors=self.target_floors.copy(),
            passenger_count=self.current_passengers,
            current_weight=self.current_weight,
            last_maintenance=self.last_maintenance,
            total_trips=self.total_trips,
            total_distance=self.total_distance,
            energy_consumed=self.energy_consumed
        )
    
    def get_elevator_info(self) -> Dict[str, Any]:
        """Get comprehensive elevator information."""
        return {
            'elevator_id': self.elevator_id,
            'elevator_type': self.elevator_type.value,
            'current_floor': self.current_floor,
            'state': self.state.value,
            'direction': self.direction.value,
            'target_floors': self.target_floors,
            'capacity': {
                'max_passengers': self.max_passengers,
                'current_passengers': self.current_passengers,
                'max_weight': self.max_weight,
                'current_weight': self.current_weight,
                'utilization': self.current_weight / self.max_weight
            },
            'performance': {
                'total_trips': self.total_trips,
                'total_distance': self.total_distance,
                'energy_consumed': self.energy_consumed,
                'efficiency': self.total_trips / max(1, self.energy_consumed)
            },
            'maintenance': {
                'last_maintenance': self.last_maintenance.isoformat(),
                'needs_maintenance': self.needs_maintenance(),
                'days_since_maintenance': (datetime.now() - self.last_maintenance).days
            },
            'access': {
                'access_levels': list(self.access_levels),
                'restricted_floors': list(self.restricted_floors)
            }
        }
    
    def __str__(self) -> str:
        return f"Elevator {self.elevator_id} - Floor {self.current_floor} ({self.state.value})"


# ============================================================================
# OBSERVER PATTERN FOR MONITORING
# ============================================================================

class ElevatorObserver(ABC):
    """Abstract elevator observer."""
    
    @abstractmethod
    def on_elevator_status_changed(self, status: ElevatorStatus) -> None:
        """Handle elevator status change."""
        pass


class ElevatorMonitor(ElevatorObserver):
    """Elevator monitoring system."""
    
    def __init__(self):
        self.status_history: List[Tuple[datetime, ElevatorStatus]] = []
        self.alerts: List[Dict[str, Any]] = []
    
    def on_elevator_status_changed(self, status: ElevatorStatus) -> None:
        """Handle elevator status change."""
        timestamp = datetime.now()
        self.status_history.append((timestamp, status))
        
        # Check for alerts
        self._check_alerts(status)
        
        # Keep only recent history (last 1000 entries)
        if len(self.status_history) > 1000:
            self.status_history = self.status_history[-1000:]
    
    def _check_alerts(self, status: ElevatorStatus) -> None:
        """Check for alert conditions."""
        # Emergency state alert
        if status.state == ElevatorState.EMERGENCY:
            self._add_alert("EMERGENCY", f"Elevator {status.elevator_id} in emergency state", "critical")
        
        # Maintenance needed alert
        maintenance_days = (datetime.now() - status.last_maintenance).days
        if maintenance_days > 90:
            self._add_alert("MAINTENANCE", f"Elevator {status.elevator_id} needs maintenance", "warning")
        
        # High energy consumption alert
        if status.energy_consumed > 1000:  # Threshold
            efficiency = status.total_trips / max(1, status.energy_consumed)
            if efficiency < 0.1:  # Low efficiency threshold
                self._add_alert("EFFICIENCY", f"Elevator {status.elevator_id} has low efficiency", "info")
    
    def _add_alert(self, alert_type: str, message: str, severity: str) -> None:
        """Add alert to the system."""
        alert = {
            'type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        }
        self.alerts.append(alert)
        
        # Keep only recent alerts (last 100)
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
    
    def get_recent_alerts(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        return self.alerts[-count:] if self.alerts else []
    
    def get_elevator_statistics(self, elevator_id: str) -> Dict[str, Any]:
        """Get statistics for a specific elevator."""
        elevator_history = [
            (timestamp, status) for timestamp, status in self.status_history
            if status.elevator_id == elevator_id
        ]
        
        if not elevator_history:
            return {}
        
        # Calculate statistics
        latest_status = elevator_history[-1][1]
        
        # Calculate average floor time
        floor_changes = []
        for i in range(1, len(elevator_history)):
            prev_status = elevator_history[i-1][1]
            curr_status = elevator_history[i][1]
            
            if prev_status.current_floor != curr_status.current_floor:
                time_diff = (elevator_history[i][0] - elevator_history[i-1][0]).total_seconds()
                floor_changes.append(time_diff)
        
        avg_floor_time = sum(floor_changes) / len(floor_changes) if floor_changes else 0
        
        return {
            'elevator_id': elevator_id,
            'total_status_updates': len(elevator_history),
            'current_status': {
                'floor': latest_status.current_floor,
                'state': latest_status.state.value,
                'trips': latest_status.total_trips,
                'distance': latest_status.total_distance,
                'energy': latest_status.energy_consumed
            },
            'performance': {
                'average_floor_time': avg_floor_time,
                'trips_per_hour': latest_status.total_trips / max(1, len(elevator_history) / 3600),
                'energy_efficiency': latest_status.total_trips / max(1, latest_status.energy_consumed)
            }
        }


# ============================================================================
# BUILDING CONTROLLER
# ============================================================================

class BuildingController:
    """Main building elevator controller."""
    
    def __init__(self, building_name: str, total_floors: int):
        self.building_name = building_name
        self.total_floors = total_floors
        self.elevators: Dict[str, Elevator] = {}
        self.request_queue: List[ElevatorRequest] = []
        self.completed_requests: List[ElevatorRequest] = []
        
        # Scheduling
        self.scheduling_strategy: SchedulingStrategy = SCANSchedulingStrategy()
        
        # Monitoring
        self.monitor = ElevatorMonitor()
        
        # Statistics
        self.total_requests = 0
        self.average_wait_time = 0.0
        self.peak_hour_requests = 0
        self.energy_consumption = 0.0
        
        # Threading
        self._lock = threading.Lock()
        self._running = False
        
        print(f"🏢 Building Controller '{building_name}' initialized ({total_floors} floors)")
    
    def add_elevator(self, elevator: Elevator) -> None:
        """Add elevator to the building."""
        with self._lock:
            self.elevators[elevator.elevator_id] = elevator
            elevator.add_observer(self.monitor)
            print(f"   ✓ Added {elevator}")
    
    def remove_elevator(self, elevator_id: str) -> bool:
        """Remove elevator from service."""
        with self._lock:
            if elevator_id in self.elevators:
                elevator = self.elevators[elevator_id]
                elevator.set_maintenance_mode(True)
                del self.elevators[elevator_id]
                return True
            return False
    
    def set_scheduling_strategy(self, strategy: SchedulingStrategy) -> None:
        """Set the scheduling strategy."""
        self.scheduling_strategy = strategy
        print(f"Scheduling strategy changed to: {strategy.get_strategy_name()}")
    
    def request_elevator(self, floor: int, direction: Direction, 
                        passenger_count: int = 1, weight: float = 70.0,
                        access_level: int = 0, priority: Priority = Priority.NORMAL) -> str:
        """Request an elevator (hall call)."""
        request = ElevatorRequest(
            request_id=str(uuid.uuid4()),
            floor=floor,
            direction=direction,
            request_type=RequestType.HALL_CALL,
            priority=priority,
            timestamp=datetime.now(),
            passenger_count=passenger_count,
            weight=weight,
            access_level=access_level
        )
        
        return self._process_request(request)
    
    def call_elevator_to_floor(self, elevator_id: str, floor: int,
                              passenger_count: int = 1, weight: float = 70.0) -> str:
        """Call specific elevator to floor (car call)."""
        request = ElevatorRequest(
            request_id=str(uuid.uuid4()),
            floor=floor,
            direction=Direction.NONE,
            request_type=RequestType.CAR_CALL,
            priority=Priority.NORMAL,
            timestamp=datetime.now(),
            passenger_count=passenger_count,
            weight=weight
        )
        
        # Assign to specific elevator
        elevator = self.elevators.get(elevator_id)
        if elevator and elevator.add_request(request):
            with self._lock:
                self.total_requests += 1
                self.completed_requests.append(request)
            return request.request_id
        
        return ""
    
    def emergency_request(self, floor: int) -> str:
        """Emergency elevator request."""
        request = ElevatorRequest(
            request_id=str(uuid.uuid4()),
            floor=floor,
            direction=Direction.NONE,
            request_type=RequestType.EMERGENCY,
            priority=Priority.EMERGENCY,
            timestamp=datetime.now()
        )
        
        return self._process_request(request)
    
    def _process_request(self, request: ElevatorRequest) -> str:
        """Process elevator request."""
        with self._lock:
            self.total_requests += 1
            
            # Find best elevator using scheduling strategy
            available_elevators = [e for e in self.elevators.values() 
                                 if e.can_handle_request(request)]
            
            if not available_elevators:
                # Add to queue for later processing
                self.request_queue.append(request)
                return request.request_id
            
            # Use scheduling strategy to select elevator
            selected_elevator = self.scheduling_strategy.schedule_request(
                request, available_elevators
            )
            
            if selected_elevator and selected_elevator.add_request(request):
                self.completed_requests.append(request)
                return request.request_id
            else:
                # Add to queue if assignment failed
                self.request_queue.append(request)
                return request.request_id
    
    def process_queued_requests(self) -> None:
        """Process queued requests."""
        with self._lock:
            processed_requests = []
            
            for request in self.request_queue:
                available_elevators = [e for e in self.elevators.values() 
                                     if e.can_handle_request(request)]
                
                if available_elevators:
                    selected_elevator = self.scheduling_strategy.schedule_request(
                        request, available_elevators
                    )
                    
                    if selected_elevator and selected_elevator.add_request(request):
                        processed_requests.append(request)
                        self.completed_requests.append(request)
            
            # Remove processed requests from queue
            for request in processed_requests:
                self.request_queue.remove(request)
    
    def emergency_stop_all(self) -> None:
        """Emergency stop all elevators."""
        for elevator in self.elevators.values():
            elevator.emergency_stop()
        print("🚨 EMERGENCY: All elevators stopped")
    
    def set_maintenance_mode(self, elevator_id: str, maintenance: bool = True) -> bool:
        """Set elevator maintenance mode."""
        elevator = self.elevators.get(elevator_id)
        if elevator:
            elevator.set_maintenance_mode(maintenance)
            return True
        return False
    
    def get_building_status(self) -> Dict[str, Any]:
        """Get comprehensive building status."""
        elevator_statuses = {}
        total_energy = 0
        total_trips = 0
        
        for elevator_id, elevator in self.elevators.items():
            status = elevator.get_elevator_info()
            elevator_statuses[elevator_id] = status
            total_energy += status['performance']['energy_consumed']
            total_trips += status['performance']['total_trips']
        
        return {
            'building_name': self.building_name,
            'total_floors': self.total_floors,
            'elevators': elevator_statuses,
            'requests': {
                'total_requests': self.total_requests,
                'queued_requests': len(self.request_queue),
                'completed_requests': len(self.completed_requests)
            },
            'performance': {
                'total_energy_consumed': total_energy,
                'total_trips': total_trips,
                'average_energy_per_trip': total_energy / max(1, total_trips),
                'scheduling_strategy': self.scheduling_strategy.get_strategy_name()
            },
            'alerts': self.monitor.get_recent_alerts()
        }
    
    def get_elevator_statistics(self) -> Dict[str, Any]:
        """Get detailed elevator statistics."""
        stats = {}
        
        for elevator_id in self.elevators.keys():
            stats[elevator_id] = self.monitor.get_elevator_statistics(elevator_id)
        
        return stats
    
    def optimize_energy_consumption(self) -> None:
        """Optimize energy consumption across elevators."""
        # Simple optimization: put least used elevators in standby
        elevator_usage = []
        
        for elevator in self.elevators.values():
            if elevator.state not in [ElevatorState.MAINTENANCE, ElevatorState.OUT_OF_SERVICE]:
                usage_score = elevator.total_trips + len(elevator.target_floors) * 0.5
                elevator_usage.append((usage_score, elevator))
        
        # Sort by usage (lowest first)
        elevator_usage.sort(key=lambda x: x[0])
        
        # Put lowest usage elevators in energy-saving mode if we have excess capacity
        if len(elevator_usage) > 2:  # Keep at least 2 elevators active
            for i in range(len(elevator_usage) // 3):  # Put 1/3 in standby
                _, elevator = elevator_usage[i]
                if elevator.state == ElevatorState.IDLE and not elevator.target_floors:
                    # Implement energy-saving mode (simplified)
                    print(f"🔋 Elevator {elevator.elevator_id} entering energy-saving mode")
    
    def start_controller(self) -> None:
        """Start the building controller."""
        self._running = True
        
        # Start background processing thread
        threading.Thread(target=self._background_processing, daemon=True).start()
        print(f"🏢 Building Controller started")
    
    def stop_controller(self) -> None:
        """Stop the building controller."""
        self._running = False
        print(f"🏢 Building Controller stopped")
    
    def _background_processing(self) -> None:
        """Background processing for queued requests and optimization."""
        while self._running:
            try:
                # Process queued requests
                self.process_queued_requests()
                
                # Optimize energy consumption periodically
                if self.total_requests % 50 == 0:  # Every 50 requests
                    self.optimize_energy_consumption()
                
                time.sleep(1)  # Process every second
                
            except Exception as e:
                print(f"Error in background processing: {e}")
                time.sleep(5)  # Wait before retrying


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def demonstrate_elevator_system():
    """Demonstrate the elevator system."""
    print("=== ELEVATOR SYSTEM DESIGN DEMONSTRATION ===\n")
    
    # Initialize building
    building = BuildingController("Tech Tower", 20)
    
    print("1. BUILDING SETUP:")
    
    # Add elevators
    elevators_config = [
        ("ELEV-A", ElevatorType.PASSENGER),
        ("ELEV-B", ElevatorType.PASSENGER),
        ("ELEV-C", ElevatorType.EXPRESS),
        ("ELEV-F", ElevatorType.FREIGHT)
    ]
    
    for elevator_id, elevator_type in elevators_config:
        elevator = Elevator(elevator_id, elevator_type, max_floor=20)
        
        # Configure access levels
        if elevator_type == ElevatorType.EXPRESS:
            elevator.add_access_level(1)  # VIP access
            elevator.add_restricted_floor(13)  # Skip floor 13
        elif elevator_type == ElevatorType.FREIGHT:
            elevator.add_access_level(2)  # Service access
        
        building.add_elevator(elevator)
    
    # Start controller
    building.start_controller()
    
    print()
    
    # Test different scheduling strategies
    print("2. SCHEDULING STRATEGY TESTING:")
    
    strategies = [
        SCANSchedulingStrategy(),
        LOOKSchedulingStrategy(),
        DestinationDispatchStrategy(),
        FCFSSchedulingStrategy()
    ]
    
    for strategy in strategies:
        building.set_scheduling_strategy(strategy)
        
        # Make some test requests
        request_id = building.request_elevator(5, Direction.UP, passenger_count=2)
        print(f"   {strategy.get_strategy_name()}: Request {request_id[:8]} processed")
    
    print()
    
    # Simulate rush hour traffic
    print("3. RUSH HOUR SIMULATION:")
    
    building.set_scheduling_strategy(LOOKSchedulingStrategy())
    
    # Generate multiple requests
    rush_requests = [
        (1, Direction.UP, 3, Priority.NORMAL),
        (1, Direction.UP, 2, Priority.NORMAL),
        (15, Direction.DOWN, 1, Priority.NORMAL),
        (8, Direction.UP, 4, Priority.NORMAL),
        (12, Direction.DOWN, 2, Priority.NORMAL),
        (1, Direction.UP, 1, Priority.HIGH),
        (20, Direction.DOWN, 3, Priority.NORMAL),
        (5, Direction.UP, 2, Priority.NORMAL),
        (18, Direction.DOWN, 1, Priority.NORMAL),
        (3, Direction.UP, 5, Priority.NORMAL)
    ]
    
    request_ids = []
    for floor, direction, passengers, priority in rush_requests:
        request_id = building.request_elevator(floor, direction, passengers, priority=priority)
        request_ids.append(request_id)
        print(f"   ✓ Request from floor {floor} ({direction.value}) - {passengers} passengers")
    
    # Wait for some processing
    time.sleep(3)
    
    print()
    
    # Test emergency scenarios
    print("4. EMERGENCY SCENARIO TESTING:")
    
    # Emergency request
    emergency_id = building.emergency_request(10)
    print(f"   🚨 Emergency request: {emergency_id[:8]}")
    
    # Emergency stop all
    time.sleep(1)
    building.emergency_stop_all()
    
    # Wait and restart
    time.sleep(2)
    
    # Reset elevators from emergency
    for elevator in building.elevators.values():
        if elevator.state == ElevatorState.EMERGENCY:
            elevator.state = ElevatorState.IDLE
            elevator._stop_requested = False
    
    print("   ✓ Elevators reset from emergency")
    
    print()
    
    # Test maintenance mode
    print("5. MAINTENANCE MODE TESTING:")
    
    # Set one elevator to maintenance
    maintenance_result = building.set_maintenance_mode("ELEV-A", True)
    print(f"   ✓ ELEV-A maintenance mode: {maintenance_result}")
    
    # Make requests with reduced capacity
    for i in range(3):
        request_id = building.request_elevator(i+2, Direction.UP)
        print(f"   Request {i+1}: {request_id[:8]}")
    
    # Bring elevator back online
    building.set_maintenance_mode("ELEV-A", False)
    print("   ✓ ELEV-A back online")
    
    print()
    
    # Test freight elevator
    print("6. FREIGHT ELEVATOR TESTING:")
    
    # Heavy cargo request
    freight_request = building.request_elevator(
        floor=5, 
        direction=Direction.UP, 
        passenger_count=1, 
        weight=2000.0,  # Heavy cargo
        access_level=2   # Service access
    )
    print(f"   ✓ Freight request: {freight_request[:8]} (2000kg cargo)")
    
    print()
    
    # Wait for processing
    time.sleep(5)
    
    # Show building status
    print("7. BUILDING STATUS REPORT:")
    
    status = building.get_building_status()
    
    print(f"   Building: {status['building_name']} ({status['total_floors']} floors)")
    print(f"   Total Requests: {status['requests']['total_requests']}")
    print(f"   Queued Requests: {status['requests']['queued_requests']}")
    print(f"   Completed Requests: {status['requests']['completed_requests']}")
    
    print(f"\n   Performance Metrics:")
    print(f"   - Total Energy Consumed: {status['performance']['total_energy_consumed']:.1f} kWh")
    print(f"   - Total Trips: {status['performance']['total_trips']}")
    print(f"   - Energy per Trip: {status['performance']['average_energy_per_trip']:.1f} kWh")
    print(f"   - Scheduling Strategy: {status['performance']['scheduling_strategy']}")
    
    print(f"\n   Elevator Status:")
    for elevator_id, elevator_info in status['elevators'].items():
        print(f"   {elevator_id}:")
        print(f"     Floor: {elevator_info['current_floor']}")
        print(f"     State: {elevator_info['state']}")
        print(f"     Targets: {elevator_info['target_floors']}")
        print(f"     Passengers: {elevator_info['capacity']['current_passengers']}/{elevator_info['capacity']['max_passengers']}")
        print(f"     Trips: {elevator_info['performance']['total_trips']}")
        print(f"     Efficiency: {elevator_info['performance']['efficiency']:.2f}")
    
    # Show alerts
    if status['alerts']:
        print(f"\n   Recent Alerts:")
        for alert in status['alerts']:
            print(f"   - {alert['severity'].upper()}: {alert['message']}")
    
    print()
    
    # Show detailed statistics
    print("8. DETAILED STATISTICS:")
    
    stats = building.get_elevator_statistics()
    
    for elevator_id, elevator_stats in stats.items():
        if elevator_stats:  # Only show if we have data
            print(f"   {elevator_id} Statistics:")
            print(f"     Status Updates: {elevator_stats['total_status_updates']}")
            print(f"     Current Floor: {elevator_stats['current_status']['floor']}")
            print(f"     Total Trips: {elevator_stats['current_status']['trips']}")
            print(f"     Distance Traveled: {elevator_stats['current_status']['distance']} floors")
            print(f"     Energy Consumed: {elevator_stats['current_status']['energy']:.1f} kWh")
            
            if 'performance' in elevator_stats:
                perf = elevator_stats['performance']
                print(f"     Avg Floor Time: {perf['average_floor_time']:.1f}s")
                print(f"     Energy Efficiency: {perf['energy_efficiency']:.2f} trips/kWh")
    
    print()
    
    # Test energy optimization
    print("9. ENERGY OPTIMIZATION:")
    
    building.optimize_energy_consumption()
    
    print("   ✓ Energy optimization completed")
    
    print()
    
    # Stop controller
    building.stop_controller()
    
    print("=== ELEVATOR SYSTEM DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_elevator_system()
