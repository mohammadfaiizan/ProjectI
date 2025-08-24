"""
1396. Design Underground System - Multiple Approaches
Difficulty: Medium

An underground railway system is keeping track of customer travel times between different stations. They are using this data to calculate the average time it takes to travel from one station to another.

Implement the UndergroundSystem class:
- void checkIn(int id, string stationName, int t) A customer with a card ID equal to id, checks in at the station stationName at time t.
- void checkOut(int id, string stationName, int t) A customer with a card ID equal to id, checks out at the station stationName at time t.
- double getAverageTime(string startStation, string endStation) Returns the average time it takes to travel from startStation to endStation.

You can assume all calls to checkIn and checkOut are consistent. If a customer checks in at time t1 then checks out at time t2, then t1 < t2. All events happen in chronological order.
"""

from typing import Dict, Tuple
from collections import defaultdict

class UndergroundSystemOptimal:
    """
    Approach 1: HashMap-based Implementation (Optimal)
    
    Use HashMaps to track check-ins and route statistics.
    
    Time: O(1) for all operations, Space: O(P + S²) where P = passengers, S = stations
    """
    
    def __init__(self):
        # id -> (station, check_in_time)
        self.check_ins = {}
        
        # (start_station, end_station) -> (total_time, trip_count)
        self.route_stats = defaultdict(lambda: [0, 0])
    
    def checkIn(self, id: int, stationName: str, t: int) -> None:
        """Customer checks in at station"""
        self.check_ins[id] = (stationName, t)
    
    def checkOut(self, id: int, stationName: str, t: int) -> None:
        """Customer checks out at station"""
        start_station, check_in_time = self.check_ins[id]
        
        # Calculate travel time
        travel_time = t - check_in_time
        
        # Update route statistics
        route = (start_station, stationName)
        self.route_stats[route][0] += travel_time  # total time
        self.route_stats[route][1] += 1            # trip count
        
        # Remove check-in record
        del self.check_ins[id]
    
    def getAverageTime(self, startStation: str, endStation: str) -> float:
        """Get average travel time between stations"""
        route = (startStation, endStation)
        total_time, trip_count = self.route_stats[route]
        
        return total_time / trip_count


class UndergroundSystemDetailed:
    """
    Approach 2: Detailed Tracking Implementation
    
    Track individual trips for more detailed analytics.
    
    Time: O(1) for check-in/out, O(n) for average (where n = trips), Space: O(P + T)
    """
    
    def __init__(self):
        # id -> (station, check_in_time)
        self.active_trips = {}
        
        # (start_station, end_station) -> [trip_times]
        self.completed_trips = defaultdict(list)
    
    def checkIn(self, id: int, stationName: str, t: int) -> None:
        """Customer checks in at station"""
        self.active_trips[id] = (stationName, t)
    
    def checkOut(self, id: int, stationName: str, t: int) -> None:
        """Customer checks out at station"""
        start_station, check_in_time = self.active_trips[id]
        
        # Calculate and store travel time
        travel_time = t - check_in_time
        route = (start_station, stationName)
        self.completed_trips[route].append(travel_time)
        
        # Remove from active trips
        del self.active_trips[id]
    
    def getAverageTime(self, startStation: str, endStation: str) -> float:
        """Get average travel time between stations"""
        route = (startStation, endStation)
        trip_times = self.completed_trips[route]
        
        return sum(trip_times) / len(trip_times)
    
    def getDetailedStats(self, startStation: str, endStation: str) -> Dict:
        """Get detailed statistics for a route"""
        route = (startStation, endStation)
        trip_times = self.completed_trips[route]
        
        if not trip_times:
            return {}
        
        return {
            "average": sum(trip_times) / len(trip_times),
            "min": min(trip_times),
            "max": max(trip_times),
            "count": len(trip_times),
            "total": sum(trip_times)
        }


class UndergroundSystemWithHistory:
    """
    Approach 3: Implementation with Trip History
    
    Maintain complete trip history for analytics.
    
    Time: O(1) for operations, Space: O(T) where T = total trips
    """
    
    def __init__(self):
        # id -> (station, check_in_time)
        self.check_ins = {}
        
        # Route statistics
        self.route_stats = defaultdict(lambda: [0, 0])
        
        # Complete trip history: [(id, start, end, start_time, end_time, duration)]
        self.trip_history = []
    
    def checkIn(self, id: int, stationName: str, t: int) -> None:
        """Customer checks in at station"""
        self.check_ins[id] = (stationName, t)
    
    def checkOut(self, id: int, stationName: str, t: int) -> None:
        """Customer checks out at station"""
        start_station, check_in_time = self.check_ins[id]
        travel_time = t - check_in_time
        
        # Update route statistics
        route = (start_station, stationName)
        self.route_stats[route][0] += travel_time
        self.route_stats[route][1] += 1
        
        # Add to trip history
        self.trip_history.append((id, start_station, stationName, check_in_time, t, travel_time))
        
        # Remove check-in record
        del self.check_ins[id]
    
    def getAverageTime(self, startStation: str, endStation: str) -> float:
        """Get average travel time between stations"""
        route = (startStation, endStation)
        total_time, trip_count = self.route_stats[route]
        
        return total_time / trip_count
    
    def getTripHistory(self, limit: int = 10) -> list:
        """Get recent trip history"""
        return self.trip_history[-limit:]
    
    def getCustomerTrips(self, customer_id: int) -> list:
        """Get all trips for a specific customer"""
        return [trip for trip in self.trip_history if trip[0] == customer_id]


def test_underground_system_implementations():
    """Test underground system implementations"""
    
    implementations = [
        ("Optimal", UndergroundSystemOptimal),
        ("Detailed", UndergroundSystemDetailed),
        ("With History", UndergroundSystemWithHistory),
    ]
    
    test_cases = [
        {
            "operations": ["checkIn", "checkIn", "checkIn", "checkOut", "checkOut", "checkOut", "getAverageTime", "getAverageTime", "checkIn", "getAverageTime", "checkOut", "getAverageTime"],
            "values": [(45,"Leyton",3), (32,"Paradise",8), (27,"Leyton",10), (45,"Waterloo",15), (27,"Waterloo",20), (32,"Cambridge",22), ("Paradise","Cambridge"), ("Leyton","Waterloo"), (10,"Leyton",24), ("Leyton","Waterloo"), (10,"Waterloo",38), ("Leyton","Waterloo")],
            "expected": [None, None, None, None, None, None, 14.0, 11.0, None, 11.0, None, 12.0],
            "description": "Example 1"
        },
        {
            "operations": ["checkIn", "checkOut", "getAverageTime"],
            "values": [(1,"A",0), (1,"B",10), ("A","B")],
            "expected": [None, None, 10.0],
            "description": "Single trip"
        },
    ]
    
    print("=== Testing Underground System Implementations ===")
    
    for impl_name, impl_class in implementations:
        print(f"\n--- {impl_name} Implementation ---")
        
        for test_case in test_cases:
            try:
                system = impl_class()
                results = []
                
                for i, op in enumerate(test_case["operations"]):
                    if op == "checkIn":
                        id, station, time = test_case["values"][i]
                        system.checkIn(id, station, time)
                        results.append(None)
                    elif op == "checkOut":
                        id, station, time = test_case["values"][i]
                        system.checkOut(id, station, time)
                        results.append(None)
                    elif op == "getAverageTime":
                        start, end = test_case["values"][i]
                        result = system.getAverageTime(start, end)
                        results.append(result)
                
                expected = test_case["expected"]
                status = "✓" if results == expected else "✗"
                
                print(f"  {test_case['description']:15} | {status} | {results}")
                if results != expected:
                    print(f"    Expected: {expected}")
                
            except Exception as e:
                print(f"  {test_case['description']:15} | ERROR: {str(e)[:40]}")


def demonstrate_underground_system():
    """Demonstrate underground system step by step"""
    print("\n=== Underground System Step-by-Step Demo ===")
    
    system = UndergroundSystemOptimal()
    
    operations = [
        ("checkIn", (45, "Leyton", 3)),
        ("checkIn", (32, "Paradise", 8)),
        ("checkOut", (45, "Waterloo", 15)),  # Travel time: 15-3 = 12
        ("checkOut", (32, "Cambridge", 22)), # Travel time: 22-8 = 14
        ("getAverageTime", ("Paradise", "Cambridge")),
        ("checkIn", (10, "Leyton", 24)),
        ("checkOut", (10, "Waterloo", 38)),  # Travel time: 38-24 = 14
        ("getAverageTime", ("Leyton", "Waterloo")),
    ]
    
    print("Strategy: Track check-ins and maintain route statistics")
    
    def print_system_state():
        """Helper to print current system state"""
        print(f"  Active check-ins: {system.check_ins}")
        route_stats = {}
        for route, (total, count) in system.route_stats.items():
            route_stats[route] = f"total={total}, count={count}, avg={total/count:.1f}"
        print(f"  Route statistics: {route_stats}")
    
    print(f"\nInitial state:")
    print_system_state()
    
    for i, (op, value) in enumerate(operations):
        print(f"\nStep {i+1}: {op}({value})")
        
        if op == "checkIn":
            id, station, time = value
            system.checkIn(id, station, time)
            print(f"  Customer {id} checked in at {station} at time {time}")
        elif op == "checkOut":
            id, station, time = value
            system.checkOut(id, station, time)
            print(f"  Customer {id} checked out at {station} at time {time}")
        elif op == "getAverageTime":
            start, end = value
            avg_time = system.getAverageTime(start, end)
            print(f"  Average time from {start} to {end}: {avg_time}")
        
        print_system_state()


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Metro system analytics
    print("1. Metro System Analytics:")
    metro = UndergroundSystemWithHistory()
    
    # Simulate rush hour traffic
    rush_hour_data = [
        (101, "Downtown", "Airport", 480, 510),    # 8:00-8:30 AM
        (102, "Downtown", "Airport", 485, 520),    # 8:05-8:40 AM
        (103, "Airport", "Downtown", 490, 515),    # 8:10-8:35 AM
        (104, "Downtown", "Mall", 495, 510),       # 8:15-8:30 AM
        (105, "Mall", "Downtown", 500, 520),       # 8:20-8:40 AM
    ]
    
    print("  Rush hour simulation:")
    for customer_id, start, end, check_in, check_out in rush_hour_data:
        metro.checkIn(customer_id, start, check_in)
        metro.checkOut(customer_id, end, check_out)
        print(f"    Customer {customer_id}: {start} -> {end} ({check_out - check_in} min)")
    
    # Get analytics
    routes = [("Downtown", "Airport"), ("Airport", "Downtown"), ("Downtown", "Mall")]
    
    print(f"\n  Route analytics:")
    for start, end in routes:
        try:
            avg_time = metro.getAverageTime(start, end)
            print(f"    {start} -> {end}: {avg_time:.1f} minutes average")
        except:
            print(f"    {start} -> {end}: No data available")
    
    # Application 2: Bus system optimization
    print(f"\n2. Bus System Route Optimization:")
    bus_system = UndergroundSystemDetailed()
    
    # Simulate bus routes
    bus_trips = [
        (201, "Central", "North", 60, 85),   # 25 min
        (202, "Central", "North", 65, 95),   # 30 min (traffic)
        (203, "North", "Central", 70, 90),   # 20 min
        (204, "Central", "South", 75, 105),  # 30 min
        (205, "South", "Central", 80, 105),  # 25 min
    ]
    
    for customer_id, start, end, check_in, check_out in bus_trips:
        bus_system.checkIn(customer_id, start, check_in)
        bus_system.checkOut(customer_id, end, check_out)
    
    print("  Bus route performance:")
    bus_routes = [("Central", "North"), ("North", "Central"), ("Central", "South")]
    
    for start, end in bus_routes:
        try:
            stats = bus_system.getDetailedStats(start, end)
            if stats:
                print(f"    {start} -> {end}:")
                print(f"      Average: {stats['average']:.1f} min")
                print(f"      Range: {stats['min']}-{stats['max']} min")
                print(f"      Trips: {stats['count']}")
        except:
            print(f"    {start} -> {end}: No data")
    
    # Application 3: Ride sharing analytics
    print(f"\n3. Ride Sharing Service Analytics:")
    rideshare = UndergroundSystemOptimal()
    
    # Simulate ride sharing trips
    rides = [
        (301, "Airport", "Hotel", 0, 25),
        (302, "Hotel", "Restaurant", 30, 45),
        (303, "Restaurant", "Hotel", 120, 135),
        (304, "Airport", "Hotel", 180, 200),
        (305, "Hotel", "Airport", 240, 270),
    ]
    
    for rider_id, pickup, dropoff, start_time, end_time in rides:
        rideshare.checkIn(rider_id, pickup, start_time)
        rideshare.checkOut(rider_id, dropoff, end_time)
    
    print("  Popular route analysis:")
    popular_routes = [("Airport", "Hotel"), ("Hotel", "Airport"), ("Hotel", "Restaurant")]
    
    for start, end in popular_routes:
        try:
            avg_time = rideshare.getAverageTime(start, end)
            print(f"    {start} -> {end}: {avg_time:.1f} minutes")
        except:
            print(f"    {start} -> {end}: No trips recorded")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Optimal", "O(1)", "O(1)", "O(1)", "O(P + S²)", "P=passengers, S=stations"),
        ("Detailed", "O(1)", "O(1)", "O(n)", "O(P + T)", "n=trips for route, T=total trips"),
        ("With History", "O(1)", "O(1)", "O(1)", "O(T)", "Complete trip history stored"),
    ]
    
    print(f"{'Approach':<15} | {'CheckIn':<8} | {'CheckOut':<9} | {'GetAvg':<8} | {'Space':<10} | {'Notes'}")
    print("-" * 80)
    
    for approach, checkin, checkout, getavg, space, notes in approaches:
        print(f"{approach:<15} | {checkin:<8} | {checkout:<9} | {getavg:<8} | {space:<10} | {notes}")
    
    print(f"\nOptimal approach provides O(1) for all operations")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    system = UndergroundSystemOptimal()
    
    edge_cases = [
        ("Single trip", lambda: (
            system.checkIn(1, "A", 0),
            system.checkOut(1, "B", 10),
            system.getAverageTime("A", "B")
        )[2], 10.0),
        
        ("Multiple same route", lambda: (
            system.checkIn(2, "A", 20),
            system.checkOut(2, "B", 25),
            system.getAverageTime("A", "B")
        )[2], 7.5),  # (10 + 5) / 2
        
        ("Different customers same time", lambda: (
            system.checkIn(3, "C", 30),
            system.checkIn(4, "C", 30),
            system.checkOut(3, "D", 40),
            system.checkOut(4, "D", 45),
            system.getAverageTime("C", "D")
        )[4], 12.5),  # (10 + 15) / 2
    ]
    
    for description, operation, expected in edge_cases:
        try:
            result = operation()
            status = "✓" if abs(result - expected) < 1e-9 else "✗"
            print(f"{description:30} | {status} | Result: {result}")
        except Exception as e:
            print(f"{description:30} | ERROR: {str(e)[:30]}")


if __name__ == "__main__":
    test_underground_system_implementations()
    demonstrate_underground_system()
    demonstrate_real_world_applications()
    analyze_time_complexity()
    test_edge_cases()

"""
Design Underground System demonstrates system design for transportation
analytics, including multiple implementation approaches for tracking
passenger journeys and calculating route statistics efficiently.
"""
