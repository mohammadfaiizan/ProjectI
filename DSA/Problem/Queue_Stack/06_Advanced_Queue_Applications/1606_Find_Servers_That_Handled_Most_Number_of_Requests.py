"""
1606. Find Servers That Handled Most Number of Requests - Multiple Approaches
Difficulty: Hard

You have k servers numbered from 0 to k-1 that are being used to handle multiple requests simultaneously. Each server has infinite computational capacity but cannot handle more than one request at a time. The requests come in and are assigned to servers using the following algorithm:

- The ith request (0-indexed) arrives.
- If all servers are busy, it is dropped.
- If the (i % k)th server is available, assign the request to it.
- Otherwise, assign the request to the next available server (wrapping around the server list as needed).

You are given a strictly increasing array arrival of positive integers, where arrival[i] represents the arrival time of the ith request, and another array load, where load[i] represents the load of the ith request (the time it takes to complete). Your goal is to find the busiest server(s). A server is considered busiest if it handled the most number of requests successfully.

Return a list of IDs of the busiest server(s) ordered in increasing order.
"""

from typing import List
import heapq

class FindBusiestServers:
    """Multiple approaches to find busiest servers"""
    
    def busiestServers_heap_approach(self, k: int, arrival: List[int], load: List[int]) -> List[int]:
        """
        Approach 1: Heap-based Approach (Optimal)
        
        Use heaps to track available and busy servers efficiently.
        
        Time: O(n log k), Space: O(k)
        """
        # Min heap for available servers (server_id)
        available = list(range(k))
        heapq.heapify(available)
        
        # Min heap for busy servers (end_time, server_id)
        busy = []
        
        # Count requests handled by each server
        request_count = [0] * k
        
        for i in range(len(arrival)):
            current_time = arrival[i]
            request_load = load[i]
            
            # Free up servers that have finished processing
            while busy and busy[0][0] <= current_time:
                _, server_id = heapq.heappop(busy)
                heapq.heappush(available, server_id)
            
            if not available:
                # All servers busy, drop request
                continue
            
            # Find the appropriate server
            preferred_server = i % k
            
            # Check if preferred server is available
            if preferred_server in available:
                # Remove preferred server from available
                available.remove(preferred_server)
                heapq.heapify(available)
                assigned_server = preferred_server
            else:
                # Find next available server >= preferred_server
                temp_available = []
                assigned_server = None
                
                while available:
                    server = heapq.heappop(available)
                    if server >= preferred_server:
                        assigned_server = server
                        break
                    temp_available.append(server)
                
                # If no server >= preferred_server, use smallest available
                if assigned_server is None and temp_available:
                    assigned_server = temp_available[0]
                    temp_available = temp_available[1:]
                
                # Put back unused servers
                for server in temp_available:
                    heapq.heappush(available, server)
            
            if assigned_server is not None:
                # Assign request to server
                end_time = current_time + request_load
                heapq.heappush(busy, (end_time, assigned_server))
                request_count[assigned_server] += 1
        
        # Find servers with maximum requests
        max_requests = max(request_count)
        return [i for i in range(k) if request_count[i] == max_requests]
    
    def busiestServers_simulation(self, k: int, arrival: List[int], load: List[int]) -> List[int]:
        """
        Approach 2: Direct Simulation
        
        Simulate the server assignment process directly.
        
        Time: O(n * k), Space: O(k)
        """
        # Track when each server becomes free
        server_free_time = [0] * k
        request_count = [0] * k
        
        for i in range(len(arrival)):
            current_time = arrival[i]
            request_load = load[i]
            preferred_server = i % k
            
            # Try to find an available server starting from preferred
            assigned_server = None
            
            # Check from preferred server to end
            for j in range(preferred_server, k):
                if server_free_time[j] <= current_time:
                    assigned_server = j
                    break
            
            # If not found, check from beginning to preferred
            if assigned_server is None:
                for j in range(preferred_server):
                    if server_free_time[j] <= current_time:
                        assigned_server = j
                        break
            
            if assigned_server is not None:
                # Assign request to server
                server_free_time[assigned_server] = current_time + request_load
                request_count[assigned_server] += 1
        
        # Find servers with maximum requests
        max_requests = max(request_count)
        return [i for i in range(k) if request_count[i] == max_requests]
    
    def busiestServers_optimized_heap(self, k: int, arrival: List[int], load: List[int]) -> List[int]:
        """
        Approach 3: Optimized Heap with Set
        
        Use heap for busy servers and set for available servers.
        
        Time: O(n log k), Space: O(k)
        """
        import bisect
        
        # Sorted list of available servers
        available = list(range(k))
        
        # Min heap for busy servers (end_time, server_id)
        busy = []
        
        # Count requests handled by each server
        request_count = [0] * k
        
        for i in range(len(arrival)):
            current_time = arrival[i]
            request_load = load[i]
            
            # Free up servers that have finished processing
            while busy and busy[0][0] <= current_time:
                _, server_id = heapq.heappop(busy)
                bisect.insort(available, server_id)
            
            if not available:
                # All servers busy, drop request
                continue
            
            # Find the appropriate server
            preferred_server = i % k
            
            # Find the first available server >= preferred_server
            idx = bisect.bisect_left(available, preferred_server)
            
            if idx < len(available):
                # Found server >= preferred_server
                assigned_server = available.pop(idx)
            else:
                # Wrap around, use first available server
                assigned_server = available.pop(0)
            
            # Assign request to server
            end_time = current_time + request_load
            heapq.heappush(busy, (end_time, assigned_server))
            request_count[assigned_server] += 1
        
        # Find servers with maximum requests
        max_requests = max(request_count)
        return [i for i in range(k) if request_count[i] == max_requests]


def test_find_busiest_servers():
    """Test find busiest servers algorithms"""
    solver = FindBusiestServers()
    
    test_cases = [
        (3, [1,2,3,4,5], [5,2,3,3,3], [1], "Example 1"),
        (3, [1,2,3,4], [1,2,1,2], [0], "Example 2"),
        (3, [1,2,3], [10,12,11], [0,1,2], "Example 3"),
        (1, [1], [1], [0], "Single server"),
        (2, [1,2,3,4], [1,1,1,1], [0,1], "Two servers"),
        (4, [1,2,3,4,5,6], [2,2,2,2,2,2], [0,1,2,3], "All servers equal"),
    ]
    
    algorithms = [
        ("Heap Approach", solver.busiestServers_heap_approach),
        ("Simulation", solver.busiestServers_simulation),
        ("Optimized Heap", solver.busiestServers_optimized_heap),
    ]
    
    print("=== Testing Find Busiest Servers ===")
    
    for k, arrival, load, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"k: {k}, arrival: {arrival}, load: {load}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(k, arrival, load)
                status = "✓" if sorted(result) == sorted(expected) else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_server_assignment():
    """Demonstrate server assignment step by step"""
    print("\n=== Server Assignment Step-by-Step Demo ===")
    
    k = 3
    arrival = [1, 2, 3, 4, 5]
    load = [5, 2, 3, 3, 3]
    
    print(f"k = {k} servers")
    print(f"Requests: arrival = {arrival}, load = {load}")
    
    server_free_time = [0] * k
    request_count = [0] * k
    
    for i in range(len(arrival)):
        current_time = arrival[i]
        request_load = load[i]
        preferred_server = i % k
        
        print(f"\nRequest {i} at time {current_time} (load: {request_load}):")
        print(f"  Preferred server: {preferred_server}")
        print(f"  Server free times: {server_free_time}")
        
        # Find available server
        assigned_server = None
        
        # Check from preferred server
        for j in range(preferred_server, k):
            if server_free_time[j] <= current_time:
                assigned_server = j
                break
        
        # Wrap around if needed
        if assigned_server is None:
            for j in range(preferred_server):
                if server_free_time[j] <= current_time:
                    assigned_server = j
                    break
        
        if assigned_server is not None:
            server_free_time[assigned_server] = current_time + request_load
            request_count[assigned_server] += 1
            print(f"  Assigned to server {assigned_server}")
            print(f"  Server {assigned_server} will be free at time {server_free_time[assigned_server]}")
        else:
            print(f"  All servers busy - request dropped")
        
        print(f"  Request counts: {request_count}")
    
    max_requests = max(request_count)
    busiest = [i for i in range(k) if request_count[i] == max_requests]
    print(f"\nBusiest servers: {busiest} (handled {max_requests} requests each)")


def visualize_server_timeline():
    """Visualize server timeline"""
    print("\n=== Server Timeline Visualization ===")
    
    k = 3
    arrival = [1, 2, 3, 4, 8]
    load = [2, 1, 3, 2, 1]
    
    print(f"Timeline for {k} servers:")
    print("Time:  0  1  2  3  4  5  6  7  8  9")
    
    server_assignments = []
    server_free_time = [0] * k
    
    for i in range(len(arrival)):
        current_time = arrival[i]
        request_load = load[i]
        preferred_server = i % k
        
        # Find available server
        assigned_server = None
        
        for j in range(preferred_server, k):
            if server_free_time[j] <= current_time:
                assigned_server = j
                break
        
        if assigned_server is None:
            for j in range(preferred_server):
                if server_free_time[j] <= current_time:
                    assigned_server = j
                    break
        
        if assigned_server is not None:
            end_time = current_time + request_load
            server_assignments.append((assigned_server, current_time, end_time, i))
            server_free_time[assigned_server] = end_time
    
    # Print timeline for each server
    for server_id in range(k):
        timeline = ['.'] * 10
        
        for assigned_server, start, end, request_id in server_assignments:
            if assigned_server == server_id:
                for t in range(start, min(end, 10)):
                    timeline[t] = str(request_id)
        
        print(f"S{server_id}:   {'  '.join(timeline)}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Web server load balancing
    print("1. Web Server Load Balancing:")
    
    # Simulate web requests
    k = 4  # 4 web servers
    arrival = [1, 2, 3, 4, 5, 6, 7, 8]  # Request arrival times
    load = [3, 2, 4, 1, 2, 3, 1, 2]     # Processing times
    
    solver = FindBusiestServers()
    busiest = solver.busiestServers_simulation(k, arrival, load)
    
    print(f"  {k} web servers handling requests")
    print(f"  Request arrivals: {arrival}")
    print(f"  Processing times: {load}")
    print(f"  Busiest servers: {busiest}")
    
    # Application 2: Database connection pooling
    print(f"\n2. Database Connection Pool:")
    
    k = 3  # 3 database connections
    arrival = [1, 1, 2, 3, 4, 5]  # Query arrival times
    load = [2, 3, 1, 2, 1, 1]     # Query execution times
    
    busiest_db = solver.busiestServers_simulation(k, arrival, load)
    
    print(f"  {k} database connections")
    print(f"  Query arrivals: {arrival}")
    print(f"  Execution times: {load}")
    print(f"  Most used connections: {busiest_db}")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Heap Approach", "O(n log k)", "O(k)", "Heap operations for server management"),
        ("Simulation", "O(n * k)", "O(k)", "Linear search for available servers"),
        ("Optimized Heap", "O(n log k)", "O(k)", "Binary search + heap operations"),
    ]
    
    print(f"{'Approach':<20} | {'Time':<12} | {'Space':<8} | {'Notes'}")
    print("-" * 65)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<20} | {time_comp:<12} | {space_comp:<8} | {notes}")
    
    print(f"\nwhere n = number of requests, k = number of servers")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = FindBusiestServers()
    
    edge_cases = [
        (1, [1], [1], [0], "Single server, single request"),
        (2, [1], [1], [0], "More servers than requests"),
        (1, [1, 2, 3], [1, 1, 1], [0], "Single server, multiple requests"),
        (3, [1, 1, 1], [10, 10, 10], [0], "Simultaneous arrivals"),
        (2, [1, 2], [10, 1], [1], "Different load times"),
        (3, [1, 5, 10], [1, 1, 1], [0, 1, 2], "Sparse arrivals"),
    ]
    
    for k, arrival, load, expected, description in edge_cases:
        try:
            result = solver.busiestServers_simulation(k, arrival, load)
            status = "✓" if sorted(result) == sorted(expected) else "✗"
            print(f"{description:30} | {status} | k={k}, result={result}")
        except Exception as e:
            print(f"{description:30} | ERROR: {str(e)[:30]}")


if __name__ == "__main__":
    test_find_busiest_servers()
    demonstrate_server_assignment()
    visualize_server_timeline()
    demonstrate_real_world_applications()
    analyze_time_complexity()
    test_edge_cases()

"""
Find Servers That Handled Most Number of Requests demonstrates advanced
queue applications for load balancing and server management, including
heap-based optimization and multiple approaches for request distribution.
"""
