"""
Real-World Graph Problems
This module implements solutions to common real-world problems that can be modeled as graph problems.
"""

from collections import defaultdict, deque
import heapq
from typing import List, Tuple, Dict, Set, Optional

class RealWorldGraphProblems:
    
    def __init__(self):
        """Initialize with utility methods for various graph problems"""
        pass
    
    # ==================== COURSE SCHEDULE PROBLEMS (TOPOLOGICAL SORT) ====================
    
    def can_finish_courses(self, num_courses: int, prerequisites: List[List[int]]) -> bool:
        """
        Course Schedule I: Determine if you can finish all courses
        
        Time Complexity: O(V + E)
        Space Complexity: O(V + E)
        
        Args:
            num_courses: Total number of courses
            prerequisites: List of [course, prerequisite] pairs
        
        Returns:
            bool: True if all courses can be finished, False otherwise
        """
        # Build adjacency list and in-degree count
        graph = defaultdict(list)
        in_degree = [0] * num_courses
        
        for course, prereq in prerequisites:
            graph[prereq].append(course)
            in_degree[course] += 1
        
        # Kahn's algorithm for topological sort
        queue = deque()
        for i in range(num_courses):
            if in_degree[i] == 0:
                queue.append(i)
        
        courses_taken = 0
        
        while queue:
            current_course = queue.popleft()
            courses_taken += 1
            
            for next_course in graph[current_course]:
                in_degree[next_course] -= 1
                if in_degree[next_course] == 0:
                    queue.append(next_course)
        
        return courses_taken == num_courses
    
    def find_course_order(self, num_courses: int, prerequisites: List[List[int]]) -> List[int]:
        """
        Course Schedule II: Find a valid order to finish all courses
        
        Args:
            num_courses: Total number of courses
            prerequisites: List of [course, prerequisite] pairs
        
        Returns:
            list: Valid course order, empty list if impossible
        """
        graph = defaultdict(list)
        in_degree = [0] * num_courses
        
        for course, prereq in prerequisites:
            graph[prereq].append(course)
            in_degree[course] += 1
        
        queue = deque()
        for i in range(num_courses):
            if in_degree[i] == 0:
                queue.append(i)
        
        course_order = []
        
        while queue:
            current_course = queue.popleft()
            course_order.append(current_course)
            
            for next_course in graph[current_course]:
                in_degree[next_course] -= 1
                if in_degree[next_course] == 0:
                    queue.append(next_course)
        
        return course_order if len(course_order) == num_courses else []
    
    def minimum_semesters(self, num_courses: int, prerequisites: List[List[int]]) -> int:
        """
        Course Schedule III: Find minimum number of semesters to finish all courses
        
        Args:
            num_courses: Total number of courses
            prerequisites: List of [course, prerequisite] pairs
        
        Returns:
            int: Minimum semesters needed, -1 if impossible
        """
        graph = defaultdict(list)
        in_degree = [0] * num_courses
        
        for course, prereq in prerequisites:
            graph[prereq].append(course)
            in_degree[course] += 1
        
        queue = deque()
        for i in range(num_courses):
            if in_degree[i] == 0:
                queue.append(i)
        
        semesters = 0
        courses_taken = 0
        
        while queue:
            semesters += 1
            semester_size = len(queue)
            
            for _ in range(semester_size):
                current_course = queue.popleft()
                courses_taken += 1
                
                for next_course in graph[current_course]:
                    in_degree[next_course] -= 1
                    if in_degree[next_course] == 0:
                        queue.append(next_course)
        
        return semesters if courses_taken == num_courses else -1
    
    # ==================== ALIEN DICTIONARY (TOPOLOGICAL SORT) ====================
    
    def alien_dictionary(self, words: List[str]) -> str:
        """
        Determine the order of letters in an alien language based on dictionary ordering
        
        Time Complexity: O(C) where C is total content of words
        Space Complexity: O(1) since at most 26 characters
        
        Args:
            words: List of words in alien dictionary order
        
        Returns:
            str: Alien alphabet order, empty string if invalid
        """
        # Build graph of character dependencies
        graph = defaultdict(set)
        in_degree = defaultdict(int)
        chars = set()
        
        # Initialize all characters
        for word in words:
            for char in word:
                chars.add(char)
                in_degree[char] = in_degree.get(char, 0)
        
        # Build edges by comparing adjacent words
        for i in range(len(words) - 1):
            word1, word2 = words[i], words[i + 1]
            min_len = min(len(word1), len(word2))
            
            # Check if word1 is prefix of word2 but longer (invalid)
            if len(word1) > len(word2) and word1[:min_len] == word2[:min_len]:
                return ""
            
            # Find first different character
            for j in range(min_len):
                if word1[j] != word2[j]:
                    if word2[j] not in graph[word1[j]]:
                        graph[word1[j]].add(word2[j])
                        in_degree[word2[j]] += 1
                    break
        
        # Topological sort using Kahn's algorithm
        queue = deque()
        for char in chars:
            if in_degree[char] == 0:
                queue.append(char)
        
        result = []
        
        while queue:
            char = queue.popleft()
            result.append(char)
            
            for neighbor in graph[char]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check if all characters are included (no cycle)
        return ''.join(result) if len(result) == len(chars) else ""
    
    # ==================== WORD LADDER (SHORTEST PATH) ====================
    
    def word_ladder_length(self, begin_word: str, end_word: str, word_list: List[str]) -> int:
        """
        Find length of shortest transformation sequence from begin_word to end_word
        
        Time Complexity: O(M^2 * N) where M is word length, N is word list size
        Space Complexity: O(M^2 * N)
        
        Args:
            begin_word: Starting word
            end_word: Target word
            word_list: List of valid words
        
        Returns:
            int: Length of shortest sequence, 0 if impossible
        """
        if end_word not in word_list:
            return 0
        
        word_set = set(word_list)
        if begin_word in word_set:
            word_set.remove(begin_word)
        
        queue = deque([(begin_word, 1)])
        
        while queue:
            current_word, length = queue.popleft()
            
            # Try changing each character
            for i in range(len(current_word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if c == current_word[i]:
                        continue
                    
                    new_word = current_word[:i] + c + current_word[i+1:]
                    
                    if new_word == end_word:
                        return length + 1
                    
                    if new_word in word_set:
                        word_set.remove(new_word)
                        queue.append((new_word, length + 1))
        
        return 0
    
    def word_ladder_all_paths(self, begin_word: str, end_word: str, word_list: List[str]) -> List[List[str]]:
        """
        Find all shortest transformation sequences from begin_word to end_word
        
        Args:
            begin_word: Starting word
            end_word: Target word  
            word_list: List of valid words
        
        Returns:
            list: All shortest transformation sequences
        """
        if end_word not in word_list:
            return []
        
        word_set = set(word_list)
        if begin_word in word_set:
            word_set.remove(begin_word)
        
        # BFS to find shortest path length and build parent mapping
        queue = deque([begin_word])
        visited = {begin_word}
        parents = defaultdict(list)
        found = False
        
        while queue and not found:
            level_visited = set()
            
            for _ in range(len(queue)):
                current_word = queue.popleft()
                
                for i in range(len(current_word)):
                    for c in 'abcdefghijklmnopqrstuvwxyz':
                        if c == current_word[i]:
                            continue
                        
                        new_word = current_word[:i] + c + current_word[i+1:]
                        
                        if new_word == end_word:
                            parents[new_word].append(current_word)
                            found = True
                        elif new_word in word_set and new_word not in visited:
                            if new_word not in level_visited:
                                level_visited.add(new_word)
                                queue.append(new_word)
                            parents[new_word].append(current_word)
            
            visited.update(level_visited)
        
        # DFS to construct all paths
        result = []
        
        def dfs(word, path):
            if word == begin_word:
                result.append([begin_word] + path[::-1])
                return
            
            for parent in parents[word]:
                dfs(parent, path + [word])
        
        if found:
            dfs(end_word, [])
        
        return result
    
    # ==================== NETWORK DELAY TIME (DIJKSTRA) ====================
    
    def network_delay_time(self, times: List[List[int]], n: int, k: int) -> int:
        """
        Find minimum time for signal to reach all nodes in network
        
        Time Complexity: O((V + E) log V)
        Space Complexity: O(V + E)
        
        Args:
            times: List of [source, target, time] for each edge
            n: Number of nodes
            k: Source node
        
        Returns:
            int: Minimum time to reach all nodes, -1 if impossible
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v, w in times:
            graph[u].append((v, w))
        
        # Dijkstra's algorithm
        distances = {}
        pq = [(0, k)]  # (distance, node)
        
        while pq:
            dist, node = heapq.heappop(pq)
            
            if node in distances:
                continue
            
            distances[node] = dist
            
            for neighbor, time in graph[node]:
                if neighbor not in distances:
                    heapq.heappush(pq, (dist + time, neighbor))
        
        # Check if all nodes are reachable
        if len(distances) != n:
            return -1
        
        return max(distances.values())
    
    # ==================== CHEAPEST FLIGHTS WITH K STOPS (MODIFIED DIJKSTRA/BELLMAN-FORD) ====================
    
    def cheapest_flights_k_stops(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        """
        Find cheapest flight path with at most k stops
        
        Time Complexity: O(E * K)
        Space Complexity: O(V * K)
        
        Args:
            n: Number of cities
            flights: List of [from, to, price] for each flight
            src: Source city
            dst: Destination city
            k: Maximum number of stops
        
        Returns:
            int: Cheapest price, -1 if no path exists
        """
        # Modified Bellman-Ford algorithm
        prices = [float('inf')] * n
        prices[src] = 0
        
        for _ in range(k + 1):
            temp_prices = prices[:]
            
            for from_city, to_city, price in flights:
                if prices[from_city] != float('inf'):
                    temp_prices[to_city] = min(temp_prices[to_city], 
                                             prices[from_city] + price)
            
            prices = temp_prices
        
        return prices[dst] if prices[dst] != float('inf') else -1
    
    def cheapest_flights_dijkstra(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        """
        Alternative solution using modified Dijkstra's algorithm
        
        Args:
            n: Number of cities
            flights: List of [from, to, price] for each flight
            src: Source city
            dst: Destination city
            k: Maximum number of stops
        
        Returns:
            int: Cheapest price, -1 if no path exists
        """
        graph = defaultdict(list)
        for from_city, to_city, price in flights:
            graph[from_city].append((to_city, price))
        
        # Priority queue: (cost, city, stops_used)
        pq = [(0, src, 0)]
        visited = {}  # (city, stops) -> min_cost
        
        while pq:
            cost, city, stops = heapq.heappop(pq)
            
            if city == dst:
                return cost
            
            if stops > k:
                continue
            
            if (city, stops) in visited and visited[(city, stops)] <= cost:
                continue
            
            visited[(city, stops)] = cost
            
            for next_city, price in graph[city]:
                new_cost = cost + price
                new_stops = stops + 1
                
                if new_stops <= k + 1:
                    heapq.heappush(pq, (new_cost, next_city, new_stops))
        
        return -1
    
    # ==================== RECONSTRUCT ITINERARY (EULERIAN PATH) ====================
    
    def find_itinerary(self, tickets: List[List[str]]) -> List[str]:
        """
        Reconstruct travel itinerary from flight tickets (Eulerian path)
        
        Time Complexity: O(E log E) where E is number of tickets
        Space Complexity: O(E)
        
        Args:
            tickets: List of [from, to] flight tickets
        
        Returns:
            list: Lexicographically smallest valid itinerary
        """
        # Build adjacency list with sorted destinations
        graph = defaultdict(list)
        for from_airport, to_airport in tickets:
            graph[from_airport].append(to_airport)
        
        # Sort destinations for lexicographic order
        for airport in graph:
            graph[airport].sort(reverse=True)  # Reverse for stack behavior
        
        # Hierholzer's algorithm for Eulerian path
        stack = ["JFK"]
        result = []
        
        while stack:
            current = stack[-1]
            
            if graph[current]:
                # Take next destination
                next_airport = graph[current].pop()
                stack.append(next_airport)
            else:
                # No more destinations, add to result
                result.append(stack.pop())
        
        return result[::-1]  # Reverse to get correct order
    
    def find_itinerary_dfs(self, tickets: List[List[str]]) -> List[str]:
        """
        Alternative DFS solution for itinerary reconstruction
        
        Args:
            tickets: List of [from, to] flight tickets
        
        Returns:
            list: Valid itinerary
        """
        graph = defaultdict(list)
        for from_airport, to_airport in tickets:
            graph[from_airport].append(to_airport)
        
        # Sort for lexicographic order
        for airport in graph:
            graph[airport].sort()
        
        route = []
        
        def dfs(airport):
            while graph[airport]:
                next_airport = graph[airport].pop(0)
                dfs(next_airport)
            route.append(airport)
        
        dfs("JFK")
        return route[::-1]
    
    # ==================== UTILITY AND HELPER METHODS ====================
    
    def is_valid_word_transformation(self, word1: str, word2: str) -> bool:
        """Check if word1 can be transformed to word2 with one character change"""
        if len(word1) != len(word2):
            return False
        
        diff_count = 0
        for i in range(len(word1)):
            if word1[i] != word2[i]:
                diff_count += 1
                if diff_count > 1:
                    return False
        
        return diff_count == 1
    
    def build_word_graph(self, word_list: List[str]) -> Dict[str, List[str]]:
        """Build adjacency list for word transformation graph"""
        graph = defaultdict(list)
        
        for i in range(len(word_list)):
            for j in range(i + 1, len(word_list)):
                if self.is_valid_word_transformation(word_list[i], word_list[j]):
                    graph[word_list[i]].append(word_list[j])
                    graph[word_list[j]].append(word_list[i])
        
        return graph
    
    def detect_cycle_in_course_dependencies(self, num_courses: int, prerequisites: List[List[int]]) -> bool:
        """Detect if there's a cycle in course dependencies"""
        return not self.can_finish_courses(num_courses, prerequisites)
    
    def shortest_path_in_word_graph(self, start: str, end: str, word_list: List[str]) -> List[str]:
        """Find shortest path between two words using BFS"""
        if end not in word_list:
            return []
        
        word_set = set(word_list)
        if start in word_set:
            word_set.remove(start)
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current_word, path = queue.popleft()
            
            if current_word == end:
                return path
            
            for i in range(len(current_word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if c == current_word[i]:
                        continue
                    
                    new_word = current_word[:i] + c + current_word[i+1:]
                    
                    if new_word in word_set and new_word not in visited:
                        visited.add(new_word)
                        queue.append((new_word, path + [new_word]))
        
        return []


# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Real-World Graph Problems Demo ===\n")
    
    solver = RealWorldGraphProblems()
    
    # Example 1: Course Schedule Problems
    print("1. Course Schedule Problems:")
    
    # Course Schedule I
    num_courses = 4
    prerequisites = [[1, 0], [2, 0], [3, 1], [3, 2]]
    
    can_finish = solver.can_finish_courses(num_courses, prerequisites)
    print(f"Can finish {num_courses} courses with prerequisites {prerequisites}: {can_finish}")
    
    # Course Schedule II
    course_order = solver.find_course_order(num_courses, prerequisites)
    print(f"Valid course order: {course_order}")
    
    # Minimum semesters
    min_semesters = solver.minimum_semesters(num_courses, prerequisites)
    print(f"Minimum semesters needed: {min_semesters}")
    print()
    
    # Example 2: Alien Dictionary
    print("2. Alien Dictionary:")
    
    alien_words = ["wrt", "wrf", "er", "ett", "rftt"]
    alien_order = solver.alien_dictionary(alien_words)
    print(f"Alien dictionary {alien_words}")
    print(f"Alien alphabet order: '{alien_order}'")
    
    # Invalid case
    invalid_words = ["z", "x", "z"]
    invalid_order = solver.alien_dictionary(invalid_words)
    print(f"Invalid dictionary {invalid_words}: '{invalid_order}'")
    print()
    
    # Example 3: Word Ladder
    print("3. Word Ladder:")
    
    begin_word = "hit"
    end_word = "cog"
    word_list = ["hot", "dot", "dog", "lot", "log", "cog"]
    
    ladder_length = solver.word_ladder_length(begin_word, end_word, word_list)
    print(f"Word ladder from '{begin_word}' to '{end_word}': length = {ladder_length}")
    
    all_ladders = solver.word_ladder_all_paths(begin_word, end_word, word_list)
    print(f"All shortest ladders: {all_ladders}")
    print()
    
    # Example 4: Network Delay Time
    print("4. Network Delay Time:")
    
    times = [[2, 1, 1], [2, 3, 1], [3, 4, 1]]
    n = 4
    k = 2
    
    delay_time = solver.network_delay_time(times, n, k)
    print(f"Network with {n} nodes, signal from {k}")
    print(f"Times: {times}")
    print(f"Minimum delay to reach all nodes: {delay_time}")
    print()
    
    # Example 5: Cheapest Flights with K Stops
    print("5. Cheapest Flights with K Stops:")
    
    n = 3
    flights = [[0, 1, 100], [1, 2, 100], [0, 2, 500]]
    src, dst, k = 0, 2, 1
    
    cheapest_price = solver.cheapest_flights_k_stops(n, flights, src, dst, k)
    print(f"Flights: {flights}")
    print(f"Cheapest price from {src} to {dst} with at most {k} stops: {cheapest_price}")
    
    # Alternative method
    cheapest_price_dijkstra = solver.cheapest_flights_dijkstra(n, flights, src, dst, k)
    print(f"Same result with Dijkstra: {cheapest_price_dijkstra}")
    print()
    
    # Example 6: Reconstruct Itinerary
    print("6. Reconstruct Itinerary:")
    
    tickets = [["MUC", "LHR"], ["JFK", "MUC"], ["SFO", "SJC"], ["LHR", "SFO"]]
    itinerary = solver.find_itinerary(tickets)
    print(f"Flight tickets: {tickets}")
    print(f"Reconstructed itinerary: {itinerary}")
    
    # More complex example
    complex_tickets = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
    complex_itinerary = solver.find_itinerary(complex_tickets)
    print(f"Complex tickets: {complex_tickets}")
    print(f"Complex itinerary: {complex_itinerary}")
    print()
    
    # Example 7: Additional Utilities
    print("7. Additional Analysis:")
    
    # Check cycle in course dependencies
    cyclic_prereqs = [[1, 0], [0, 1]]
    has_cycle = solver.detect_cycle_in_course_dependencies(2, cyclic_prereqs)
    print(f"Prerequisites {cyclic_prereqs} have cycle: {has_cycle}")
    
    # Word transformation check
    word1, word2 = "hit", "hot"
    can_transform = solver.is_valid_word_transformation(word1, word2)
    print(f"Can transform '{word1}' to '{word2}': {can_transform}")
    
    # Shortest path in word graph
    shortest_word_path = solver.shortest_path_in_word_graph("hit", "cog", word_list)
    print(f"Shortest word path: {shortest_word_path}")
    
    print("\n=== Demo Complete ===") 