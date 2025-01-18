import heapq
import sys

class ShortestPathAlgorithms:
    def __init__(self, graph):
        self.graph = graph

    def dijkstra(self, start):
        """
        Dijkstra's Algorithm
        - Finds the shortest path from a single source to all nodes in a graph with non-negative weights.
        - Time Complexity: O((V + E) log V), where V is the number of nodes and E is the number of edges.
        - Space Complexity: O(V) for the distance and priority queue.
        """
        n = len(self.graph)
        dist = {}
        for node in self.graph:
            dist[node] = float('inf')
        
        dist[start] = 0
        min_heap = []
        heapq.heappush(min_heap, (0, start))  # (distance, node)

        while min_heap:
            current_dist, current_node = heapq.heappop(min_heap)

            if current_dist > dist[current_node]:
                continue

            for neighbor, weight in self.graph[current_node]:
                new_dist = current_dist + weight
                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    heapq.heappush(min_heap, (new_dist, neighbor))

        return dist

    def bellman_ford(self, start):
        """
        Bellman-Ford Algorithm
        - Finds the shortest path from a single source to all nodes and detects negative weight cycles.
        - Time Complexity: O(V * E), where V is the number of nodes and E is the number of edges.
        - Space Complexity: O(V) for the distance array.
        """
        n = len(self.graph)
        dist = {node: float('inf') for node in self.graph}
        dist[start] = 0

        # Relax edges V-1 times
        for _ in range(n - 1):
            for node in self.graph:
                for neighbor, weight in self.graph[node]:
                    if dist[node] + weight < dist[neighbor]:
                        dist[neighbor] = dist[node] + weight

        # Check for negative weight cycles
        for node in self.graph:
            for neighbor, weight in self.graph[node]:
                if dist[node] + weight < dist[neighbor]:
                    return "Graph contains a negative weight cycle."

        return dist

    def floyd_warshall(self):
        """
        Floyd-Warshall Algorithm
        - Finds shortest paths between all pairs of nodes.
        - Time Complexity: O(V^3), where V is the number of nodes.
        - Space Complexity: O(V^2) for the distance matrix.
        """
        nodes = list(self.graph.keys())
        n = len(nodes)
        dist = {i: {j: float('inf') for j in nodes} for i in nodes}

        # Initialize distances
        for node in nodes:
            dist[node][node] = 0
            for neighbor, weight in self.graph[node]:
                dist[node][neighbor] = weight

        # Update distances using intermediate nodes
        for k in nodes:
            for i in nodes:
                for j in nodes:
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

        return dist


# Main
if __name__ == "__main__":
    # Example graph (directed, weighted)
    graph = {
        0: [(1, 4), (2, 1)],
        1: [(3, 1)],
        2: [(1, 2), (3, 5)],
        3: []
    }

    # Initialize the class
    algo = ShortestPathAlgorithms(graph)

    # Run Dijkstra's Algorithm
    print("Dijkstra's Algorithm:")
    print(algo.dijkstra(0))  # Shortest paths from node 0

    # Run Bellman-Ford Algorithm
    print("\nBellman-Ford Algorithm:")
    print(algo.bellman_ford(0))  # Shortest paths from node 0

    # Run Floyd-Warshall Algorithm
    print("\nFloyd-Warshall Algorithm:")
    fw_result = algo.floyd_warshall()
    for src in fw_result:
        print(f"From node {src}: {fw_result[src]}")
