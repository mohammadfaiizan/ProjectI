import matplotlib.pyplot as plt
import networkx as nx

class Graph:
    def __init__(self):
        self.graph = {}

    # Add an edge to the graph
    def add_edge(self, u, v, directed=False):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)
        if not directed:
            if v not in self.graph:
                self.graph[v] = []
            self.graph[v].append(u)

    # Create a graph using an adjacency list
    def create_from_adj_list(self, adj_list):
        self.graph = adj_list

    # Create a graph using an adjacency matrix
    def create_from_adj_matrix(self, matrix):
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] != 0:
                    self.add_edge(i, j, directed=True)

    # BFS Traversal
    def bfs(self, start):
        visited = set()
        queue = [start]
        result = []

        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                visited.add(vertex)
                result.append(vertex)
                for v in self.graph.get(vertex, []):  # Iterate through neighbors
                    if v not in visited:
                        queue.append(v)

        return result

    # DFS Traversal
    def dfs(self, start):
        visited = set()
        stack = [start]
        result = []

        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                visited.add(vertex)
                result.append(vertex)
                stack.extend([v for v in self.graph.get(vertex, []) if v not in visited])

        return result

    # Find all connected components
    def connected_components(self):
        visited = set()
        components = []

        for vertex in self.graph:
            if vertex not in visited:
                component = self.dfs(vertex)
                components.append(component)
                visited.update(component)

        return components

    # Detect a cycle in the graph (for directed graphs)
    def detect_cycle_directed(self):
        visited = set()
        rec_stack = set()

        def dfs_cycle(v):
            visited.add(v)
            rec_stack.add(v)
            for neighbor in self.graph.get(v, []):
                if neighbor not in visited:
                    if dfs_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(v)
            return False

        for node in self.graph:
            if node not in visited:
                if dfs_cycle(node):
                    return True

        return False

    # Detect a cycle in the graph (for undirected graphs)
    def detect_cycle_undirected(self):
        visited = set()

        def dfs_cycle(v, parent):
            visited.add(v)
            for neighbor in self.graph.get(v, []):
                if neighbor not in visited:
                    if dfs_cycle(neighbor, v):
                        return True
                elif neighbor != parent:
                    return True
            return False

        for node in self.graph:
            if node not in visited:
                if dfs_cycle(node, None):
                    return True

        return False

    # Topological Sort
    def topological_sort(self):
        visited = set()
        stack = []

        def dfs(v):
            visited.add(v)
            for neighbor in self.graph.get(v, []):
                if neighbor not in visited:
                    dfs(neighbor)
            stack.append(v)

        for node in self.graph:
            if node not in visited:
                dfs(node)

        return stack[::-1]

    # Shortest Path using BFS (unweighted graph)
    def shortest_path_unweighted(self, start, end):
        visited = set()
        queue = [(start, [start])]

        while queue:
            vertex, path = queue.pop(0)
            if vertex == end:
                return path

            if vertex not in visited:
                visited.add(vertex)
                for neighbor in self.graph.get(vertex, []):
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))

        return None

    # Display the graph
    def display_graph(self):
        for node, edges in self.graph.items():
            print(f"{node} -> {edges}")

    # Visualize the graph using Matplotlib
    def visualize_graph(self):
        edges = []
        print(self.graph)
        for node, neighbors in self.graph.items():
            for neighbor in neighbors:
                edges.append((node, neighbor))

        G = nx.DiGraph() if any(len(v) != len(set(v)) for v in self.graph.values()) else nx.Graph()
        G.add_edges_from(edges)

        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='gray')
        plt.title("Graph Visualization")
        plt.show()

# Example Usage
if __name__ == "__main__":
    GraphInitMethod = "MATRIX"

    if GraphInitMethod == "EDGE":
        # Initialize graph using edge
        g = Graph()
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(1, 2)
        g.add_edge(2, 3, directed=True)
        print("Graph:")
        g.display_graph()
        print("\nVisualizing Graph:")
        g.visualize_graph()

    
    if GraphInitMethod == "ADJLIST":
        # Initialize graph using adjacency list
        adj_list = {
            0: [1, 2],
            1: [2],
            2: [3],
            3: [2]
        }
        g = Graph()
        g.create_from_adj_list(adj_list)
        print("Graph (from adjacency list):")
        g.display_graph()
        print("\nVisualizing Graph (Adjacency List):")
        g.visualize_graph()

    if GraphInitMethod == "MATRIX":
        # Initialize graph using adjacency matrix
        adj_matrix = [
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ]
        g = Graph()
        g.create_from_adj_matrix(adj_matrix)
        print("\nGraph (from adjacency matrix):")
        g.display_graph()
        # print("\nVisualizing Graph (Adjacency Matrix):")
        # g.visualize_graph()
        print("DFS Traversal:", g.dfs(0))

    """
    print("\nBFS Traversal:", g.bfs(0))    
    print("Connected Components:", g.connected_components())
    print("Cycle Detection (Directed):", g.detect_cycle_directed())
    print("Cycle Detection (Undirected):", g.detect_cycle_undirected())
    print("Topological Sort:", g.topological_sort())
    print("Shortest Path (0 to 3):", g.shortest_path_unweighted(0, 3))
    """
