/*
 * =============================================================================
 * COMPLETE GRAPH GUIDE - All Representations & Algorithms
 * =============================================================================
 * 
 * This file covers:
 * 1. Graph representations (Adjacency Matrix, Adjacency List, Edge List)
 * 2. Graph traversal algorithms (DFS, BFS)
 * 3. Shortest path algorithms (Dijkstra, Floyd-Warshall)
 * 4. Minimum spanning tree algorithms (Kruskal, Prim)
 * 5. Topological sorting
 * 6. Cycle detection and other graph algorithms
 * 
 * =============================================================================
 */

#include <iostream>
#include <vector>
#include <list>
#include <queue>
#include <stack>
#include <set>
#include <map>
#include <algorithm>
#include <climits>
#include <iomanip>
using namespace std;

// =============================================================================
// ADJACENCY MATRIX REPRESENTATION
// =============================================================================

class AdjacencyMatrix {
private:
    vector<vector<int>> matrix;
    int vertices;
    bool is_directed;

public:
    // Constructor
    AdjacencyMatrix(int v, bool directed = false) 
        : vertices(v), is_directed(directed) {
        matrix.resize(v, vector<int>(v, 0));
        cout << "Adjacency matrix created for " << v << " vertices ("
             << (directed ? "directed" : "undirected") << ")" << endl;
    }
    
    // Add edge
    void addEdge(int src, int dest, int weight = 1) {
        if (src >= 0 && src < vertices && dest >= 0 && dest < vertices) {
            matrix[src][dest] = weight;
            if (!is_directed) {
                matrix[dest][src] = weight;
            }
            cout << "Added edge: " << src << " -> " << dest 
                 << " (weight: " << weight << ")" << endl;
        } else {
            cout << "Invalid vertices for edge: " << src << " -> " << dest << endl;
        }
    }
    
    // Remove edge
    void removeEdge(int src, int dest) {
        if (src >= 0 && src < vertices && dest >= 0 && dest < vertices) {
            matrix[src][dest] = 0;
            if (!is_directed) {
                matrix[dest][src] = 0;
            }
            cout << "Removed edge: " << src << " -> " << dest << endl;
        }
    }
    
    // Check if edge exists
    bool hasEdge(int src, int dest) const {
        if (src >= 0 && src < vertices && dest >= 0 && dest < vertices) {
            return matrix[src][dest] != 0;
        }
        return false;
    }
    
    // Get edge weight
    int getEdgeWeight(int src, int dest) const {
        if (hasEdge(src, dest)) {
            return matrix[src][dest];
        }
        return 0;
    }
    
    // Get neighbors
    vector<int> getNeighbors(int vertex) const {
        vector<int> neighbors;
        if (vertex >= 0 && vertex < vertices) {
            for (int i = 0; i < vertices; i++) {
                if (matrix[vertex][i] != 0) {
                    neighbors.push_back(i);
                }
            }
        }
        return neighbors;
    }
    
    // Display matrix
    void display() const {
        cout << "Adjacency Matrix:" << endl;
        cout << "   ";
        for (int i = 0; i < vertices; i++) {
            cout << setw(3) << i;
        }
        cout << endl;
        
        for (int i = 0; i < vertices; i++) {
            cout << setw(2) << i << " ";
            for (int j = 0; j < vertices; j++) {
                cout << setw(3) << matrix[i][j];
            }
            cout << endl;
        }
    }
    
    // DFS traversal
    void dfs(int start) const {
        vector<bool> visited(vertices, false);
        cout << "DFS from vertex " << start << ": ";
        dfsHelper(start, visited);
        cout << endl;
    }
    
    void dfsHelper(int vertex, vector<bool>& visited) const {
        visited[vertex] = true;
        cout << vertex << " ";
        
        for (int i = 0; i < vertices; i++) {
            if (matrix[vertex][i] != 0 && !visited[i]) {
                dfsHelper(i, visited);
            }
        }
    }
    
    // BFS traversal
    void bfs(int start) const {
        vector<bool> visited(vertices, false);
        queue<int> q;
        
        visited[start] = true;
        q.push(start);
        
        cout << "BFS from vertex " << start << ": ";
        
        while (!q.empty()) {
            int current = q.front();
            q.pop();
            cout << current << " ";
            
            for (int i = 0; i < vertices; i++) {
                if (matrix[current][i] != 0 && !visited[i]) {
                    visited[i] = true;
                    q.push(i);
                }
            }
        }
        cout << endl;
    }
    
    // Get number of vertices
    int getVertices() const { return vertices; }
    
    // Get matrix for external algorithms
    const vector<vector<int>>& getMatrix() const { return matrix; }
};

// =============================================================================
// ADJACENCY LIST REPRESENTATION
// =============================================================================

class AdjacencyList {
private:
    vector<list<pair<int, int>>> adj_list; // pair<vertex, weight>
    int vertices;
    bool is_directed;

public:
    // Constructor
    AdjacencyList(int v, bool directed = false) 
        : vertices(v), is_directed(directed) {
        adj_list.resize(v);
        cout << "Adjacency list created for " << v << " vertices ("
             << (directed ? "directed" : "undirected") << ")" << endl;
    }
    
    // Add edge
    void addEdge(int src, int dest, int weight = 1) {
        if (src >= 0 && src < vertices && dest >= 0 && dest < vertices) {
            adj_list[src].push_back({dest, weight});
            if (!is_directed) {
                adj_list[dest].push_back({src, weight});
            }
            cout << "Added edge: " << src << " -> " << dest 
                 << " (weight: " << weight << ")" << endl;
        } else {
            cout << "Invalid vertices for edge: " << src << " -> " << dest << endl;
        }
    }
    
    // Remove edge
    void removeEdge(int src, int dest) {
        if (src >= 0 && src < vertices && dest >= 0 && dest < vertices) {
            adj_list[src].remove_if([dest](const pair<int, int>& p) {
                return p.first == dest;
            });
            
            if (!is_directed) {
                adj_list[dest].remove_if([src](const pair<int, int>& p) {
                    return p.first == src;
                });
            }
            cout << "Removed edge: " << src << " -> " << dest << endl;
        }
    }
    
    // Check if edge exists
    bool hasEdge(int src, int dest) const {
        if (src >= 0 && src < vertices && dest >= 0 && dest < vertices) {
            for (const auto& neighbor : adj_list[src]) {
                if (neighbor.first == dest) {
                    return true;
                }
            }
        }
        return false;
    }
    
    // Get neighbors
    vector<pair<int, int>> getNeighbors(int vertex) const {
        vector<pair<int, int>> neighbors;
        if (vertex >= 0 && vertex < vertices) {
            for (const auto& neighbor : adj_list[vertex]) {
                neighbors.push_back(neighbor);
            }
        }
        return neighbors;
    }
    
    // Display list
    void display() const {
        cout << "Adjacency List:" << endl;
        for (int i = 0; i < vertices; i++) {
            cout << "Vertex " << i << ": ";
            for (const auto& neighbor : adj_list[i]) {
                cout << "(" << neighbor.first << "," << neighbor.second << ") ";
            }
            cout << endl;
        }
    }
    
    // DFS traversal
    void dfs(int start) const {
        vector<bool> visited(vertices, false);
        cout << "DFS from vertex " << start << ": ";
        dfsHelper(start, visited);
        cout << endl;
    }
    
    void dfsHelper(int vertex, vector<bool>& visited) const {
        visited[vertex] = true;
        cout << vertex << " ";
        
        for (const auto& neighbor : adj_list[vertex]) {
            if (!visited[neighbor.first]) {
                dfsHelper(neighbor.first, visited);
            }
        }
    }
    
    // BFS traversal
    void bfs(int start) const {
        vector<bool> visited(vertices, false);
        queue<int> q;
        
        visited[start] = true;
        q.push(start);
        
        cout << "BFS from vertex " << start << ": ";
        
        while (!q.empty()) {
            int current = q.front();
            q.pop();
            cout << current << " ";
            
            for (const auto& neighbor : adj_list[current]) {
                if (!visited[neighbor.first]) {
                    visited[neighbor.first] = true;
                    q.push(neighbor.first);
                }
            }
        }
        cout << endl;
    }
    
    // Topological sort (for directed graphs)
    void topologicalSort() const {
        if (!is_directed) {
            cout << "Topological sort only applies to directed graphs" << endl;
            return;
        }
        
        vector<int> in_degree(vertices, 0);
        
        // Calculate in-degrees
        for (int i = 0; i < vertices; i++) {
            for (const auto& neighbor : adj_list[i]) {
                in_degree[neighbor.first]++;
            }
        }
        
        queue<int> q;
        for (int i = 0; i < vertices; i++) {
            if (in_degree[i] == 0) {
                q.push(i);
            }
        }
        
        cout << "Topological sort: ";
        while (!q.empty()) {
            int current = q.front();
            q.pop();
            cout << current << " ";
            
            for (const auto& neighbor : adj_list[current]) {
                in_degree[neighbor.first]--;
                if (in_degree[neighbor.first] == 0) {
                    q.push(neighbor.first);
                }
            }
        }
        cout << endl;
    }
    
    // Detect cycle in directed graph
    bool hasCycleDirected() const {
        if (!is_directed) {
            cout << "This cycle detection is for directed graphs" << endl;
            return false;
        }
        
        vector<int> state(vertices, 0); // 0: unvisited, 1: visiting, 2: visited
        
        for (int i = 0; i < vertices; i++) {
            if (state[i] == 0) {
                if (hasCycleDirectedHelper(i, state)) {
                    return true;
                }
            }
        }
        return false;
    }
    
    bool hasCycleDirectedHelper(int vertex, vector<int>& state) const {
        state[vertex] = 1; // visiting
        
        for (const auto& neighbor : adj_list[vertex]) {
            if (state[neighbor.first] == 1) {
                return true; // back edge found
            }
            if (state[neighbor.first] == 0 && hasCycleDirectedHelper(neighbor.first, state)) {
                return true;
            }
        }
        
        state[vertex] = 2; // visited
        return false;
    }
    
    // Get number of vertices
    int getVertices() const { return vertices; }
    
    // Get adjacency list for external algorithms
    const vector<list<pair<int, int>>>& getList() const { return adj_list; }
};

// =============================================================================
// EDGE LIST REPRESENTATION
// =============================================================================

struct Edge {
    int src, dest, weight;
    
    Edge(int s, int d, int w) : src(s), dest(d), weight(w) {}
    
    bool operator<(const Edge& other) const {
        return weight < other.weight;
    }
};

class EdgeList {
private:
    vector<Edge> edges;
    int vertices;
    bool is_directed;

public:
    // Constructor
    EdgeList(int v, bool directed = false) 
        : vertices(v), is_directed(directed) {
        cout << "Edge list created for " << v << " vertices ("
             << (directed ? "directed" : "undirected") << ")" << endl;
    }
    
    // Add edge
    void addEdge(int src, int dest, int weight = 1) {
        if (src >= 0 && src < vertices && dest >= 0 && dest < vertices) {
            edges.emplace_back(src, dest, weight);
            if (!is_directed) {
                edges.emplace_back(dest, src, weight);
            }
            cout << "Added edge: " << src << " -> " << dest 
                 << " (weight: " << weight << ")" << endl;
        } else {
            cout << "Invalid vertices for edge: " << src << " -> " << dest << endl;
        }
    }
    
    // Display edges
    void display() const {
        cout << "Edge List:" << endl;
        for (const auto& edge : edges) {
            cout << "Edge: " << edge.src << " -> " << edge.dest 
                 << " (weight: " << edge.weight << ")" << endl;
        }
    }
    
    // Get edges
    const vector<Edge>& getEdges() const { return edges; }
    
    // Get number of vertices
    int getVertices() const { return vertices; }
};

// =============================================================================
// GRAPH ALGORITHMS
// =============================================================================

class GraphAlgorithms {
public:
    // Dijkstra's shortest path algorithm
    static void dijkstra(const AdjacencyList& graph, int start) {
        int vertices = graph.getVertices();
        vector<int> dist(vertices, INT_MAX);
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
        
        dist[start] = 0;
        pq.push({0, start});
        
        cout << "Dijkstra's shortest paths from vertex " << start << ":" << endl;
        
        while (!pq.empty()) {
            int u = pq.top().second;
            pq.pop();
            
            auto neighbors = graph.getNeighbors(u);
            for (const auto& neighbor : neighbors) {
                int v = neighbor.first;
                int weight = neighbor.second;
                
                if (dist[u] + weight < dist[v]) {
                    dist[v] = dist[u] + weight;
                    pq.push({dist[v], v});
                }
            }
        }
        
        for (int i = 0; i < vertices; i++) {
            cout << "Distance to vertex " << i << ": ";
            if (dist[i] == INT_MAX) {
                cout << "INF";
            } else {
                cout << dist[i];
            }
            cout << endl;
        }
    }
    
    // Floyd-Warshall algorithm
    static void floydWarshall(const AdjacencyMatrix& graph) {
        int vertices = graph.getVertices();
        vector<vector<int>> dist = graph.getMatrix();
        
        // Initialize distances
        for (int i = 0; i < vertices; i++) {
            for (int j = 0; j < vertices; j++) {
                if (i == j) {
                    dist[i][j] = 0;
                } else if (dist[i][j] == 0) {
                    dist[i][j] = INT_MAX;
                }
            }
        }
        
        // Floyd-Warshall algorithm
        for (int k = 0; k < vertices; k++) {
            for (int i = 0; i < vertices; i++) {
                for (int j = 0; j < vertices; j++) {
                    if (dist[i][k] != INT_MAX && dist[k][j] != INT_MAX) {
                        dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
                    }
                }
            }
        }
        
        cout << "Floyd-Warshall all-pairs shortest paths:" << endl;
        cout << "   ";
        for (int i = 0; i < vertices; i++) {
            cout << setw(4) << i;
        }
        cout << endl;
        
        for (int i = 0; i < vertices; i++) {
            cout << setw(2) << i << " ";
            for (int j = 0; j < vertices; j++) {
                if (dist[i][j] == INT_MAX) {
                    cout << setw(4) << "INF";
                } else {
                    cout << setw(4) << dist[i][j];
                }
            }
            cout << endl;
        }
    }
    
    // Disjoint Set Union (Union-Find)
    class DSU {
    private:
        vector<int> parent, rank;
    
    public:
        DSU(int n) : parent(n), rank(n, 0) {
            for (int i = 0; i < n; i++) {
                parent[i] = i;
            }
        }
        
        int find(int x) {
            if (parent[x] != x) {
                parent[x] = find(parent[x]);
            }
            return parent[x];
        }
        
        bool unite(int x, int y) {
            int px = find(x), py = find(y);
            if (px == py) return false;
            
            if (rank[px] < rank[py]) {
                parent[px] = py;
            } else if (rank[px] > rank[py]) {
                parent[py] = px;
            } else {
                parent[py] = px;
                rank[px]++;
            }
            return true;
        }
    };
    
    // Kruskal's Minimum Spanning Tree
    static void kruskalMST(EdgeList& edge_list) {
        int vertices = edge_list.getVertices();
        vector<Edge> edges = edge_list.getEdges();
        
        // Sort edges by weight
        sort(edges.begin(), edges.end());
        
        DSU dsu(vertices);
        vector<Edge> mst;
        int total_weight = 0;
        
        cout << "Kruskal's Minimum Spanning Tree:" << endl;
        
        for (const auto& edge : edges) {
            if (dsu.unite(edge.src, edge.dest)) {
                mst.push_back(edge);
                total_weight += edge.weight;
                cout << "Added edge: " << edge.src << " -> " << edge.dest 
                     << " (weight: " << edge.weight << ")" << endl;
                
                if (mst.size() == vertices - 1) {
                    break;
                }
            }
        }
        
        cout << "Total MST weight: " << total_weight << endl;
    }
    
    // Prim's Minimum Spanning Tree
    static void primMST(const AdjacencyList& graph, int start) {
        int vertices = graph.getVertices();
        vector<bool> in_mst(vertices, false);
        vector<int> key(vertices, INT_MAX);
        vector<int> parent(vertices, -1);
        
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
        
        key[start] = 0;
        pq.push({0, start});
        
        cout << "Prim's Minimum Spanning Tree starting from vertex " << start << ":" << endl;
        
        int total_weight = 0;
        
        while (!pq.empty()) {
            int u = pq.top().second;
            pq.pop();
            
            if (in_mst[u]) continue;
            in_mst[u] = true;
            
            if (parent[u] != -1) {
                cout << "Added edge: " << parent[u] << " -> " << u 
                     << " (weight: " << key[u] << ")" << endl;
                total_weight += key[u];
            }
            
            auto neighbors = graph.getNeighbors(u);
            for (const auto& neighbor : neighbors) {
                int v = neighbor.first;
                int weight = neighbor.second;
                
                if (!in_mst[v] && weight < key[v]) {
                    key[v] = weight;
                    parent[v] = u;
                    pq.push({key[v], v});
                }
            }
        }
        
        cout << "Total MST weight: " << total_weight << endl;
    }
    
    // Check if graph is connected (for undirected graphs)
    static bool isConnected(const AdjacencyList& graph) {
        int vertices = graph.getVertices();
        vector<bool> visited(vertices, false);
        
        // Start DFS from vertex 0
        stack<int> st;
        st.push(0);
        visited[0] = true;
        int visited_count = 1;
        
        while (!st.empty()) {
            int current = st.top();
            st.pop();
            
            auto neighbors = graph.getNeighbors(current);
            for (const auto& neighbor : neighbors) {
                if (!visited[neighbor.first]) {
                    visited[neighbor.first] = true;
                    st.push(neighbor.first);
                    visited_count++;
                }
            }
        }
        
        return visited_count == vertices;
    }
    
    // Find strongly connected components (Kosaraju's algorithm)
    static void findSCCs(const AdjacencyList& graph) {
        int vertices = graph.getVertices();
        vector<bool> visited(vertices, false);
        stack<int> finish_stack;
        
        // Step 1: Fill vertices in stack according to their finishing times
        for (int i = 0; i < vertices; i++) {
            if (!visited[i]) {
                fillOrder(graph, i, visited, finish_stack);
            }
        }
        
        // Step 2: Create transpose graph
        AdjacencyList transpose(vertices, true);
        for (int i = 0; i < vertices; i++) {
            auto neighbors = graph.getNeighbors(i);
            for (const auto& neighbor : neighbors) {
                transpose.addEdge(neighbor.first, i);
            }
        }
        
        // Step 3: Process vertices in order defined by stack
        fill(visited.begin(), visited.end(), false);
        int scc_count = 0;
        
        cout << "Strongly Connected Components:" << endl;
        while (!finish_stack.empty()) {
            int v = finish_stack.top();
            finish_stack.pop();
            
            if (!visited[v]) {
                cout << "SCC " << ++scc_count << ": ";
                printSCC(transpose, v, visited);
                cout << endl;
            }
        }
    }
    
    static void fillOrder(const AdjacencyList& graph, int v, vector<bool>& visited, stack<int>& finish_stack) {
        visited[v] = true;
        
        auto neighbors = graph.getNeighbors(v);
        for (const auto& neighbor : neighbors) {
            if (!visited[neighbor.first]) {
                fillOrder(graph, neighbor.first, visited, finish_stack);
            }
        }
        
        finish_stack.push(v);
    }
    
    static void printSCC(const AdjacencyList& graph, int v, vector<bool>& visited) {
        visited[v] = true;
        cout << v << " ";
        
        auto neighbors = graph.getNeighbors(v);
        for (const auto& neighbor : neighbors) {
            if (!visited[neighbor.first]) {
                printSCC(graph, neighbor.first, visited);
            }
        }
    }
};

// =============================================================================
// DEMONSTRATION FUNCTIONS
// =============================================================================

void demonstrate_adjacency_matrix() {
    cout << "\n=== ADJACENCY MATRIX DEMONSTRATION ===" << endl;
    
    AdjacencyMatrix graph(5, false);
    
    // Add edges
    graph.addEdge(0, 1, 2);
    graph.addEdge(0, 3, 6);
    graph.addEdge(1, 2, 3);
    graph.addEdge(1, 3, 8);
    graph.addEdge(1, 4, 5);
    graph.addEdge(2, 4, 7);
    graph.addEdge(3, 4, 9);
    
    graph.display();
    
    // Traversals
    cout << "\n--- Graph Traversals ---" << endl;
    graph.dfs(0);
    graph.bfs(0);
    
    // Edge operations
    cout << "\n--- Edge Operations ---" << endl;
    cout << "Edge 0->1 exists: " << (graph.hasEdge(0, 1) ? "Yes" : "No") << endl;
    cout << "Edge 0->2 exists: " << (graph.hasEdge(0, 2) ? "Yes" : "No") << endl;
    cout << "Weight of edge 0->1: " << graph.getEdgeWeight(0, 1) << endl;
    
    auto neighbors = graph.getNeighbors(1);
    cout << "Neighbors of vertex 1: ";
    for (int neighbor : neighbors) {
        cout << neighbor << " ";
    }
    cout << endl;
}

void demonstrate_adjacency_list() {
    cout << "\n=== ADJACENCY LIST DEMONSTRATION ===" << endl;
    
    AdjacencyList graph(5, false);
    
    // Add edges
    graph.addEdge(0, 1, 2);
    graph.addEdge(0, 3, 6);
    graph.addEdge(1, 2, 3);
    graph.addEdge(1, 3, 8);
    graph.addEdge(1, 4, 5);
    graph.addEdge(2, 4, 7);
    graph.addEdge(3, 4, 9);
    
    graph.display();
    
    // Traversals
    cout << "\n--- Graph Traversals ---" << endl;
    graph.dfs(0);
    graph.bfs(0);
    
    // Directed graph operations
    cout << "\n--- Directed Graph Operations ---" << endl;
    AdjacencyList directed_graph(4, true);
    directed_graph.addEdge(0, 1);
    directed_graph.addEdge(1, 2);
    directed_graph.addEdge(2, 3);
    directed_graph.addEdge(3, 1);
    
    directed_graph.display();
    directed_graph.topologicalSort();
    
    cout << "Has cycle: " << (directed_graph.hasCycleDirected() ? "Yes" : "No") << endl;
}

void demonstrate_edge_list() {
    cout << "\n=== EDGE LIST DEMONSTRATION ===" << endl;
    
    EdgeList graph(5, false);
    
    // Add edges
    graph.addEdge(0, 1, 2);
    graph.addEdge(0, 3, 6);
    graph.addEdge(1, 2, 3);
    graph.addEdge(1, 3, 8);
    graph.addEdge(1, 4, 5);
    graph.addEdge(2, 4, 7);
    graph.addEdge(3, 4, 9);
    
    graph.display();
}

void demonstrate_graph_algorithms() {
    cout << "\n=== GRAPH ALGORITHMS DEMONSTRATION ===" << endl;
    
    // Dijkstra's algorithm
    cout << "\n--- Dijkstra's Algorithm ---" << endl;
    AdjacencyList dijkstra_graph(5, true);
    dijkstra_graph.addEdge(0, 1, 10);
    dijkstra_graph.addEdge(0, 4, 5);
    dijkstra_graph.addEdge(1, 2, 1);
    dijkstra_graph.addEdge(1, 4, 2);
    dijkstra_graph.addEdge(2, 3, 4);
    dijkstra_graph.addEdge(3, 2, 6);
    dijkstra_graph.addEdge(3, 0, 7);
    dijkstra_graph.addEdge(4, 1, 3);
    dijkstra_graph.addEdge(4, 2, 9);
    dijkstra_graph.addEdge(4, 3, 2);
    
    GraphAlgorithms::dijkstra(dijkstra_graph, 0);
    
    // Floyd-Warshall algorithm
    cout << "\n--- Floyd-Warshall Algorithm ---" << endl;
    AdjacencyMatrix floyd_graph(4, true);
    floyd_graph.addEdge(0, 1, 5);
    floyd_graph.addEdge(0, 3, 10);
    floyd_graph.addEdge(1, 2, 3);
    floyd_graph.addEdge(2, 3, 1);
    
    GraphAlgorithms::floydWarshall(floyd_graph);
    
    // Minimum Spanning Tree algorithms
    cout << "\n--- Minimum Spanning Tree ---" << endl;
    EdgeList mst_edge_list(4, false);
    mst_edge_list.addEdge(0, 1, 10);
    mst_edge_list.addEdge(0, 2, 6);
    mst_edge_list.addEdge(0, 3, 5);
    mst_edge_list.addEdge(1, 3, 15);
    mst_edge_list.addEdge(2, 3, 4);
    
    GraphAlgorithms::kruskalMST(mst_edge_list);
    
    AdjacencyList mst_adj_list(4, false);
    mst_adj_list.addEdge(0, 1, 10);
    mst_adj_list.addEdge(0, 2, 6);
    mst_adj_list.addEdge(0, 3, 5);
    mst_adj_list.addEdge(1, 3, 15);
    mst_adj_list.addEdge(2, 3, 4);
    
    GraphAlgorithms::primMST(mst_adj_list, 0);
    
    // Graph connectivity
    cout << "\n--- Graph Connectivity ---" << endl;
    cout << "Graph is connected: " << (GraphAlgorithms::isConnected(mst_adj_list) ? "Yes" : "No") << endl;
    
    // Strongly Connected Components
    cout << "\n--- Strongly Connected Components ---" << endl;
    AdjacencyList scc_graph(5, true);
    scc_graph.addEdge(1, 0);
    scc_graph.addEdge(0, 2);
    scc_graph.addEdge(2, 1);
    scc_graph.addEdge(0, 3);
    scc_graph.addEdge(3, 4);
    
    GraphAlgorithms::findSCCs(scc_graph);
}

// =============================================================================
// MAIN FUNCTION
// =============================================================================

int main() {
    cout << "=== COMPLETE GRAPH GUIDE ===" << endl;
    
    demonstrate_adjacency_matrix();
    demonstrate_adjacency_list();
    demonstrate_edge_list();
    demonstrate_graph_algorithms();
    
    cout << "\n=== SUMMARY ===" << endl;
    cout << "1. Adjacency Matrix: O(V²) space, O(1) edge queries" << endl;
    cout << "2. Adjacency List: O(V+E) space, efficient for sparse graphs" << endl;
    cout << "3. Edge List: Simple representation, good for MST algorithms" << endl;
    cout << "4. Traversals: DFS (stack/recursion), BFS (queue)" << endl;
    cout << "5. Shortest Paths: Dijkstra (single source), Floyd-Warshall (all pairs)" << endl;
    cout << "6. MST: Kruskal (edge-based), Prim (vertex-based)" << endl;
    cout << "7. Applications: Social networks, routing, scheduling" << endl;
    
    return 0;
} 