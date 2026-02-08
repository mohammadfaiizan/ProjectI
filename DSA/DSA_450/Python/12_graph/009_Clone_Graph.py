"""
Problem: Clone a Graph
URL: https://leetcode.com/problems/clone-graph/

Problem Statement:
Given a reference to a node in a connected undirected graph, return a deep copy.

Sample Input/Output:
Input: Graph with nodes
Output: Cloned graph
"""


class GraphNode:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


class Solution:
    def Clone_Graph_DFS_Helper(self, node, visited):
        if node in visited:
            return visited[node]
        
        clone_node = GraphNode(node.val)
        visited[node] = clone_node
        
        for neighbor in node.neighbors:
            clone_node.neighbors.append(self.Clone_Graph_DFS_Helper(neighbor, visited))
        
        return clone_node

    def Clone_Graph_DFS(self, node):
        """
        DFS with Hashmap
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        """
        if not node:
            return None
        
        visited = {}
        return self.Clone_Graph_DFS_Helper(node, visited)

    def Clone_Graph_BFS(self, node):
        """
        BFS with Hashmap
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        """
        if not node:
            return None
        
        visited = {}
        from collections import deque
        q = deque([node])
        
        clone_node = GraphNode(node.val)
        visited[node] = clone_node
        
        while q:
            current = q.popleft()
            
            for neighbor in current.neighbors:
                if neighbor not in visited:
                    clone_neighbor = GraphNode(neighbor.val)
                    visited[neighbor] = clone_neighbor
                    q.append(neighbor)
                visited[current].neighbors.append(visited[neighbor])
        
        return clone_node


def Test_Clone_Graph():
    solution = Solution()
    
    print("Test: Clone Graph")
    
    node1 = GraphNode(1)
    node2 = GraphNode(2)
    node3 = GraphNode(3)
    node4 = GraphNode(4)
    
    node1.neighbors = [node2, node4]
    node2.neighbors = [node1, node3]
    node3.neighbors = [node2, node4]
    node4.neighbors = [node1, node3]
    
    cloned1 = solution.Clone_Graph_DFS(node1)
    print("Cloned graph (DFS) - Node values: ", end="")
    from collections import deque
    q = deque([cloned1])
    visited = {cloned1}
    
    while q:
        current = q.popleft()
        print(current.val, end=" ")
        
        for neighbor in current.neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                q.append(neighbor)
    print()
    
    cloned2 = solution.Clone_Graph_BFS(node1)
    print("Cloned graph (BFS) - Node values: ", end="")
    q2 = deque([cloned2])
    visited2 = {cloned2}
    
    while q2:
        current = q2.popleft()
        print(current.val, end=" ")
        
        for neighbor in current.neighbors:
            if neighbor not in visited2:
                visited2.add(neighbor)
                q2.append(neighbor)
    print()


if __name__ == "__main__":
    Test_Clone_Graph()
