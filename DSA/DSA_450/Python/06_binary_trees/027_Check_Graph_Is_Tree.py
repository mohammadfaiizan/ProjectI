"""
Problem: Check Graph Is Tree
URL: https://www.geeksforgeeks.org/check-given-graph-tree/

Problem Statement:
Check if an undirected graph is a tree (no cycle + all connected).

Sample Input/Output:
Input: 
Vertices: 5, Edges: 4
Edges: (0,1), (0,2), (0,3), (1,4)

Output: true
Explanation: Graph is connected and has no cycles.
"""

from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def Build_Tree(vals):
    if not vals or vals[0] == -1:
        return None
    root = TreeNode(vals[0])
    q = deque([root])
    i = 1
    while q and i < len(vals):
        node = q.popleft()
        if i < len(vals) and vals[i] != -1:
            node.left = TreeNode(vals[i])
            q.append(node.left)
        i += 1
        if i < len(vals) and vals[i] != -1:
            node.right = TreeNode(vals[i])
            q.append(node.right)
        i += 1
    return root


def Print_Tree(root):
    if not root:
        return
    q = deque([root])
    result = []
    while q:
        node = q.popleft()
        result.append(str(node.val))
        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)
    print(" ".join(result))


class Solution:
    def Has_Cycle_DFS(self, graph, node, parent, visited):
        visited[node] = True
        for neighbor in graph[node]:
            if not visited[neighbor]:
                if self.Has_Cycle_DFS(graph, neighbor, node, visited):
                    return True
            elif neighbor != parent:
                return True
        return False

    def Is_Tree_DFS(self, graph, V):
        """
        DFS cycle detection + connectivity check: Check for cycles and connectivity
        Time Complexity: O(V+E) where V is vertices and E is edges
        Space Complexity: O(V) for visited array and recursion stack
        """
        visited = [False] * V
        if self.Has_Cycle_DFS(graph, 0, -1, visited):
            return False
        for i in range(V):
            if not visited[i]:
                return False
        return True

    def Is_Tree_BFS(self, graph, V):
        """
        BFS approach: Use BFS to check cycles and connectivity
        Time Complexity: O(V+E) where V is vertices and E is edges
        Space Complexity: O(V) for visited array and queue
        """
        visited = [False] * V
        q = deque([(0, -1)])
        visited[0] = True
        while q:
            node, parent = q.popleft()
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    q.append((neighbor, node))
                elif neighbor != parent:
                    return False
        for i in range(V):
            if not visited[i]:
                return False
        return True


def Test_Check_Graph_Is_Tree():
    solution = Solution()
    
    V1 = 5
    graph1 = [[] for _ in range(V1)]
    graph1[0].extend([1, 2, 3])
    graph1[1].extend([0, 4])
    graph1[2].append(0)
    graph1[3].append(0)
    graph1[4].append(1)
    print("Graph 1 (DFS):", solution.Is_Tree_DFS(graph1, V1))
    print("Graph 1 (BFS):", solution.Is_Tree_BFS(graph1, V1))
    
    V2 = 3
    graph2 = [[] for _ in range(V2)]
    graph2[0].append(1)
    graph2[1].extend([0, 2])
    graph2[2].extend([1, 0])
    graph2[0].append(2)
    print("Graph 2 (DFS):", solution.Is_Tree_DFS(graph2, V2))
    print("Graph 2 (BFS):", solution.Is_Tree_BFS(graph2, V2))


if __name__ == "__main__":
    Test_Check_Graph_Is_Tree()
