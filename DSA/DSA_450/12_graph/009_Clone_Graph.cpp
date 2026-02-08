/*
Problem: Clone a Graph
URL: https://leetcode.com/problems/clone-graph/

Problem Statement:
Given a reference to a node in a connected undirected graph, return a deep copy.

Sample Input/Output:
Input: Graph with nodes
Output: Cloned graph
*/

#include <bits/stdc++.h>
using namespace std;

class GraphNode {
public:
    int val;
    vector<GraphNode*> neighbors;
    
    GraphNode() {
        val = 0;
        neighbors = vector<GraphNode*>();
    }
    
    GraphNode(int _val) {
        val = _val;
        neighbors = vector<GraphNode*>();
    }
    
    GraphNode(int _val, vector<GraphNode*> _neighbors) {
        val = _val;
        neighbors = _neighbors;
    }
};

class Solution {
public:
    GraphNode* Clone_Graph_DFS_Helper(GraphNode* node, unordered_map<GraphNode*, GraphNode*>& visited) {
        if (visited.find(node) != visited.end()) {
            return visited[node];
        }
        
        GraphNode* cloneNode = new GraphNode(node->val);
        visited[node] = cloneNode;
        
        for (GraphNode* neighbor : node->neighbors) {
            cloneNode->neighbors.push_back(Clone_Graph_DFS_Helper(neighbor, visited));
        }
        
        return cloneNode;
    }

    GraphNode* Clone_Graph_DFS(GraphNode* node) {
        /*
        DFS with Hashmap
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        */
        if (!node) return nullptr;
        
        unordered_map<GraphNode*, GraphNode*> visited;
        return Clone_Graph_DFS_Helper(node, visited);
    }

    GraphNode* Clone_Graph_BFS(GraphNode* node) {
        /*
        BFS with Hashmap
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        */
        if (!node) return nullptr;
        
        unordered_map<GraphNode*, GraphNode*> visited;
        queue<GraphNode*> q;
        
        GraphNode* cloneNode = new GraphNode(node->val);
        visited[node] = cloneNode;
        q.push(node);
        
        while (!q.empty()) {
            GraphNode* current = q.front();
            q.pop();
            
            for (GraphNode* neighbor : current->neighbors) {
                if (visited.find(neighbor) == visited.end()) {
                    GraphNode* cloneNeighbor = new GraphNode(neighbor->val);
                    visited[neighbor] = cloneNeighbor;
                    q.push(neighbor);
                }
                visited[current]->neighbors.push_back(visited[neighbor]);
            }
        }
        
        return cloneNode;
    }
};

void Test_Clone_Graph() {
    Solution solution;
    
    cout << "Test: Clone Graph" << endl;
    
    GraphNode* node1 = new GraphNode(1);
    GraphNode* node2 = new GraphNode(2);
    GraphNode* node3 = new GraphNode(3);
    GraphNode* node4 = new GraphNode(4);
    
    node1->neighbors = {node2, node4};
    node2->neighbors = {node1, node3};
    node3->neighbors = {node2, node4};
    node4->neighbors = {node1, node3};
    
    GraphNode* cloned1 = solution.Clone_Graph_DFS(node1);
    cout << "Cloned graph (DFS) - Node values: ";
    queue<GraphNode*> q;
    unordered_set<GraphNode*> visited;
    q.push(cloned1);
    visited.insert(cloned1);
    
    while (!q.empty()) {
        GraphNode* current = q.front();
        q.pop();
        cout << current->val << " ";
        
        for (GraphNode* neighbor : current->neighbors) {
            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
                q.push(neighbor);
            }
        }
    }
    cout << endl;
    
    GraphNode* cloned2 = solution.Clone_Graph_BFS(node1);
    cout << "Cloned graph (BFS) - Node values: ";
    queue<GraphNode*> q2;
    unordered_set<GraphNode*> visited2;
    q2.push(cloned2);
    visited2.insert(cloned2);
    
    while (!q2.empty()) {
        GraphNode* current = q2.front();
        q2.pop();
        cout << current->val << " ";
        
        for (GraphNode* neighbor : current->neighbors) {
            if (visited2.find(neighbor) == visited2.end()) {
                visited2.insert(neighbor);
                q2.push(neighbor);
            }
        }
    }
    cout << endl;
}

int main() {
    Test_Clone_Graph();
    return 0;
}
