/*
Problem: Find Whether It Is Possible to Finish All Tasks (Course Schedule)
URL: https://leetcode.com/problems/course-schedule/

Problem Statement:
Given numCourses and prerequisites array, determine if all courses can be finished. This is equivalent to checking if the dependency graph has a cycle.

Sample Input/Output:
Input: numCourses=4, prerequisites=[[1,0],[2,1],[3,2]]
Output: true
Input: numCourses=2, prerequisites=[[1,0],[0,1]]
Output: false
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Course_Schedule_DFS(int numCourses, vector<vector<int>>& prerequisites) {
        /*
        Cycle detection with coloring
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        */
        vector<vector<int>> adj(numCourses);
        for (auto& edge : prerequisites) {
            adj[edge[1]].push_back(edge[0]);
        }
        
        vector<int> color(numCourses, 0);
        
        function<bool(int)> hasCycle = [&](int u) -> bool {
            if (color[u] == 1) return true;
            if (color[u] == 2) return false;
            
            color[u] = 1;
            for (int v : adj[u]) {
                if (hasCycle(v)) return true;
            }
            color[u] = 2;
            return false;
        };
        
        for (int i = 0; i < numCourses; i++) {
            if (color[i] == 0 && hasCycle(i)) {
                return false;
            }
        }
        return true;
    }
    
    bool Course_Schedule_BFS_Kahn(int numCourses, vector<vector<int>>& prerequisites) {
        /*
        Kahn's topological sort
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        */
        vector<vector<int>> adj(numCourses);
        vector<int> inDegree(numCourses, 0);
        
        for (auto& edge : prerequisites) {
            adj[edge[1]].push_back(edge[0]);
            inDegree[edge[0]]++;
        }
        
        queue<int> q;
        for (int i = 0; i < numCourses; i++) {
            if (inDegree[i] == 0) {
                q.push(i);
            }
        }
        
        int count = 0;
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            count++;
            
            for (int v : adj[u]) {
                inDegree[v]--;
                if (inDegree[v] == 0) {
                    q.push(v);
                }
            }
        }
        
        return count == numCourses;
    }
};

void Test_Course_Schedule() {
    Solution solution;
    
    cout << "Test Case 1: Valid schedule (no cycle)" << endl;
    int numCourses1 = 4;
    vector<vector<int>> prerequisites1 = {{1,0}, {2,1}, {3,2}};
    cout << "DFS Result: " << (solution.Course_Schedule_DFS(numCourses1, prerequisites1) ? "true" : "false") << endl;
    cout << "BFS Result: " << (solution.Course_Schedule_BFS_Kahn(numCourses1, prerequisites1) ? "true" : "false") << endl;
    
    cout << "\nTest Case 2: Invalid schedule (cycle)" << endl;
    int numCourses2 = 2;
    vector<vector<int>> prerequisites2 = {{1,0}, {0,1}};
    cout << "DFS Result: " << (solution.Course_Schedule_DFS(numCourses2, prerequisites2) ? "true" : "false") << endl;
    cout << "BFS Result: " << (solution.Course_Schedule_BFS_Kahn(numCourses2, prerequisites2) ? "true" : "false") << endl;
    
    cout << "\nTest Case 3: No prerequisites" << endl;
    int numCourses3 = 3;
    vector<vector<int>> prerequisites3 = {};
    cout << "DFS Result: " << (solution.Course_Schedule_DFS(numCourses3, prerequisites3) ? "true" : "false") << endl;
    cout << "BFS Result: " << (solution.Course_Schedule_BFS_Kahn(numCourses3, prerequisites3) ? "true" : "false") << endl;
}

int main() {
    Test_Course_Schedule();
    return 0;
}
