/*
Problem: Minimum Time Taken by Each Job in a DAG
URL: https://www.geeksforgeeks.org/find-the-ordering-of-tasks-from-given-dependencies/

Problem Statement:
Given a DAG of jobs with dependencies, find the minimum time taken by each job. Each job takes 1 unit of time. A job can only start after all its dependencies are completed.

Sample Input/Output:
Input: 10 jobs, dependencies: 1->2, 1->3, 2->4, 3->4, 4->5
Output: Job 1: 1, Job 2: 2, Job 3: 2, Job 4: 3, Job 5: 4
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> Min_Time_Jobs_Topological(int V, vector<int> adj[]) {
        /*
        Topological sort + BFS level computation
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        */
        vector<int> inDegree(V, 0);
        for (int u = 0; u < V; u++) {
            for (int v : adj[u]) {
                inDegree[v]++;
            }
        }
        
        queue<int> q;
        vector<int> time(V, 0);
        
        for (int i = 0; i < V; i++) {
            if (inDegree[i] == 0) {
                q.push(i);
                time[i] = 1;
            }
        }
        
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            
            for (int v : adj[u]) {
                inDegree[v]--;
                if (inDegree[v] == 0) {
                    time[v] = time[u] + 1;
                    q.push(v);
                }
            }
        }
        
        return time;
    }
};

void Test_Min_Time_Jobs() {
    Solution solution;
    
    cout << "Test Case 1: 10 jobs with dependencies" << endl;
    int V = 10;
    vector<int> adj[10];
    adj[0].push_back(1);
    adj[0].push_back(2);
    adj[1].push_back(3);
    adj[2].push_back(3);
    adj[3].push_back(4);
    
    vector<int> time = solution.Min_Time_Jobs_Topological(V, adj);
    for (int i = 0; i < V; i++) {
        cout << "Job " << i << ": " << time[i] << " unit(s)" << endl;
    }
    
    cout << "\nTest Case 2: Linear chain" << endl;
    int V2 = 5;
    vector<int> adj2[5];
    adj2[0].push_back(1);
    adj2[1].push_back(2);
    adj2[2].push_back(3);
    adj2[3].push_back(4);
    
    vector<int> time2 = solution.Min_Time_Jobs_Topological(V2, adj2);
    for (int i = 0; i < V2; i++) {
        cout << "Job " << i << ": " << time2[i] << " unit(s)" << endl;
    }
}

int main() {
    Test_Min_Time_Jobs();
    return 0;
}
