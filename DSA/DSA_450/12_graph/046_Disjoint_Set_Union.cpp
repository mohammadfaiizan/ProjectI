/*
Problem: Disjoint Set Union (Union-Find)
URL: https://www.geeksforgeeks.org/disjoint-set-data-structures/

Problem Statement:
Implement DSU with union by rank and path compression.

Sample Input/Output:
Input: Operations to unite elements and check connectivity
Output: Results of connectivity checks
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    class DSU {
    private:
        vector<int> parent;
        vector<int> rank;
        
    public:
        DSU(int n) {
            parent.resize(n);
            rank.resize(n, 0);
            for (int i = 0; i < n; i++) {
                parent[i] = i;
            }
        }
        
        int Find(int x) {
            if (parent[x] != x) {
                parent[x] = Find(parent[x]);
            }
            return parent[x];
        }
        
        void Unite(int x, int y) {
            int px = Find(x);
            int py = Find(y);
            
            if (px == py) return;
            
            if (rank[px] < rank[py]) {
                parent[px] = py;
            } else if (rank[px] > rank[py]) {
                parent[py] = px;
            } else {
                parent[py] = px;
                rank[px]++;
            }
        }
        
        bool Connected(int x, int y) {
            return Find(x) == Find(y);
        }
    };

    DSU DSU_Rank_Path_Compression(int n) {
        /*
        Union by rank + path compression
        Time Complexity: Near O(1) amortized per operation
        Space Complexity: O(n)
        */
        return DSU(n);
    }
};

void Test_DSU() {
    Solution solution;
    
    cout << "Test Case 1:" << endl;
    Solution::DSU dsu1 = solution.DSU_Rank_Path_Compression(5);
    dsu1.Unite(0, 1);
    dsu1.Unite(2, 3);
    dsu1.Unite(1, 2);
    
    cout << "0 and 3 connected: " << (dsu1.Connected(0, 3) ? "Yes" : "No") << endl;
    cout << "0 and 4 connected: " << (dsu1.Connected(0, 4) ? "Yes" : "No") << endl;
    cout << endl;
    
    cout << "Test Case 2:" << endl;
    Solution::DSU dsu2 = solution.DSU_Rank_Path_Compression(7);
    dsu2.Unite(0, 1);
    dsu2.Unite(1, 2);
    dsu2.Unite(3, 4);
    dsu2.Unite(5, 6);
    dsu2.Unite(2, 3);
    
    cout << "0 and 4 connected: " << (dsu2.Connected(0, 4) ? "Yes" : "No") << endl;
    cout << "0 and 5 connected: " << (dsu2.Connected(0, 5) ? "Yes" : "No") << endl;
    cout << "5 and 6 connected: " << (dsu2.Connected(5, 6) ? "Yes" : "No") << endl;
    cout << endl;
    
    cout << "Test Case 3:" << endl;
    Solution::DSU dsu3 = solution.DSU_Rank_Path_Compression(4);
    dsu3.Unite(0, 1);
    dsu3.Unite(2, 3);
    
    cout << "0 and 1 connected: " << (dsu3.Connected(0, 1) ? "Yes" : "No") << endl;
    cout << "2 and 3 connected: " << (dsu3.Connected(2, 3) ? "Yes" : "No") << endl;
    cout << "0 and 2 connected: " << (dsu3.Connected(0, 2) ? "Yes" : "No") << endl;
    dsu3.Unite(1, 2);
    cout << "After uniting 1 and 2:" << endl;
    cout << "0 and 2 connected: " << (dsu3.Connected(0, 2) ? "Yes" : "No") << endl;
    cout << "0 and 3 connected: " << (dsu3.Connected(0, 3) ? "Yes" : "No") << endl;
}

int main() {
    Test_DSU();
    return 0;
}
