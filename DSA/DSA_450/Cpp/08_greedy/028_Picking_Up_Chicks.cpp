/*
Problem: Picking Up Chicks
URL: https://www.spoj.com/problems/GCJ101BB/

Problem Statement:
N chicks on a road, barn at position B. Each chick has position x[i] and speed v[i]. Time T seconds. Need at least K chicks to reach barn. Find min swaps, or IMPOSSIBLE.

Sample Input/Output:
Input: N=5, K=3, B=10, T=5, positions=[0,2,5,6,7], speeds=[1,1,1,1,4]
Output: 0
Explanation: Greedy right-to-left approach finds minimum swaps needed.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Min_Swaps_Or_Impossible(int N, int K, int B, int T, vector<int>& positions, vector<int>& speeds) {
        /*
        Greedy right-to-left approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        vector<bool> can_reach(N, false);
        
        for (int i = 0; i < N; i++) {
            int distance = B - positions[i];
            if (distance <= T * speeds[i]) {
                can_reach[i] = true;
            }
        }
        
        int reached = 0;
        int swaps = 0;
        int slow_behind = 0;
        
        for (int i = N - 1; i >= 0; i--) {
            if (can_reach[i]) {
                reached++;
                swaps += slow_behind;
            } else {
                slow_behind++;
            }
            
            if (reached >= K) {
                return swaps;
            }
        }
        
        return -1;
    }
};

void Test_Picking_Up_Chicks() {
    Solution solution;
    
    vector<int> positions1 = {0, 2, 5, 6, 7};
    vector<int> speeds1 = {1, 1, 1, 1, 4};
    int result1 = solution.Min_Swaps_Or_Impossible(5, 3, 10, 5, positions1, speeds1);
    if (result1 == -1) {
        cout << "Test 1: IMPOSSIBLE" << endl;
    } else {
        cout << "Test 1: " << result1 << endl;
    }
    
    vector<int> positions2 = {0, 1, 2, 3, 4};
    vector<int> speeds2 = {1, 1, 1, 1, 1};
    int result2 = solution.Min_Swaps_Or_Impossible(5, 3, 10, 5, positions2, speeds2);
    if (result2 == -1) {
        cout << "Test 2: IMPOSSIBLE" << endl;
    } else {
        cout << "Test 2: " << result2 << endl;
    }
}

int main() {
    Test_Picking_Up_Chicks();
    return 0;
}
