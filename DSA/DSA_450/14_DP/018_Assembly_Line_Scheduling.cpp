/*
Problem: Assembly Line Scheduling
URL: https://www.geeksforgeeks.org/assembly-line-scheduling-dp-34/

Problem Statement:
A car factory has two assembly lines, each with n stations. A station is denoted by Si,j where i is either 1 or 2 and indicates the assembly line the station is on, and j indicates the number of the station. The time taken per station is denoted by ai,j. Each station is dedicated to some sort of work like engine fitting, body fitting, painting and so on. So, a car chassis must pass through each of the n stations in order before exiting the factory. The parallel stations of the two assembly lines perform the same task. After it passes through station Si,j, it will continue to station Si,j+1 unless it takes a transfer to the other line. Continuing on the same line incurs no extra cost, but transferring from line i at station j-1 to station j on the other line takes time ti,j. Each assembly line takes an entry time ei and exit time xi which may be different for the two lines. Give an algorithm for computing the minimum time it will take to build a car chassis.

Sample Input/Output:
Input: Two assembly lines with station times and transfer costs
Output: Minimum time to build chassis
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Assembly_Line_DP(vector<vector<int>>& a, vector<vector<int>>& t, vector<int>& e, vector<int>& x, int n) {
        /*
        DP approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        vector<int> T1(n), T2(n);
        T1[0] = e[0] + a[0][0];
        T2[0] = e[1] + a[1][0];
        for (int i = 1; i < n; i++) {
            T1[i] = min(T1[i-1] + a[0][i], T2[i-1] + t[1][i] + a[0][i]);
            T2[i] = min(T2[i-1] + a[1][i], T1[i-1] + t[0][i] + a[1][i]);
        }
        return min(T1[n-1] + x[0], T2[n-1] + x[1]);
    }

    int Assembly_Line_Space(vector<vector<int>>& a, vector<vector<int>>& t, vector<int>& e, vector<int>& x, int n) {
        /*
        Space optimized approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int T1 = e[0] + a[0][0];
        int T2 = e[1] + a[1][0];
        for (int i = 1; i < n; i++) {
            int new_T1 = min(T1 + a[0][i], T2 + t[1][i] + a[0][i]);
            int new_T2 = min(T2 + a[1][i], T1 + t[0][i] + a[1][i]);
            T1 = new_T1;
            T2 = new_T2;
        }
        return min(T1 + x[0], T2 + x[1]);
    }
};

void Test_Assembly_Line() {
    Solution solution;
    int n = 4;
    vector<vector<int>> a = {{4, 5, 3, 2}, {2, 10, 1, 4}};
    vector<vector<int>> t = {{0, 7, 4, 5}, {0, 9, 2, 8}};
    vector<int> e = {10, 12};
    vector<int> x = {18, 7};
    cout << "DP: " << solution.Assembly_Line_DP(a, t, e, x, n) << endl;
    cout << "Space Optimized: " << solution.Assembly_Line_Space(a, t, e, x, n) << endl;
}

int main() {
    Test_Assembly_Line();
    return 0;
}
