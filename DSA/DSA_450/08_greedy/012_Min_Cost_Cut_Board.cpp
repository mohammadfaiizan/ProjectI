/*
Problem: Minimum Cost to Cut Board into Squares
URL: https://www.geeksforgeeks.org/minimum-cost-cut-board-squares/

Problem Statement:
A board of length m and width n is given, the task is to break this board into m*n squares such that cost of breaking is minimum. The cutting cost for each edge will be given. In short, we need to choose such a sequence of cutting such that cost is minimized.

Sample Input/Output:
Input: m = 6, n = 4, X[] = {2, 1, 3, 1, 4}, Y[] = {4, 1, 2}
Output: 42
Explanation: Cut horizontally first (cost 4*6=24), then vertically (cost 1*4+2*4+1*4+3*4=28). Total = 52. Better: Cut vertical first.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Min_Cost_Cut_Board_Sort_Greedy(int m, int n, vector<int>& X, vector<int>& Y) {
        /*
        Sort cuts in descending order, greedily make expensive cuts first when more pieces exist
        Time Complexity: O((m+n) log(m+n))
        Space Complexity: O(1)
        */
        sort(X.begin(), X.end(), greater<int>());
        sort(Y.begin(), Y.end(), greater<int>());
        
        int horizontal_pieces = 1;
        int vertical_pieces = 1;
        int cost = 0;
        int i = 0;
        int j = 0;
        
        while (i < X.size() && j < Y.size()) {
            if (X[i] > Y[j]) {
                cost += X[i] * vertical_pieces;
                horizontal_pieces++;
                i++;
            } else {
                cost += Y[j] * horizontal_pieces;
                vertical_pieces++;
                j++;
            }
        }
        
        while (i < X.size()) {
            cost += X[i] * vertical_pieces;
            i++;
        }
        
        while (j < Y.size()) {
            cost += Y[j] * horizontal_pieces;
            j++;
        }
        
        return cost;
    }
};

void Test_Min_Cost_Cut_Board() {
    Solution solution;
    int m = 6, n = 4;
    vector<int> X = {2, 1, 3, 1, 4};
    vector<int> Y = {4, 1, 2};
    cout << "Minimum cost: " << solution.Min_Cost_Cut_Board_Sort_Greedy(m, n, X, Y) << endl;
}

int main() {
    Test_Min_Cost_Cut_Board();
    return 0;
}
