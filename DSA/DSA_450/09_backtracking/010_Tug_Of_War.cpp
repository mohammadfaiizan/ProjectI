/*
Problem: Tug Of War
URL: https://www.geeksforgeeks.org/tug-of-war/

Problem Statement:
Divide set of integers into two subsets of sizes n/2 and ceil(n/2) such that difference of their sums is minimized.

Sample Input/Output:
Input: arr[]={23,45,-34,12,0,98,-99,4,189,-1,4}
Output: Min difference: 1
Explanation: Subsets with minimum difference
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Tug_Of_War_Backtracking(vector<int> &arr) {
        /*
        Backtracking with subset size constraint
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        */
        int n = arr.size();
        int total_sum = accumulate(arr.begin(), arr.end(), 0);
        int min_diff = INT_MAX;
        vector<bool> selected(n, false);
        vector<bool> best_selection(n, false);
        
        function<void(int, int, int)> backtrack = [&](int index, int count, int current_sum) {
            if (count == n / 2) {
                int diff = abs(total_sum - 2 * current_sum);
                if (diff < min_diff) {
                    min_diff = diff;
                    best_selection = selected;
                }
                return;
            }
            
            if (index >= n) return;
            
            selected[index] = true;
            backtrack(index + 1, count + 1, current_sum + arr[index]);
            
            selected[index] = false;
            backtrack(index + 1, count, current_sum);
        };
        
        backtrack(0, 0, 0);
        return min_diff;
    }
};

void Test_Tug_Of_War() {
    Solution solution;
    
    vector<int> arr = {23, 45, -34, 12, 0, 98, -99, 4, 189, -1, 4};
    int min_diff = solution.Tug_Of_War_Backtracking(arr);
    
    cout << "Minimum difference: " << min_diff << endl;
}

int main() {
    Test_Tug_Of_War();
    return 0;
}
