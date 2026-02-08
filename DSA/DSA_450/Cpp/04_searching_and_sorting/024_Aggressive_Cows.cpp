/*
 * Problem: Aggressive Cows
 * URL: https://www.spoj.com/problems/AGGRCOW/
 * Problem Statement:
 * Place C cows in N stalls to maximize the minimum distance between any two cows.
 * Binary search on the answer (minimum distance).
 * 
 * Sample Input:
 * 5 3
 * 1 2 8 4 9
 * 
 * Sample Output:
 * 3
 */

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Solve_Binary_Search_On_Answer(vector<int>& stalls, int cows) {
        /*
         * Approach: Binary search on minimum distance. For each distance,
         * check if we can place all cows with at least that distance.
         * Time Complexity: O(n log n + n log(max_pos)) where n = stalls
         * Space Complexity: O(1)
         */
        sort(stalls.begin(), stalls.end());
        int n = stalls.size();
        int left = 0, right = stalls[n - 1] - stalls[0];
        int result = 0;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (Can_Place_Cows(stalls, cows, mid)) {
                result = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return result;
    }
    
private:
    bool Can_Place_Cows(vector<int>& stalls, int cows, int min_distance) {
        int count = 1;
        int last_pos = stalls[0];
        
        for (int i = 1; i < stalls.size(); i++) {
            if (stalls[i] - last_pos >= min_distance) {
                count++;
                last_pos = stalls[i];
                if (count >= cows) {
                    return true;
                }
            }
        }
        
        return count >= cows;
    }
};

void Test_Aggressive_Cows() {
    Solution sol;
    
    vector<int> stalls1 = {1, 2, 8, 4, 9};
    assert(sol.Solve_Binary_Search_On_Answer(stalls1, 3) == 3);
    
    vector<int> stalls2 = {1, 2, 4, 8, 9};
    assert(sol.Solve_Binary_Search_On_Answer(stalls2, 3) == 3);
    
    vector<int> stalls3 = {1, 5, 10};
    assert(sol.Solve_Binary_Search_On_Answer(stalls3, 2) == 9);
    
    vector<int> stalls4 = {1, 2, 3, 4, 5};
    assert(sol.Solve_Binary_Search_On_Answer(stalls4, 3) == 2);
    
    cout << "All test cases passed!" << endl;
}

int main() {
    Test_Aggressive_Cows();
    return 0;
}
