/*
 * Problem: Bishu and Soldiers
 * URL: https://www.hackerearth.com/practice/algorithms/searching/binary-search/practice-problems/algorithm/bishu-and-soldiers/
 * Problem Statement:
 * Bishu is fighting with soldiers. Each soldier has a power level.
 * In each round, Bishu has a power level. Find how many soldiers Bishu can defeat
 * (soldiers with power <= Bishu's power) and the sum of their powers.
 * 
 * Sample Input:
 * 7
 * 1 2 3 4 5 6 7
 * 3
 * 3
 * 10
 * 2
 * 
 * Sample Output:
 * 3 6
 * 7 28
 * 2 3
 */

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<pair<int, int>> Solve_Binary_Search(vector<int>& soldiers, vector<int>& rounds) {
        /*
         * Approach: Sort soldiers, then for each round use binary search (upper_bound)
         * to find count of soldiers with power <= round power, and sum their powers.
         * Time Complexity: O(n log n + q log n) where n = soldiers, q = rounds
         * Space Complexity: O(n) for sorted array
         */
        sort(soldiers.begin(), soldiers.end());
        vector<int> prefix_sum(soldiers.size() + 1, 0);
        for (int i = 0; i < soldiers.size(); i++) {
            prefix_sum[i + 1] = prefix_sum[i] + soldiers[i];
        }
        
        vector<pair<int, int>> result;
        for (int power : rounds) {
            int idx = upper_bound(soldiers.begin(), soldiers.end(), power) - soldiers.begin();
            int count = idx;
            int sum = prefix_sum[idx];
            result.push_back({count, sum});
        }
        return result;
    }
    
    vector<pair<int, int>> Solve_Prefix_Sum_Binary_Search(vector<int>& soldiers, vector<int>& rounds) {
        /*
         * Approach: Sort soldiers, build prefix sum array, then binary search for each round.
         * Time Complexity: O(n log n + q log n)
         * Space Complexity: O(n) for prefix sum
         */
        sort(soldiers.begin(), soldiers.end());
        vector<int> prefix_sum(soldiers.size() + 1, 0);
        for (int i = 0; i < soldiers.size(); i++) {
            prefix_sum[i + 1] = prefix_sum[i] + soldiers[i];
        }
        
        vector<pair<int, int>> result;
        for (int power : rounds) {
            int idx = upper_bound(soldiers.begin(), soldiers.end(), power) - soldiers.begin();
            result.push_back({idx, prefix_sum[idx]});
        }
        return result;
    }
};

void Test_Bishu_And_Soldiers() {
    Solution sol;
    
    vector<int> soldiers1 = {1, 2, 3, 4, 5, 6, 7};
    vector<int> rounds1 = {3, 10, 2};
    vector<pair<int, int>> result1 = sol.Solve_Binary_Search(soldiers1, rounds1);
    assert(result1[0].first == 3 && result1[0].second == 6);
    assert(result1[1].first == 7 && result1[1].second == 28);
    assert(result1[2].first == 2 && result1[2].second == 3);
    
    vector<int> soldiers2 = {5, 3, 1, 4, 2};
    vector<int> rounds2 = {3, 6};
    vector<pair<int, int>> result2 = sol.Solve_Prefix_Sum_Binary_Search(soldiers2, rounds2);
    assert(result2[0].first == 3 && result2[0].second == 6);
    assert(result2[1].first == 5 && result2[1].second == 15);
    
    cout << "All test cases passed!" << endl;
}

int main() {
    Test_Bishu_And_Soldiers();
    return 0;
}
