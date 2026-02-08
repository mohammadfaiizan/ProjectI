/*
 * Problem: Kth Smallest Number Again
 * URL: https://www.hackerearth.com/practice/algorithms/searching/binary-search/practice-problems/algorithm/kth-smallest-number-again-2/
 * Problem Statement:
 * Given N ranges [a,b], merge overlapping ranges and find kth smallest number
 * across all ranges.
 * 
 * Sample Input:
 * 2
 * 1 3
 * 4 6
 * 2
 * 2
 * 5
 * 
 * Sample Output:
 * 2
 * -1
 */

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Solve_Sort_Merge_Intervals(vector<pair<int, int>>& ranges, int k) {
        /*
         * Approach: Sort ranges by start, merge overlapping intervals,
         * then linearly scan to find which merged interval contains kth element.
         * Time Complexity: O(n log n + q*n) where n = ranges, q = queries
         * Space Complexity: O(n) for merged intervals
         */
        if (ranges.empty()) return -1;
        
        sort(ranges.begin(), ranges.end());
        vector<pair<int, int>> merged;
        merged.push_back(ranges[0]);
        
        for (int i = 1; i < ranges.size(); i++) {
            if (ranges[i].first <= merged.back().second) {
                merged.back().second = max(merged.back().second, ranges[i].second);
            } else {
                merged.push_back(ranges[i]);
            }
        }
        
        int current = 1;
        for (auto& interval : merged) {
            int count = interval.second - interval.first + 1;
            if (current <= k && k <= current + count - 1) {
                return interval.first + (k - current);
            }
            current += count;
        }
        
        return -1;
    }
    
    int Solve_Linear_Scan(vector<pair<int, int>>& ranges, int k) {
        /*
         * Approach: Merge intervals first, then scan linearly to find kth element.
         * Time Complexity: O(n log n + q*n)
         * Space Complexity: O(n)
         */
        return Solve_Sort_Merge_Intervals(ranges, k);
    }
};

void Test_Kth_Smallest_Number_Again() {
    Solution sol;
    
    vector<pair<int, int>> ranges1 = {{1, 3}, {4, 6}};
    assert(sol.Solve_Sort_Merge_Intervals(ranges1, 2) == 2);
    assert(sol.Solve_Sort_Merge_Intervals(ranges1, 5) == -1);
    
    vector<pair<int, int>> ranges2 = {{1, 5}, {3, 7}};
    assert(sol.Solve_Linear_Scan(ranges2, 4) == 4);
    assert(sol.Solve_Linear_Scan(ranges2, 8) == 7);
    
    vector<pair<int, int>> ranges3 = {{10, 12}};
    assert(sol.Solve_Sort_Merge_Intervals(ranges3, 1) == 10);
    assert(sol.Solve_Sort_Merge_Intervals(ranges3, 3) == 12);
    assert(sol.Solve_Sort_Merge_Intervals(ranges3, 4) == -1);
    
    cout << "All test cases passed!" << endl;
}

int main() {
    Test_Kth_Smallest_Number_Again();
    return 0;
}
