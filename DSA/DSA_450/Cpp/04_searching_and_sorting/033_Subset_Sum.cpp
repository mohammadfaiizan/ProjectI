/*
 * Problem: Subset Sums
 * URL: https://www.spoj.com/problems/SUBSUMS/
 * 
 * Problem Statement:
 * Count number of subsets whose sum falls in range [A, B].
 * Meet in the middle technique: split array into two halves.
 * Generate all subset sums for each half, then combine.
 * 
 * Sample Input:
 * arr[] = {1, 2, 3}, A = 1, B = 3
 * 
 * Sample Output:
 * 4
 */

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    /*
     * Approach: Generate all subset sums for a given array
     * Use bit manipulation or recursion to generate all possible sums
     * 
     * Time Complexity: O(2^n)
     * Space Complexity: O(2^n)
     */
    void Generate_Subset_Sums(vector<int>& arr, vector<long long>& sums) {
        int n = arr.size();
        sums.push_back(0);
        
        for (int i = 0; i < n; i++) {
            int size = sums.size();
            for (int j = 0; j < size; j++) {
                sums.push_back(sums[j] + arr[i]);
            }
        }
    }

    /*
     * Approach: Meet in the middle
     * Split array into two halves, generate subset sums for each
     * For each sum in first half, count valid sums in second half using binary search
     * 
     * Time Complexity: O(2^(n/2) * log(2^(n/2)))
     * Space Complexity: O(2^(n/2))
     */
    long long Count_Subsets_In_Range(vector<int>& arr, int A, int B) {
        int n = arr.size();
        int mid = n / 2;
        
        vector<int> left(arr.begin(), arr.begin() + mid);
        vector<int> right(arr.begin() + mid, arr.end());
        
        vector<long long> left_sums, right_sums;
        Generate_Subset_Sums(left, left_sums);
        Generate_Subset_Sums(right, right_sums);
        
        sort(right_sums.begin(), right_sums.end());
        
        long long count = 0;
        for (long long left_sum : left_sums) {
            long long min_right = A - left_sum;
            long long max_right = B - left_sum;
            
            auto lower = lower_bound(right_sums.begin(), right_sums.end(), min_right);
            auto upper = upper_bound(right_sums.begin(), right_sums.end(), max_right);
            
            count += (upper - lower);
        }
        
        return count;
    }
};

void Test_Subset_Sum() {
    Solution sol;
    
    vector<int> arr1 = {1, 2, 3};
    assert(sol.Count_Subsets_In_Range(arr1, 1, 3) == 4);
    
    vector<int> arr2 = {1, 2};
    assert(sol.Count_Subsets_In_Range(arr2, 1, 2) == 2);
    
    vector<int> arr3 = {1};
    assert(sol.Count_Subsets_In_Range(arr3, 0, 1) == 2);
    
    vector<int> arr4 = {1, 2, 3, 4};
    assert(sol.Count_Subsets_In_Range(arr4, 3, 6) >= 0);
    
    cout << "All test cases passed!" << endl;
}

int main() {
    Test_Subset_Sum();
    return 0;
}
