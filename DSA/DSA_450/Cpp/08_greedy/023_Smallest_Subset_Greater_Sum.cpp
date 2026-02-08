/*
Problem: Smallest Subset Greater Sum
URL: https://www.geeksforgeeks.org/smallest-subset-sum-greater-elements/

Problem Statement:
Find minimum number of elements such that their sum is greater than sum of rest.

Sample Input/Output:
Input: arr[] = {3, 1, 7, 1}
Output: 1
Explanation: Subset {7} has sum 7 > sum of rest (3+1+1=5). Minimum size is 1.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Smallest_Subset_Greater_Sum_Sort_Descending(vector<int>& arr) {
        /*
        Sort descending + greedy pick greedy approach
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        */
        int total_sum = 0;
        for (int num : arr) {
            total_sum += num;
        }
        
        sort(arr.begin(), arr.end(), greater<int>());
        
        int subset_sum = 0;
        int count = 0;
        
        for (int num : arr) {
            subset_sum += num;
            count++;
            if (subset_sum > total_sum - subset_sum) {
                return count;
            }
        }
        
        return count;
    }
};

void Test_Smallest_Subset_Greater_Sum() {
    Solution solution;
    
    vector<int> arr1 = {3, 1, 7, 1};
    cout << "Test 1: " << solution.Smallest_Subset_Greater_Sum_Sort_Descending(arr1) << endl;
    
    vector<int> arr2 = {2, 1, 2};
    cout << "Test 2: " << solution.Smallest_Subset_Greater_Sum_Sort_Descending(arr2) << endl;
    
    vector<int> arr3 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    cout << "Test 3: " << solution.Smallest_Subset_Greater_Sum_Sort_Descending(arr3) << endl;
}

int main() {
    Test_Smallest_Subset_Greater_Sum();
    return 0;
}
