/*
Problem: Maximum Sum Absolute Difference
URL: https://www.geeksforgeeks.org/maximum-sum-absolute-difference-array/

Problem Statement:
Rearrange array to maximize sum of |arr[i]-arr[i+1]| (circular).

Sample Input/Output:
Input: arr[] = {1, 2, 4, 8}
Output: 18
Explanation: Rearrange to {1, 8, 2, 4}. Sum = |1-8| + |8-2| + |2-4| + |4-1| = 7 + 6 + 2 + 3 = 18
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Max_Sum_Absolute_Difference_Sort_Interleave(vector<int>& arr) {
        /*
        Sort + interleave small/large greedy approach
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        */
        sort(arr.begin(), arr.end());
        int n = arr.size();
        vector<int> result(n);
        
        int left = 0, right = n - 1;
        for (int i = 0; i < n; i++) {
            if (i % 2 == 0) {
                result[i] = arr[left++];
            } else {
                result[i] = arr[right--];
            }
        }
        
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += abs(result[i] - result[(i + 1) % n]);
        }
        
        return sum;
    }
    
    int Max_Sum_Absolute_Difference_Sort_Double_Difference(vector<int>& arr) {
        /*
        Sort + double difference greedy approach
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        */
        sort(arr.begin(), arr.end());
        int n = arr.size();
        int sum = 0;
        
        for (int i = 0; i < n / 2; i++) {
            sum -= 2 * arr[i];
            sum += 2 * arr[n - 1 - i];
        }
        
        return sum;
    }
};

void Test_Max_Sum_Absolute_Difference() {
    Solution solution;
    
    vector<int> arr1 = {1, 2, 4, 8};
    cout << "Test 1 (Interleave): " << solution.Max_Sum_Absolute_Difference_Sort_Interleave(arr1) << endl;
    
    vector<int> arr2 = {1, 2, 4, 8};
    cout << "Test 1 (Double Diff): " << solution.Max_Sum_Absolute_Difference_Sort_Double_Difference(arr2) << endl;
    
    vector<int> arr3 = {4, 2, 1, 8};
    cout << "Test 2 (Interleave): " << solution.Max_Sum_Absolute_Difference_Sort_Interleave(arr3) << endl;
}

int main() {
    Test_Max_Sum_Absolute_Difference();
    return 0;
}
