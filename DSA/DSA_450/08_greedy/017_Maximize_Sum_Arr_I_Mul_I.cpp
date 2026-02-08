/*
Problem: Maximize Sum Arr[i]*i
URL: https://practice.geeksforgeeks.org/problems/maximize-arrii-of-an-array0026/1

Problem Statement:
Maximize sum of arr[i]*i by rearranging the array.

Sample Input/Output:
Input: arr[] = {3, 5, 6, 1}
Output: 31
Explanation: Rearrange to {1, 3, 5, 6}. Sum = 0*1 + 1*3 + 2*5 + 3*6 = 31
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Maximize_Sum_Arr_I_Mul_I_Sort_Ascending(vector<int>& arr) {
        /*
        Sort ascending greedy approach: Smallest element at smallest index
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        */
        sort(arr.begin(), arr.end());
        long long sum = 0;
        int mod = 1000000007;
        
        for (int i = 0; i < arr.size(); i++) {
            sum = (sum + (long long)arr[i] * i) % mod;
        }
        
        return sum;
    }
};

void Test_Maximize_Sum_Arr_I_Mul_I() {
    Solution solution;
    
    vector<int> arr1 = {3, 5, 6, 1};
    cout << "Test 1: " << solution.Maximize_Sum_Arr_I_Mul_I_Sort_Ascending(arr1) << endl;
    
    vector<int> arr2 = {1, 2, 3};
    cout << "Test 2: " << solution.Maximize_Sum_Arr_I_Mul_I_Sort_Ascending(arr2) << endl;
    
    vector<int> arr3 = {5, 3, 2, 4, 1};
    cout << "Test 3: " << solution.Maximize_Sum_Arr_I_Mul_I_Sort_Ascending(arr3) << endl;
}

int main() {
    Test_Maximize_Sum_Arr_I_Mul_I();
    return 0;
}
