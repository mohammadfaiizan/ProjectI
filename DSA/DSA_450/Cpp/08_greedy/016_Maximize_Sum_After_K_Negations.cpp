/*
Problem: Maximize Sum After K Negations
URL: https://practice.geeksforgeeks.org/problems/maximize-sum-after-k-negations1149/1

Problem Statement:
Given array and K, negate K elements to maximize sum.

Sample Input/Output:
Input: arr[] = {-2, 0, 5, -1, 2}, K = 4
Output: 10
Explanation: Negate -2, -1, 0, 5. Array becomes {2, 0, -5, 1, 2}. Sum = 0.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    long long Maximize_Sum_After_K_Negations_Sort(vector<int>& arr, int K) {
        /*
        Sort + negate negatives, handle remaining K greedy approach
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        */
        sort(arr.begin(), arr.end());
        int n = arr.size();
        
        for (int i = 0; i < n && K > 0; i++) {
            if (arr[i] < 0) {
                arr[i] = -arr[i];
                K--;
            } else {
                break;
            }
        }
        
        if (K > 0 && K % 2 == 1) {
            sort(arr.begin(), arr.end());
            arr[0] = -arr[0];
        }
        
        long long sum = 0;
        for (int num : arr) {
            sum += num;
        }
        
        return sum;
    }
    
    long long Maximize_Sum_After_K_Negations_Min_Heap(vector<int>& arr, int K) {
        /*
        Min-heap approach: Always negate smallest element
        Time Complexity: O(n + k log n)
        Space Complexity: O(n)
        */
        priority_queue<int, vector<int>, greater<int>> pq(arr.begin(), arr.end());
        
        for (int i = 0; i < K; i++) {
            int min_val = pq.top();
            pq.pop();
            pq.push(-min_val);
        }
        
        long long sum = 0;
        while (!pq.empty()) {
            sum += pq.top();
            pq.pop();
        }
        
        return sum;
    }
};

void Test_Maximize_Sum_After_K_Negations() {
    Solution solution;
    
    vector<int> arr1 = {-2, 0, 5, -1, 2};
    cout << "Test 1 (Sort): " << solution.Maximize_Sum_After_K_Negations_Sort(arr1, 4) << endl;
    
    vector<int> arr2 = {-2, 0, 5, -1, 2};
    cout << "Test 1 (Heap): " << solution.Maximize_Sum_After_K_Negations_Min_Heap(arr2, 4) << endl;
    
    vector<int> arr3 = {9, 8, 8, 5};
    cout << "Test 2 (Sort): " << solution.Maximize_Sum_After_K_Negations_Sort(arr3, 3) << endl;
}

int main() {
    Test_Maximize_Sum_After_K_Negations();
    return 0;
}
