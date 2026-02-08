/*
Problem: First Negative Integer in Every Window of Size K
URL: https://practice.geeksforgeeks.org/problems/first-negative-integer-in-every-window-of-size-k3345/1

Problem Statement:
Given an array and a positive integer k, find the first negative integer for each and every contiguous subarray of size k.
If a window does not contain a negative integer, then return 0 for that window.

Sample Input/Output:
Input: arr[] = {-8,2,3,-6,10}, k = 2
Output: [-8,0,-6,-6]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<long long> First_Negative_Window_K_Deque(vector<long long>& arr, int k) {
        /*
        Deque-based sliding window
        Time Complexity: O(n)
        Space Complexity: O(k)
        */
        int n = arr.size();
        vector<long long> result;
        deque<int> dq;
        
        for (int i = 0; i < k; i++) {
            if (arr[i] < 0) {
                dq.push_back(i);
            }
        }
        
        if (!dq.empty()) {
            result.push_back(arr[dq.front()]);
        } else {
            result.push_back(0);
        }
        
        for (int i = k; i < n; i++) {
            if (!dq.empty() && dq.front() == i - k) {
                dq.pop_front();
            }
            
            if (arr[i] < 0) {
                dq.push_back(i);
            }
            
            if (!dq.empty()) {
                result.push_back(arr[dq.front()]);
            } else {
                result.push_back(0);
            }
        }
        
        return result;
    }
    
    vector<long long> First_Negative_Window_K_Brute_Force(vector<long long>& arr, int k) {
        /*
        Brute force
        Time Complexity: O(n*k)
        Space Complexity: O(1)
        */
        int n = arr.size();
        vector<long long> result;
        
        for (int i = 0; i <= n - k; i++) {
            long long firstNeg = 0;
            for (int j = i; j < i + k; j++) {
                if (arr[j] < 0) {
                    firstNeg = arr[j];
                    break;
                }
            }
            result.push_back(firstNeg);
        }
        
        return result;
    }
};

void Test_First_Negative_Window_K() {
    Solution solution;
    
    vector<long long> arr1 = {-8, 2, 3, -6, 10};
    int k1 = 2;
    vector<long long> result1 = solution.First_Negative_Window_K_Deque(arr1, k1);
    cout << "Test 1 - Deque: ";
    for (long long val : result1) cout << val << " ";
    cout << endl;
    
    vector<long long> arr2 = {12, -1, -7, 8, -15, 30, 16, 28};
    int k2 = 3;
    vector<long long> result2 = solution.First_Negative_Window_K_Deque(arr2, k2);
    cout << "Test 2 - Deque: ";
    for (long long val : result2) cout << val << " ";
    cout << endl;
    
    vector<long long> arr3 = {-1, -2, -3, -4, -5};
    int k3 = 2;
    vector<long long> result3 = solution.First_Negative_Window_K_Deque(arr3, k3);
    cout << "Test 3 - Deque: ";
    for (long long val : result3) cout << val << " ";
    cout << endl;
}

int main() {
    Test_First_Negative_Window_K();
    return 0;
}
