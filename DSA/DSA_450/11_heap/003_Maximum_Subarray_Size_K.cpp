/*
Problem: Maximum of All Subarrays of Size K (Sliding Window Maximum)
URL: https://practice.geeksforgeeks.org/problems/maximum-of-all-subarrays-of-size-k3101/1

Problem Statement:
Given an array and integer K, find the maximum for each contiguous subarray of size K.

Sample Input/Output:
Input: [1,2,3,1,4,5,2,3,6], k=3
Output: [3,3,4,5,5,5,6]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> Max_Subarray_K_Deque(vector<int>& arr, int k) {
        /*
        Deque Based Approach
        Time Complexity: O(n)
        Space Complexity: O(k)
        */
        vector<int> result;
        deque<int> dq;
        
        for (int i = 0; i < arr.size(); i++) {
            while (!dq.empty() && dq.front() <= i - k) {
                dq.pop_front();
            }
            
            while (!dq.empty() && arr[dq.back()] <= arr[i]) {
                dq.pop_back();
            }
            
            dq.push_back(i);
            
            if (i >= k - 1) {
                result.push_back(arr[dq.front()]);
            }
        }
        
        return result;
    }

    vector<int> Max_Subarray_K_Heap(vector<int>& arr, int k) {
        /*
        Max Heap with Lazy Deletion
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        */
        vector<int> result;
        priority_queue<pair<int, int>> pq;
        
        for (int i = 0; i < arr.size(); i++) {
            pq.push({arr[i], i});
            
            if (i >= k - 1) {
                while (!pq.empty() && pq.top().second <= i - k) {
                    pq.pop();
                }
                result.push_back(pq.top().first);
            }
        }
        
        return result;
    }

    vector<int> Max_Subarray_K_Brute(vector<int>& arr, int k) {
        /*
        Brute Force Approach
        Time Complexity: O(n*k)
        Space Complexity: O(1)
        */
        vector<int> result;
        
        for (int i = 0; i <= (int)arr.size() - k; i++) {
            int max_val = arr[i];
            for (int j = i + 1; j < i + k; j++) {
                max_val = max(max_val, arr[j]);
            }
            result.push_back(max_val);
        }
        
        return result;
    }
};

void Test_Max_Subarray_K() {
    Solution solution;
    
    vector<int> arr1 = {1, 2, 3, 1, 4, 5, 2, 3, 6};
    int k1 = 3;
    
    cout << "Array: ";
    for (int x : arr1) cout << x << " ";
    cout << ", k = " << k1 << endl;
    
    vector<int> res1 = solution.Max_Subarray_K_Deque(arr1, k1);
    cout << "Deque Result: ";
    for (int x : res1) cout << x << " ";
    cout << endl;
    
    vector<int> res2 = solution.Max_Subarray_K_Heap(arr1, k1);
    cout << "Heap Result: ";
    for (int x : res2) cout << x << " ";
    cout << endl;
    
    vector<int> res3 = solution.Max_Subarray_K_Brute(arr1, k1);
    cout << "Brute Result: ";
    for (int x : res3) cout << x << " ";
    cout << endl;
    
    vector<int> arr2 = {8, 5, 10, 7, 9, 4, 15, 12, 90, 13};
    int k2 = 4;
    
    cout << "\nArray: ";
    for (int x : arr2) cout << x << " ";
    cout << ", k = " << k2 << endl;
    
    vector<int> res4 = solution.Max_Subarray_K_Deque(arr2, k2);
    cout << "Deque Result: ";
    for (int x : res4) cout << x << " ";
    cout << endl;
    
    vector<int> arr3 = {1, 3, -1, -3, 5, 3, 6, 7};
    int k3 = 3;
    
    cout << "\nArray: ";
    for (int x : arr3) cout << x << " ";
    cout << ", k = " << k3 << endl;
    
    vector<int> res5 = solution.Max_Subarray_K_Deque(arr3, k3);
    cout << "Deque Result: ";
    for (int x : res5) cout << x << " ";
    cout << endl;
}

int main() {
    Test_Max_Subarray_K();
    return 0;
}
