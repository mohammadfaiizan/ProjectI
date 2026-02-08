/*
Problem: Sum of Minimum and Maximum Elements of All Subarrays of Size K
URL: https://www.geeksforgeeks.org/sum-minimum-maximum-elements-subarrays-size-k/

Problem Statement:
Given an array of size N and an integer K, find the sum of minimum and maximum elements of all contiguous subarrays of size K.

Sample Input/Output:
Input: arr[] = {2,5,-1,7,-3,-1,-2}, k = 3
Output: 18
Explanation: Subarrays of size 3: [2,5,-1] -> min=-1, max=5, sum=4
            [5,-1,7] -> min=-1, max=7, sum=6
            [-1,7,-3] -> min=-3, max=7, sum=4
            [7,-3,-1] -> min=-3, max=7, sum=4
            Total = 4+6+4+4 = 18
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    long long Sum_Min_Max_Subarrays_K_Deque(vector<int>& arr, int k) {
        /*
        Deque-based sliding window with two deques
        Time Complexity: O(n)
        Space Complexity: O(k)
        */
        int n = arr.size();
        deque<int> minDeque;
        deque<int> maxDeque;
        long long sum = 0;
        
        for (int i = 0; i < k; i++) {
            while (!minDeque.empty() && arr[minDeque.back()] >= arr[i]) {
                minDeque.pop_back();
            }
            while (!maxDeque.empty() && arr[maxDeque.back()] <= arr[i]) {
                maxDeque.pop_back();
            }
            minDeque.push_back(i);
            maxDeque.push_back(i);
        }
        
        sum += arr[minDeque.front()] + arr[maxDeque.front()];
        
        for (int i = k; i < n; i++) {
            while (!minDeque.empty() && minDeque.front() <= i - k) {
                minDeque.pop_front();
            }
            while (!maxDeque.empty() && maxDeque.front() <= i - k) {
                maxDeque.pop_front();
            }
            
            while (!minDeque.empty() && arr[minDeque.back()] >= arr[i]) {
                minDeque.pop_back();
            }
            while (!maxDeque.empty() && arr[maxDeque.back()] <= arr[i]) {
                maxDeque.pop_back();
            }
            
            minDeque.push_back(i);
            maxDeque.push_back(i);
            
            sum += arr[minDeque.front()] + arr[maxDeque.front()];
        }
        
        return sum;
    }
    
    long long Sum_Min_Max_Subarrays_K_Brute_Force(vector<int>& arr, int k) {
        /*
        Brute force
        Time Complexity: O(n*k)
        Space Complexity: O(1)
        */
        int n = arr.size();
        long long sum = 0;
        
        for (int i = 0; i <= n - k; i++) {
            int minVal = arr[i];
            int maxVal = arr[i];
            
            for (int j = i; j < i + k; j++) {
                minVal = min(minVal, arr[j]);
                maxVal = max(maxVal, arr[j]);
            }
            
            sum += minVal + maxVal;
        }
        
        return sum;
    }
};

void Test_Sum_Min_Max_Subarrays_K() {
    Solution solution;
    
    vector<int> arr1 = {2, 5, -1, 7, -3, -1, -2};
    int k1 = 3;
    cout << "Test 1 - Deque: " << solution.Sum_Min_Max_Subarrays_K_Deque(arr1, k1) << endl;
    cout << "Test 1 - Brute Force: " << solution.Sum_Min_Max_Subarrays_K_Brute_Force(arr1, k1) << endl;
    
    vector<int> arr2 = {1, 2, 3, 4, 5};
    int k2 = 3;
    cout << "Test 2 - Deque: " << solution.Sum_Min_Max_Subarrays_K_Deque(arr2, k2) << endl;
    cout << "Test 2 - Brute Force: " << solution.Sum_Min_Max_Subarrays_K_Brute_Force(arr2, k2) << endl;
    
    vector<int> arr3 = {5, 4, 3, 2, 1};
    int k3 = 2;
    cout << "Test 3 - Deque: " << solution.Sum_Min_Max_Subarrays_K_Deque(arr3, k3) << endl;
    cout << "Test 3 - Brute Force: " << solution.Sum_Min_Max_Subarrays_K_Brute_Force(arr3, k3) << endl;
}

int main() {
    Test_Sum_Min_Max_Subarrays_K();
    return 0;
}
