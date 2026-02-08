/*
Problem: Kth Largest Sum of Contiguous Subarrays
URL: https://www.geeksforgeeks.org/k-th-largest-sum-contiguous-subarray/

Problem Statement:
Find the Kth largest sum among all contiguous subarrays.

Sample Input/Output:
Input: [20,-5,-1], k=3
Output: 14
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Kth_Largest_Sum_Min_Heap(vector<int>& arr, int k) {
        /*
        Min Heap of Size K
        Time Complexity: O(n^2 log k)
        Space Complexity: O(k)
        */
        priority_queue<int, vector<int>, greater<int>> min_heap;
        
        for (int i = 0; i < arr.size(); i++) {
            int sum = 0;
            for (int j = i; j < arr.size(); j++) {
                sum += arr[j];
                
                if (min_heap.size() < k) {
                    min_heap.push(sum);
                } else if (sum > min_heap.top()) {
                    min_heap.pop();
                    min_heap.push(sum);
                }
            }
        }
        
        return min_heap.top();
    }

    int Kth_Largest_Sum_Sort(vector<int>& arr, int k) {
        /*
        Store All Sums and Sort
        Time Complexity: O(n^2 log n)
        Space Complexity: O(n^2)
        */
        vector<int> sums;
        
        for (int i = 0; i < arr.size(); i++) {
            int sum = 0;
            for (int j = i; j < arr.size(); j++) {
                sum += arr[j];
                sums.push_back(sum);
            }
        }
        
        sort(sums.begin(), sums.end(), greater<int>());
        
        return sums[k - 1];
    }
};

void Test_Kth_Largest_Sum() {
    Solution solution;
    
    vector<int> arr1 = {20, -5, -1};
    int k1 = 3;
    
    cout << "Array: ";
    for (int x : arr1) cout << x << " ";
    cout << ", k = " << k1 << endl;
    
    int result1 = solution.Kth_Largest_Sum_Min_Heap(arr1, k1);
    cout << "Min Heap Result: " << result1 << endl;
    
    int result2 = solution.Kth_Largest_Sum_Sort(arr1, k1);
    cout << "Sort Result: " << result2 << endl;
    
    vector<int> arr2 = {10, -10, 20, -40};
    int k2 = 6;
    
    cout << "\nArray: ";
    for (int x : arr2) cout << x << " ";
    cout << ", k = " << k2 << endl;
    
    int result3 = solution.Kth_Largest_Sum_Min_Heap(arr2, k2);
    cout << "Min Heap Result: " << result3 << endl;
    
    vector<int> arr3 = {1, 2, 3, 4};
    int k3 = 3;
    
    cout << "\nArray: ";
    for (int x : arr3) cout << x << " ";
    cout << ", k = " << k3 << endl;
    
    int result4 = solution.Kth_Largest_Sum_Min_Heap(arr3, k3);
    cout << "Min Heap Result: " << result4 << endl;
    
    vector<int> arr4 = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
    int k4 = 2;
    
    cout << "\nArray: ";
    for (int x : arr4) cout << x << " ";
    cout << ", k = " << k4 << endl;
    
    int result5 = solution.Kth_Largest_Sum_Min_Heap(arr4, k4);
    cout << "Min Heap Result: " << result5 << endl;
}

int main() {
    Test_Kth_Largest_Sum();
    return 0;
}
