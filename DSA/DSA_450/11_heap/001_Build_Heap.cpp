/*
Problem: Build Max Heap and Min Heap
URL: https://www.geeksforgeeks.org/building-heap-from-array/

Problem Statement:
Build a max heap and min heap from an array using heapify (top-down recursive). Show both max heap and min heap construction.

Sample Input/Output:
Input: [1,3,5,4,6,13,10,9,8,15,17]
Output: Max Heap: [17,15,13,9,6,5,10,4,8,3,1]
        Min Heap: [1,3,5,4,6,13,10,9,8,15,17]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    void Build_Max_Heap_Recursive(vector<int>& arr) {
        /*
        Build Max Heap Recursive
        Time Complexity: O(n)
        Space Complexity: O(log n) for recursion stack
        */
        int n = arr.size();
        for (int i = n / 2 - 1; i >= 0; i--) {
            Max_Heapify(arr, n, i);
        }
    }

    void Build_Min_Heap_Recursive(vector<int>& arr) {
        /*
        Build Min Heap Recursive
        Time Complexity: O(n)
        Space Complexity: O(log n) for recursion stack
        */
        int n = arr.size();
        for (int i = n / 2 - 1; i >= 0; i--) {
            Min_Heapify(arr, n, i);
        }
    }

private:
    void Max_Heapify(vector<int>& arr, int n, int i) {
        int largest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;

        if (left < n && arr[left] > arr[largest]) {
            largest = left;
        }

        if (right < n && arr[right] > arr[largest]) {
            largest = right;
        }

        if (largest != i) {
            swap(arr[i], arr[largest]);
            Max_Heapify(arr, n, largest);
        }
    }

    void Min_Heapify(vector<int>& arr, int n, int i) {
        int smallest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;

        if (left < n && arr[left] < arr[smallest]) {
            smallest = left;
        }

        if (right < n && arr[right] < arr[smallest]) {
            smallest = right;
        }

        if (smallest != i) {
            swap(arr[i], arr[smallest]);
            Min_Heapify(arr, n, smallest);
        }
    }
};

void Test_Build_Heap() {
    Solution solution;
    
    vector<int> arr1 = {1, 3, 5, 4, 6, 13, 10, 9, 8, 15, 17};
    vector<int> arr2 = arr1;
    
    cout << "Original array: ";
    for (int x : arr1) cout << x << " ";
    cout << endl;
    
    solution.Build_Max_Heap_Recursive(arr1);
    cout << "Max Heap: ";
    for (int x : arr1) cout << x << " ";
    cout << endl;
    
    solution.Build_Min_Heap_Recursive(arr2);
    cout << "Min Heap: ";
    for (int x : arr2) cout << x << " ";
    cout << endl;
    
    vector<int> arr3 = {10, 20, 15, 30, 40};
    vector<int> arr4 = arr3;
    
    cout << "\nOriginal array: ";
    for (int x : arr3) cout << x << " ";
    cout << endl;
    
    solution.Build_Max_Heap_Recursive(arr3);
    cout << "Max Heap: ";
    for (int x : arr3) cout << x << " ";
    cout << endl;
    
    solution.Build_Min_Heap_Recursive(arr4);
    cout << "Min Heap: ";
    for (int x : arr4) cout << x << " ";
    cout << endl;
}

int main() {
    Test_Build_Heap();
    return 0;
}
