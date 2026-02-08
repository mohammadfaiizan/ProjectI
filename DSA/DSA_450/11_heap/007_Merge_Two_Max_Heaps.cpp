/*
Problem: Merge Two Binary Max Heaps
URL: https://practice.geeksforgeeks.org/problems/merge-two-binary-max-heap0144/1

Problem Statement:
Given two max heaps, merge them into a single max heap.

Sample Input/Output:
Input: heap1=[10,5,6,2], heap2=[12,7,9]
Output: Merged max heap
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> Merge_Heaps_Rebuild(vector<int>& heap1, vector<int>& heap2) {
        /*
        Concatenate Arrays and Rebuild Heap
        Time Complexity: O(n+m)
        Space Complexity: O(n+m)
        */
        vector<int> merged;
        
        for (int x : heap1) {
            merged.push_back(x);
        }
        
        for (int x : heap2) {
            merged.push_back(x);
        }
        
        int n = merged.size();
        for (int i = n / 2 - 1; i >= 0; i--) {
            Max_Heapify(merged, n, i);
        }
        
        return merged;
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
};

void Test_Merge_Heaps() {
    Solution solution;
    
    vector<int> heap1 = {10, 5, 6, 2};
    vector<int> heap2 = {12, 7, 9};
    
    cout << "Heap 1: ";
    for (int x : heap1) cout << x << " ";
    cout << endl;
    
    cout << "Heap 2: ";
    for (int x : heap2) cout << x << " ";
    cout << endl;
    
    vector<int> merged = solution.Merge_Heaps_Rebuild(heap1, heap2);
    cout << "Merged Max Heap: ";
    for (int x : merged) cout << x << " ";
    cout << endl;
    
    vector<int> heap3 = {20, 10, 15, 8, 5};
    vector<int> heap4 = {25, 18, 12};
    
    cout << "\nHeap 1: ";
    for (int x : heap3) cout << x << " ";
    cout << endl;
    
    cout << "Heap 2: ";
    for (int x : heap4) cout << x << " ";
    cout << endl;
    
    vector<int> merged2 = solution.Merge_Heaps_Rebuild(heap3, heap4);
    cout << "Merged Max Heap: ";
    for (int x : merged2) cout << x << " ";
    cout << endl;
    
    vector<int> heap5 = {30};
    vector<int> heap6 = {40, 35};
    
    cout << "\nHeap 1: ";
    for (int x : heap5) cout << x << " ";
    cout << endl;
    
    cout << "Heap 2: ";
    for (int x : heap6) cout << x << " ";
    cout << endl;
    
    vector<int> merged3 = solution.Merge_Heaps_Rebuild(heap5, heap6);
    cout << "Merged Max Heap: ";
    for (int x : merged3) cout << x << " ";
    cout << endl;
}

int main() {
    Test_Merge_Heaps();
    return 0;
}
