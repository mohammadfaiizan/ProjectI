/*
Problem: Merge K Sorted Arrays
URL: https://practice.geeksforgeeks.org/problems/merge-k-sorted-arrays/1

Problem Statement:
Merge K sorted arrays into a single sorted array.

Sample Input/Output:
Input: [[1,2,3],[4,5,6],[7,8,9]]
Output: [1,2,3,4,5,6,7,8,9]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> Merge_K_Sorted_Min_Heap(vector<vector<int>>& arrays) {
        /*
        Min Heap with {value, array_idx, element_idx}
        Time Complexity: O(n*k log k)
        Space Complexity: O(k)
        */
        vector<int> result;
        priority_queue<pair<int, pair<int, int>>, vector<pair<int, pair<int, int>>>, greater<pair<int, pair<int, int>>>> min_heap;
        
        for (int i = 0; i < arrays.size(); i++) {
            if (!arrays[i].empty()) {
                min_heap.push({arrays[i][0], {i, 0}});
            }
        }
        
        while (!min_heap.empty()) {
            auto top = min_heap.top();
            min_heap.pop();
            
            int value = top.first;
            int array_idx = top.second.first;
            int element_idx = top.second.second;
            
            result.push_back(value);
            
            if (element_idx + 1 < arrays[array_idx].size()) {
                min_heap.push({arrays[array_idx][element_idx + 1], {array_idx, element_idx + 1}});
            }
        }
        
        return result;
    }

    vector<int> Merge_K_Sorted_Divide_Conquer(vector<vector<int>>& arrays) {
        /*
        Divide and Conquer (Merge Two at a Time)
        Time Complexity: O(n*k log k)
        Space Complexity: O(n*k)
        */
        if (arrays.empty()) return {};
        if (arrays.size() == 1) return arrays[0];
        
        return Merge_Helper(arrays, 0, arrays.size() - 1);
    }

private:
    vector<int> Merge_Helper(vector<vector<int>>& arrays, int left, int right) {
        if (left == right) {
            return arrays[left];
        }
        
        int mid = left + (right - left) / 2;
        vector<int> left_merged = Merge_Helper(arrays, left, mid);
        vector<int> right_merged = Merge_Helper(arrays, mid + 1, right);
        
        return Merge_Two_Arrays(left_merged, right_merged);
    }

    vector<int> Merge_Two_Arrays(vector<int>& arr1, vector<int>& arr2) {
        vector<int> result;
        int i = 0, j = 0;
        
        while (i < arr1.size() && j < arr2.size()) {
            if (arr1[i] <= arr2[j]) {
                result.push_back(arr1[i++]);
            } else {
                result.push_back(arr2[j++]);
            }
        }
        
        while (i < arr1.size()) {
            result.push_back(arr1[i++]);
        }
        
        while (j < arr2.size()) {
            result.push_back(arr2[j++]);
        }
        
        return result;
    }
};

void Test_Merge_K_Sorted() {
    Solution solution;
    
    vector<vector<int>> arrays1 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    
    cout << "Input Arrays:" << endl;
    for (int i = 0; i < arrays1.size(); i++) {
        cout << "Array " << i << ": ";
        for (int x : arrays1[i]) cout << x << " ";
        cout << endl;
    }
    
    vector<int> res1 = solution.Merge_K_Sorted_Min_Heap(arrays1);
    cout << "Min Heap Result: ";
    for (int x : res1) cout << x << " ";
    cout << endl;
    
    vector<int> res2 = solution.Merge_K_Sorted_Divide_Conquer(arrays1);
    cout << "Divide Conquer Result: ";
    for (int x : res2) cout << x << " ";
    cout << endl;
    
    vector<vector<int>> arrays2 = {{1, 3, 5, 7}, {2, 4, 6, 8}, {0, 9, 10, 11}};
    
    cout << "\nInput Arrays:" << endl;
    for (int i = 0; i < arrays2.size(); i++) {
        cout << "Array " << i << ": ";
        for (int x : arrays2[i]) cout << x << " ";
        cout << endl;
    }
    
    vector<int> res3 = solution.Merge_K_Sorted_Min_Heap(arrays2);
    cout << "Min Heap Result: ";
    for (int x : res3) cout << x << " ";
    cout << endl;
    
    vector<vector<int>> arrays3 = {{1, 4, 7}, {2, 5, 8}, {3, 6, 9}, {10, 11, 12}};
    
    cout << "\nInput Arrays:" << endl;
    for (int i = 0; i < arrays3.size(); i++) {
        cout << "Array " << i << ": ";
        for (int x : arrays3[i]) cout << x << " ";
        cout << endl;
    }
    
    vector<int> res4 = solution.Merge_K_Sorted_Min_Heap(arrays3);
    cout << "Min Heap Result: ";
    for (int x : res4) cout << x << " ";
    cout << endl;
}

int main() {
    Test_Merge_K_Sorted();
    return 0;
}
