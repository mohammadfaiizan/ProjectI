/*
Problem: Majority Element
URL: https://practice.geeksforgeeks.org/problems/majority-element-1587115620/1

Problem Statement:
Given an array A of N elements. Find the majority element in the array. A majority element in an array A of size N is an element that appears more than N/2 times in the array.

Sample Input/Output:
Input: N = 3, A[] = {1,2,3}
Output: -1

Input: N = 5, A[] = {3,1,3,3,2}
Output: 3
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Majority_Element_HashMap(vector<int>& a, int size) {
        /*
        Using hashmap to count occurrences
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        unordered_map<int, int> count;
        for (int i = 0; i < size; i++) {
            count[a[i]]++;
            if (count[a[i]] > size / 2) {
                return a[i];
            }
        }
        return -1;
    }

    int Majority_Element_Sorting(vector<int>& a, int size) {
        /*
        Sort array and check middle element
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        */
        sort(a.begin(), a.end());
        int candidate = a[size / 2];
        int count = 0;
        for (int i = 0; i < size; i++) {
            if (a[i] == candidate) {
                count++;
            }
        }
        return (count > size / 2) ? candidate : -1;
    }

    int Majority_Element_Moore_Voting(vector<int>& a, int size) {
        /*
        Moore's Voting Algorithm
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int candidate = -1, votes = 0;
        
        for (int i = 0; i < size; i++) {
            if (votes == 0) {
                candidate = a[i];
                votes = 1;
            } else {
                if (a[i] == candidate) {
                    votes++;
                } else {
                    votes--;
                }
            }
        }
        
        int count = 0;
        for (int i = 0; i < size; i++) {
            if (a[i] == candidate) {
                count++;
            }
        }
        
        return (count > size / 2) ? candidate : -1;
    }
};

void Test_Majority_Element() {
    Solution sol;
    vector<vector<int>> tests = {
        {1, 2, 3},
        {3, 1, 3, 3, 2},
        {1, 1, 1, 2, 2},
        {1},
        {1, 2, 2, 2, 3}
    };

    for (auto& arr : tests) {
        int size = arr.size();
        vector<int> arr1 = arr, arr2 = arr;
        
        cout << "Array: ";
        for (int num : arr) cout << num << " ";
        cout << endl;
        
        int res1 = sol.Majority_Element_HashMap(arr, size);
        cout << "HashMap: " << res1 << endl;
        
        int res2 = sol.Majority_Element_Sorting(arr1, size);
        cout << "Sorting: " << res2 << endl;
        
        int res3 = sol.Majority_Element_Moore_Voting(arr2, size);
        cout << "Moore's Voting: " << res3 << endl;
        
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Majority_Element();
    return 0;
}
