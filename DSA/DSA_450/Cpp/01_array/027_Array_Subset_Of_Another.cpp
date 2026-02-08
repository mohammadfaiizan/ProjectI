/*
Problem: Array Subset of Another Array
URL: https://practice.geeksforgeeks.org/problems/array-subset-of-another-array2317/1

Problem Statement:
Given two arrays a1[] and a2[], determine if a2[] is a subset of a1[].
Both arrays can have duplicates.

Sample Input/Output:
Input: a1 = [11, 1, 13, 21, 3, 7], a2 = [11, 3, 7, 1]
Output: Yes

Input: a1 = [10, 5, 2, 23, 19], a2 = [19, 5, 3]
Output: No
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    string Is_Subset_HashSet_Optimal(vector<int>& a1, vector<int>& a2) {
        /*
        HashSet Approach - Check all elements of a2 exist in a1
        Time Complexity: O(n + m)
        Space Complexity: O(n)
        */
        unordered_set<int> s(a1.begin(), a1.end());
        for (int x : a2) {
            if (s.find(x) == s.end()) return "No";
        }
        return "Yes";
    }

    string Is_Subset_HashMap(vector<int>& a1, vector<int>& a2) {
        /*
        HashMap Approach - Handle duplicate frequencies
        Time Complexity: O(n + m)
        Space Complexity: O(n)
        */
        unordered_map<int, int> freq;
        for (int x : a1) freq[x]++;
        for (int x : a2) {
            if (freq[x] <= 0) return "No";
            freq[x]--;
        }
        return "Yes";
    }

    string Is_Subset_Sorting(vector<int> a1, vector<int> a2) {
        /*
        Sorting + Two Pointers - Sort both and compare
        Time Complexity: O(n log n + m log m)
        Space Complexity: O(1)
        */
        sort(a1.begin(), a1.end());
        sort(a2.begin(), a2.end());
        int i = 0, j = 0;
        while (i < (int)a1.size() && j < (int)a2.size()) {
            if (a1[i] == a2[j]) { i++; j++; }
            else if (a1[i] < a2[j]) i++;
            else return "No";
        }
        return (j == (int)a2.size()) ? "Yes" : "No";
    }
};

void Test_Array_Subset() {
    Solution solution;

    struct TestCase {
        vector<int> a1, a2;
        string expected;
    };

    vector<TestCase> test_cases = {
        {{11, 1, 13, 21, 3, 7}, {11, 3, 7, 1}, "Yes"},
        {{1, 2, 3, 4, 5, 6}, {1, 2, 4}, "Yes"},
        {{10, 5, 2, 23, 19}, {19, 5, 3}, "No"}
    };

    for (auto& tc : test_cases) {
        cout << "a1: ";
        for (int x : tc.a1) cout << x << " ";
        cout << ", a2: ";
        for (int x : tc.a2) cout << x << " ";
        cout << ", Expected: " << tc.expected << endl;

        cout << "HashSet: " << solution.Is_Subset_HashSet_Optimal(tc.a1, tc.a2) << endl;
        cout << "HashMap: " << solution.Is_Subset_HashMap(tc.a1, tc.a2) << endl;
        cout << "Sorting: " << solution.Is_Subset_Sorting(tc.a1, tc.a2) << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Array_Subset();
    return 0;
}
