/*
Problem: Common Elements in Three Sorted Arrays
URL: https://practice.geeksforgeeks.org/problems/common-elements1132/1

Problem Statement:
Given three arrays sorted in increasing order, find the elements that are common in all three.

Sample Input/Output:
Input: A = [1, 5, 10, 20, 40, 80], B = [6, 7, 20, 80, 100], C = [3, 4, 15, 20, 30, 70, 80, 120]
Output: [20, 80]

Input: A = [1, 5, 5], B = [3, 4, 5, 5, 10], C = [5, 5, 10, 20]
Output: [5, 5]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> Common_Elements_Three_Pointers_Optimal(vector<int>& A, vector<int>& B, vector<int>& C) {
        /*
        Three Pointers - Advance pointer of smallest element
        Time Complexity: O(n1 + n2 + n3)
        Space Complexity: O(1) excluding result
        */
        vector<int> result;
        int i = 0, j = 0, k = 0;
        int prev = INT_MIN;
        while (i < (int)A.size() && j < (int)B.size() && k < (int)C.size()) {
            if (A[i] == B[j] && B[j] == C[k]) {
                if (A[i] != prev) {
                    result.push_back(A[i]);
                    prev = A[i];
                }
                i++; j++; k++;
            } else if (A[i] < B[j]) i++;
            else if (B[j] < C[k]) j++;
            else k++;
        }
        return result;
    }

    vector<int> Common_Elements_Hashing(vector<int>& A, vector<int>& B, vector<int>& C) {
        /*
        Hashing Approach - Use maps to count occurrences
        Time Complexity: O(n1 + n2 + n3)
        Space Complexity: O(n1 + n2 + n3)
        */
        unordered_map<int, int> freqA, freqB;
        for (int x : A) freqA[x]++;
        for (int x : B) freqB[x]++;
        vector<int> result;
        unordered_set<int> added;
        for (int x : C) {
            if (freqA[x] > 0 && freqB[x] > 0 && !added.count(x)) {
                result.push_back(x);
                added.insert(x);
            }
        }
        return result;
    }
};

void Test_Common_Elements() {
    Solution solution;

    struct TestCase {
        vector<int> A, B, C;
    };

    vector<TestCase> test_cases = {
        {{1, 5, 10, 20, 40, 80}, {6, 7, 20, 80, 100}, {3, 4, 15, 20, 30, 70, 80, 120}},
        {{1, 5, 5}, {3, 4, 5, 5, 10}, {5, 5, 10, 20}},
        {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
    };

    for (auto& tc : test_cases) {
        cout << "A: ";
        for (int x : tc.A) cout << x << " ";
        cout << ", B: ";
        for (int x : tc.B) cout << x << " ";
        cout << ", C: ";
        for (int x : tc.C) cout << x << " ";
        cout << endl;

        auto r1 = solution.Common_Elements_Three_Pointers_Optimal(tc.A, tc.B, tc.C);
        cout << "Three Pointers: ";
        for (int x : r1) cout << x << " ";
        cout << endl;

        auto r2 = solution.Common_Elements_Hashing(tc.A, tc.B, tc.C);
        cout << "Hashing: ";
        for (int x : r2) cout << x << " ";
        cout << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Common_Elements();
    return 0;
}
