/*
Problem: Median of Two Sorted Arrays of Equal Size
URL: https://www.geeksforgeeks.org/median-of-two-sorted-arrays/

Problem Statement:
Given two sorted arrays ar1[] and ar2[] of the same size n, find the median of the
merged array (without actually merging). Median is average of elements at n-1 and n.

Sample Input/Output:
Input: ar1 = [1, 12, 15, 26, 38], ar2 = [2, 13, 17, 30, 45]
Output: 16
Explanation: Merged = [1,2,12,13,15,17,26,30,38,45], median = (15+17)/2 = 16.

Input: ar1 = [1, 2, 3, 6], ar2 = [4, 6, 8, 10]
Output: 5
Explanation: Merged = [1,2,3,4,6,6,8,10], median = (4+6)/2 = 5.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Median_Binary_Search_Optimal(vector<int>& ar1, vector<int>& ar2) {
        /*
        Divide and Conquer - Compare medians and recurse on halves
        Time Complexity: O(log n)
        Space Complexity: O(log n) recursion stack
        */
        return Find_Median_Recursive(ar1, ar2, 0, 0, ar1.size());
    }

    int Median_Merge_Count(vector<int>& ar1, vector<int>& ar2) {
        /*
        Merge Count - Count while merging until reaching median position
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int n = ar1.size();
        int i = 0, j = 0;
        int m1 = -1, m2 = -1;
        for (int count = 0; count <= n; count++) {
            m1 = m2;
            if (i == n) { m2 = ar2[j++]; }
            else if (j == n) { m2 = ar1[i++]; }
            else if (ar1[i] <= ar2[j]) { m2 = ar1[i++]; }
            else { m2 = ar2[j++]; }
        }
        return (m1 + m2) / 2;
    }

    int Median_Full_Merge(vector<int>& ar1, vector<int>& ar2) {
        /*
        Full Merge - Merge both arrays and find median
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        int n = ar1.size();
        vector<int> merged;
        int i = 0, j = 0;
        while (i < n && j < n) {
            if (ar1[i] <= ar2[j]) merged.push_back(ar1[i++]);
            else merged.push_back(ar2[j++]);
        }
        while (i < n) merged.push_back(ar1[i++]);
        while (j < n) merged.push_back(ar2[j++]);
        return (merged[n - 1] + merged[n]) / 2;
    }

private:
    int Median_Single(vector<int>& arr, int start, int n) {
        if (n % 2 == 0) return (arr[start + n / 2] + arr[start + n / 2 - 1]) / 2;
        return arr[start + n / 2];
    }

    int Find_Median_Recursive(vector<int>& ar1, vector<int>& ar2, int s1, int s2, int n) {
        if (n <= 0) return -1;
        if (n == 1) return (ar1[s1] + ar2[s2]) / 2;
        if (n == 2) return (max(ar1[s1], ar2[s2]) + min(ar1[s1 + 1], ar2[s2 + 1])) / 2;

        int m1 = Median_Single(ar1, s1, n);
        int m2 = Median_Single(ar2, s2, n);
        if (m1 == m2) return m1;

        int half = (n % 2 == 0) ? n / 2 - 1 : n / 2;
        if (m1 < m2)
            return Find_Median_Recursive(ar1, ar2, s1 + half, s2, n - half);
        return Find_Median_Recursive(ar1, ar2, s1, s2 + half, n - half);
    }
};

void Test_Median_Equal_Size() {
    Solution solution;

    struct TestCase {
        vector<int> ar1, ar2;
        int expected;
    };

    vector<TestCase> test_cases = {
        {{1, 12, 15, 26, 38}, {2, 13, 17, 30, 45}, 16},
        {{1, 2, 3, 6}, {4, 6, 8, 10}, 5}
    };

    for (auto& tc : test_cases) {
        cout << "ar1: ";
        for (int x : tc.ar1) cout << x << " ";
        cout << ", ar2: ";
        for (int x : tc.ar2) cout << x << " ";
        cout << ", Expected: " << tc.expected << endl;

        cout << "Binary Search: " << solution.Median_Binary_Search_Optimal(tc.ar1, tc.ar2) << endl;
        cout << "Merge Count: " << solution.Median_Merge_Count(tc.ar1, tc.ar2) << endl;
        cout << "Full Merge: " << solution.Median_Full_Merge(tc.ar1, tc.ar2) << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Median_Equal_Size();
    return 0;
}
