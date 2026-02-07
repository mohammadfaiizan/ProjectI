/*
Problem: Median of Two Sorted Arrays of Different Size
URL: https://www.geeksforgeeks.org/median-of-two-sorted-arrays-of-different-sizes/

Problem Statement:
Given two sorted arrays of different sizes, find the median of the merged array.
If the merged array has even size, median is the average of the two middle elements.

Sample Input/Output:
Input: ar1 = [900], ar2 = [5, 8, 10, 20]
Output: 10
Explanation: Merged = [5, 8, 10, 20, 900], median = 10.

Input: ar1 = [-5, 3, 6, 12, 15], ar2 = [-12, -10, -6, -3, 4, 10]
Output: 3
Explanation: Merged = [-12,-10,-6,-5,-3,3,4,6,10,12,15], median = 3.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    double Median_Binary_Search_Optimal(vector<int>& nums1, vector<int>& nums2) {
        /*
        Binary Search on Smaller Array - Partition both arrays around median
        Time Complexity: O(log(min(n, m)))
        Space Complexity: O(1)
        */
        if (nums1.size() > nums2.size()) return Median_Binary_Search_Optimal(nums2, nums1);
        int n = nums1.size(), m = nums2.size();
        int low = 0, high = n;
        while (low <= high) {
            int cut1 = (low + high) / 2;
            int cut2 = (n + m + 1) / 2 - cut1;
            int left1 = (cut1 == 0) ? INT_MIN : nums1[cut1 - 1];
            int left2 = (cut2 == 0) ? INT_MIN : nums2[cut2 - 1];
            int right1 = (cut1 == n) ? INT_MAX : nums1[cut1];
            int right2 = (cut2 == m) ? INT_MAX : nums2[cut2];
            if (left1 <= right2 && left2 <= right1) {
                if ((n + m) % 2 == 0)
                    return (max(left1, left2) + min(right1, right2)) / 2.0;
                return max(left1, left2);
            } else if (left1 > right2) {
                high = cut1 - 1;
            } else {
                low = cut1 + 1;
            }
        }
        return 0.0;
    }

    double Median_Merge_Count(vector<int>& ar1, vector<int>& ar2) {
        /*
        Merge and Count - Walk merge to find median position
        Time Complexity: O(n + m)
        Space Complexity: O(1)
        */
        int n = ar1.size(), m = ar2.size();
        int i = 0, j = 0;
        int m1 = -1, m2 = -1;
        int target = (n + m) / 2;
        for (int count = 0; count <= target; count++) {
            m2 = m1;
            if (i < n && j < m) {
                m1 = (ar1[i] <= ar2[j]) ? ar1[i++] : ar2[j++];
            } else if (i < n) {
                m1 = ar1[i++];
            } else {
                m1 = ar2[j++];
            }
        }
        if ((n + m) % 2 == 1) return m1;
        return (m1 + m2) / 2.0;
    }

    double Median_Full_Merge(vector<int>& ar1, vector<int>& ar2) {
        /*
        Full Merge - Merge into new array and find median
        Time Complexity: O(n + m)
        Space Complexity: O(n + m)
        */
        vector<int> merged;
        int i = 0, j = 0;
        while (i < (int)ar1.size() && j < (int)ar2.size()) {
            if (ar1[i] <= ar2[j]) merged.push_back(ar1[i++]);
            else merged.push_back(ar2[j++]);
        }
        while (i < (int)ar1.size()) merged.push_back(ar1[i++]);
        while (j < (int)ar2.size()) merged.push_back(ar2[j++]);
        int total = merged.size();
        if (total % 2 == 1) return merged[total / 2];
        return (merged[total / 2 - 1] + merged[total / 2]) / 2.0;
    }
};

void Test_Median_Different_Size() {
    Solution solution;

    struct TestCase {
        vector<int> ar1, ar2;
        double expected;
    };

    vector<TestCase> test_cases = {
        {{900}, {5, 8, 10, 20}, 10},
        {{-5, 3, 6, 12, 15}, {-12, -10, -6, -3, 4, 10}, 3},
        {{1, 3}, {2}, 2},
        {{1, 2}, {3, 4}, 2.5}
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
    Test_Median_Different_Size();
    return 0;
}
