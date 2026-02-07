/*
Problem: Union and Intersection of Two Arrays
URL: https://practice.geeksforgeeks.org/problems/union-of-two-arrays3538/1

Problem Statement:
Given two arrays a[] and b[], find the number of elements in the union and intersection.
Union: Set of all distinct elements from both arrays.
Intersection: Set of all elements common to both arrays.

Sample Input/Output:
Input: a = [1, 2, 3, 4, 5], b = [1, 2, 3]
Output: Union = 5, Intersection = 3

Input: a = [85, 25, 1, 32, 54, 6], b = [85, 2]
Output: Union = 7, Intersection = 1
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Union_Set_Optimal(vector<int>& a, vector<int>& b) {
        /*
        Set Based Union - Insert all elements into set
        Time Complexity: O(m + n)
        Space Complexity: O(m + n)
        */
        unordered_set<int> s;
        for (int x : a) s.insert(x);
        for (int x : b) s.insert(x);
        return s.size();
    }

    int Intersection_Sorting(vector<int> a, vector<int> b) {
        /*
        Sorting + Two Pointers - Sort both and merge to find common
        Time Complexity: O(m log m + n log n)
        Space Complexity: O(1)
        */
        sort(a.begin(), a.end());
        sort(b.begin(), b.end());
        int i = 0, j = 0, count = 0;
        while (i < (int)a.size() && j < (int)b.size()) {
            if (a[i] == b[j]) { count++; i++; j++; }
            else if (a[i] < b[j]) i++;
            else j++;
        }
        return count;
    }

    int Intersection_Hashing_Optimal(vector<int>& a, vector<int>& b) {
        /*
        Hashing Approach - Use map to count and find common
        Time Complexity: O(m + n)
        Space Complexity: O(min(m, n))
        */
        unordered_map<int, int> freq;
        for (int x : a) freq[x]++;
        int count = 0;
        for (int x : b) {
            if (freq[x] > 0) { count++; freq[x]--; }
        }
        return count;
    }
};

void Test_Union_And_Intersection() {
    Solution solution;

    struct TestCase {
        vector<int> a, b;
    };

    vector<TestCase> test_cases = {
        {{1, 2, 3, 4, 5}, {1, 2, 3}},
        {{85, 25, 1, 32, 54, 6}, {85, 2}},
        {{1, 2, 3}, {4, 5, 6}},
        {{1, 1, 1}, {1, 1}}
    };

    for (auto& tc : test_cases) {
        cout << "A: ";
        for (int x : tc.a) cout << x << " ";
        cout << ", B: ";
        for (int x : tc.b) cout << x << " ";
        cout << endl;

        cout << "Union (Set): " << solution.Union_Set_Optimal(tc.a, tc.b) << endl;
        cout << "Intersection (Sorting): " << solution.Intersection_Sorting(tc.a, tc.b) << endl;
        cout << "Intersection (Hashing): " << solution.Intersection_Hashing_Optimal(tc.a, tc.b) << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Union_And_Intersection();
    return 0;
}
