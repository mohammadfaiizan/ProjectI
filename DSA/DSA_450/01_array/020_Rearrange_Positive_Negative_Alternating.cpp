/*
Problem: Rearrange Array Alternating Positive and Negative
URL: https://www.geeksforgeeks.org/rearrange-array-alternating-positive-negative-items-o1-extra-space/

Problem Statement:
Given an array of positive and negative numbers, arrange them in an alternate fashion
such that every positive number is followed by a negative and vice-versa. Order of elements
should be maintained. Extra positive or negative elements are placed at the end.

Sample Input/Output:
Input: arr = [1, 2, 3, -4, -1, 4]
Output: [-4, 1, -1, 2, 3, 4]

Input: arr = [-5, -2, 5, 2, 4, 7, 1, 8, 0, -8]
Output: [-5, 5, -2, 2, -8, 4, 7, 1, 8, 0]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    void Rearrange_Alternating_Right_Rotate_Optimal(vector<int>& arr) {
        /*
        Right Rotate Based - Track out-of-place elements and rotate
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        */
        int n = arr.size();
        int outofplace = -1;
        for (int index = 0; index < n; index++) {
            if (outofplace >= 0) {
                if ((arr[index] >= 0 && arr[outofplace] < 0) ||
                    (arr[index] < 0 && arr[outofplace] >= 0)) {
                    Right_Rotate(arr, outofplace, index);
                    if (index - outofplace >= 2) outofplace += 2;
                    else outofplace = -1;
                }
            }
            if (outofplace == -1) {
                if ((arr[index] >= 0 && !(index & 1)) ||
                    (arr[index] < 0 && (index & 1))) {
                    outofplace = index;
                }
            }
        }
    }

    void Rearrange_Alternating_Extra_Space(vector<int>& arr) {
        /*
        Extra Space - Separate positives and negatives then interleave
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        vector<int> pos, neg;
        for (int x : arr) {
            if (x >= 0) pos.push_back(x);
            else neg.push_back(x);
        }
        int i = 0, p = 0, n_idx = 0;
        bool place_neg = true;
        while (p < (int)pos.size() && n_idx < (int)neg.size()) {
            if (place_neg) arr[i++] = neg[n_idx++];
            else arr[i++] = pos[p++];
            place_neg = !place_neg;
        }
        while (p < (int)pos.size()) arr[i++] = pos[p++];
        while (n_idx < (int)neg.size()) arr[i++] = neg[n_idx++];
    }

private:
    void Right_Rotate(vector<int>& arr, int outofplace, int cur) {
        int temp = arr[cur];
        for (int i = cur; i > outofplace; i--)
            arr[i] = arr[i - 1];
        arr[outofplace] = temp;
    }
};

void Test_Rearrange_Alternating() {
    Solution solution;

    vector<vector<int>> test_cases = {
        {1, 2, 3, -4, -1, 4},
        {-5, -2, 5, 2, 4, 7, 1, 8, 0, -8},
        {-1, -2, -3, 1, 2, 3}
    };

    for (auto& arr : test_cases) {
        cout << "Original: ";
        for (int x : arr) cout << x << " ";
        cout << endl;

        vector<int> arr1 = arr, arr2 = arr;

        solution.Rearrange_Alternating_Right_Rotate_Optimal(arr1);
        cout << "Right Rotate: ";
        for (int x : arr1) cout << x << " ";
        cout << endl;

        solution.Rearrange_Alternating_Extra_Space(arr2);
        cout << "Extra Space: ";
        for (int x : arr2) cout << x << " ";
        cout << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Rearrange_Alternating();
    return 0;
}
