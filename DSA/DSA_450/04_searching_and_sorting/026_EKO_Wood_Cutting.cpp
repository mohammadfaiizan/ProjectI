/*
 * Problem: EKO - Eko Wood Cutting
 * URL: https://www.spoj.com/problems/EKO/
 * Problem Statement:
 * Find maximum height to cut trees to get at least M meters of wood.
 * Binary search on the answer (cutting height).
 * 
 * Sample Input:
 * 4 7
 * 20 15 10 17
 * 
 * Sample Output:
 * 15
 */

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Solve_Binary_Search(vector<int>& trees, int required_wood) {
        /*
         * Approach: Binary search on cutting height. For each height,
         * calculate total wood obtained and adjust search range.
         * Time Complexity: O(n log(max_height)) where n = trees
         * Space Complexity: O(1)
         */
        int left = 0;
        int right = *max_element(trees.begin(), trees.end());
        int result = 0;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            long long wood_obtained = Get_Wood_At_Height(trees, mid);
            
            if (wood_obtained >= required_wood) {
                result = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return result;
    }
    
    int Solve_Sorting_Math(vector<int>& trees, int required_wood) {
        /*
         * Approach: Sort trees, then calculate cumulative wood from top.
         * Find the height where cumulative wood >= required.
         * Time Complexity: O(n log n)
         * Space Complexity: O(1)
         */
        sort(trees.begin(), trees.end(), greater<int>());
        int n = trees.size();
        long long cumulative_wood = 0;
        
        for (int i = 0; i < n - 1; i++) {
            cumulative_wood += (long long)(trees[i] - trees[i + 1]) * (i + 1);
            if (cumulative_wood >= required_wood) {
                return trees[i + 1] + (cumulative_wood - required_wood) / (i + 1);
            }
        }
        
        cumulative_wood += (long long)trees[n - 1] * n;
        if (cumulative_wood >= required_wood) {
            return (cumulative_wood - required_wood) / n;
        }
        
        return 0;
    }
    
private:
    long long Get_Wood_At_Height(vector<int>& trees, int height) {
        long long total = 0;
        for (int tree : trees) {
            if (tree > height) {
                total += tree - height;
            }
        }
        return total;
    }
};

void Test_EKO_Wood_Cutting() {
    Solution sol;
    
    vector<int> trees1 = {20, 15, 10, 17};
    assert(sol.Solve_Binary_Search(trees1, 7) == 15);
    
    vector<int> trees2 = {4, 42, 40, 26, 46};
    int result2 = sol.Solve_Binary_Search(trees2, 20);
    assert(result2 >= 36);
    
    vector<int> trees3 = {10, 10, 10};
    assert(sol.Solve_Binary_Search(trees3, 15) == 5);
    
    vector<int> trees4 = {1, 2, 3, 4, 5};
    assert(sol.Solve_Binary_Search(trees4, 5) == 3);
    
    cout << "All test cases passed!" << endl;
}

int main() {
    Test_EKO_Wood_Cutting();
    return 0;
}
