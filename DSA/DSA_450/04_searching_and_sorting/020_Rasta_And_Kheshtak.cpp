/*
 * Problem: Rasta and Kheshtak
 * URL: https://www.hackerearth.com/practice/algorithms/searching/binary-search/practice-problems/algorithm/rasta-and-kheshtak/
 * Problem Statement:
 * Given two binary matrices, find the largest square submatrix of matrix1
 * that appears in matrix2.
 * 
 * Sample Input:
 * 3 3
 * 1 0 1
 * 0 1 0
 * 1 0 1
 * 4 4
 * 1 0 1 0
 * 0 1 0 1
 * 1 0 1 0
 * 0 1 0 1
 * 
 * Sample Output:
 * 2
 */

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Solve_2D_Hashing_Binary_Search(vector<vector<int>>& matrix1, vector<vector<int>>& matrix2) {
        /*
         * Approach: Use 2D hashing to compute hash values for all submatrices,
         * then binary search on the answer (size of square) to find maximum size.
         * Time Complexity: O(n^2 * m^2 * log(min(n,m))) where n,m are matrix dimensions
         * Space Complexity: O(n^2 * m^2) for hash storage
         */
        int n1 = matrix1.size(), m1 = matrix1[0].size();
        int n2 = matrix2.size(), m2 = matrix2[0].size();
        
        int max_size = 0;
        int low = 1, high = min(min(n1, m1), min(n2, m2));
        
        while (low <= high) {
            int mid = (low + high) / 2;
            if (Has_Submatrix_Of_Size(matrix1, matrix2, mid)) {
                max_size = mid;
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        
        return max_size;
    }
    
private:
    bool Has_Submatrix_Of_Size(vector<vector<int>>& mat1, vector<vector<int>>& mat2, int size) {
        int n1 = mat1.size(), m1 = mat1[0].size();
        int n2 = mat2.size(), m2 = mat2[0].size();
        
        set<vector<vector<int>>> submatrices;
        
        for (int i = 0; i <= n1 - size; i++) {
            for (int j = 0; j <= m1 - size; j++) {
                vector<vector<int>> sub(size, vector<int>(size));
                for (int x = 0; x < size; x++) {
                    for (int y = 0; y < size; y++) {
                        sub[x][y] = mat1[i + x][j + y];
                    }
                }
                submatrices.insert(sub);
            }
        }
        
        for (int i = 0; i <= n2 - size; i++) {
            for (int j = 0; j <= m2 - size; j++) {
                vector<vector<int>> sub(size, vector<int>(size));
                for (int x = 0; x < size; x++) {
                    for (int y = 0; y < size; y++) {
                        sub[x][y] = mat2[i + x][j + y];
                    }
                }
                if (submatrices.find(sub) != submatrices.end()) {
                    return true;
                }
            }
        }
        
        return false;
    }
};

void Test_Rasta_And_Kheshtak() {
    Solution sol;
    
    vector<vector<int>> matrix1 = {
        {1, 0, 1},
        {0, 1, 0},
        {1, 0, 1}
    };
    vector<vector<int>> matrix2 = {
        {1, 0, 1, 0},
        {0, 1, 0, 1},
        {1, 0, 1, 0},
        {0, 1, 0, 1}
    };
    int result1 = sol.Solve_2D_Hashing_Binary_Search(matrix1, matrix2);
    assert(result1 == 2);
    
    vector<vector<int>> matrix3 = {
        {1, 1},
        {1, 1}
    };
    vector<vector<int>> matrix4 = {
        {1, 1, 0},
        {1, 1, 0},
        {0, 0, 0}
    };
    int result2 = sol.Solve_2D_Hashing_Binary_Search(matrix3, matrix4);
    assert(result2 == 2);
    
    cout << "All test cases passed!" << endl;
}

int main() {
    Test_Rasta_And_Kheshtak();
    return 0;
}
