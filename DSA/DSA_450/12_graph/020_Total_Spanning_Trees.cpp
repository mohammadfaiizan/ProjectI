/*
Problem: Total Number of Spanning Trees in a Graph
URL: https://www.geeksforgeeks.org/total-number-spanning-trees-graph/

Problem Statement:
Count the total number of spanning trees in a connected undirected graph using Kirchhoff's Matrix Tree Theorem. The theorem states that the number of spanning trees equals any cofactor of the Laplacian matrix.

Sample Input/Output:
Input: Triangle graph (3 vertices, 3 edges)
Output: 3 spanning trees
Input: Complete graph K4 (4 vertices, 6 edges)
Output: 16 spanning trees
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Spanning_Trees_Kirchhoff(int V, vector<vector<int>>& graph) {
        /*
        Construct Laplacian matrix, compute cofactor/determinant
        Time Complexity: O(V^3)
        Space Complexity: O(V^2)
        */
        vector<vector<int>> laplacian(V, vector<int>(V, 0));
        
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                if (i != j && graph[i][j] > 0) {
                    laplacian[i][j] = -graph[i][j];
                    laplacian[i][i] += graph[i][j];
                }
            }
        }
        
        vector<vector<double>> matrix(V - 1, vector<double>(V - 1));
        for (int i = 1; i < V; i++) {
            for (int j = 1; j < V; j++) {
                matrix[i - 1][j - 1] = laplacian[i][j];
            }
        }
        
        return (int)round(Determinant(matrix));
    }
    
private:
    double Determinant(vector<vector<double>>& mat) {
        int n = mat.size();
        if (n == 1) return mat[0][0];
        if (n == 2) return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
        
        double det = 1.0;
        for (int i = 0; i < n; i++) {
            int maxRow = i;
            for (int k = i + 1; k < n; k++) {
                if (abs(mat[k][i]) > abs(mat[maxRow][i])) {
                    maxRow = k;
                }
            }
            
            if (maxRow != i) {
                swap(mat[i], mat[maxRow]);
                det *= -1;
            }
            
            if (abs(mat[i][i]) < 1e-9) return 0;
            
            for (int k = i + 1; k < n; k++) {
                double factor = mat[k][i] / mat[i][i];
                for (int j = i; j < n; j++) {
                    mat[k][j] -= factor * mat[i][j];
                }
            }
        }
        
        for (int i = 0; i < n; i++) {
            det *= mat[i][i];
        }
        
        return det;
    }
};

void Test_Spanning_Trees() {
    Solution solution;
    
    cout << "Test Case 1: Triangle graph (3 vertices)" << endl;
    int V1 = 3;
    vector<vector<int>> graph1 = {
        {0, 1, 1},
        {1, 0, 1},
        {1, 1, 0}
    };
    cout << "Number of spanning trees: " << solution.Spanning_Trees_Kirchhoff(V1, graph1) << endl;
    
    cout << "\nTest Case 2: Complete graph K4 (4 vertices)" << endl;
    int V2 = 4;
    vector<vector<int>> graph2 = {
        {0, 1, 1, 1},
        {1, 0, 1, 1},
        {1, 1, 0, 1},
        {1, 1, 1, 0}
    };
    cout << "Number of spanning trees: " << solution.Spanning_Trees_Kirchhoff(V2, graph2) << endl;
    
    cout << "\nTest Case 3: Simple path graph (4 vertices)" << endl;
    int V3 = 4;
    vector<vector<int>> graph3 = {
        {0, 1, 0, 0},
        {1, 0, 1, 0},
        {0, 1, 0, 1},
        {0, 0, 1, 0}
    };
    cout << "Number of spanning trees: " << solution.Spanning_Trees_Kirchhoff(V3, graph3) << endl;
    
    cout << "\nTest Case 4: Star graph (5 vertices)" << endl;
    int V4 = 5;
    vector<vector<int>> graph4 = {
        {0, 1, 1, 1, 1},
        {1, 0, 0, 0, 0},
        {1, 0, 0, 0, 0},
        {1, 0, 0, 0, 0},
        {1, 0, 0, 0, 0}
    };
    cout << "Number of spanning trees: " << solution.Spanning_Trees_Kirchhoff(V4, graph4) << endl;
}

int main() {
    Test_Spanning_Trees();
    return 0;
}
