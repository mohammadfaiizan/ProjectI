/*
 * Problem: Roti Prata
 * URL: https://www.spoj.com/problems/PRATA/
 * 
 * Problem Statement:
 * Assign parathas to cooks with different ranks.
 * ith paratha by cook of rank R takes R*i time.
 * Minimize total time. Binary search on answer.
 * 
 * Sample Input:
 * P = 10, L = 4, ranks[] = {1, 2, 3, 4}
 * 
 * Sample Output:
 * 12
 */

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    /*
     * Approach: Count how many parathas can be made in given time
     * For each cook with rank R, count parathas: floor((-1 + sqrt(1 + 8*time/R)) / 2)
     * Or use iterative approach: count parathas where R*i*(i+1)/2 <= time
     * 
     * Time Complexity: O(n) where n is number of cooks
     * Space Complexity: O(1)
     */
    long long Count_Parathas(int ranks[], int n, long long time_limit) {
        long long total_parathas = 0;
        for (int i = 0; i < n; i++) {
            long long count = 0;
            long long time_used = 0;
            int paratha_num = 1;
            
            while (time_used + ranks[i] * paratha_num <= time_limit) {
                time_used += ranks[i] * paratha_num;
                count++;
                paratha_num++;
            }
            total_parathas += count;
        }
        return total_parathas;
    }

    /*
     * Approach: Binary search on answer
     * Search for minimum time needed to make P parathas
     * Low = 0, High = P * (P + 1) * max_rank / 2 (upper bound)
     * 
     * Time Complexity: O(n * log(max_time))
     * Space Complexity: O(1)
     */
    long long Min_Time_To_Make_Parathas(int P, int ranks[], int n) {
        long long low = 0;
        int max_rank = *max_element(ranks, ranks + n);
        long long high = (long long)P * (P + 1) * max_rank / 2;
        long long result = high;
        
        while (low <= high) {
            long long mid = low + (high - low) / 2;
            long long parathas_made = Count_Parathas(ranks, n, mid);
            
            if (parathas_made >= P) {
                result = mid;
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        
        return result;
    }
};

void Test_Roti_Prata() {
    Solution sol;
    
    int ranks1[] = {1, 2, 3, 4};
    assert(sol.Min_Time_To_Make_Parathas(10, ranks1, 4) == 12);
    
    int ranks2[] = {1};
    assert(sol.Min_Time_To_Make_Parathas(8, ranks2, 1) == 36);
    
    int ranks3[] = {1, 1, 1, 1};
    assert(sol.Min_Time_To_Make_Parathas(10, ranks3, 4) == 4);
    
    assert(sol.Count_Parathas(ranks1, 4, 12) >= 10);
    assert(sol.Count_Parathas(ranks1, 4, 11) < 10);
    
    cout << "All test cases passed!" << endl;
}

int main() {
    Test_Roti_Prata();
    return 0;
}
