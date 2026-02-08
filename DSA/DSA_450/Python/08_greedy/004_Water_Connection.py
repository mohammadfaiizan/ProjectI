"""
Problem: Water Connection
URL: https://practice.geeksforgeeks.org/problems/water-connection-problem5822/1

Problem Statement:
There are n houses and p water pipes in Geek Colony. Each house has at most one pipe going into it and at most one pipe going out of it. Geek Colony needs to install water tanks and taps in the colony. Houses with no incoming pipe get a water tank and houses with no outgoing pipe get a tap. Find the source houses (with tanks), destination houses (with taps), and minimum diameter along each path.

Sample Input/Output:
Input: n = 9, p = 6, a[] = {7,5,4,2,9,3}, b[] = {4,9,6,8,7,1}, d[] = {98,72,10,22,17,66}
Output: 3
        2 8 22
        3 1 66
        5 6 10
Explanation: Three paths: 2->8 (diameter 22), 3->1 (diameter 66), 5->6 (diameter 10)
"""


class Solution:
    def Solve_Water_Connection_DFS(self, n, p, a, b, d):
        """
        DFS traversal to find source-destination paths with minimum diameter
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        in_degree = [0] * (n + 1)
        out_degree = [0] * (n + 1)
        graph = [(-1, -1)] * (n + 1)
        
        for i in range(p):
            graph[a[i]] = (b[i], d[i])
            in_degree[b[i]] += 1
            out_degree[a[i]] += 1
        
        result = []
        
        for i in range(1, n + 1):
            if in_degree[i] == 0 and out_degree[i] > 0:
                start = i
                end_node = i
                min_diameter = float('inf')
                
                while graph[end_node][0] != -1:
                    min_diameter = min(min_diameter, graph[end_node][1])
                    end_node = graph[end_node][0]
                
                if end_node != start:
                    result.append([start, end_node, min_diameter])
        
        return result


def Test_Water_Connection():
    solution = Solution()
    n, p = 9, 6
    a = [7, 5, 4, 2, 9, 3]
    b = [4, 9, 6, 8, 7, 1]
    d = [98, 72, 10, 22, 17, 66]
    result = solution.Solve_Water_Connection_DFS(n, p, a, b, d)
    print(f"Number of paths: {len(result)}")
    for path in result:
        print(f"{path[0]} {path[1]} {path[2]}")


if __name__ == "__main__":
    Test_Water_Connection()
