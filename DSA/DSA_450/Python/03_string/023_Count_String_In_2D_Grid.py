"""
Problem: Count of Number of Given String in a 2D Character Array
URL: https://www.geeksforgeeks.org/find-count-number-given-string-present-2d-character-array/

Problem Statement:
Given a 2D character array and a string, find the count of occurrences of the
string in the 2D array. The string can be searched in all 4 directions
(up, down, left, right) and can bend at any point.

Sample Input/Output:
Input: grid = {{"BBABBM"}, {"CBMBBA"}, {"IBABBG"}, {"GOZBBI"}, {"ABBBBC"}, {"MCIGAM"}}
       word = "MAGIC"
Output: 3
"""


class Solution:
    def Count_String_Backtrack(self, grid, word):
        """
        Backtracking DFS - search from every cell
        Time Complexity: O(R * C * 4^L) where L = word length
        Space Complexity: O(L) recursion stack
        """
        R = len(grid)
        C = len(grid[0])
        found = 0
        for i in range(R):
            for j in range(C):
                found += self.DFS(grid, word, i, j, 0, R, C)
        return found

    def Count_String_Eight_Dir(self, grid, word):
        """
        Search in all 8 directions (straight lines only, no bending)
        Time Complexity: O(R * C * 8 * L)
        Space Complexity: O(1)
        """
        R = len(grid)
        C = len(grid[0])
        dx = [-1, -1, -1, 0, 0, 1, 1, 1]
        dy = [-1, 0, 1, -1, 1, -1, 0, 1]
        count = 0

        for i in range(R):
            for j in range(C):
                for d in range(8):
                    rd, cd = i, j
                    k = 0
                    while k < len(word):
                        if rd < 0 or rd >= R or cd < 0 or cd >= C:
                            break
                        if grid[rd][cd] != word[k]:
                            break
                        rd += dx[d]
                        cd += dy[d]
                        k += 1
                    if k == len(word):
                        count += 1

        return count

    def DFS(self, grid, word, r, c, idx, R, C):
        if idx == len(word):
            return 1
        if r < 0 or r >= R or c < 0 or c >= C:
            return 0
        if grid[r][c] != word[idx]:
            return 0

        temp = grid[r][c]
        grid_list = [list(row) for row in grid]
        grid_list[r][c] = '#'
        grid = [''.join(row) for row in grid_list]

        found = 0
        dr = [0, 0, 1, -1]
        dc = [1, -1, 0, 0]
        for d in range(4):
            found += self.DFS(grid, word, r + dr[d], c + dc[d], idx + 1, R, C)

        grid_list[r][c] = temp
        grid = [''.join(row) for row in grid_list]
        return found


def Test_Count_String_In_2D_Grid():
    sol = Solution()

    grid1 = ["BBABBM", "CBMBBA", "IBABBG", "GOZBBI", "ABBBBC", "MCIGAM"]
    word1 = "MAGIC"
    print(f"Grid 1, Word: {word1}")
    print(f"Backtrack: {sol.Count_String_Backtrack(grid1, word1)}")

    grid2 = [
        ['A', 'B', 'C'],
        ['D', 'E', 'F'],
        ['G', 'H', 'I']
    ]
    word2 = "ABC"
    print(f"Grid 2, Word: {word2}")
    grid2_str = [''.join(row) for row in grid2]
    print(f"Eight Dir: {sol.Count_String_Eight_Dir(grid2_str, word2)}")

    print('-' * 50)


if __name__ == "__main__":
    Test_Count_String_In_2D_Grid()
