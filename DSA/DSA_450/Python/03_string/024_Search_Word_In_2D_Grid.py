"""
Problem: Search a Word in a 2D Grid (8 directions)
URL: https://practice.geeksforgeeks.org/problems/find-the-string-in-grid0111/1

Problem Statement:
Given a 2D grid of characters and a word, find all occurrences of the word in the grid.
The word can be matched in all 8 directions. Return the starting coordinates.

Sample Input/Output:
Input: grid = {{'a','b','c'},{'d','r','f'},{'g','h','i'}}, word = "abc"
Output: {{0,0}}
"""


class Solution:
    def Search_Word_Eight_Dir(self, grid, word):
        """
        Search in all 8 directions from each cell
        Time Complexity: O(R * C * 8 * L)
        Space Complexity: O(1) excluding result
        """
        R = len(grid)
        C = len(grid[0])
        dx = [-1, -1, -1, 0, 0, 1, 1, 1]
        dy = [-1, 0, 1, -1, 1, -1, 0, 1]
        ans = []

        for i in range(R):
            for j in range(C):
                if grid[i][j] != word[0]:
                    continue
                for d in range(8):
                    rd, cd = i + dx[d], j + dy[d]
                    k = 1
                    while k < len(word):
                        if rd < 0 or rd >= R or cd < 0 or cd >= C:
                            break
                        if grid[rd][cd] != word[k]:
                            break
                        rd += dx[d]
                        cd += dy[d]
                        k += 1
                    if k == len(word):
                        ans.append([i, j])
                        break

        return ans

    def Search_Word_DFS(self, grid, word):
        """
        DFS from each cell (4 directions with bending allowed)
        Time Complexity: O(R * C * 4^L)
        Space Complexity: O(L) recursion stack
        """
        R = len(grid)
        C = len(grid[0])
        ans = []
        visited = [[False] * C for _ in range(R)]

        for i in range(R):
            for j in range(C):
                if self.DFS(grid, word, i, j, 0, visited):
                    ans.append([i, j])

        return ans

    def DFS(self, grid, word, r, c, idx, visited):
        if idx == len(word):
            return True
        R = len(grid)
        C = len(grid[0])
        if r < 0 or r >= R or c < 0 or c >= C:
            return False
        if visited[r][c] or grid[r][c] != word[idx]:
            return False

        visited[r][c] = True
        dr = [0, 0, 1, -1]
        dc = [1, -1, 0, 0]
        for d in range(4):
            if self.DFS(grid, word, r + dr[d], c + dc[d], idx + 1, visited):
                visited[r][c] = False
                return True
        visited[r][c] = False
        return False


def Test_Search_Word_In_2D_Grid():
    sol = Solution()
    grid = [
        ['a', 'b', 'c', 'd'],
        ['e', 'f', 'c', 'h'],
        ['i', 'j', 'b', 'a'],
        ['m', 'n', 'o', 'p']
    ]

    words = ["abc", "abcba", "afj"]
    for word in words:
        print(f"Word: {word}")

        r1 = sol.Search_Word_Eight_Dir(grid, word)
        print(f"Eight Dir: ", end="")
        for pos in r1:
            print(f"[{pos[0]},{pos[1]}]", end=" ")
        print()

        r2 = sol.Search_Word_DFS(grid, word)
        print(f"DFS: ", end="")
        for pos in r2:
            print(f"[{pos[0]},{pos[1]}]", end=" ")
        print()

        print('-' * 50)


if __name__ == "__main__":
    Test_Search_Word_In_2D_Grid()
