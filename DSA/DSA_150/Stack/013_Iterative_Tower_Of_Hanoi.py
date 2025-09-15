"""
Problem: Iterative Tower of Hanoi
URL: https://www.geeksforgeeks.org/problems/tower-of-hanoi-1587115621/1

Problem Statement:
The Tower of Hanoi is a classic problem where you have three rods and n disks of different sizes.
Initially, all disks are stacked on the first rod in decreasing order of size (largest at bottom).
The goal is to move all disks to the third rod following these rules:
1. Only one disk can be moved at a time
2. Only the top disk from any rod can be moved
3. A disk cannot be placed on top of a smaller disk

Sample Input/Output:
Input: n = 2
Output: ["move disk 1 from rod A to rod B", "move disk 2 from rod A to rod C", "move disk 1 from rod B to rod C"]
Explanation: Move 2 disks from A to C using B as auxiliary

Input: n = 3
Output: 7 moves total (2^3 - 1 = 7)
"""

from typing import List

class Solution:
    def Tower_Of_Hanoi_Recursive(self, n: int) -> List[str]:
        """
        Recursive Approach - Classic recursive solution
        Time Complexity: O(2^n)
        Space Complexity: O(n) for recursion stack
        """
        moves = []
        
        def Hanoi(n: int, source: str, destination: str, auxiliary: str):
            if n == 1:
                moves.append(f"move disk {n} from rod {source} to rod {destination}")
                return
            
            Hanoi(n - 1, source, auxiliary, destination)
            moves.append(f"move disk {n} from rod {source} to rod {destination}")
            Hanoi(n - 1, auxiliary, destination, source)
        
        Hanoi(n, 'A', 'C', 'B')
        return moves
    
    def Tower_Of_Hanoi_Stack_Iterative(self, n: int) -> List[str]:
        """
        Stack Iterative Approach - Using stack to simulate recursion
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        """
        moves = []
        stack = [(n, 'A', 'C', 'B', False)]
        
        while stack:
            disks, src, dest, aux, processed = stack.pop()
            
            if disks == 1:
                moves.append(f"move disk {disks} from rod {src} to rod {dest}")
            elif not processed:
                stack.append((disks - 1, aux, dest, src, False))
                stack.append((disks, src, dest, aux, True))
                stack.append((disks - 1, src, aux, dest, False))
            else:
                moves.append(f"move disk {disks} from rod {src} to rod {dest}")
        
        return moves
    
    def Tower_Of_Hanoi_Iterative_Optimal(self, n: int) -> List[str]:
        """
        Iterative Optimal Approach - Without recursion using pattern
        Time Complexity: O(2^n)
        Space Complexity: O(1)
        """
        moves = []
        total_moves = (1 << n) - 1
        
        source = [i for i in range(n, 0, -1)]
        auxiliary = []
        destination = []
        
        rods = {'A': source, 'B': auxiliary, 'C': destination}
        rod_names = ['A', 'B', 'C']
        
        if n % 2 == 0:
            rod_names[1], rod_names[2] = rod_names[2], rod_names[1]
        
        for i in range(total_moves):
            from_rod = rod_names[self.Get_From_Rod(i + 1, n)]
            to_rod = rod_names[self.Get_To_Rod(i + 1, n)]
            
            if not rods[from_rod]:
                from_rod, to_rod = to_rod, from_rod
            elif rods[to_rod] and rods[from_rod][-1] > rods[to_rod][-1]:
                from_rod, to_rod = to_rod, from_rod
            
            disk = rods[from_rod].pop()
            rods[to_rod].append(disk)
            moves.append(f"move disk {disk} from rod {from_rod} to rod {to_rod}")
        
        return moves
    
    def Get_From_Rod(self, move_num: int, n: int) -> int:
        return (move_num & (move_num - 1)) % 3
    
    def Get_To_Rod(self, move_num: int, n: int) -> int:
        return ((move_num | (move_num - 1)) + 1) % 3
    
    def Tower_Of_Hanoi_Binary_Pattern(self, n: int) -> List[str]:
        """
        Binary Pattern Approach - Using binary representation
        Time Complexity: O(2^n)
        Space Complexity: O(1)
        """
        moves = []
        total_moves = (1 << n) - 1
        
        for i in range(1, total_moves + 1):
            from_peg = (i & i - 1) % 3
            to_peg = ((i | i - 1) + 1) % 3
            
            if n % 2 == 0:
                if from_peg == 1:
                    from_peg = 2
                elif from_peg == 2:
                    from_peg = 1
                
                if to_peg == 1:
                    to_peg = 2
                elif to_peg == 2:
                    to_peg = 1
            
            disk = self.Find_Disk_To_Move(i, n)
            from_rod = chr(ord('A') + from_peg)
            to_rod = chr(ord('A') + to_peg)
            
            moves.append(f"move disk {disk} from rod {from_rod} to rod {to_rod}")
        
        return moves
    
    def Find_Disk_To_Move(self, move_num: int, n: int) -> int:
        return (move_num & -move_num).bit_length()
    
    def Tower_Of_Hanoi_State_Machine(self, n: int) -> List[str]:
        """
        State Machine Approach - Track state of each rod
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        """
        moves = []
        rods = {
            'A': list(range(n, 0, -1)),
            'B': [],
            'C': []
        }
        
        total_moves = (1 << n) - 1
        
        for move in range(total_moves):
            smallest_disk = float('inf')
            from_rod = to_rod = None
            
            for rod_name in ['A', 'B', 'C']:
                if rods[rod_name] and rods[rod_name][-1] < smallest_disk:
                    if (rods[rod_name][-1] + move) % 2 == 1:
                        smallest_disk = rods[rod_name][-1]
                        from_rod = rod_name
            
            if from_rod:
                for rod_name in ['A', 'B', 'C']:
                    if rod_name != from_rod:
                        if not rods[rod_name] or rods[rod_name][-1] > smallest_disk:
                            to_rod = rod_name
                            break
                
                disk = rods[from_rod].pop()
                rods[to_rod].append(disk)
                moves.append(f"move disk {disk} from rod {from_rod} to rod {to_rod}")
        
        return moves
    
    def Tower_Of_Hanoi_Mathematical(self, n: int) -> List[str]:
        """
        Mathematical Approach - Using mathematical formula
        Time Complexity: O(2^n)
        Space Complexity: O(1)
        """
        moves = []
        
        def Get_Rod_Name(rod_num: int, n: int) -> str:
            if n % 2 == 1:
                return ['A', 'B', 'C'][rod_num]
            else:
                return ['A', 'C', 'B'][rod_num]
        
        for i in range(1, (1 << n)):
            from_rod = Get_Rod_Name((i & i - 1) % 3, n)
            to_rod = Get_Rod_Name(((i | i - 1) + 1) % 3, n)
            
            disk = bin(i & -i).count('1')
            moves.append(f"move disk {disk} from rod {from_rod} to rod {to_rod}")
        
        return moves

def Test_Tower_Of_Hanoi():
    solution = Solution()
    
    test_cases = [1, 2, 3, 4]
    
    for n in test_cases:
        print(f"Testing n = {n}:")
        
        result1 = solution.Tower_Of_Hanoi_Recursive(n)
        result2 = solution.Tower_Of_Hanoi_Stack_Iterative(n)
        result3 = solution.Tower_Of_Hanoi_Iterative_Optimal(n)
        result4 = solution.Tower_Of_Hanoi_Binary_Pattern(n)
        result5 = solution.Tower_Of_Hanoi_Mathematical(n)
        
        expected_moves = (1 << n) - 1
        
        print(f"Expected moves: {expected_moves}")
        print(f"Recursive moves: {len(result1)}")
        print(f"Stack Iterative moves: {len(result2)}")
        print(f"Iterative Optimal moves: {len(result3)}")
        print(f"Binary Pattern moves: {len(result4)}")
        print(f"Mathematical moves: {len(result5)}")
        
        if n <= 3:
            print("Recursive solution:")
            for move in result1:
                print(f"  {move}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Tower_Of_Hanoi()
