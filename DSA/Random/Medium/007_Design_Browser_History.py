"""
Problem: Design Browser History
URL: https://leetcode.com/problems/design-browser-history/

Problem Statement:
You have a browser of one tab where you start on the homepage and you can visit another url, 
get back in the history number of steps or move forward in the history number of steps.

Implement the BrowserHistory class:
- BrowserHistory(string homepage) Initializes the object with the homepage of the browser.
- void visit(string url) Visits url from the current page. It clears up all the forward history.
- string back(int steps) Move steps back in history. Return the current url after moving back.
- string forward(int steps) Move steps forward in history. Return the current url.

Sample Input/Output:
Input:
["BrowserHistory","visit","visit","visit","back","back","forward","visit","forward","back","back"]
[["leetcode.com"],["google.com"],["facebook.com"],["youtube.com"],[1],[1],[1],["linkedin.com"],[2],[2],[7]]
Output:
[null,null,null,null,"facebook.com","google.com","facebook.com",null,"linkedin.com","google.com","leetcode.com"]
"""

from typing import List

class Browser_History_List:
    """
    List Approach
    Time Complexity: O(1) for all operations
    Space Complexity: O(n)
    """
    def __init__(self, homepage: str):
        self.history = [homepage]
        self.current = 0
    
    def Visit(self, url: str) -> None:
        self.current += 1
        self.history = self.history[:self.current]
        self.history.append(url)
    
    def Back(self, steps: int) -> str:
        self.current = max(0, self.current - steps)
        return self.history[self.current]
    
    def Forward(self, steps: int) -> str:
        self.current = min(len(self.history) - 1, self.current + steps)
        return self.history[self.current]

class Browser_History_Stack:
    """
    Two Stacks Approach
    Time Complexity: O(n) for back/forward
    Space Complexity: O(n)
    """
    def __init__(self, homepage: str):
        self.back_stack = [homepage]
        self.forward_stack = []
    
    def Visit(self, url: str) -> None:
        self.back_stack.append(url)
        self.forward_stack = []
    
    def Back(self, steps: int) -> str:
        while steps > 0 and len(self.back_stack) > 1:
            self.forward_stack.append(self.back_stack.pop())
            steps -= 1
        return self.back_stack[-1]
    
    def Forward(self, steps: int) -> str:
        while steps > 0 and self.forward_stack:
            self.back_stack.append(self.forward_stack.pop())
            steps -= 1
        return self.back_stack[-1]

class Doubly_List_Node:
    def __init__(self, url: str, prev=None, next=None):
        self.url = url
        self.prev = prev
        self.next = next

class Browser_History_Doubly_Linked_List:
    """
    Doubly Linked List Approach
    Time Complexity: O(n) for back/forward
    Space Complexity: O(n)
    """
    def __init__(self, homepage: str):
        self.current = Doubly_List_Node(homepage)
    
    def Visit(self, url: str) -> None:
        new_node = Doubly_List_Node(url, self.current)
        self.current.next = new_node
        self.current = new_node
    
    def Back(self, steps: int) -> str:
        while steps > 0 and self.current.prev:
            self.current = self.current.prev
            steps -= 1
        return self.current.url
    
    def Forward(self, steps: int) -> str:
        while steps > 0 and self.current.next:
            self.current = self.current.next
            steps -= 1
        return self.current.url

class Browser_History_Deque:
    """
    Deque Approach
    Time Complexity: O(1) amortized
    Space Complexity: O(n)
    """
    def __init__(self, homepage: str):
        from collections import deque
        self.history = deque([homepage])
        self.current_index = 0
    
    def Visit(self, url: str) -> None:
        while len(self.history) > self.current_index + 1:
            self.history.pop()
        self.history.append(url)
        self.current_index += 1
    
    def Back(self, steps: int) -> str:
        self.current_index = max(0, self.current_index - steps)
        return self.history[self.current_index]
    
    def Forward(self, steps: int) -> str:
        self.current_index = min(len(self.history) - 1, self.current_index + steps)
        return self.history[self.current_index]

def Test_Browser_History():
    operations = ["BrowserHistory","visit","visit","visit","back","back","forward","visit","forward","back","back"]
    values = [["leetcode.com"],["google.com"],["facebook.com"],["youtube.com"],[1],[1],[1],["linkedin.com"],[2],[2],[7]]
    expected = [None,None,None,None,"facebook.com","google.com","facebook.com",None,"linkedin.com","google.com","leetcode.com"]
    
    for approach_name, BrowserClass in [
        ("List", Browser_History_List),
        ("Stack", Browser_History_Stack),
        ("Doubly Linked List", Browser_History_Doubly_Linked_List),
        ("Deque", Browser_History_Deque)
    ]:
        print(f"Testing {approach_name} Approach:")
        browser = None
        results = []
        
        for i, op in enumerate(operations):
            if op == "BrowserHistory":
                browser = BrowserClass(values[i][0])
                results.append(None)
            elif op == "visit":
                browser.Visit(values[i][0])
                results.append(None)
            elif op == "back":
                results.append(browser.Back(values[i][0]))
            elif op == "forward":
                results.append(browser.Forward(values[i][0]))
        
        print(f"Expected: {expected}")
        print(f"Got:      {results}")
        print(f"Match: {results == expected}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Browser_History()

