"""
1472. Design Browser History - Multiple Approaches
Difficulty: Easy

You have a browser of one tab where you start on the homepage and you can visit another url, get back in the history number of steps or move forward in the history number of steps.

Implement the BrowserHistory class:
- BrowserHistory(string homepage) Initializes the object with the homepage of the browser.
- void visit(string url) Visits url from the current page. It clears up all the forward history.
- string back(int steps) Move steps back in history. Return the current url after moving back in history at most steps.
- string forward(int steps) Move steps forward in history. Return the current url after moving forward in history at most steps.
"""

from typing import List

class BrowserHistoryStack:
    """
    Approach 1: Two Stacks Implementation (Optimal)
    
    Use two stacks to maintain back and forward history.
    
    Time: O(1) for visit, O(min(steps, history_size)) for back/forward
    Space: O(n) where n is number of pages visited
    """
    
    def __init__(self, homepage: str):
        self.current = homepage
        self.back_stack = []
        self.forward_stack = []
    
    def visit(self, url: str) -> None:
        """Visit a new URL"""
        # Push current page to back stack
        self.back_stack.append(self.current)
        
        # Set new current page
        self.current = url
        
        # Clear forward history
        self.forward_stack.clear()
    
    def back(self, steps: int) -> str:
        """Go back in history"""
        # Move back at most 'steps' or until we reach the beginning
        actual_steps = min(steps, len(self.back_stack))
        
        for _ in range(actual_steps):
            # Push current to forward stack
            self.forward_stack.append(self.current)
            
            # Pop from back stack to current
            self.current = self.back_stack.pop()
        
        return self.current
    
    def forward(self, steps: int) -> str:
        """Go forward in history"""
        # Move forward at most 'steps' or until we reach the end
        actual_steps = min(steps, len(self.forward_stack))
        
        for _ in range(actual_steps):
            # Push current to back stack
            self.back_stack.append(self.current)
            
            # Pop from forward stack to current
            self.current = self.forward_stack.pop()
        
        return self.current


class BrowserHistoryArray:
    """
    Approach 2: Array Implementation
    
    Use array with pointers to track history.
    
    Time: O(1) for all operations, Space: O(n)
    """
    
    def __init__(self, homepage: str):
        self.history = [homepage]
        self.current_index = 0
        self.max_index = 0  # Furthest we can go forward
    
    def visit(self, url: str) -> None:
        """Visit a new URL"""
        self.current_index += 1
        
        # If we're at the end, append new page
        if self.current_index >= len(self.history):
            self.history.append(url)
        else:
            # Overwrite existing page (clearing forward history)
            self.history[self.current_index] = url
        
        # Update max index (clear forward history)
        self.max_index = self.current_index
    
    def back(self, steps: int) -> str:
        """Go back in history"""
        self.current_index = max(0, self.current_index - steps)
        return self.history[self.current_index]
    
    def forward(self, steps: int) -> str:
        """Go forward in history"""
        self.current_index = min(self.max_index, self.current_index + steps)
        return self.history[self.current_index]


class BrowserHistoryLinkedList:
    """
    Approach 3: Doubly Linked List Implementation
    
    Use doubly linked list for navigation.
    
    Time: O(min(steps, history_size)) for back/forward, O(1) for visit
    Space: O(n)
    """
    
    class Node:
        def __init__(self, url: str):
            self.url = url
            self.prev = None
            self.next = None
    
    def __init__(self, homepage: str):
        self.current = self.Node(homepage)
    
    def visit(self, url: str) -> None:
        """Visit a new URL"""
        new_node = self.Node(url)
        
        # Link new node to current
        new_node.prev = self.current
        self.current.next = new_node
        
        # Move to new node
        self.current = new_node
    
    def back(self, steps: int) -> str:
        """Go back in history"""
        for _ in range(steps):
            if self.current.prev:
                self.current = self.current.prev
            else:
                break
        
        return self.current.url
    
    def forward(self, steps: int) -> str:
        """Go forward in history"""
        for _ in range(steps):
            if self.current.next:
                self.current = self.current.next
            else:
                break
        
        return self.current.url


class BrowserHistoryDeque:
    """
    Approach 4: Deque Implementation
    
    Use deque for efficient operations at both ends.
    
    Time: O(min(steps, history_size)) for back/forward, O(1) for visit
    Space: O(n)
    """
    
    def __init__(self, homepage: str):
        from collections import deque
        self.history = deque([homepage])
        self.current_index = 0
    
    def visit(self, url: str) -> None:
        """Visit a new URL"""
        # Remove all pages after current (forward history)
        while len(self.history) > self.current_index + 1:
            self.history.pop()
        
        # Add new page
        self.history.append(url)
        self.current_index += 1
    
    def back(self, steps: int) -> str:
        """Go back in history"""
        self.current_index = max(0, self.current_index - steps)
        return self.history[self.current_index]
    
    def forward(self, steps: int) -> str:
        """Go forward in history"""
        self.current_index = min(len(self.history) - 1, self.current_index + steps)
        return self.history[self.current_index]


def test_browser_history_implementations():
    """Test browser history implementations"""
    
    implementations = [
        ("Two Stacks", BrowserHistoryStack),
        ("Array", BrowserHistoryArray),
        ("Linked List", BrowserHistoryLinkedList),
        ("Deque", BrowserHistoryDeque),
    ]
    
    test_cases = [
        {
            "homepage": "leetcode.com",
            "operations": ["visit", "visit", "back", "back", "forward", "visit", "forward", "back", "back"],
            "values": ["google.com", "facebook.com", 1, 1, 1, "youtube.com", 2, 2, 7],
            "expected": [None, None, "google.com", "leetcode.com", "google.com", None, "google.com", "leetcode.com", "leetcode.com"],
            "description": "Example 1"
        },
        {
            "homepage": "home.com",
            "operations": ["visit", "back", "forward"],
            "values": ["page1.com", 1, 1],
            "expected": [None, "home.com", "page1.com"],
            "description": "Simple navigation"
        },
        {
            "homepage": "start.com",
            "operations": ["visit", "visit", "visit", "back", "visit", "forward"],
            "values": ["page1.com", "page2.com", "page3.com", 2, "new.com", 1],
            "expected": [None, None, None, "page1.com", None, "new.com"],
            "description": "Visit clears forward history"
        },
    ]
    
    print("=== Testing Browser History Implementations ===")
    
    for impl_name, impl_class in implementations:
        print(f"\n--- {impl_name} Implementation ---")
        
        for test_case in test_cases:
            try:
                browser = impl_class(test_case["homepage"])
                results = []
                
                for i, op in enumerate(test_case["operations"]):
                    if op == "visit":
                        browser.visit(test_case["values"][i])
                        results.append(None)
                    elif op == "back":
                        result = browser.back(test_case["values"][i])
                        results.append(result)
                    elif op == "forward":
                        result = browser.forward(test_case["values"][i])
                        results.append(result)
                
                expected = test_case["expected"]
                status = "✓" if results == expected else "✗"
                
                print(f"  {test_case['description']:20} | {status} | {results}")
                if results != expected:
                    print(f"    Expected: {expected}")
                
            except Exception as e:
                print(f"  {test_case['description']:20} | ERROR: {str(e)[:40]}")


def demonstrate_two_stacks_approach():
    """Demonstrate two stacks approach step by step"""
    print("\n=== Two Stacks Approach Step-by-Step Demo ===")
    
    browser = BrowserHistoryStack("leetcode.com")
    
    operations = [
        ("visit", "google.com"),
        ("visit", "facebook.com"),
        ("back", 1),
        ("back", 1),
        ("forward", 1),
        ("visit", "youtube.com"),
        ("forward", 2),
    ]
    
    print("Strategy: Use two stacks for back and forward history")
    print(f"Initial state: current = '{browser.current}'")
    
    for i, (op, value) in enumerate(operations):
        print(f"\nStep {i+1}: {op}({value})")
        print(f"  Before: current = '{browser.current}'")
        print(f"  Back stack: {browser.back_stack}")
        print(f"  Forward stack: {browser.forward_stack}")
        
        if op == "visit":
            browser.visit(value)
            result = None
        elif op == "back":
            result = browser.back(value)
        elif op == "forward":
            result = browser.forward(value)
        
        print(f"  After: current = '{browser.current}'")
        print(f"  Back stack: {browser.back_stack}")
        print(f"  Forward stack: {browser.forward_stack}")
        if result:
            print(f"  Returned: '{result}'")


def visualize_browser_navigation():
    """Visualize browser navigation"""
    print("\n=== Browser Navigation Visualization ===")
    
    browser = BrowserHistoryArray("home.com")
    
    operations = [
        ("visit", "page1.com"),
        ("visit", "page2.com"),
        ("visit", "page3.com"),
        ("back", 2),
        ("visit", "new.com"),
        ("forward", 1),
    ]
    
    print("Array-based implementation visualization:")
    
    for i, (op, value) in enumerate(operations):
        print(f"\nStep {i+1}: {op}({value})")
        print(f"  History: {browser.history[:browser.max_index+1]}")
        print(f"  Current index: {browser.current_index} -> '{browser.history[browser.current_index]}'")
        
        if op == "visit":
            browser.visit(value)
        elif op == "back":
            result = browser.back(value)
            print(f"  Returned: '{result}'")
        elif op == "forward":
            result = browser.forward(value)
            print(f"  Returned: '{result}'")
        
        print(f"  After: History: {browser.history[:browser.max_index+1]}")
        print(f"  Current index: {browser.current_index} -> '{browser.history[browser.current_index]}'")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Web browser implementation
    print("1. Web Browser Implementation:")
    browser = BrowserHistoryStack("https://www.google.com")
    
    browsing_session = [
        ("visit", "https://www.stackoverflow.com"),
        ("visit", "https://www.github.com"),
        ("visit", "https://www.leetcode.com"),
        ("back", 2),  # Go back to stackoverflow
        ("visit", "https://www.reddit.com"),  # This clears forward history
        ("back", 1),  # Go back to stackoverflow
        ("forward", 1),  # Go forward to reddit
    ]
    
    print("  Browsing session:")
    for op, value in browsing_session:
        if op == "visit":
            browser.visit(value)
            print(f"    Visited: {value}")
        elif op == "back":
            result = browser.back(value)
            print(f"    Back {value} steps -> {result}")
        elif op == "forward":
            result = browser.forward(value)
            print(f"    Forward {value} steps -> {result}")
    
    # Application 2: Document editor undo/redo
    print(f"\n2. Document Editor Undo/Redo System:")
    
    class DocumentEditor:
        def __init__(self):
            self.browser = BrowserHistoryStack("empty_document")
            self.current_content = "empty_document"
        
        def type_text(self, text):
            self.browser.visit(f"{self.current_content}+{text}")
            self.current_content = f"{self.current_content}+{text}"
        
        def undo(self, steps=1):
            result = self.browser.back(steps)
            self.current_content = result
            return result
        
        def redo(self, steps=1):
            result = self.browser.forward(steps)
            self.current_content = result
            return result
    
    editor = DocumentEditor()
    
    editing_session = [
        ("type", "Hello"),
        ("type", " World"),
        ("type", "!"),
        ("undo", 2),
        ("type", " Python"),
        ("undo", 1),
        ("redo", 1),
    ]
    
    print("  Document editing session:")
    for action, value in editing_session:
        if action == "type":
            editor.type_text(value)
            print(f"    Typed '{value}' -> Document: {editor.current_content}")
        elif action == "undo":
            result = editor.undo(value)
            print(f"    Undo {value} -> Document: {result}")
        elif action == "redo":
            result = editor.redo(value)
            print(f"    Redo {value} -> Document: {result}")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Two Stacks", "O(1)", "O(k)", "O(k)", "O(n)", "k = steps"),
        ("Array", "O(1)", "O(1)", "O(1)", "O(n)", "Index-based navigation"),
        ("Linked List", "O(1)", "O(k)", "O(k)", "O(n)", "Traverse k nodes"),
        ("Deque", "O(1)", "O(1)", "O(1)", "O(n)", "Index-based with deque"),
    ]
    
    print(f"{'Approach':<15} | {'Visit':<8} | {'Back':<8} | {'Forward':<8} | {'Space':<8} | {'Notes'}")
    print("-" * 80)
    
    for approach, visit, back, forward, space, notes in approaches:
        print(f"{approach:<15} | {visit:<8} | {back:<8} | {forward:<8} | {space:<8} | {notes}")
    
    print(f"\nArray approach is optimal for most use cases")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    browser = BrowserHistoryStack("home.com")
    
    edge_cases = [
        ("Back from homepage", lambda: browser.back(5), "home.com"),
        ("Forward with no forward history", lambda: browser.forward(3), "home.com"),
        ("Visit then immediate back", lambda: (browser.visit("page1.com"), browser.back(1))[1], "home.com"),
        ("Multiple visits then back beyond start", lambda: (
            browser.visit("page2.com"), 
            browser.visit("page3.com"), 
            browser.back(10)
        )[2], "home.com"),
    ]
    
    for description, operation, expected in edge_cases:
        try:
            # Reset browser for each test
            browser = BrowserHistoryStack("home.com")
            result = operation()
            status = "✓" if result == expected else "✗"
            print(f"{description:35} | {status} | Result: {result}")
        except Exception as e:
            print(f"{description:35} | ERROR: {str(e)[:30]}")


def benchmark_implementations():
    """Benchmark different implementations"""
    import time
    
    implementations = [
        ("Two Stacks", BrowserHistoryStack),
        ("Array", BrowserHistoryArray),
        ("Deque", BrowserHistoryDeque),
    ]
    
    n_operations = 10000
    
    print(f"\n=== Performance Benchmark ===")
    print(f"Operations: {n_operations} mixed visit/back/forward operations")
    
    for impl_name, impl_class in implementations:
        try:
            browser = impl_class("home.com")
            
            start_time = time.time()
            
            # Mixed operations
            for i in range(n_operations):
                op_type = i % 4
                
                if op_type == 0:
                    browser.visit(f"page{i}.com")
                elif op_type == 1:
                    browser.back(1)
                elif op_type == 2:
                    browser.forward(1)
                else:
                    browser.visit(f"new{i}.com")
            
            end_time = time.time()
            
            print(f"{impl_name:15} | Time: {end_time - start_time:.4f}s")
            
        except Exception as e:
            print(f"{impl_name:15} | ERROR: {str(e)[:30]}")


if __name__ == "__main__":
    test_browser_history_implementations()
    demonstrate_two_stacks_approach()
    visualize_browser_navigation()
    demonstrate_real_world_applications()
    analyze_time_complexity()
    test_edge_cases()
    benchmark_implementations()

"""
Design Browser History demonstrates system design with stacks and queues
for navigation systems, including multiple implementation approaches
for browser history management and undo/redo functionality.
"""
