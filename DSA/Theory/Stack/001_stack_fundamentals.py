"""
Stack Fundamentals - Core Concepts and Implementation
====================================================

Topics: Stack definition, operations, implementations, basic applications
Companies: All tech companies test stack fundamentals
Difficulty: Easy to Medium
Time Complexity: O(1) for all basic operations
Space Complexity: O(n) where n is number of elements
"""

from typing import List, Optional, Any, Generic, TypeVar
import sys
from collections import deque

T = TypeVar('T')

class StackFundamentals:
    
    def __init__(self):
        """Initialize with demonstration tracking"""
        self.operation_count = 0
        self.demo_stack = []
    
    # ==========================================
    # 1. WHAT IS A STACK?
    # ==========================================
    
    def explain_stack_concept(self) -> None:
        """
        Explain the fundamental concept of a stack data structure
        
        Stack is a LIFO (Last In, First Out) data structure
        Think of it like a stack of plates - you can only add/remove from the top
        """
        print("=== WHAT IS A STACK? ===")
        print("A Stack is a linear data structure that follows LIFO principle:")
        print("• LIFO = Last In, First Out")
        print("• Elements are added and removed from the same end (called 'top')")
        print("• Only the top element is accessible at any time")
        print()
        print("Real-world Analogies:")
        print("• Stack of plates: Add/remove only from top")
        print("• Stack of books: Last book placed is first to be removed")
        print("• Browser history: Back button follows LIFO order")
        print("• Function call stack: Last called function returns first")
        print()
        print("Key Characteristics:")
        print("• Insertion and deletion happen at the same end")
        print("• No random access to middle elements")
        print("• Simple and efficient for specific use cases")
        print("• Essential for recursion and expression evaluation")
    
    def stack_vs_other_structures(self) -> None:
        """Compare stack with other data structures"""
        print("=== STACK VS OTHER DATA STRUCTURES ===")
        print()
        print("Stack vs Array:")
        print("  Array: Random access to any element")
        print("  Stack: Access only to top element")
        print("  Use Stack when: You need LIFO behavior")
        print()
        print("Stack vs Queue:")
        print("  Queue: FIFO (First In, First Out)")
        print("  Stack: LIFO (Last In, First Out)")
        print("  Use Stack when: Processing needs to be in reverse order")
        print()
        print("Stack vs Linked List:")
        print("  Linked List: Can insert/delete anywhere")
        print("  Stack: Insert/delete only at top")
        print("  Use Stack when: You want restricted access pattern")
        print()
        print("When to Use Stack:")
        print("• Function call management")
        print("• Expression evaluation and parsing")
        print("• Undo operations in applications")
        print("• Backtracking algorithms")
        print("• Memory management")
    
    # ==========================================
    # 2. STACK OPERATIONS
    # ==========================================
    
    def demonstrate_basic_operations(self) -> None:
        """
        Demonstrate all basic stack operations with detailed explanation
        """
        print("=== BASIC STACK OPERATIONS ===")
        
        stack = []
        print(f"Initial stack: {stack}")
        print()
        
        # 1. PUSH Operation
        print("1. PUSH Operation (Add element to top):")
        elements_to_push = [10, 20, 30, 40]
        
        for element in elements_to_push:
            stack.append(element)
            print(f"   Push {element}: {stack} (top = {stack[-1] if stack else 'None'})")
        print()
        
        # 2. POP Operation
        print("2. POP Operation (Remove element from top):")
        for _ in range(2):
            if stack:
                popped = stack.pop()
                print(f"   Pop {popped}: {stack} (top = {stack[-1] if stack else 'None'})")
        print()
        
        # 3. PEEK/TOP Operation
        print("3. PEEK/TOP Operation (View top element without removing):")
        if stack:
            top_element = stack[-1]
            print(f"   Peek: {top_element} (stack remains: {stack})")
        print()
        
        # 4. isEmpty Operation
        print("4. isEmpty Operation:")
        print(f"   Is stack empty? {len(stack) == 0}")
        print()
        
        # 5. size Operation
        print("5. size Operation:")
        print(f"   Stack size: {len(stack)}")
        print()
        
        # Empty the stack
        print("6. Emptying the stack:")
        while stack:
            popped = stack.pop()
            print(f"   Pop {popped}: {stack}")
        
        print(f"   Final stack: {stack}")
        print(f"   Is empty now? {len(stack) == 0}")
    
    # ==========================================
    # 3. STACK IMPLEMENTATIONS
    # ==========================================

class ArrayBasedStack(Generic[T]):
    """
    Array-based stack implementation with fixed capacity
    
    Pros: Memory efficient, fast operations, cache-friendly
    Cons: Fixed size, potential overflow
    """
    
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.stack = [None] * capacity
        self.top_index = -1
        self.operation_count = 0
    
    def push(self, item: T) -> bool:
        """
        Push element to top of stack
        
        Time: O(1), Space: O(1)
        Returns: True if successful, False if stack overflow
        """
        self.operation_count += 1
        
        print(f"Push operation #{self.operation_count}: Adding {item}")
        
        if self.is_full():
            print(f"   ✗ Stack Overflow! Cannot push {item}")
            return False
        
        self.top_index += 1
        self.stack[self.top_index] = item
        
        print(f"   ✓ Pushed {item} at index {self.top_index}")
        print(f"   Stack state: {self.get_elements()}")
        return True
    
    def pop(self) -> Optional[T]:
        """
        Pop element from top of stack
        
        Time: O(1), Space: O(1)
        Returns: Popped element or None if empty
        """
        self.operation_count += 1
        
        print(f"Pop operation #{self.operation_count}:")
        
        if self.is_empty():
            print(f"   ✗ Stack Underflow! Cannot pop from empty stack")
            return None
        
        item = self.stack[self.top_index]
        self.stack[self.top_index] = None  # Clear reference
        self.top_index -= 1
        
        print(f"   ✓ Popped {item}")
        print(f"   Stack state: {self.get_elements()}")
        return item
    
    def peek(self) -> Optional[T]:
        """
        Get top element without removing it
        
        Time: O(1), Space: O(1)
        """
        if self.is_empty():
            print("   ✗ Cannot peek: Stack is empty")
            return None
        
        top_element = self.stack[self.top_index]
        print(f"   Top element: {top_element}")
        return top_element
    
    def is_empty(self) -> bool:
        """Check if stack is empty"""
        return self.top_index == -1
    
    def is_full(self) -> bool:
        """Check if stack is full"""
        return self.top_index == self.capacity - 1
    
    def size(self) -> int:
        """Get current size of stack"""
        return self.top_index + 1
    
    def get_elements(self) -> List[T]:
        """Get current elements in stack (for visualization)"""
        return [self.stack[i] for i in range(self.top_index + 1)]
    
    def display(self) -> None:
        """Display stack in visual format"""
        print("Stack visualization (top to bottom):")
        if self.is_empty():
            print("   (empty)")
        else:
            for i in range(self.top_index, -1, -1):
                marker = " <- TOP" if i == self.top_index else ""
                print(f"   [{i}] {self.stack[i]}{marker}")


class LinkedListBasedStack(Generic[T]):
    """
    Linked list based stack implementation with dynamic size
    
    Pros: Dynamic size, no overflow issues
    Cons: Extra memory overhead for pointers, less cache-friendly
    """
    
    class Node:
        def __init__(self, data: T):
            self.data = data
            self.next = None
    
    def __init__(self):
        self.top = None
        self.stack_size = 0
        self.operation_count = 0
    
    def push(self, item: T) -> None:
        """
        Push element to top of stack
        
        Time: O(1), Space: O(1)
        """
        self.operation_count += 1
        
        print(f"Push operation #{self.operation_count}: Adding {item}")
        
        # Create new node
        new_node = self.Node(item)
        new_node.next = self.top
        self.top = new_node
        self.stack_size += 1
        
        print(f"   ✓ Pushed {item}")
        print(f"   Stack state: {self.get_elements()}")
    
    def pop(self) -> Optional[T]:
        """
        Pop element from top of stack
        
        Time: O(1), Space: O(1)
        """
        self.operation_count += 1
        
        print(f"Pop operation #{self.operation_count}:")
        
        if self.is_empty():
            print(f"   ✗ Stack Underflow! Cannot pop from empty stack")
            return None
        
        item = self.top.data
        self.top = self.top.next
        self.stack_size -= 1
        
        print(f"   ✓ Popped {item}")
        print(f"   Stack state: {self.get_elements()}")
        return item
    
    def peek(self) -> Optional[T]:
        """Get top element without removing it"""
        if self.is_empty():
            print("   ✗ Cannot peek: Stack is empty")
            return None
        
        top_element = self.top.data
        print(f"   Top element: {top_element}")
        return top_element
    
    def is_empty(self) -> bool:
        """Check if stack is empty"""
        return self.top is None
    
    def size(self) -> int:
        """Get current size of stack"""
        return self.stack_size
    
    def get_elements(self) -> List[T]:
        """Get current elements in stack (for visualization)"""
        elements = []
        current = self.top
        while current:
            elements.append(current.data)
            current = current.next
        return elements
    
    def display(self) -> None:
        """Display stack in visual format"""
        print("Linked List Stack visualization (top to bottom):")
        if self.is_empty():
            print("   (empty)")
        else:
            current = self.top
            index = 0
            while current:
                marker = " <- TOP" if index == 0 else ""
                print(f"   {current.data}{marker}")
                current = current.next
                index += 1


class DequeBasedStack(Generic[T]):
    """
    Stack implementation using collections.deque
    
    Pros: Optimized for both ends, thread-safe operations
    Cons: Slight overhead compared to simple list
    """
    
    def __init__(self):
        self.stack = deque()
        self.operation_count = 0
    
    def push(self, item: T) -> None:
        """Push element to top of stack"""
        self.operation_count += 1
        self.stack.append(item)
        print(f"Push operation #{self.operation_count}: Added {item}")
        print(f"   Stack state: {list(self.stack)}")
    
    def pop(self) -> Optional[T]:
        """Pop element from top of stack"""
        self.operation_count += 1
        
        if not self.stack:
            print(f"Pop operation #{self.operation_count}: Stack is empty")
            return None
        
        item = self.stack.pop()
        print(f"Pop operation #{self.operation_count}: Removed {item}")
        print(f"   Stack state: {list(self.stack)}")
        return item
    
    def peek(self) -> Optional[T]:
        """Get top element without removing it"""
        if not self.stack:
            print("   ✗ Cannot peek: Stack is empty")
            return None
        
        top_element = self.stack[-1]
        print(f"   Top element: {top_element}")
        return top_element
    
    def is_empty(self) -> bool:
        """Check if stack is empty"""
        return len(self.stack) == 0
    
    def size(self) -> int:
        """Get current size of stack"""
        return len(self.stack)


# ==========================================
# 4. BASIC STACK APPLICATIONS
# ==========================================

class BasicStackApplications:
    
    def __init__(self):
        self.demo_count = 0
    
    def reverse_string(self, s: str) -> str:
        """
        Reverse a string using stack
        
        Company: Basic interview question
        Difficulty: Easy
        Time: O(n), Space: O(n)
        """
        stack = []
        
        print(f"Reversing string: '{s}'")
        
        # Push all characters to stack
        print("Pushing characters:")
        for char in s:
            stack.append(char)
            print(f"   Push '{char}': {stack}")
        
        # Pop all characters to form reversed string
        reversed_str = ""
        print("Popping characters:")
        while stack:
            char = stack.pop()
            reversed_str += char
            print(f"   Pop '{char}': {stack} -> reversed so far: '{reversed_str}'")
        
        print(f"Final reversed string: '{reversed_str}'")
        return reversed_str
    
    def check_balanced_parentheses(self, expression: str) -> bool:
        """
        Check if parentheses are balanced using stack
        
        Company: Amazon, Microsoft, Google
        Difficulty: Easy
        Time: O(n), Space: O(n)
        """
        stack = []
        opening = {'(', '[', '{'}
        closing = {')', ']', '}'}
        pairs = {'(': ')', '[': ']', '{': '}'}
        
        print(f"Checking balanced parentheses in: '{expression}'")
        
        for i, char in enumerate(expression):
            print(f"Step {i+1}: Processing '{char}'")
            
            if char in opening:
                stack.append(char)
                print(f"   Opening bracket found. Stack: {stack}")
            
            elif char in closing:
                if not stack:
                    print(f"   ✗ Closing bracket '{char}' without matching opening")
                    return False
                
                top = stack.pop()
                if pairs[top] != char:
                    print(f"   ✗ Mismatched brackets: '{top}' and '{char}'")
                    return False
                
                print(f"   ✓ Matched '{top}' with '{char}'. Stack: {stack}")
        
        is_balanced = len(stack) == 0
        print(f"Final result: {'Balanced' if is_balanced else 'Unbalanced'}")
        if not is_balanced:
            print(f"   Unmatched opening brackets: {stack}")
        
        return is_balanced
    
    def decimal_to_binary(self, decimal: int) -> str:
        """
        Convert decimal to binary using stack
        
        Time: O(log n), Space: O(log n)
        """
        if decimal == 0:
            return "0"
        
        stack = []
        original = decimal
        
        print(f"Converting decimal {decimal} to binary:")
        
        # Push remainders onto stack
        while decimal > 0:
            remainder = decimal % 2
            stack.append(remainder)
            print(f"   {decimal} ÷ 2 = {decimal // 2}, remainder = {remainder}")
            print(f"   Push {remainder}: Stack = {stack}")
            decimal //= 2
        
        # Pop remainders to form binary number
        binary = ""
        print("Building binary number by popping stack:")
        while stack:
            bit = stack.pop()
            binary += str(bit)
            print(f"   Pop {bit}: Stack = {stack}, Binary so far = '{binary}'")
        
        print(f"Decimal {original} = Binary {binary}")
        return binary
    
    def next_greater_element(self, arr: List[int]) -> List[int]:
        """
        Find next greater element for each element using stack
        
        Company: Amazon, Microsoft
        Difficulty: Medium
        Time: O(n), Space: O(n)
        """
        result = [-1] * len(arr)
        stack = []  # Store indices
        
        print(f"Finding next greater elements for: {arr}")
        print("Stack will store indices of elements waiting for their next greater element")
        print()
        
        for i in range(len(arr)):
            print(f"Step {i+1}: Processing arr[{i}] = {arr[i]}")
            
            # Pop elements from stack while current element is greater
            while stack and arr[stack[-1]] < arr[i]:
                index = stack.pop()
                result[index] = arr[i]
                print(f"   Found next greater for arr[{index}] = {arr[index]}: {arr[i]}")
                print(f"   Stack after pop: {[arr[idx] for idx in stack] if stack else 'empty'}")
            
            # Push current index to stack
            stack.append(i)
            print(f"   Push index {i} (value {arr[i]}) to stack")
            print(f"   Stack: {[arr[idx] for idx in stack]}")
            print()
        
        print(f"Final result: {result}")
        print("Elements still in stack have no next greater element")
        
        return result


# ==========================================
# 5. PERFORMANCE ANALYSIS
# ==========================================

def analyze_stack_performance():
    """
    Analyze performance characteristics of different stack implementations
    """
    print("=== STACK PERFORMANCE ANALYSIS ===")
    print()
    
    print("TIME COMPLEXITY:")
    print("• Push:     O(1) - Constant time insertion")
    print("• Pop:      O(1) - Constant time deletion")
    print("• Peek:     O(1) - Constant time access to top")
    print("• isEmpty:  O(1) - Constant time check")
    print("• Size:     O(1) - Constant time if maintained")
    print()
    
    print("SPACE COMPLEXITY:")
    print("• Array-based:      O(n) where n is capacity")
    print("• Linked-list:      O(n) where n is number of elements")
    print("• Additional space: O(1) per element for array, O(1) + pointer for linked list")
    print()
    
    print("IMPLEMENTATION COMPARISON:")
    print("┌─────────────────┬──────────────┬───────────────┬─────────────────┐")
    print("│ Implementation  │ Memory Usage │ Cache Perf.   │ Use Case        │")
    print("├─────────────────┼──────────────┼───────────────┼─────────────────┤")
    print("│ Array-based     │ Low          │ Excellent     │ Known max size  │")
    print("│ Linked-list     │ High         │ Good          │ Dynamic size    │")
    print("│ Deque-based     │ Medium       │ Very Good     │ General purpose │")
    print("└─────────────────┴──────────────┴───────────────┴─────────────────┘")
    print()
    
    print("WHEN TO USE EACH:")
    print("• Array-based: When maximum size is known, memory is constrained")
    print("• Linked-list: When size varies greatly, frequent push/pop operations")
    print("• Deque-based: When you need both stack and queue operations")


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_stack_fundamentals():
    """Demonstrate all stack fundamental concepts"""
    print("=== STACK FUNDAMENTALS DEMONSTRATION ===\n")
    
    fundamentals = StackFundamentals()
    
    # 1. Concept explanation
    fundamentals.explain_stack_concept()
    print("\n" + "="*60 + "\n")
    
    # 2. Comparison with other structures
    fundamentals.stack_vs_other_structures()
    print("\n" + "="*60 + "\n")
    
    # 3. Basic operations demonstration
    fundamentals.demonstrate_basic_operations()
    print("\n" + "="*60 + "\n")
    
    # 4. Array-based stack implementation
    print("=== ARRAY-BASED STACK IMPLEMENTATION ===")
    array_stack = ArrayBasedStack[int](capacity=5)
    
    # Test operations
    for value in [10, 20, 30, 40, 50]:
        array_stack.push(value)
    
    print("\nTrying to push to full stack:")
    array_stack.push(60)  # Should fail
    
    print("\nPeek operation:")
    array_stack.peek()
    
    print("\nPopping elements:")
    for _ in range(3):
        array_stack.pop()
    
    print("\nFinal display:")
    array_stack.display()
    
    print("\n" + "="*60 + "\n")
    
    # 5. Linked list based stack
    print("=== LINKED LIST BASED STACK IMPLEMENTATION ===")
    ll_stack = LinkedListBasedStack[str]()
    
    for value in ['A', 'B', 'C', 'D']:
        ll_stack.push(value)
    
    print("\nPeek operation:")
    ll_stack.peek()
    
    print("\nPopping elements:")
    for _ in range(2):
        ll_stack.pop()
    
    print("\nFinal display:")
    ll_stack.display()
    
    print("\n" + "="*60 + "\n")
    
    # 6. Basic applications
    print("=== BASIC STACK APPLICATIONS ===")
    apps = BasicStackApplications()
    
    # String reversal
    print("1. String Reversal:")
    apps.reverse_string("HELLO")
    print()
    
    # Balanced parentheses
    print("2. Balanced Parentheses:")
    test_expressions = ["()", "({[]})", "(()", "([)]", "{[()]}"]
    for expr in test_expressions:
        apps.check_balanced_parentheses(expr)
        print()
    
    # Decimal to binary
    print("3. Decimal to Binary Conversion:")
    apps.decimal_to_binary(13)
    print()
    
    # Next greater element
    print("4. Next Greater Element:")
    apps.next_greater_element([4, 5, 2, 25, 7, 8])
    print()
    
    # 7. Performance analysis
    analyze_stack_performance()


if __name__ == "__main__":
    demonstrate_stack_fundamentals()
    
    print("\n" + "="*60)
    print("=== STACK MASTERY GUIDE ===")
    print("="*60)
    
    print("\n🎯 WHEN TO USE STACK:")
    print("✅ Function call management (recursion)")
    print("✅ Expression evaluation and parsing")
    print("✅ Undo operations in applications")
    print("✅ Backtracking algorithms")
    print("✅ Browser history management")
    print("✅ Syntax parsing in compilers")
    
    print("\n📋 STACK IMPLEMENTATION CHECKLIST:")
    print("1. Choose appropriate implementation based on requirements")
    print("2. Handle stack overflow and underflow conditions")
    print("3. Consider thread safety for concurrent applications")
    print("4. Optimize for memory usage vs performance trade-offs")
    print("5. Implement proper error handling and edge cases")
    
    print("\n⚡ OPTIMIZATION STRATEGIES:")
    print("• Pre-allocate array size if maximum known")
    print("• Use deque for better performance in Python")
    print("• Consider memory pooling for frequent allocations")
    print("• Implement stack using existing array/list when possible")
    print("• Cache size information to avoid recalculation")
    
    print("\n🚨 COMMON PITFALLS:")
    print("• Not checking for empty stack before pop/peek")
    print("• Integer overflow in fixed-size implementations")
    print("• Memory leaks in linked list implementations")
    print("• Not handling edge cases (empty input, single element)")
    print("• Confusing stack with queue behavior")
    
    print("\n🎓 LEARNING PROGRESSION:")
    print("1. Master basic stack operations and concepts")
    print("2. Implement different stack variations")
    print("3. Practice stack-based algorithms")
    print("4. Learn advanced applications (expression parsing)")
    print("5. Study memory management and optimization")
    
    print("\n📚 PROBLEM CATEGORIES TO PRACTICE:")
    print("• Parentheses and bracket matching")
    print("• Expression evaluation and conversion")
    print("• Next/previous greater/smaller element")
    print("• String manipulation and reversal")
    print("• Function call simulation")
    print("• Histogram and area calculation problems")

