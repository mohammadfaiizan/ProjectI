/*
 * =============================================================================
 * COMPLETE STACK GUIDE - All Implementations & Operations
 * =============================================================================
 * 
 * This file covers:
 * 1. Array-based Stack implementation
 * 2. Linked List-based Stack implementation
 * 3. STL Stack operations
 * 4. Different ways to use stack
 * 5. Stack applications and algorithms
 * 6. Common patterns and problems
 * 
 * =============================================================================
 */

#include <iostream>
#include <stack>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
using namespace std;

// =============================================================================
// ARRAY-BASED STACK IMPLEMENTATION
// =============================================================================

template<typename T>
class ArrayStack {
private:
    T* data;
    int capacity;
    int top_index;
    
    void resize() {
        int new_capacity = capacity * 2;
        T* new_data = new T[new_capacity];
        for (int i = 0; i <= top_index; i++) {
            new_data[i] = data[i];
        }
        delete[] data;
        data = new_data;
        capacity = new_capacity;
        cout << "Stack resized to capacity: " << capacity << endl;
    }

public:
    // Constructor
    ArrayStack(int initial_capacity = 10) 
        : capacity(initial_capacity), top_index(-1) {
        data = new T[capacity];
        cout << "Array stack created with capacity: " << capacity << endl;
    }
    
    // Destructor
    ~ArrayStack() {
        delete[] data;
        cout << "Array stack destroyed" << endl;
    }
    
    // Copy constructor
    ArrayStack(const ArrayStack& other) 
        : capacity(other.capacity), top_index(other.top_index) {
        data = new T[capacity];
        for (int i = 0; i <= top_index; i++) {
            data[i] = other.data[i];
        }
        cout << "Array stack copied" << endl;
    }
    
    // Assignment operator
    ArrayStack& operator=(const ArrayStack& other) {
        if (this != &other) {
            delete[] data;
            capacity = other.capacity;
            top_index = other.top_index;
            data = new T[capacity];
            for (int i = 0; i <= top_index; i++) {
                data[i] = other.data[i];
            }
        }
        return *this;
    }
    
    // Basic operations
    void push(const T& value) {
        if (top_index >= capacity - 1) {
            resize();
        }
        data[++top_index] = value;
        cout << "Pushed " << value << " to array stack" << endl;
    }
    
    void pop() {
        if (isEmpty()) {
            cout << "Error: Stack is empty!" << endl;
            return;
        }
        cout << "Popped " << data[top_index] << " from array stack" << endl;
        top_index--;
    }
    
    T& top() {
        if (isEmpty()) {
            throw runtime_error("Stack is empty!");
        }
        return data[top_index];
    }
    
    const T& top() const {
        if (isEmpty()) {
            throw runtime_error("Stack is empty!");
        }
        return data[top_index];
    }
    
    // Utility methods
    bool isEmpty() const { return top_index == -1; }
    int size() const { return top_index + 1; }
    int getCapacity() const { return capacity; }
    
    // Display methods
    void display() const {
        if (isEmpty()) {
            cout << "Array stack is empty" << endl;
            return;
        }
        cout << "Array stack (top to bottom): ";
        for (int i = top_index; i >= 0; i--) {
            cout << data[i] << " ";
        }
        cout << endl;
    }
    
    void displayWithIndex() const {
        if (isEmpty()) {
            cout << "Array stack is empty" << endl;
            return;
        }
        cout << "Array stack with indices:" << endl;
        for (int i = top_index; i >= 0; i--) {
            cout << "Index " << i << ": " << data[i] << endl;
        }
    }
    
    // Advanced operations
    void clear() {
        top_index = -1;
        cout << "Array stack cleared" << endl;
    }
    
    T peek(int depth) const {
        if (depth < 0 || depth > top_index) {
            throw runtime_error("Invalid depth!");
        }
        return data[top_index - depth];
    }
    
    void printInfo() const {
        cout << "Array Stack Info - Size: " << size() << ", Capacity: " << capacity 
             << ", Top: " << (isEmpty() ? "Empty" : to_string(top())) << endl;
    }
};

// =============================================================================
// LINKED LIST-BASED STACK IMPLEMENTATION
// =============================================================================

template<typename T>
class LinkedStack {
private:
    struct Node {
        T data;
        Node* next;
        
        Node(const T& value) : data(value), next(nullptr) {}
    };
    
    Node* top_node;
    int stack_size;

public:
    // Constructor
    LinkedStack() : top_node(nullptr), stack_size(0) {
        cout << "Linked stack created" << endl;
    }
    
    // Destructor
    ~LinkedStack() {
        clear();
        cout << "Linked stack destroyed" << endl;
    }
    
    // Copy constructor
    LinkedStack(const LinkedStack& other) : top_node(nullptr), stack_size(0) {
        if (other.top_node) {
            vector<T> temp;
            Node* current = other.top_node;
            while (current) {
                temp.push_back(current->data);
                current = current->next;
            }
            for (int i = temp.size() - 1; i >= 0; i--) {
                push(temp[i]);
            }
        }
        cout << "Linked stack copied" << endl;
    }
    
    // Assignment operator
    LinkedStack& operator=(const LinkedStack& other) {
        if (this != &other) {
            clear();
            if (other.top_node) {
                vector<T> temp;
                Node* current = other.top_node;
                while (current) {
                    temp.push_back(current->data);
                    current = current->next;
                }
                for (int i = temp.size() - 1; i >= 0; i--) {
                    push(temp[i]);
                }
            }
        }
        return *this;
    }
    
    // Basic operations
    void push(const T& value) {
        Node* new_node = new Node(value);
        new_node->next = top_node;
        top_node = new_node;
        stack_size++;
        cout << "Pushed " << value << " to linked stack" << endl;
    }
    
    void pop() {
        if (isEmpty()) {
            cout << "Error: Stack is empty!" << endl;
            return;
        }
        Node* temp = top_node;
        top_node = top_node->next;
        cout << "Popped " << temp->data << " from linked stack" << endl;
        delete temp;
        stack_size--;
    }
    
    T& top() {
        if (isEmpty()) {
            throw runtime_error("Stack is empty!");
        }
        return top_node->data;
    }
    
    const T& top() const {
        if (isEmpty()) {
            throw runtime_error("Stack is empty!");
        }
        return top_node->data;
    }
    
    // Utility methods
    bool isEmpty() const { return top_node == nullptr; }
    int size() const { return stack_size; }
    
    // Display methods
    void display() const {
        if (isEmpty()) {
            cout << "Linked stack is empty" << endl;
            return;
        }
        cout << "Linked stack (top to bottom): ";
        Node* current = top_node;
        while (current) {
            cout << current->data << " ";
            current = current->next;
        }
        cout << endl;
    }
    
    void displayWithPosition() const {
        if (isEmpty()) {
            cout << "Linked stack is empty" << endl;
            return;
        }
        cout << "Linked stack with positions:" << endl;
        Node* current = top_node;
        int position = 0;
        while (current) {
            cout << "Position " << position << ": " << current->data << endl;
            current = current->next;
            position++;
        }
    }
    
    // Advanced operations
    void clear() {
        while (!isEmpty()) {
            pop();
        }
    }
    
    T peek(int depth) const {
        if (depth < 0 || depth >= stack_size) {
            throw runtime_error("Invalid depth!");
        }
        Node* current = top_node;
        for (int i = 0; i < depth; i++) {
            current = current->next;
        }
        return current->data;
    }
    
    void printInfo() const {
        cout << "Linked Stack Info - Size: " << stack_size 
             << ", Top: " << (isEmpty() ? "Empty" : to_string(top())) << endl;
    }
};

// =============================================================================
// STL STACK OPERATIONS
// =============================================================================

void demonstrate_stl_stack() {
    cout << "\n=== STL STACK OPERATIONS ===" << endl;
    
    // Different ways to create STL stack
    cout << "\n--- STL Stack Creation ---" << endl;
    
    // Method 1: Default stack
    stack<int> s1;
    cout << "Default stack created" << endl;
    
    // Method 2: Stack with vector as underlying container
    stack<int, vector<int>> s2;
    cout << "Stack with vector container created" << endl;
    
    // Method 3: Copy constructor
    s1.push(10);
    s1.push(20);
    stack<int> s3(s1);
    cout << "Stack copied using copy constructor" << endl;
    
    // Basic operations
    cout << "\n--- Basic Operations ---" << endl;
    
    // Push elements
    for (int i = 1; i <= 5; i++) {
        s1.push(i * 10);
        cout << "Pushed " << i * 10 << " to STL stack" << endl;
    }
    
    // Stack properties
    cout << "Stack size: " << s1.size() << endl;
    cout << "Stack empty: " << (s1.empty() ? "Yes" : "No") << endl;
    cout << "Top element: " << s1.top() << endl;
    
    // Pop elements
    cout << "\n--- Popping Elements ---" << endl;
    while (!s1.empty()) {
        cout << "Popped: " << s1.top() << endl;
        s1.pop();
    }
    
    cout << "Stack size after popping all: " << s1.size() << endl;
    cout << "Stack empty: " << (s1.empty() ? "Yes" : "No") << endl;
}

// =============================================================================
// STACK APPLICATIONS
// =============================================================================

class StackApplications {
public:
    // Application 1: Balanced Parentheses
    static bool isBalanced(const string& expr) {
        stack<char> s;
        
        for (char c : expr) {
            if (c == '(' || c == '[' || c == '{') {
                s.push(c);
            } else if (c == ')' || c == ']' || c == '}') {
                if (s.empty()) return false;
                
                char top = s.top();
                s.pop();
                
                if ((c == ')' && top != '(') ||
                    (c == ']' && top != '[') ||
                    (c == '}' && top != '{')) {
                    return false;
                }
            }
        }
        
        return s.empty();
    }
    
    // Application 2: Infix to Postfix
    static string infixToPostfix(const string& infix) {
        stack<char> s;
        string postfix = "";
        
        auto precedence = [](char op) -> int {
            switch (op) {
                case '+': case '-': return 1;
                case '*': case '/': return 2;
                case '^': return 3;
                default: return 0;
            }
        };
        
        auto isOperator = [](char c) -> bool {
            return c == '+' || c == '-' || c == '*' || c == '/' || c == '^';
        };
        
        for (char c : infix) {
            if (isalnum(c)) {
                postfix += c;
            } else if (c == '(') {
                s.push(c);
            } else if (c == ')') {
                while (!s.empty() && s.top() != '(') {
                    postfix += s.top();
                    s.pop();
                }
                if (!s.empty()) s.pop(); // Remove '('
            } else if (isOperator(c)) {
                while (!s.empty() && precedence(s.top()) >= precedence(c)) {
                    postfix += s.top();
                    s.pop();
                }
                s.push(c);
            }
        }
        
        while (!s.empty()) {
            postfix += s.top();
            s.pop();
        }
        
        return postfix;
    }
    
    // Application 3: Postfix Evaluation
    static int evaluatePostfix(const string& postfix) {
        stack<int> s;
        
        for (char c : postfix) {
            if (isdigit(c)) {
                s.push(c - '0');
            } else {
                int b = s.top(); s.pop();
                int a = s.top(); s.pop();
                
                switch (c) {
                    case '+': s.push(a + b); break;
                    case '-': s.push(a - b); break;
                    case '*': s.push(a * b); break;
                    case '/': s.push(a / b); break;
                    case '^': s.push(pow(a, b)); break;
                }
            }
        }
        
        return s.top();
    }
    
    // Application 4: Reverse String
    static string reverseString(const string& str) {
        stack<char> s;
        
        for (char c : str) {
            s.push(c);
        }
        
        string reversed = "";
        while (!s.empty()) {
            reversed += s.top();
            s.pop();
        }
        
        return reversed;
    }
    
    // Application 5: Stock Span Problem
    static vector<int> stockSpan(const vector<int>& prices) {
        vector<int> span(prices.size());
        stack<int> s; // Stack to store indices
        
        for (int i = 0; i < prices.size(); i++) {
            while (!s.empty() && prices[s.top()] <= prices[i]) {
                s.pop();
            }
            
            span[i] = s.empty() ? i + 1 : i - s.top();
            s.push(i);
        }
        
        return span;
    }
    
    // Application 6: Next Greater Element
    static vector<int> nextGreaterElement(const vector<int>& arr) {
        vector<int> result(arr.size(), -1);
        stack<int> s;
        
        for (int i = 0; i < arr.size(); i++) {
            while (!s.empty() && arr[s.top()] < arr[i]) {
                result[s.top()] = arr[i];
                s.pop();
            }
            s.push(i);
        }
        
        return result;
    }
    
    // Application 7: Largest Rectangle in Histogram
    static int largestRectangleArea(const vector<int>& heights) {
        stack<int> s;
        int max_area = 0;
        
        for (int i = 0; i < heights.size(); i++) {
            while (!s.empty() && heights[s.top()] > heights[i]) {
                int height = heights[s.top()];
                s.pop();
                int width = s.empty() ? i : i - s.top() - 1;
                max_area = max(max_area, height * width);
            }
            s.push(i);
        }
        
        while (!s.empty()) {
            int height = heights[s.top()];
            s.pop();
            int width = s.empty() ? heights.size() : heights.size() - s.top() - 1;
            max_area = max(max_area, height * width);
        }
        
        return max_area;
    }
    
    // Application 8: Valid Parentheses with Multiple Types
    static bool isValidParentheses(const string& s) {
        stack<char> st;
        
        for (char c : s) {
            if (c == '(' || c == '[' || c == '{') {
                st.push(c);
            } else {
                if (st.empty()) return false;
                
                char top = st.top();
                st.pop();
                
                if ((c == ')' && top != '(') ||
                    (c == ']' && top != '[') ||
                    (c == '}' && top != '{')) {
                    return false;
                }
            }
        }
        
        return st.empty();
    }
};

// =============================================================================
// STACK PATTERNS AND ALGORITHMS
// =============================================================================

void demonstrate_stack_patterns() {
    cout << "\n=== STACK PATTERNS & ALGORITHMS ===" << endl;
    
    // Pattern 1: Monotonic Stack
    cout << "\n--- Monotonic Stack Pattern ---" << endl;
    vector<int> arr = {2, 1, 2, 4, 3, 1};
    cout << "Array: ";
    for (int x : arr) cout << x << " ";
    cout << endl;
    
    vector<int> next_greater = StackApplications::nextGreaterElement(arr);
    cout << "Next greater elements: ";
    for (int x : next_greater) cout << x << " ";
    cout << endl;
    
    // Pattern 2: Two Stacks in One Array
    cout << "\n--- Two Stacks in One Array ---" << endl;
    class TwoStacks {
        int* arr;
        int size;
        int top1, top2;
    public:
        TwoStacks(int n) : size(n), top1(-1), top2(n) {
            arr = new int[n];
        }
        
        ~TwoStacks() { delete[] arr; }
        
        void push1(int x) {
            if (top1 < top2 - 1) {
                arr[++top1] = x;
                cout << "Pushed " << x << " to stack1" << endl;
            } else {
                cout << "Stack overflow for stack1" << endl;
            }
        }
        
        void push2(int x) {
            if (top1 < top2 - 1) {
                arr[--top2] = x;
                cout << "Pushed " << x << " to stack2" << endl;
            } else {
                cout << "Stack overflow for stack2" << endl;
            }
        }
        
        int pop1() {
            if (top1 >= 0) {
                return arr[top1--];
            }
            return -1;
        }
        
        int pop2() {
            if (top2 < size) {
                return arr[top2++];
            }
            return -1;
        }
    };
    
    TwoStacks ts(5);
    ts.push1(5);
    ts.push2(10);
    ts.push1(15);
    ts.push2(20);
    
    cout << "Popped from stack1: " << ts.pop1() << endl;
    cout << "Popped from stack2: " << ts.pop2() << endl;
    
    // Pattern 3: Stack with getMin operation
    cout << "\n--- Stack with getMin Operation ---" << endl;
    class MinStack {
        stack<int> s;
        stack<int> min_s;
    public:
        void push(int x) {
            s.push(x);
            if (min_s.empty() || x <= min_s.top()) {
                min_s.push(x);
            }
        }
        
        void pop() {
            if (!s.empty()) {
                if (s.top() == min_s.top()) {
                    min_s.pop();
                }
                s.pop();
            }
        }
        
        int top() {
            return s.empty() ? -1 : s.top();
        }
        
        int getMin() {
            return min_s.empty() ? -1 : min_s.top();
        }
    };
    
    MinStack min_stack;
    min_stack.push(3);
    min_stack.push(5);
    min_stack.push(2);
    min_stack.push(1);
    
    cout << "Top: " << min_stack.top() << endl;
    cout << "Min: " << min_stack.getMin() << endl;
    
    min_stack.pop();
    cout << "After pop - Top: " << min_stack.top() << ", Min: " << min_stack.getMin() << endl;
}

// =============================================================================
// DEMONSTRATION FUNCTIONS
// =============================================================================

void demonstrate_array_stack() {
    cout << "\n=== ARRAY STACK DEMONSTRATION ===" << endl;
    
    ArrayStack<int> arr_stack(5);
    
    // Push elements
    for (int i = 1; i <= 7; i++) {
        arr_stack.push(i * 10);
    }
    
    arr_stack.display();
    arr_stack.displayWithIndex();
    arr_stack.printInfo();
    
    // Pop elements
    cout << "\n--- Popping Elements ---" << endl;
    while (!arr_stack.isEmpty()) {
        cout << "Top element: " << arr_stack.top() << endl;
        arr_stack.pop();
    }
    
    arr_stack.printInfo();
}

void demonstrate_linked_stack() {
    cout << "\n=== LINKED STACK DEMONSTRATION ===" << endl;
    
    LinkedStack<int> linked_stack;
    
    // Push elements
    for (int i = 1; i <= 5; i++) {
        linked_stack.push(i * 10);
    }
    
    linked_stack.display();
    linked_stack.displayWithPosition();
    linked_stack.printInfo();
    
    // Pop elements
    cout << "\n--- Popping Elements ---" << endl;
    while (!linked_stack.isEmpty()) {
        cout << "Top element: " << linked_stack.top() << endl;
        linked_stack.pop();
    }
    
    linked_stack.printInfo();
}

void demonstrate_stack_applications() {
    cout << "\n=== STACK APPLICATIONS DEMONSTRATION ===" << endl;
    
    // Balanced parentheses
    cout << "\n--- Balanced Parentheses ---" << endl;
    string expr1 = "((()))";
    string expr2 = "((())";
    cout << expr1 << " is balanced: " << (StackApplications::isBalanced(expr1) ? "Yes" : "No") << endl;
    cout << expr2 << " is balanced: " << (StackApplications::isBalanced(expr2) ? "Yes" : "No") << endl;
    
    // Infix to postfix
    cout << "\n--- Infix to Postfix ---" << endl;
    string infix = "a+b*c";
    string postfix = StackApplications::infixToPostfix(infix);
    cout << "Infix: " << infix << " -> Postfix: " << postfix << endl;
    
    // Postfix evaluation
    cout << "\n--- Postfix Evaluation ---" << endl;
    string postfix_expr = "23*4+";
    int result = StackApplications::evaluatePostfix(postfix_expr);
    cout << "Postfix: " << postfix_expr << " = " << result << endl;
    
    // Reverse string
    cout << "\n--- Reverse String ---" << endl;
    string original = "Hello";
    string reversed = StackApplications::reverseString(original);
    cout << "Original: " << original << " -> Reversed: " << reversed << endl;
    
    // Stock span
    cout << "\n--- Stock Span Problem ---" << endl;
    vector<int> prices = {100, 80, 60, 70, 60, 75, 85};
    vector<int> span = StackApplications::stockSpan(prices);
    cout << "Prices: ";
    for (int p : prices) cout << p << " ";
    cout << endl;
    cout << "Span:   ";
    for (int s : span) cout << s << " ";
    cout << endl;
    
    // Largest rectangle
    cout << "\n--- Largest Rectangle in Histogram ---" << endl;
    vector<int> heights = {2, 1, 5, 6, 2, 3};
    int max_area = StackApplications::largestRectangleArea(heights);
    cout << "Heights: ";
    for (int h : heights) cout << h << " ";
    cout << endl;
    cout << "Largest rectangle area: " << max_area << endl;
}

// =============================================================================
// MAIN FUNCTION
// =============================================================================

int main() {
    cout << "=== COMPLETE STACK GUIDE ===" << endl;
    
    demonstrate_array_stack();
    demonstrate_linked_stack();
    demonstrate_stl_stack();
    demonstrate_stack_applications();
    demonstrate_stack_patterns();
    
    cout << "\n=== SUMMARY ===" << endl;
    cout << "1. Array Stack: Fast access, fixed/resizable capacity" << endl;
    cout << "2. Linked Stack: Dynamic size, extra memory for pointers" << endl;
    cout << "3. STL Stack: Ready-to-use, optimized implementation" << endl;
    cout << "4. Applications: Parentheses, expressions, undo operations" << endl;
    cout << "5. Patterns: Monotonic stack, two stacks, min stack" << endl;
    
    return 0;
} 