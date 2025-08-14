/*
 * =============================================================================
 * COMPLETE LINKED LIST GUIDE - All Types & Operations
 * =============================================================================
 * 
 * This file covers:
 * 1. Singly Linked List - all operations
 * 2. Doubly Linked List - all operations
 * 3. Circular Linked List - all operations
 * 4. Different traversal methods
 * 5. Common patterns and algorithms
 * 6. Memory management techniques
 * 
 * =============================================================================
 */

#include <iostream>
#include <vector>
#include <stack>
using namespace std;

// =============================================================================
// SINGLY LINKED LIST - COMPLETE IMPLEMENTATION
// =============================================================================

struct SinglyNode {
    int data;
    SinglyNode* next;
    
    SinglyNode(int val) : data(val), next(nullptr) {}
};

class SinglyLinkedList {
private:
    SinglyNode* head;
    SinglyNode* tail;
    int size;

public:
    SinglyLinkedList() : head(nullptr), tail(nullptr), size(0) {}
    
    ~SinglyLinkedList() { clear(); }
    
    // Different ways to insert
    void insertAtBeginning(int val) {
        SinglyNode* newNode = new SinglyNode(val);
        if (!head) {
            head = tail = newNode;
        } else {
            newNode->next = head;
            head = newNode;
        }
        size++;
        cout << "Inserted " << val << " at beginning" << endl;
    }
    
    void insertAtEnd(int val) {
        SinglyNode* newNode = new SinglyNode(val);
        if (!head) {
            head = tail = newNode;
        } else {
            tail->next = newNode;
            tail = newNode;
        }
        size++;
        cout << "Inserted " << val << " at end" << endl;
    }
    
    void insertAtPosition(int pos, int val) {
        if (pos < 0 || pos > size) {
            cout << "Invalid position!" << endl;
            return;
        }
        
        if (pos == 0) {
            insertAtBeginning(val);
            return;
        }
        
        if (pos == size) {
            insertAtEnd(val);
            return;
        }
        
        SinglyNode* newNode = new SinglyNode(val);
        SinglyNode* current = head;
        for (int i = 0; i < pos - 1; i++) {
            current = current->next;
        }
        
        newNode->next = current->next;
        current->next = newNode;
        size++;
        cout << "Inserted " << val << " at position " << pos << endl;
    }
    
    void insertAfterValue(int target, int val) {
        SinglyNode* current = head;
        while (current && current->data != target) {
            current = current->next;
        }
        
        if (current) {
            SinglyNode* newNode = new SinglyNode(val);
            newNode->next = current->next;
            current->next = newNode;
            if (current == tail) tail = newNode;
            size++;
            cout << "Inserted " << val << " after " << target << endl;
        } else {
            cout << "Value " << target << " not found!" << endl;
        }
    }
    
    void insertBeforeValue(int target, int val) {
        if (!head) return;
        
        if (head->data == target) {
            insertAtBeginning(val);
            return;
        }
        
        SinglyNode* current = head;
        while (current->next && current->next->data != target) {
            current = current->next;
        }
        
        if (current->next) {
            SinglyNode* newNode = new SinglyNode(val);
            newNode->next = current->next;
            current->next = newNode;
            size++;
            cout << "Inserted " << val << " before " << target << endl;
        } else {
            cout << "Value " << target << " not found!" << endl;
        }
    }
    
    // Different ways to delete
    void deleteFromBeginning() {
        if (!head) {
            cout << "List is empty!" << endl;
            return;
        }
        
        SinglyNode* temp = head;
        head = head->next;
        cout << "Deleted " << temp->data << " from beginning" << endl;
        delete temp;
        size--;
        
        if (!head) tail = nullptr;
    }
    
    void deleteFromEnd() {
        if (!head) {
            cout << "List is empty!" << endl;
            return;
        }
        
        if (head == tail) {
            cout << "Deleted " << head->data << " from end" << endl;
            delete head;
            head = tail = nullptr;
        } else {
            SinglyNode* current = head;
            while (current->next != tail) {
                current = current->next;
            }
            cout << "Deleted " << tail->data << " from end" << endl;
            delete tail;
            tail = current;
            tail->next = nullptr;
        }
        size--;
    }
    
    void deleteAtPosition(int pos) {
        if (pos < 0 || pos >= size) {
            cout << "Invalid position!" << endl;
            return;
        }
        
        if (pos == 0) {
            deleteFromBeginning();
            return;
        }
        
        SinglyNode* current = head;
        for (int i = 0; i < pos - 1; i++) {
            current = current->next;
        }
        
        SinglyNode* toDelete = current->next;
        current->next = toDelete->next;
        if (toDelete == tail) tail = current;
        cout << "Deleted " << toDelete->data << " from position " << pos << endl;
        delete toDelete;
        size--;
    }
    
    void deleteValue(int val) {
        if (!head) return;
        
        if (head->data == val) {
            deleteFromBeginning();
            return;
        }
        
        SinglyNode* current = head;
        while (current->next && current->next->data != val) {
            current = current->next;
        }
        
        if (current->next) {
            SinglyNode* toDelete = current->next;
            current->next = toDelete->next;
            if (toDelete == tail) tail = current;
            cout << "Deleted " << val << " from list" << endl;
            delete toDelete;
            size--;
        } else {
            cout << "Value " << val << " not found!" << endl;
        }
    }
    
    void deleteAllOccurrences(int val) {
        int count = 0;
        while (head && head->data == val) {
            deleteFromBeginning();
            count++;
        }
        
        SinglyNode* current = head;
        while (current && current->next) {
            if (current->next->data == val) {
                SinglyNode* toDelete = current->next;
                current->next = toDelete->next;
                if (toDelete == tail) tail = current;
                delete toDelete;
                size--;
                count++;
            } else {
                current = current->next;
            }
        }
        cout << "Deleted " << count << " occurrences of " << val << endl;
    }
    
    // Different search methods
    bool search(int val) {
        SinglyNode* current = head;
        while (current) {
            if (current->data == val) return true;
            current = current->next;
        }
        return false;
    }
    
    int findIndex(int val) {
        SinglyNode* current = head;
        int index = 0;
        while (current) {
            if (current->data == val) return index;
            current = current->next;
            index++;
        }
        return -1;
    }
    
    vector<int> findAllIndices(int val) {
        vector<int> indices;
        SinglyNode* current = head;
        int index = 0;
        while (current) {
            if (current->data == val) {
                indices.push_back(index);
            }
            current = current->next;
            index++;
        }
        return indices;
    }
    
    // Different access methods
    int getAt(int pos) {
        if (pos < 0 || pos >= size) {
            cout << "Invalid position!" << endl;
            return -1;
        }
        
        SinglyNode* current = head;
        for (int i = 0; i < pos; i++) {
            current = current->next;
        }
        return current->data;
    }
    
    int getFirst() {
        return head ? head->data : -1;
    }
    
    int getLast() {
        return tail ? tail->data : -1;
    }
    
    // Different traversal methods
    void displayForward() {
        cout << "Forward: ";
        SinglyNode* current = head;
        while (current) {
            cout << current->data << " ";
            current = current->next;
        }
        cout << "-> NULL" << endl;
    }
    
    void displayRecursiveHelper(SinglyNode* node) {
        if (!node) return;
        cout << node->data << " ";
        displayRecursiveHelper(node->next);
    }
    
    void displayRecursive() {
        cout << "Recursive: ";
        displayRecursiveHelper(head);
        cout << "-> NULL" << endl;
    }
    
    void displayReverseRecursiveHelper(SinglyNode* node) {
        if (!node) return;
        displayReverseRecursiveHelper(node->next);
        cout << node->data << " ";
    }
    
    void displayReverseRecursive() {
        cout << "Reverse Recursive: ";
        displayReverseRecursiveHelper(head);
        cout << endl;
    }
    
    void displayReverseIterative() {
        cout << "Reverse Iterative: ";
        stack<int> st;
        SinglyNode* current = head;
        while (current) {
            st.push(current->data);
            current = current->next;
        }
        while (!st.empty()) {
            cout << st.top() << " ";
            st.pop();
        }
        cout << endl;
    }
    
    // List manipulation methods
    void reverse() {
        SinglyNode* prev = nullptr;
        SinglyNode* current = head;
        SinglyNode* next = nullptr;
        
        tail = head;
        
        while (current) {
            next = current->next;
            current->next = prev;
            prev = current;
            current = next;
        }
        head = prev;
        cout << "List reversed!" << endl;
    }
    
    void sortList() {
        if (!head || !head->next) return;
        
        // Simple bubble sort for demonstration
        bool swapped;
        do {
            swapped = false;
            SinglyNode* current = head;
            while (current->next) {
                if (current->data > current->next->data) {
                    swap(current->data, current->next->data);
                    swapped = true;
                }
                current = current->next;
            }
        } while (swapped);
        cout << "List sorted!" << endl;
    }
    
    void removeDuplicates() {
        if (!head) return;
        
        SinglyNode* current = head;
        while (current->next) {
            if (current->data == current->next->data) {
                SinglyNode* duplicate = current->next;
                current->next = duplicate->next;
                if (duplicate == tail) tail = current;
                delete duplicate;
                size--;
            } else {
                current = current->next;
            }
        }
        cout << "Duplicates removed!" << endl;
    }
    
    // Utility methods
    int getSize() { return size; }
    bool isEmpty() { return size == 0; }
    
    void clear() {
        while (head) {
            SinglyNode* temp = head;
            head = head->next;
            delete temp;
        }
        tail = nullptr;
        size = 0;
    }
    
    void printInfo() {
        cout << "Size: " << size << ", Head: " << (head ? to_string(head->data) : "NULL") 
             << ", Tail: " << (tail ? to_string(tail->data) : "NULL") << endl;
    }
};

// =============================================================================
// DOUBLY LINKED LIST - COMPLETE IMPLEMENTATION
// =============================================================================

struct DoublyNode {
    int data;
    DoublyNode* next;
    DoublyNode* prev;
    
    DoublyNode(int val) : data(val), next(nullptr), prev(nullptr) {}
};

class DoublyLinkedList {
private:
    DoublyNode* head;
    DoublyNode* tail;
    int size;

public:
    DoublyLinkedList() : head(nullptr), tail(nullptr), size(0) {}
    
    ~DoublyLinkedList() { clear(); }
    
    // Different ways to insert
    void insertAtBeginning(int val) {
        DoublyNode* newNode = new DoublyNode(val);
        if (!head) {
            head = tail = newNode;
        } else {
            newNode->next = head;
            head->prev = newNode;
            head = newNode;
        }
        size++;
        cout << "Inserted " << val << " at beginning" << endl;
    }
    
    void insertAtEnd(int val) {
        DoublyNode* newNode = new DoublyNode(val);
        if (!tail) {
            head = tail = newNode;
        } else {
            tail->next = newNode;
            newNode->prev = tail;
            tail = newNode;
        }
        size++;
        cout << "Inserted " << val << " at end" << endl;
    }
    
    void insertAtPosition(int pos, int val) {
        if (pos < 0 || pos > size) {
            cout << "Invalid position!" << endl;
            return;
        }
        
        if (pos == 0) {
            insertAtBeginning(val);
            return;
        }
        
        if (pos == size) {
            insertAtEnd(val);
            return;
        }
        
        DoublyNode* newNode = new DoublyNode(val);
        DoublyNode* current = head;
        for (int i = 0; i < pos; i++) {
            current = current->next;
        }
        
        newNode->next = current;
        newNode->prev = current->prev;
        current->prev->next = newNode;
        current->prev = newNode;
        size++;
        cout << "Inserted " << val << " at position " << pos << endl;
    }
    
    // Different ways to delete
    void deleteFromBeginning() {
        if (!head) {
            cout << "List is empty!" << endl;
            return;
        }
        
        DoublyNode* temp = head;
        head = head->next;
        if (head) {
            head->prev = nullptr;
        } else {
            tail = nullptr;
        }
        cout << "Deleted " << temp->data << " from beginning" << endl;
        delete temp;
        size--;
    }
    
    void deleteFromEnd() {
        if (!tail) {
            cout << "List is empty!" << endl;
            return;
        }
        
        DoublyNode* temp = tail;
        tail = tail->prev;
        if (tail) {
            tail->next = nullptr;
        } else {
            head = nullptr;
        }
        cout << "Deleted " << temp->data << " from end" << endl;
        delete temp;
        size--;
    }
    
    void deleteAtPosition(int pos) {
        if (pos < 0 || pos >= size) {
            cout << "Invalid position!" << endl;
            return;
        }
        
        if (pos == 0) {
            deleteFromBeginning();
            return;
        }
        
        if (pos == size - 1) {
            deleteFromEnd();
            return;
        }
        
        DoublyNode* current = head;
        for (int i = 0; i < pos; i++) {
            current = current->next;
        }
        
        current->prev->next = current->next;
        current->next->prev = current->prev;
        cout << "Deleted " << current->data << " from position " << pos << endl;
        delete current;
        size--;
    }
    
    // Different traversal methods
    void displayForward() {
        cout << "Forward: NULL <-> ";
        DoublyNode* current = head;
        while (current) {
            cout << current->data << " <-> ";
            current = current->next;
        }
        cout << "NULL" << endl;
    }
    
    void displayBackward() {
        cout << "Backward: NULL <-> ";
        DoublyNode* current = tail;
        while (current) {
            cout << current->data << " <-> ";
            current = current->prev;
        }
        cout << "NULL" << endl;
    }
    
    // Search methods
    bool search(int val) {
        DoublyNode* current = head;
        while (current) {
            if (current->data == val) return true;
            current = current->next;
        }
        return false;
    }
    
    int findIndex(int val) {
        DoublyNode* current = head;
        int index = 0;
        while (current) {
            if (current->data == val) return index;
            current = current->next;
            index++;
        }
        return -1;
    }
    
    // Access methods
    int getAt(int pos) {
        if (pos < 0 || pos >= size) {
            cout << "Invalid position!" << endl;
            return -1;
        }
        
        DoublyNode* current;
        if (pos < size / 2) {
            // Search from beginning
            current = head;
            for (int i = 0; i < pos; i++) {
                current = current->next;
            }
        } else {
            // Search from end (optimization for doubly linked list)
            current = tail;
            for (int i = size - 1; i > pos; i--) {
                current = current->prev;
            }
        }
        return current->data;
    }
    
    // Utility methods
    int getSize() { return size; }
    bool isEmpty() { return size == 0; }
    
    void clear() {
        while (head) {
            DoublyNode* temp = head;
            head = head->next;
            delete temp;
        }
        tail = nullptr;
        size = 0;
    }
    
    void printInfo() {
        cout << "Size: " << size << ", Head: " << (head ? to_string(head->data) : "NULL") 
             << ", Tail: " << (tail ? to_string(tail->data) : "NULL") << endl;
    }
};

// =============================================================================
// CIRCULAR LINKED LIST - COMPLETE IMPLEMENTATION
// =============================================================================

struct CircularNode {
    int data;
    CircularNode* next;
    
    CircularNode(int val) : data(val), next(nullptr) {}
};

class CircularLinkedList {
private:
    CircularNode* head;
    CircularNode* tail;
    int size;

public:
    CircularLinkedList() : head(nullptr), tail(nullptr), size(0) {}
    
    ~CircularLinkedList() { clear(); }
    
    // Different ways to insert
    void insertAtBeginning(int val) {
        CircularNode* newNode = new CircularNode(val);
        if (!head) {
            head = tail = newNode;
            newNode->next = newNode;
        } else {
            newNode->next = head;
            head = newNode;
            tail->next = head;
        }
        size++;
        cout << "Inserted " << val << " at beginning" << endl;
    }
    
    void insertAtEnd(int val) {
        CircularNode* newNode = new CircularNode(val);
        if (!head) {
            head = tail = newNode;
            newNode->next = newNode;
        } else {
            tail->next = newNode;
            newNode->next = head;
            tail = newNode;
        }
        size++;
        cout << "Inserted " << val << " at end" << endl;
    }
    
    void insertAtPosition(int pos, int val) {
        if (pos < 0 || pos > size) {
            cout << "Invalid position!" << endl;
            return;
        }
        
        if (pos == 0) {
            insertAtBeginning(val);
            return;
        }
        
        if (pos == size) {
            insertAtEnd(val);
            return;
        }
        
        CircularNode* newNode = new CircularNode(val);
        CircularNode* current = head;
        for (int i = 0; i < pos - 1; i++) {
            current = current->next;
        }
        
        newNode->next = current->next;
        current->next = newNode;
        size++;
        cout << "Inserted " << val << " at position " << pos << endl;
    }
    
    // Different ways to delete
    void deleteFromBeginning() {
        if (!head) {
            cout << "List is empty!" << endl;
            return;
        }
        
        if (head == tail) {
            cout << "Deleted " << head->data << " from beginning" << endl;
            delete head;
            head = tail = nullptr;
        } else {
            CircularNode* temp = head;
            head = head->next;
            tail->next = head;
            cout << "Deleted " << temp->data << " from beginning" << endl;
            delete temp;
        }
        size--;
    }
    
    void deleteFromEnd() {
        if (!head) {
            cout << "List is empty!" << endl;
            return;
        }
        
        if (head == tail) {
            cout << "Deleted " << head->data << " from end" << endl;
            delete head;
            head = tail = nullptr;
        } else {
            CircularNode* current = head;
            while (current->next != tail) {
                current = current->next;
            }
            cout << "Deleted " << tail->data << " from end" << endl;
            delete tail;
            tail = current;
            tail->next = head;
        }
        size--;
    }
    
    // Display methods
    void display() {
        if (!head) {
            cout << "List is empty!" << endl;
            return;
        }
        
        cout << "Circular List: ";
        CircularNode* current = head;
        do {
            cout << current->data << " -> ";
            current = current->next;
        } while (current != head);
        cout << "(back to " << head->data << ")" << endl;
    }
    
    void displayNTimes(int n) {
        if (!head) {
            cout << "List is empty!" << endl;
            return;
        }
        
        cout << "Circular List (" << n << " rounds): ";
        CircularNode* current = head;
        for (int i = 0; i < n * size; i++) {
            cout << current->data << " ";
            current = current->next;
        }
        cout << endl;
    }
    
    // Search methods
    bool search(int val) {
        if (!head) return false;
        
        CircularNode* current = head;
        do {
            if (current->data == val) return true;
            current = current->next;
        } while (current != head);
        return false;
    }
    
    int findIndex(int val) {
        if (!head) return -1;
        
        CircularNode* current = head;
        int index = 0;
        do {
            if (current->data == val) return index;
            current = current->next;
            index++;
        } while (current != head);
        return -1;
    }
    
    // Utility methods
    int getSize() { return size; }
    bool isEmpty() { return size == 0; }
    
    void clear() {
        if (!head) return;
        
        CircularNode* current = head;
        do {
            CircularNode* temp = current;
            current = current->next;
            delete temp;
        } while (current != head);
        
        head = tail = nullptr;
        size = 0;
    }
    
    void printInfo() {
        cout << "Size: " << size << ", Head: " << (head ? to_string(head->data) : "NULL") 
             << ", Tail: " << (tail ? to_string(tail->data) : "NULL") << endl;
    }
};

// =============================================================================
// DEMONSTRATION FUNCTIONS
// =============================================================================

void demonstrateSinglyLinkedList() {
    cout << "\n=== SINGLY LINKED LIST DEMONSTRATION ===" << endl;
    
    SinglyLinkedList sll;
    
    cout << "\n--- Insertion Methods ---" << endl;
    sll.insertAtBeginning(10);
    sll.insertAtEnd(20);
    sll.insertAtEnd(30);
    sll.insertAtPosition(1, 15);
    sll.insertAfterValue(15, 17);
    sll.insertBeforeValue(20, 18);
    sll.displayForward();
    sll.printInfo();
    
    cout << "\n--- Search Methods ---" << endl;
    cout << "Search 15: " << (sll.search(15) ? "Found" : "Not found") << endl;
    cout << "Index of 17: " << sll.findIndex(17) << endl;
    cout << "Element at position 2: " << sll.getAt(2) << endl;
    
    cout << "\n--- Different Display Methods ---" << endl;
    sll.displayForward();
    sll.displayRecursive();
    sll.displayReverseRecursive();
    sll.displayReverseIterative();
    
    cout << "\n--- List Manipulation ---" << endl;
    sll.reverse();
    sll.displayForward();
    
    sll.insertAtEnd(15);
    sll.insertAtEnd(17);
    sll.displayForward();
    sll.removeDuplicates();
    sll.displayForward();
    
    cout << "\n--- Deletion Methods ---" << endl;
    sll.deleteFromBeginning();
    sll.deleteFromEnd();
    sll.deleteAtPosition(1);
    sll.displayForward();
    sll.printInfo();
}

void demonstrateDoublyLinkedList() {
    cout << "\n=== DOUBLY LINKED LIST DEMONSTRATION ===" << endl;
    
    DoublyLinkedList dll;
    
    cout << "\n--- Insertion Methods ---" << endl;
    dll.insertAtBeginning(10);
    dll.insertAtEnd(20);
    dll.insertAtEnd(30);
    dll.insertAtPosition(1, 15);
    dll.displayForward();
    dll.printInfo();
    
    cout << "\n--- Different Display Methods ---" << endl;
    dll.displayForward();
    dll.displayBackward();
    
    cout << "\n--- Search and Access ---" << endl;
    cout << "Search 15: " << (dll.search(15) ? "Found" : "Not found") << endl;
    cout << "Index of 20: " << dll.findIndex(20) << endl;
    cout << "Element at position 2: " << dll.getAt(2) << endl;
    
    cout << "\n--- Deletion Methods ---" << endl;
    dll.deleteFromBeginning();
    dll.deleteFromEnd();
    dll.deleteAtPosition(1);
    dll.displayForward();
    dll.printInfo();
}

void demonstrateCircularLinkedList() {
    cout << "\n=== CIRCULAR LINKED LIST DEMONSTRATION ===" << endl;
    
    CircularLinkedList cll;
    
    cout << "\n--- Insertion Methods ---" << endl;
    cll.insertAtBeginning(10);
    cll.insertAtEnd(20);
    cll.insertAtEnd(30);
    cll.insertAtPosition(1, 15);
    cll.display();
    cll.printInfo();
    
    cout << "\n--- Circular Nature ---" << endl;
    cll.displayNTimes(2);
    
    cout << "\n--- Search Methods ---" << endl;
    cout << "Search 15: " << (cll.search(15) ? "Found" : "Not found") << endl;
    cout << "Index of 20: " << cll.findIndex(20) << endl;
    
    cout << "\n--- Deletion Methods ---" << endl;
    cll.deleteFromBeginning();
    cll.deleteFromEnd();
    cll.display();
    cll.printInfo();
}

void demonstrateLinkedListPatterns() {
    cout << "\n=== LINKED LIST PATTERNS & ALGORITHMS ===" << endl;
    
    SinglyLinkedList list;
    
    cout << "\n--- Pattern 1: Building List ---" << endl;
    for (int i = 1; i <= 5; i++) {
        list.insertAtEnd(i * 10);
    }
    list.displayForward();
    
    cout << "\n--- Pattern 2: Find Middle Element ---" << endl;
    // Using two pointers (slow and fast)
    cout << "Middle element using two pointers technique" << endl;
    int middle_pos = list.getSize() / 2;
    cout << "Middle element: " << list.getAt(middle_pos) << endl;
    
    cout << "\n--- Pattern 3: Reverse List ---" << endl;
    list.reverse();
    list.displayForward();
    
    cout << "\n--- Pattern 4: Detect Patterns ---" << endl;
    list.insertAtEnd(50);
    list.insertAtEnd(40);
    list.displayForward();
    
    bool has_duplicates = false;
    for (int i = 0; i < list.getSize(); i++) {
        for (int j = i + 1; j < list.getSize(); j++) {
            if (list.getAt(i) == list.getAt(j)) {
                has_duplicates = true;
                break;
            }
        }
        if (has_duplicates) break;
    }
    cout << "List has duplicates: " << (has_duplicates ? "Yes" : "No") << endl;
    
    cout << "\n--- Pattern 5: Sort List ---" << endl;
    list.sortList();
    list.displayForward();
}

// =============================================================================
// MAIN FUNCTION
// =============================================================================

int main() {
    cout << "=== COMPLETE LINKED LIST GUIDE ===" << endl;
    
    demonstrateSinglyLinkedList();
    demonstrateDoublyLinkedList();
    demonstrateCircularLinkedList();
    demonstrateLinkedListPatterns();
    
    cout << "\n=== SUMMARY ===" << endl;
    cout << "1. Singly Linked List: Simple, memory efficient, one-way traversal" << endl;
    cout << "2. Doubly Linked List: Two-way traversal, more memory usage" << endl;
    cout << "3. Circular Linked List: Last node connects to first, useful for round-robin" << endl;
    cout << "4. Each has different use cases and trade-offs" << endl;
    
    return 0;
} 