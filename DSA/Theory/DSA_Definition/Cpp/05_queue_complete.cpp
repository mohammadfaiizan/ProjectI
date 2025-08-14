/*
 * =============================================================================
 * COMPLETE QUEUE GUIDE - All Implementations & Operations
 * =============================================================================
 * 
 * This file covers:
 * 1. Array-based Queue implementation
 * 2. Linked List-based Queue implementation
 * 3. Circular Queue implementation
 * 4. Priority Queue implementation
 * 5. STL Queue operations
 * 6. Queue applications and algorithms
 * 
 * =============================================================================
 */

#include <iostream>
#include <queue>
#include <priority_queue>
#include <deque>
#include <vector>
#include <string>
using namespace std;

// =============================================================================
// ARRAY-BASED QUEUE IMPLEMENTATION
// =============================================================================

template<typename T>
class ArrayQueue {
private:
    T* data;
    int front;
    int rear;
    int capacity;
    int size;

public:
    // Constructor
    ArrayQueue(int cap = 10) : capacity(cap), front(0), rear(-1), size(0) {
        data = new T[capacity];
        cout << "Array queue created with capacity: " << capacity << endl;
    }
    
    // Destructor
    ~ArrayQueue() {
        delete[] data;
        cout << "Array queue destroyed" << endl;
    }
    
    // Copy constructor
    ArrayQueue(const ArrayQueue& other) 
        : capacity(other.capacity), front(other.front), rear(other.rear), size(other.size) {
        data = new T[capacity];
        for (int i = 0; i < capacity; i++) {
            data[i] = other.data[i];
        }
        cout << "Array queue copied" << endl;
    }
    
    // Assignment operator
    ArrayQueue& operator=(const ArrayQueue& other) {
        if (this != &other) {
            delete[] data;
            capacity = other.capacity;
            front = other.front;
            rear = other.rear;
            size = other.size;
            data = new T[capacity];
            for (int i = 0; i < capacity; i++) {
                data[i] = other.data[i];
            }
        }
        return *this;
    }
    
    // Basic operations
    void enqueue(const T& value) {
        if (isFull()) {
            cout << "Queue overflow! Cannot enqueue " << value << endl;
            return;
        }
        rear = (rear + 1) % capacity;
        data[rear] = value;
        size++;
        cout << "Enqueued " << value << " to array queue" << endl;
    }
    
    void dequeue() {
        if (isEmpty()) {
            cout << "Queue underflow! Cannot dequeue" << endl;
            return;
        }
        cout << "Dequeued " << data[front] << " from array queue" << endl;
        front = (front + 1) % capacity;
        size--;
    }
    
    T& frontElement() {
        if (isEmpty()) {
            throw runtime_error("Queue is empty!");
        }
        return data[front];
    }
    
    const T& frontElement() const {
        if (isEmpty()) {
            throw runtime_error("Queue is empty!");
        }
        return data[front];
    }
    
    T& rearElement() {
        if (isEmpty()) {
            throw runtime_error("Queue is empty!");
        }
        return data[rear];
    }
    
    const T& rearElement() const {
        if (isEmpty()) {
            throw runtime_error("Queue is empty!");
        }
        return data[rear];
    }
    
    // Utility methods
    bool isEmpty() const { return size == 0; }
    bool isFull() const { return size == capacity; }
    int getSize() const { return size; }
    int getCapacity() const { return capacity; }
    
    // Display methods
    void display() const {
        if (isEmpty()) {
            cout << "Array queue is empty" << endl;
            return;
        }
        cout << "Array queue (front to rear): ";
        for (int i = 0; i < size; i++) {
            int index = (front + i) % capacity;
            cout << data[index] << " ";
        }
        cout << endl;
    }
    
    void displayWithIndices() const {
        if (isEmpty()) {
            cout << "Array queue is empty" << endl;
            return;
        }
        cout << "Array queue with indices:" << endl;
        for (int i = 0; i < size; i++) {
            int index = (front + i) % capacity;
            cout << "Position " << i << " (Index " << index << "): " << data[index] << endl;
        }
    }
    
    void printInfo() const {
        cout << "Array Queue Info - Size: " << size << ", Capacity: " << capacity 
             << ", Front: " << front << ", Rear: " << rear << endl;
        if (!isEmpty()) {
            cout << "Front element: " << frontElement() << ", Rear element: " << rearElement() << endl;
        }
    }
    
    // Advanced operations
    void clear() {
        front = 0;
        rear = -1;
        size = 0;
        cout << "Array queue cleared" << endl;
    }
    
    T peek(int position) const {
        if (position < 0 || position >= size) {
            throw runtime_error("Invalid position!");
        }
        int index = (front + position) % capacity;
        return data[index];
    }
};

// =============================================================================
// LINKED LIST-BASED QUEUE IMPLEMENTATION
// =============================================================================

template<typename T>
class LinkedQueue {
private:
    struct Node {
        T data;
        Node* next;
        
        Node(const T& value) : data(value), next(nullptr) {}
    };
    
    Node* front;
    Node* rear;
    int size;

public:
    // Constructor
    LinkedQueue() : front(nullptr), rear(nullptr), size(0) {
        cout << "Linked queue created" << endl;
    }
    
    // Destructor
    ~LinkedQueue() {
        clear();
        cout << "Linked queue destroyed" << endl;
    }
    
    // Copy constructor
    LinkedQueue(const LinkedQueue& other) : front(nullptr), rear(nullptr), size(0) {
        if (other.front) {
            Node* current = other.front;
            while (current) {
                enqueue(current->data);
                current = current->next;
            }
        }
        cout << "Linked queue copied" << endl;
    }
    
    // Assignment operator
    LinkedQueue& operator=(const LinkedQueue& other) {
        if (this != &other) {
            clear();
            if (other.front) {
                Node* current = other.front;
                while (current) {
                    enqueue(current->data);
                    current = current->next;
                }
            }
        }
        return *this;
    }
    
    // Basic operations
    void enqueue(const T& value) {
        Node* new_node = new Node(value);
        if (isEmpty()) {
            front = rear = new_node;
        } else {
            rear->next = new_node;
            rear = new_node;
        }
        size++;
        cout << "Enqueued " << value << " to linked queue" << endl;
    }
    
    void dequeue() {
        if (isEmpty()) {
            cout << "Queue underflow! Cannot dequeue" << endl;
            return;
        }
        Node* temp = front;
        front = front->next;
        if (front == nullptr) {
            rear = nullptr;
        }
        cout << "Dequeued " << temp->data << " from linked queue" << endl;
        delete temp;
        size--;
    }
    
    T& frontElement() {
        if (isEmpty()) {
            throw runtime_error("Queue is empty!");
        }
        return front->data;
    }
    
    const T& frontElement() const {
        if (isEmpty()) {
            throw runtime_error("Queue is empty!");
        }
        return front->data;
    }
    
    T& rearElement() {
        if (isEmpty()) {
            throw runtime_error("Queue is empty!");
        }
        return rear->data;
    }
    
    const T& rearElement() const {
        if (isEmpty()) {
            throw runtime_error("Queue is empty!");
        }
        return rear->data;
    }
    
    // Utility methods
    bool isEmpty() const { return front == nullptr; }
    int getSize() const { return size; }
    
    // Display methods
    void display() const {
        if (isEmpty()) {
            cout << "Linked queue is empty" << endl;
            return;
        }
        cout << "Linked queue (front to rear): ";
        Node* current = front;
        while (current) {
            cout << current->data << " ";
            current = current->next;
        }
        cout << endl;
    }
    
    void displayWithPosition() const {
        if (isEmpty()) {
            cout << "Linked queue is empty" << endl;
            return;
        }
        cout << "Linked queue with positions:" << endl;
        Node* current = front;
        int position = 0;
        while (current) {
            cout << "Position " << position << ": " << current->data << endl;
            current = current->next;
            position++;
        }
    }
    
    void printInfo() const {
        cout << "Linked Queue Info - Size: " << size << endl;
        if (!isEmpty()) {
            cout << "Front element: " << frontElement() << ", Rear element: " << rearElement() << endl;
        }
    }
    
    // Advanced operations
    void clear() {
        while (!isEmpty()) {
            dequeue();
        }
    }
    
    T peek(int position) const {
        if (position < 0 || position >= size) {
            throw runtime_error("Invalid position!");
        }
        Node* current = front;
        for (int i = 0; i < position; i++) {
            current = current->next;
        }
        return current->data;
    }
};

// =============================================================================
// CIRCULAR QUEUE IMPLEMENTATION
// =============================================================================

template<typename T>
class CircularQueue {
private:
    T* data;
    int front;
    int rear;
    int capacity;
    int size;

public:
    // Constructor
    CircularQueue(int cap = 10) : capacity(cap), front(0), rear(0), size(0) {
        data = new T[capacity];
        cout << "Circular queue created with capacity: " << capacity << endl;
    }
    
    // Destructor
    ~CircularQueue() {
        delete[] data;
        cout << "Circular queue destroyed" << endl;
    }
    
    // Basic operations
    void enqueue(const T& value) {
        if (isFull()) {
            cout << "Circular queue overflow! Cannot enqueue " << value << endl;
            return;
        }
        data[rear] = value;
        rear = (rear + 1) % capacity;
        size++;
        cout << "Enqueued " << value << " to circular queue" << endl;
    }
    
    void dequeue() {
        if (isEmpty()) {
            cout << "Circular queue underflow! Cannot dequeue" << endl;
            return;
        }
        cout << "Dequeued " << data[front] << " from circular queue" << endl;
        front = (front + 1) % capacity;
        size--;
    }
    
    T& frontElement() {
        if (isEmpty()) {
            throw runtime_error("Queue is empty!");
        }
        return data[front];
    }
    
    const T& frontElement() const {
        if (isEmpty()) {
            throw runtime_error("Queue is empty!");
        }
        return data[front];
    }
    
    T& rearElement() {
        if (isEmpty()) {
            throw runtime_error("Queue is empty!");
        }
        int rear_index = (rear - 1 + capacity) % capacity;
        return data[rear_index];
    }
    
    // Utility methods
    bool isEmpty() const { return size == 0; }
    bool isFull() const { return size == capacity; }
    int getSize() const { return size; }
    int getCapacity() const { return capacity; }
    
    // Display methods
    void display() const {
        if (isEmpty()) {
            cout << "Circular queue is empty" << endl;
            return;
        }
        cout << "Circular queue (front to rear): ";
        for (int i = 0; i < size; i++) {
            int index = (front + i) % capacity;
            cout << data[index] << " ";
        }
        cout << endl;
    }
    
    void displayCircular() const {
        cout << "Circular queue array representation:" << endl;
        for (int i = 0; i < capacity; i++) {
            cout << "Index " << i << ": ";
            if (i == front && i == ((rear - 1 + capacity) % capacity) && size > 0) {
                cout << data[i] << " (F,R)";
            } else if (i == front && size > 0) {
                cout << data[i] << " (F)";
            } else if (i == ((rear - 1 + capacity) % capacity) && size > 0) {
                cout << data[i] << " (R)";
            } else if (size > 0) {
                bool inQueue = false;
                for (int j = 0; j < size; j++) {
                    if (((front + j) % capacity) == i) {
                        inQueue = true;
                        break;
                    }
                }
                if (inQueue) {
                    cout << data[i];
                } else {
                    cout << "empty";
                }
            } else {
                cout << "empty";
            }
            cout << endl;
        }
    }
    
    void printInfo() const {
        cout << "Circular Queue Info - Size: " << size << ", Capacity: " << capacity 
             << ", Front: " << front << ", Rear: " << rear << endl;
        if (!isEmpty()) {
            cout << "Front element: " << frontElement() << ", Rear element: " << rearElement() << endl;
        }
    }
};

// =============================================================================
// PRIORITY QUEUE IMPLEMENTATION
// =============================================================================

template<typename T>
class SimplePriorityQueue {
private:
    struct PriorityNode {
        T data;
        int priority;
        
        PriorityNode(const T& value, int prio) : data(value), priority(prio) {}
    };
    
    vector<PriorityNode> queue;

public:
    // Constructor
    SimplePriorityQueue() {
        cout << "Simple priority queue created" << endl;
    }
    
    // Destructor
    ~SimplePriorityQueue() {
        cout << "Simple priority queue destroyed" << endl;
    }
    
    // Basic operations
    void enqueue(const T& value, int priority) {
        queue.emplace_back(value, priority);
        cout << "Enqueued " << value << " with priority " << priority << endl;
    }
    
    void dequeue() {
        if (isEmpty()) {
            cout << "Priority queue underflow! Cannot dequeue" << endl;
            return;
        }
        
        // Find highest priority element
        int maxPriorityIndex = 0;
        for (int i = 1; i < queue.size(); i++) {
            if (queue[i].priority > queue[maxPriorityIndex].priority) {
                maxPriorityIndex = i;
            }
        }
        
        cout << "Dequeued " << queue[maxPriorityIndex].data 
             << " with priority " << queue[maxPriorityIndex].priority << endl;
        
        queue.erase(queue.begin() + maxPriorityIndex);
    }
    
    T& front() {
        if (isEmpty()) {
            throw runtime_error("Priority queue is empty!");
        }
        
        int maxPriorityIndex = 0;
        for (int i = 1; i < queue.size(); i++) {
            if (queue[i].priority > queue[maxPriorityIndex].priority) {
                maxPriorityIndex = i;
            }
        }
        
        return queue[maxPriorityIndex].data;
    }
    
    // Utility methods
    bool isEmpty() const { return queue.empty(); }
    int getSize() const { return queue.size(); }
    
    // Display methods
    void display() const {
        if (isEmpty()) {
            cout << "Priority queue is empty" << endl;
            return;
        }
        cout << "Priority queue (data, priority): ";
        for (const auto& node : queue) {
            cout << "(" << node.data << "," << node.priority << ") ";
        }
        cout << endl;
    }
    
    void displaySorted() const {
        if (isEmpty()) {
            cout << "Priority queue is empty" << endl;
            return;
        }
        
        vector<PriorityNode> sorted_queue = queue;
        sort(sorted_queue.begin(), sorted_queue.end(), 
             [](const PriorityNode& a, const PriorityNode& b) {
                 return a.priority > b.priority;
             });
        
        cout << "Priority queue sorted by priority: ";
        for (const auto& node : sorted_queue) {
            cout << "(" << node.data << "," << node.priority << ") ";
        }
        cout << endl;
    }
};

// =============================================================================
// STL QUEUE OPERATIONS
// =============================================================================

void demonstrate_stl_queue() {
    cout << "\n=== STL QUEUE OPERATIONS ===" << endl;
    
    // Different ways to create STL queue
    cout << "\n--- STL Queue Creation ---" << endl;
    
    // Method 1: Default queue (uses deque as underlying container)
    queue<int> q1;
    cout << "Default queue created" << endl;
    
    // Method 2: Queue with vector as underlying container
    queue<int, vector<int>> q2;
    cout << "Queue with vector container created" << endl;
    
    // Method 3: Queue with deque as underlying container (explicit)
    queue<int, deque<int>> q3;
    cout << "Queue with deque container created" << endl;
    
    // Basic operations
    cout << "\n--- Basic Operations ---" << endl;
    
    // Push elements
    for (int i = 1; i <= 5; i++) {
        q1.push(i * 10);
        cout << "Pushed " << i * 10 << " to STL queue" << endl;
    }
    
    // Queue properties
    cout << "Queue size: " << q1.size() << endl;
    cout << "Queue empty: " << (q1.empty() ? "Yes" : "No") << endl;
    cout << "Front element: " << q1.front() << endl;
    cout << "Back element: " << q1.back() << endl;
    
    // Pop elements
    cout << "\n--- Popping Elements ---" << endl;
    while (!q1.empty()) {
        cout << "Front: " << q1.front() << endl;
        q1.pop();
    }
    
    cout << "Queue size after popping all: " << q1.size() << endl;
    cout << "Queue empty: " << (q1.empty() ? "Yes" : "No") << endl;
    
    // Priority queue operations
    cout << "\n--- STL Priority Queue ---" << endl;
    
    // Max heap (default)
    priority_queue<int> max_pq;
    
    // Min heap
    priority_queue<int, vector<int>, greater<int>> min_pq;
    
    // Add elements
    int arr[] = {3, 1, 4, 1, 5, 9, 2, 6};
    for (int x : arr) {
        max_pq.push(x);
        min_pq.push(x);
    }
    
    cout << "Max heap elements: ";
    while (!max_pq.empty()) {
        cout << max_pq.top() << " ";
        max_pq.pop();
    }
    cout << endl;
    
    cout << "Min heap elements: ";
    while (!min_pq.empty()) {
        cout << min_pq.top() << " ";
        min_pq.pop();
    }
    cout << endl;
}

// =============================================================================
// QUEUE APPLICATIONS
// =============================================================================

class QueueApplications {
public:
    // Application 1: BFS Traversal (conceptual)
    static void bfsTraversal() {
        cout << "BFS Traversal simulation:" << endl;
        queue<int> q;
        vector<bool> visited(10, false);
        
        q.push(0);
        visited[0] = true;
        
        cout << "BFS order: ";
        while (!q.empty()) {
            int current = q.front();
            q.pop();
            cout << current << " ";
            
            // Simulate adding neighbors
            for (int neighbor = current + 1; neighbor <= current + 2 && neighbor < 10; neighbor++) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    q.push(neighbor);
                }
            }
        }
        cout << endl;
    }
    
    // Application 2: Level Order Traversal
    static void levelOrderTraversal() {
        cout << "Level order traversal simulation:" << endl;
        queue<pair<int, int>> q; // (value, level)
        
        q.push({1, 0});
        q.push({2, 1});
        q.push({3, 1});
        q.push({4, 2});
        q.push({5, 2});
        q.push({6, 2});
        q.push({7, 2});
        
        int current_level = -1;
        while (!q.empty()) {
            auto [value, level] = q.front();
            q.pop();
            
            if (level != current_level) {
                cout << "\nLevel " << level << ": ";
                current_level = level;
            }
            cout << value << " ";
        }
        cout << endl;
    }
    
    // Application 3: Sliding Window Maximum
    static vector<int> slidingWindowMaximum(const vector<int>& arr, int k) {
        deque<int> dq;
        vector<int> result;
        
        for (int i = 0; i < arr.size(); i++) {
            // Remove elements outside current window
            while (!dq.empty() && dq.front() <= i - k) {
                dq.pop_front();
            }
            
            // Remove elements smaller than current element
            while (!dq.empty() && arr[dq.back()] <= arr[i]) {
                dq.pop_back();
            }
            
            dq.push_back(i);
            
            // Add maximum to result if window is complete
            if (i >= k - 1) {
                result.push_back(arr[dq.front()]);
            }
        }
        
        return result;
    }
    
    // Application 4: First Non-Repeating Character
    static char firstNonRepeatingChar(const string& stream) {
        queue<char> q;
        int freq[256] = {0};
        
        for (char c : stream) {
            freq[c]++;
            q.push(c);
            
            while (!q.empty() && freq[q.front()] > 1) {
                q.pop();
            }
        }
        
        return q.empty() ? '\0' : q.front();
    }
    
    // Application 5: Generate Binary Numbers
    static vector<string> generateBinaryNumbers(int n) {
        vector<string> result;
        queue<string> q;
        
        q.push("1");
        
        for (int i = 0; i < n; i++) {
            string current = q.front();
            q.pop();
            result.push_back(current);
            
            q.push(current + "0");
            q.push(current + "1");
        }
        
        return result;
    }
};

// =============================================================================
// DEMONSTRATION FUNCTIONS
// =============================================================================

void demonstrate_array_queue() {
    cout << "\n=== ARRAY QUEUE DEMONSTRATION ===" << endl;
    
    ArrayQueue<int> arr_queue(5);
    
    // Enqueue elements
    for (int i = 1; i <= 6; i++) {
        arr_queue.enqueue(i * 10);
    }
    
    arr_queue.display();
    arr_queue.displayWithIndices();
    arr_queue.printInfo();
    
    // Dequeue elements
    cout << "\n--- Dequeuing Elements ---" << endl;
    arr_queue.dequeue();
    arr_queue.dequeue();
    arr_queue.display();
    arr_queue.printInfo();
    
    // Add more elements
    arr_queue.enqueue(70);
    arr_queue.enqueue(80);
    arr_queue.display();
    arr_queue.printInfo();
}

void demonstrate_linked_queue() {
    cout << "\n=== LINKED QUEUE DEMONSTRATION ===" << endl;
    
    LinkedQueue<int> linked_queue;
    
    // Enqueue elements
    for (int i = 1; i <= 5; i++) {
        linked_queue.enqueue(i * 10);
    }
    
    linked_queue.display();
    linked_queue.displayWithPosition();
    linked_queue.printInfo();
    
    // Dequeue elements
    cout << "\n--- Dequeuing Elements ---" << endl;
    linked_queue.dequeue();
    linked_queue.dequeue();
    linked_queue.display();
    linked_queue.printInfo();
}

void demonstrate_circular_queue() {
    cout << "\n=== CIRCULAR QUEUE DEMONSTRATION ===" << endl;
    
    CircularQueue<int> circular_queue(5);
    
    // Enqueue elements
    for (int i = 1; i <= 5; i++) {
        circular_queue.enqueue(i * 10);
    }
    
    circular_queue.display();
    circular_queue.displayCircular();
    circular_queue.printInfo();
    
    // Dequeue and enqueue to show circular nature
    cout << "\n--- Dequeue and Enqueue ---" << endl;
    circular_queue.dequeue();
    circular_queue.dequeue();
    circular_queue.enqueue(60);
    circular_queue.enqueue(70);
    
    circular_queue.display();
    circular_queue.displayCircular();
    circular_queue.printInfo();
}

void demonstrate_priority_queue() {
    cout << "\n=== PRIORITY QUEUE DEMONSTRATION ===" << endl;
    
    SimplePriorityQueue<string> pq;
    
    // Enqueue elements with different priorities
    pq.enqueue("Low", 1);
    pq.enqueue("High", 5);
    pq.enqueue("Medium", 3);
    pq.enqueue("Urgent", 10);
    pq.enqueue("Normal", 2);
    
    pq.display();
    pq.displaySorted();
    
    cout << "\n--- Dequeuing Elements ---" << endl;
    while (!pq.isEmpty()) {
        cout << "Front: " << pq.front() << endl;
        pq.dequeue();
    }
}

void demonstrate_queue_applications() {
    cout << "\n=== QUEUE APPLICATIONS DEMONSTRATION ===" << endl;
    
    // BFS Traversal
    cout << "\n--- BFS Traversal ---" << endl;
    QueueApplications::bfsTraversal();
    
    // Level Order Traversal
    cout << "\n--- Level Order Traversal ---" << endl;
    QueueApplications::levelOrderTraversal();
    
    // Sliding Window Maximum
    cout << "\n--- Sliding Window Maximum ---" << endl;
    vector<int> arr = {1, 2, 3, 1, 4, 5, 2, 3, 6};
    vector<int> result = QueueApplications::slidingWindowMaximum(arr, 3);
    cout << "Array: ";
    for (int x : arr) cout << x << " ";
    cout << endl;
    cout << "Sliding window max (k=3): ";
    for (int x : result) cout << x << " ";
    cout << endl;
    
    // First Non-Repeating Character
    cout << "\n--- First Non-Repeating Character ---" << endl;
    string stream = "abccba";
    char first_non_rep = QueueApplications::firstNonRepeatingChar(stream);
    cout << "Stream: " << stream << endl;
    cout << "First non-repeating: " << (first_non_rep ? first_non_rep : '0') << endl;
    
    // Generate Binary Numbers
    cout << "\n--- Generate Binary Numbers ---" << endl;
    vector<string> binary_nums = QueueApplications::generateBinaryNumbers(10);
    cout << "First 10 binary numbers: ";
    for (const string& binary : binary_nums) {
        cout << binary << " ";
    }
    cout << endl;
}

// =============================================================================
// MAIN FUNCTION
// =============================================================================

int main() {
    cout << "=== COMPLETE QUEUE GUIDE ===" << endl;
    
    demonstrate_array_queue();
    demonstrate_linked_queue();
    demonstrate_circular_queue();
    demonstrate_priority_queue();
    demonstrate_stl_queue();
    demonstrate_queue_applications();
    
    cout << "\n=== SUMMARY ===" << endl;
    cout << "1. Array Queue: Fixed capacity, circular implementation possible" << endl;
    cout << "2. Linked Queue: Dynamic size, no capacity limit" << endl;
    cout << "3. Circular Queue: Efficient space utilization" << endl;
    cout << "4. Priority Queue: Elements dequeued by priority" << endl;
    cout << "5. STL Queue: Ready-to-use, optimized implementation" << endl;
    cout << "6. Applications: BFS, level order, scheduling, sliding window" << endl;
    
    return 0;
} 