/*
 * =============================================================================
 * COMPLETE HEAP GUIDE - All Implementations & Operations
 * =============================================================================
 * 
 * This file covers:
 * 1. Min-Heap implementation
 * 2. Max-Heap implementation
 * 3. Heap operations (insert, extract, heapify)
 * 4. Priority Queue using heap
 * 5. STL heap operations
 * 6. Heap applications and algorithms
 * 
 * =============================================================================
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
#include <functional>
#include <climits>
using namespace std;

// =============================================================================
// MIN-HEAP IMPLEMENTATION
// =============================================================================

class MinHeap {
private:
    vector<int> heap;
    
    // Helper methods
    int parent(int i) const { return (i - 1) / 2; }
    int leftChild(int i) const { return 2 * i + 1; }
    int rightChild(int i) const { return 2 * i + 2; }
    
    void heapifyUp(int index) {
        while (index > 0 && heap[index] < heap[parent(index)]) {
            swap(heap[index], heap[parent(index)]);
            index = parent(index);
        }
    }
    
    void heapifyDown(int index) {
        int smallest = index;
        int left = leftChild(index);
        int right = rightChild(index);
        
        if (left < heap.size() && heap[left] < heap[smallest]) {
            smallest = left;
        }
        
        if (right < heap.size() && heap[right] < heap[smallest]) {
            smallest = right;
        }
        
        if (smallest != index) {
            swap(heap[index], heap[smallest]);
            heapifyDown(smallest);
        }
    }

public:
    // Constructor
    MinHeap() {
        cout << "Min-heap created" << endl;
    }
    
    // Constructor with initial values
    MinHeap(const vector<int>& values) {
        heap = values;
        buildHeap();
        cout << "Min-heap created from array" << endl;
    }
    
    // Build heap from array
    void buildHeap() {
        for (int i = parent(heap.size() - 1); i >= 0; i--) {
            heapifyDown(i);
        }
        cout << "Heap built from array" << endl;
    }
    
    // Insert element
    void insert(int value) {
        heap.push_back(value);
        heapifyUp(heap.size() - 1);
        cout << "Inserted " << value << " into min-heap" << endl;
    }
    
    // Extract minimum
    int extractMin() {
        if (isEmpty()) {
            cout << "Heap is empty!" << endl;
            return INT_MAX;
        }
        
        int min_val = heap[0];
        heap[0] = heap[heap.size() - 1];
        heap.pop_back();
        
        if (!isEmpty()) {
            heapifyDown(0);
        }
        
        cout << "Extracted minimum: " << min_val << endl;
        return min_val;
    }
    
    // Get minimum without removing
    int getMin() const {
        if (isEmpty()) {
            cout << "Heap is empty!" << endl;
            return INT_MAX;
        }
        return heap[0];
    }
    
    // Decrease key
    void decreaseKey(int index, int new_value) {
        if (index >= heap.size()) {
            cout << "Invalid index!" << endl;
            return;
        }
        
        if (new_value > heap[index]) {
            cout << "New value is greater than current value!" << endl;
            return;
        }
        
        heap[index] = new_value;
        heapifyUp(index);
        cout << "Decreased key at index " << index << " to " << new_value << endl;
    }
    
    // Delete element at index
    void deleteKey(int index) {
        if (index >= heap.size()) {
            cout << "Invalid index!" << endl;
            return;
        }
        
        decreaseKey(index, INT_MIN);
        extractMin();
        cout << "Deleted element at index " << index << endl;
    }
    
    // Search for value
    int search(int value) const {
        for (int i = 0; i < heap.size(); i++) {
            if (heap[i] == value) {
                return i;
            }
        }
        return -1;
    }
    
    // Display heap
    void display() const {
        cout << "Min-heap: ";
        for (int value : heap) {
            cout << value << " ";
        }
        cout << endl;
    }
    
    // Display heap as tree
    void displayAsTree() const {
        if (isEmpty()) {
            cout << "Heap is empty!" << endl;
            return;
        }
        
        cout << "Min-heap as tree:" << endl;
        displayTreeHelper(0, 0);
    }
    
    void displayTreeHelper(int index, int depth) const {
        if (index >= heap.size()) return;
        
        displayTreeHelper(rightChild(index), depth + 1);
        
        for (int i = 0; i < depth; i++) {
            cout << "    ";
        }
        cout << heap[index] << endl;
        
        displayTreeHelper(leftChild(index), depth + 1);
    }
    
    // Heap properties
    bool isEmpty() const { return heap.empty(); }
    int size() const { return heap.size(); }
    
    // Verify heap property
    bool isValidHeap() const {
        for (int i = 0; i < heap.size(); i++) {
            int left = leftChild(i);
            int right = rightChild(i);
            
            if (left < heap.size() && heap[i] > heap[left]) {
                return false;
            }
            
            if (right < heap.size() && heap[i] > heap[right]) {
                return false;
            }
        }
        return true;
    }
    
    void printInfo() const {
        cout << "Min-heap info:" << endl;
        cout << "Size: " << size() << endl;
        cout << "Is empty: " << (isEmpty() ? "Yes" : "No") << endl;
        cout << "Is valid heap: " << (isValidHeap() ? "Yes" : "No") << endl;
        if (!isEmpty()) {
            cout << "Minimum element: " << getMin() << endl;
        }
    }
    
    // Get heap array (for external use)
    const vector<int>& getHeap() const { return heap; }
};

// =============================================================================
// MAX-HEAP IMPLEMENTATION
// =============================================================================

class MaxHeap {
private:
    vector<int> heap;
    
    // Helper methods
    int parent(int i) const { return (i - 1) / 2; }
    int leftChild(int i) const { return 2 * i + 1; }
    int rightChild(int i) const { return 2 * i + 2; }
    
    void heapifyUp(int index) {
        while (index > 0 && heap[index] > heap[parent(index)]) {
            swap(heap[index], heap[parent(index)]);
            index = parent(index);
        }
    }
    
    void heapifyDown(int index) {
        int largest = index;
        int left = leftChild(index);
        int right = rightChild(index);
        
        if (left < heap.size() && heap[left] > heap[largest]) {
            largest = left;
        }
        
        if (right < heap.size() && heap[right] > heap[largest]) {
            largest = right;
        }
        
        if (largest != index) {
            swap(heap[index], heap[largest]);
            heapifyDown(largest);
        }
    }

public:
    // Constructor
    MaxHeap() {
        cout << "Max-heap created" << endl;
    }
    
    // Constructor with initial values
    MaxHeap(const vector<int>& values) {
        heap = values;
        buildHeap();
        cout << "Max-heap created from array" << endl;
    }
    
    // Build heap from array
    void buildHeap() {
        for (int i = parent(heap.size() - 1); i >= 0; i--) {
            heapifyDown(i);
        }
        cout << "Heap built from array" << endl;
    }
    
    // Insert element
    void insert(int value) {
        heap.push_back(value);
        heapifyUp(heap.size() - 1);
        cout << "Inserted " << value << " into max-heap" << endl;
    }
    
    // Extract maximum
    int extractMax() {
        if (isEmpty()) {
            cout << "Heap is empty!" << endl;
            return INT_MIN;
        }
        
        int max_val = heap[0];
        heap[0] = heap[heap.size() - 1];
        heap.pop_back();
        
        if (!isEmpty()) {
            heapifyDown(0);
        }
        
        cout << "Extracted maximum: " << max_val << endl;
        return max_val;
    }
    
    // Get maximum without removing
    int getMax() const {
        if (isEmpty()) {
            cout << "Heap is empty!" << endl;
            return INT_MIN;
        }
        return heap[0];
    }
    
    // Increase key
    void increaseKey(int index, int new_value) {
        if (index >= heap.size()) {
            cout << "Invalid index!" << endl;
            return;
        }
        
        if (new_value < heap[index]) {
            cout << "New value is smaller than current value!" << endl;
            return;
        }
        
        heap[index] = new_value;
        heapifyUp(index);
        cout << "Increased key at index " << index << " to " << new_value << endl;
    }
    
    // Delete element at index
    void deleteKey(int index) {
        if (index >= heap.size()) {
            cout << "Invalid index!" << endl;
            return;
        }
        
        increaseKey(index, INT_MAX);
        extractMax();
        cout << "Deleted element at index " << index << endl;
    }
    
    // Display heap
    void display() const {
        cout << "Max-heap: ";
        for (int value : heap) {
            cout << value << " ";
        }
        cout << endl;
    }
    
    // Display heap as tree
    void displayAsTree() const {
        if (isEmpty()) {
            cout << "Heap is empty!" << endl;
            return;
        }
        
        cout << "Max-heap as tree:" << endl;
        displayTreeHelper(0, 0);
    }
    
    void displayTreeHelper(int index, int depth) const {
        if (index >= heap.size()) return;
        
        displayTreeHelper(rightChild(index), depth + 1);
        
        for (int i = 0; i < depth; i++) {
            cout << "    ";
        }
        cout << heap[index] << endl;
        
        displayTreeHelper(leftChild(index), depth + 1);
    }
    
    // Heap properties
    bool isEmpty() const { return heap.empty(); }
    int size() const { return heap.size(); }
    
    // Verify heap property
    bool isValidHeap() const {
        for (int i = 0; i < heap.size(); i++) {
            int left = leftChild(i);
            int right = rightChild(i);
            
            if (left < heap.size() && heap[i] < heap[left]) {
                return false;
            }
            
            if (right < heap.size() && heap[i] < heap[right]) {
                return false;
            }
        }
        return true;
    }
    
    void printInfo() const {
        cout << "Max-heap info:" << endl;
        cout << "Size: " << size() << endl;
        cout << "Is empty: " << (isEmpty() ? "Yes" : "No") << endl;
        cout << "Is valid heap: " << (isValidHeap() ? "Yes" : "No") << endl;
        if (!isEmpty()) {
            cout << "Maximum element: " << getMax() << endl;
        }
    }
    
    // Get heap array (for external use)
    const vector<int>& getHeap() const { return heap; }
};

// =============================================================================
// GENERIC HEAP IMPLEMENTATION
// =============================================================================

template<typename T, typename Compare = less<T>>
class GenericHeap {
private:
    vector<T> heap;
    Compare comp;
    
    // Helper methods
    int parent(int i) const { return (i - 1) / 2; }
    int leftChild(int i) const { return 2 * i + 1; }
    int rightChild(int i) const { return 2 * i + 2; }
    
    void heapifyUp(int index) {
        while (index > 0 && comp(heap[index], heap[parent(index)])) {
            swap(heap[index], heap[parent(index)]);
            index = parent(index);
        }
    }
    
    void heapifyDown(int index) {
        int target = index;
        int left = leftChild(index);
        int right = rightChild(index);
        
        if (left < heap.size() && comp(heap[left], heap[target])) {
            target = left;
        }
        
        if (right < heap.size() && comp(heap[right], heap[target])) {
            target = right;
        }
        
        if (target != index) {
            swap(heap[index], heap[target]);
            heapifyDown(target);
        }
    }

public:
    // Constructor
    GenericHeap(Compare c = Compare()) : comp(c) {
        cout << "Generic heap created" << endl;
    }
    
    // Insert element
    void insert(const T& value) {
        heap.push_back(value);
        heapifyUp(heap.size() - 1);
        cout << "Inserted " << value << " into generic heap" << endl;
    }
    
    // Extract top element
    T extractTop() {
        if (isEmpty()) {
            throw runtime_error("Heap is empty!");
        }
        
        T top_val = heap[0];
        heap[0] = heap[heap.size() - 1];
        heap.pop_back();
        
        if (!isEmpty()) {
            heapifyDown(0);
        }
        
        cout << "Extracted top: " << top_val << endl;
        return top_val;
    }
    
    // Get top element without removing
    const T& top() const {
        if (isEmpty()) {
            throw runtime_error("Heap is empty!");
        }
        return heap[0];
    }
    
    // Display heap
    void display() const {
        cout << "Generic heap: ";
        for (const T& value : heap) {
            cout << value << " ";
        }
        cout << endl;
    }
    
    // Heap properties
    bool isEmpty() const { return heap.empty(); }
    int size() const { return heap.size(); }
};

// =============================================================================
// STL HEAP OPERATIONS
// =============================================================================

void demonstrate_stl_heap() {
    cout << "\n=== STL HEAP OPERATIONS ===" << endl;
    
    // Using STL heap functions
    cout << "\n--- STL Heap Functions ---" << endl;
    
    vector<int> vec = {3, 1, 4, 1, 5, 9, 2, 6};
    cout << "Original vector: ";
    for (int x : vec) cout << x << " ";
    cout << endl;
    
    // Make heap (max-heap by default)
    make_heap(vec.begin(), vec.end());
    cout << "After make_heap: ";
    for (int x : vec) cout << x << " ";
    cout << endl;
    
    // Push heap
    vec.push_back(8);
    push_heap(vec.begin(), vec.end());
    cout << "After push_heap(8): ";
    for (int x : vec) cout << x << " ";
    cout << endl;
    
    // Pop heap
    pop_heap(vec.begin(), vec.end());
    int popped = vec.back();
    vec.pop_back();
    cout << "After pop_heap, popped: " << popped << endl;
    cout << "Heap now: ";
    for (int x : vec) cout << x << " ";
    cout << endl;
    
    // Sort heap
    sort_heap(vec.begin(), vec.end());
    cout << "After sort_heap: ";
    for (int x : vec) cout << x << " ";
    cout << endl;
    
    // Min-heap using STL
    cout << "\n--- STL Min-Heap ---" << endl;
    vector<int> min_vec = {3, 1, 4, 1, 5, 9, 2, 6};
    make_heap(min_vec.begin(), min_vec.end(), greater<int>());
    cout << "Min-heap: ";
    for (int x : min_vec) cout << x << " ";
    cout << endl;
    
    // Priority queue
    cout << "\n--- STL Priority Queue ---" << endl;
    
    // Max-heap priority queue (default)
    priority_queue<int> max_pq;
    for (int x : {3, 1, 4, 1, 5, 9, 2, 6}) {
        max_pq.push(x);
    }
    
    cout << "Max-heap priority queue: ";
    while (!max_pq.empty()) {
        cout << max_pq.top() << " ";
        max_pq.pop();
    }
    cout << endl;
    
    // Min-heap priority queue
    priority_queue<int, vector<int>, greater<int>> min_pq;
    for (int x : {3, 1, 4, 1, 5, 9, 2, 6}) {
        min_pq.push(x);
    }
    
    cout << "Min-heap priority queue: ";
    while (!min_pq.empty()) {
        cout << min_pq.top() << " ";
        min_pq.pop();
    }
    cout << endl;
    
    // Custom comparator
    cout << "\n--- Custom Comparator ---" << endl;
    
    struct Person {
        string name;
        int age;
        
        Person(string n, int a) : name(n), age(a) {}
    };
    
    auto person_comp = [](const Person& a, const Person& b) {
        return a.age > b.age; // Min-heap based on age
    };
    
    priority_queue<Person, vector<Person>, decltype(person_comp)> person_pq(person_comp);
    person_pq.emplace("Alice", 30);
    person_pq.emplace("Bob", 25);
    person_pq.emplace("Charlie", 35);
    
    cout << "Person priority queue (by age): ";
    while (!person_pq.empty()) {
        cout << person_pq.top().name << "(" << person_pq.top().age << ") ";
        person_pq.pop();
    }
    cout << endl;
}

// =============================================================================
// HEAP APPLICATIONS
// =============================================================================

class HeapApplications {
public:
    // Application 1: Heap Sort
    static void heapSort(vector<int>& arr) {
        cout << "Heap sort process:" << endl;
        
        // Build max-heap
        make_heap(arr.begin(), arr.end());
        cout << "After building heap: ";
        for (int x : arr) cout << x << " ";
        cout << endl;
        
        // Extract elements one by one
        for (int i = arr.size() - 1; i > 0; i--) {
            pop_heap(arr.begin(), arr.begin() + i + 1);
            cout << "After extracting " << arr[i] << ": ";
            for (int j = 0; j < i; j++) cout << arr[j] << " ";
            cout << "| " << arr[i] << endl;
        }
        
        cout << "Final sorted array: ";
        for (int x : arr) cout << x << " ";
        cout << endl;
    }
    
    // Application 2: Find K largest elements
    static vector<int> findKLargest(const vector<int>& arr, int k) {
        priority_queue<int, vector<int>, greater<int>> min_heap;
        
        for (int num : arr) {
            if (min_heap.size() < k) {
                min_heap.push(num);
            } else if (num > min_heap.top()) {
                min_heap.pop();
                min_heap.push(num);
            }
        }
        
        vector<int> result;
        while (!min_heap.empty()) {
            result.push_back(min_heap.top());
            min_heap.pop();
        }
        
        reverse(result.begin(), result.end());
        return result;
    }
    
    // Application 3: Merge K sorted arrays
    static vector<int> mergeKSortedArrays(const vector<vector<int>>& arrays) {
        struct Element {
            int value;
            int array_index;
            int element_index;
            
            bool operator>(const Element& other) const {
                return value > other.value;
            }
        };
        
        priority_queue<Element, vector<Element>, greater<Element>> min_heap;
        
        // Add first element from each array
        for (int i = 0; i < arrays.size(); i++) {
            if (!arrays[i].empty()) {
                min_heap.push({arrays[i][0], i, 0});
            }
        }
        
        vector<int> result;
        
        while (!min_heap.empty()) {
            Element current = min_heap.top();
            min_heap.pop();
            
            result.push_back(current.value);
            
            // Add next element from the same array
            if (current.element_index + 1 < arrays[current.array_index].size()) {
                min_heap.push({
                    arrays[current.array_index][current.element_index + 1],
                    current.array_index,
                    current.element_index + 1
                });
            }
        }
        
        return result;
    }
    
    // Application 4: Running median
    class RunningMedian {
    private:
        priority_queue<int> max_heap; // for smaller half
        priority_queue<int, vector<int>, greater<int>> min_heap; // for larger half
        
    public:
        void addNumber(int num) {
            // Add to appropriate heap
            if (max_heap.empty() || num <= max_heap.top()) {
                max_heap.push(num);
            } else {
                min_heap.push(num);
            }
            
            // Balance heaps
            if (max_heap.size() > min_heap.size() + 1) {
                min_heap.push(max_heap.top());
                max_heap.pop();
            } else if (min_heap.size() > max_heap.size() + 1) {
                max_heap.push(min_heap.top());
                min_heap.pop();
            }
        }
        
        double getMedian() {
            if (max_heap.size() == min_heap.size()) {
                return (max_heap.top() + min_heap.top()) / 2.0;
            } else if (max_heap.size() > min_heap.size()) {
                return max_heap.top();
            } else {
                return min_heap.top();
            }
        }
    };
    
    // Application 5: Task Scheduler
    static int taskScheduler(const vector<char>& tasks, int n) {
        map<char, int> task_count;
        for (char task : tasks) {
            task_count[task]++;
        }
        
        priority_queue<int> max_heap;
        for (const auto& pair : task_count) {
            max_heap.push(pair.second);
        }
        
        int time = 0;
        
        while (!max_heap.empty()) {
            vector<int> temp;
            int cycle = 0;
            
            // Process tasks in one cycle
            for (int i = 0; i <= n; i++) {
                if (!max_heap.empty()) {
                    int count = max_heap.top();
                    max_heap.pop();
                    
                    if (count > 1) {
                        temp.push_back(count - 1);
                    }
                    cycle++;
                }
            }
            
            // Put back remaining tasks
            for (int count : temp) {
                max_heap.push(count);
            }
            
            time += max_heap.empty() ? cycle : n + 1;
        }
        
        return time;
    }
};

// =============================================================================
// DEMONSTRATION FUNCTIONS
// =============================================================================

void demonstrate_min_heap() {
    cout << "\n=== MIN-HEAP DEMONSTRATION ===" << endl;
    
    MinHeap min_heap;
    
    // Insert elements
    vector<int> values = {4, 1, 3, 2, 16, 9, 10, 14, 8, 7};
    for (int val : values) {
        min_heap.insert(val);
    }
    
    min_heap.display();
    min_heap.displayAsTree();
    min_heap.printInfo();
    
    // Extract operations
    cout << "\n--- Extract Operations ---" << endl;
    cout << "Minimum: " << min_heap.getMin() << endl;
    min_heap.extractMin();
    min_heap.extractMin();
    min_heap.display();
    
    // Key operations
    cout << "\n--- Key Operations ---" << endl;
    int index = min_heap.search(16);
    if (index != -1) {
        min_heap.decreaseKey(index, 1);
    }
    min_heap.display();
    
    // Build heap from array
    cout << "\n--- Build Heap from Array ---" << endl;
    vector<int> arr = {20, 15, 8, 10, 5, 7, 6, 2, 9, 1};
    MinHeap heap_from_array(arr);
    heap_from_array.display();
    heap_from_array.printInfo();
}

void demonstrate_max_heap() {
    cout << "\n=== MAX-HEAP DEMONSTRATION ===" << endl;
    
    MaxHeap max_heap;
    
    // Insert elements
    vector<int> values = {4, 1, 3, 2, 16, 9, 10, 14, 8, 7};
    for (int val : values) {
        max_heap.insert(val);
    }
    
    max_heap.display();
    max_heap.displayAsTree();
    max_heap.printInfo();
    
    // Extract operations
    cout << "\n--- Extract Operations ---" << endl;
    cout << "Maximum: " << max_heap.getMax() << endl;
    max_heap.extractMax();
    max_heap.extractMax();
    max_heap.display();
    
    // Key operations
    cout << "\n--- Key Operations ---" << endl;
    int index = max_heap.search(7);
    if (index != -1) {
        max_heap.increaseKey(index, 20);
    }
    max_heap.display();
}

void demonstrate_generic_heap() {
    cout << "\n=== GENERIC HEAP DEMONSTRATION ===" << endl;
    
    // Min-heap for integers
    cout << "\n--- Generic Min-Heap ---" << endl;
    GenericHeap<int> min_heap;
    for (int val : {4, 1, 3, 2, 16, 9, 10}) {
        min_heap.insert(val);
    }
    min_heap.display();
    
    // Max-heap for integers
    cout << "\n--- Generic Max-Heap ---" << endl;
    GenericHeap<int, greater<int>> max_heap;
    for (int val : {4, 1, 3, 2, 16, 9, 10}) {
        max_heap.insert(val);
    }
    max_heap.display();
    
    // String heap
    cout << "\n--- String Heap ---" << endl;
    GenericHeap<string> string_heap;
    for (const string& word : {"apple", "banana", "cherry", "date"}) {
        string_heap.insert(word);
    }
    string_heap.display();
}

void demonstrate_heap_applications() {
    cout << "\n=== HEAP APPLICATIONS DEMONSTRATION ===" << endl;
    
    // Heap sort
    cout << "\n--- Heap Sort ---" << endl;
    vector<int> arr = {12, 11, 13, 5, 6, 7};
    cout << "Original array: ";
    for (int x : arr) cout << x << " ";
    cout << endl;
    
    HeapApplications::heapSort(arr);
    
    // Find K largest
    cout << "\n--- Find K Largest Elements ---" << endl;
    vector<int> nums = {3, 2, 1, 5, 6, 4};
    int k = 3;
    vector<int> largest = HeapApplications::findKLargest(nums, k);
    cout << "Array: ";
    for (int x : nums) cout << x << " ";
    cout << endl;
    cout << k << " largest elements: ";
    for (int x : largest) cout << x << " ";
    cout << endl;
    
    // Merge K sorted arrays
    cout << "\n--- Merge K Sorted Arrays ---" << endl;
    vector<vector<int>> arrays = {
        {1, 4, 7},
        {2, 5, 8},
        {3, 6, 9}
    };
    
    cout << "Input arrays:" << endl;
    for (const auto& arr : arrays) {
        for (int x : arr) cout << x << " ";
        cout << endl;
    }
    
    vector<int> merged = HeapApplications::mergeKSortedArrays(arrays);
    cout << "Merged array: ";
    for (int x : merged) cout << x << " ";
    cout << endl;
    
    // Running median
    cout << "\n--- Running Median ---" << endl;
    HeapApplications::RunningMedian rm;
    vector<int> stream = {5, 15, 1, 3, 8, 7, 9, 2, 4, 6};
    
    cout << "Stream: ";
    for (int num : stream) {
        rm.addNumber(num);
        cout << num << "(median:" << rm.getMedian() << ") ";
    }
    cout << endl;
}

// =============================================================================
// MAIN FUNCTION
// =============================================================================

int main() {
    cout << "=== COMPLETE HEAP GUIDE ===" << endl;
    
    demonstrate_min_heap();
    demonstrate_max_heap();
    demonstrate_generic_heap();
    demonstrate_stl_heap();
    demonstrate_heap_applications();
    
    cout << "\n=== SUMMARY ===" << endl;
    cout << "1. Min-Heap: Root is minimum, parent ≤ children" << endl;
    cout << "2. Max-Heap: Root is maximum, parent ≥ children" << endl;
    cout << "3. Operations: Insert O(log n), Extract O(log n), Peek O(1)" << endl;
    cout << "4. Build Heap: O(n) from array" << endl;
    cout << "5. STL: priority_queue, make_heap, push_heap, pop_heap" << endl;
    cout << "6. Applications: Priority queues, heap sort, median finding" << endl;
    cout << "7. Space Complexity: O(n)" << endl;
    
    return 0;
} 