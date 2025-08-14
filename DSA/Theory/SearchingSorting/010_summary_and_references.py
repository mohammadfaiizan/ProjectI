"""
Searching & Sorting - Summary and References
===========================================

Complete study guide and reference for all searching and sorting algorithms
Companies: All major tech companies
Difficulty: All levels
"""

class SearchingSortingSummary:
    
    def get_algorithm_summary(self):
        """Complete summary of all algorithms covered"""
        return {
            "Searching Algorithms": {
                "Linear Search": {
                    "Time": "O(n)", "Space": "O(1)",
                    "Use Case": "Unsorted data, small datasets",
                    "LeetCode": "Basic implementation"
                },
                "Binary Search": {
                    "Time": "O(log n)", "Space": "O(1)",
                    "Use Case": "Sorted data, large datasets",
                    "LeetCode": "704, 35, 33, 81, 153, 154"
                },
                "Jump Search": {
                    "Time": "O(√n)", "Space": "O(1)",
                    "Use Case": "Sorted data, block-wise access",
                    "LeetCode": "Custom implementations"
                },
                "Interpolation Search": {
                    "Time": "O(log log n)", "Space": "O(1)",
                    "Use Case": "Uniformly distributed sorted data",
                    "LeetCode": "Custom implementations"
                },
                "Exponential Search": {
                    "Time": "O(log n)", "Space": "O(1)",
                    "Use Case": "Unbounded/infinite arrays",
                    "LeetCode": "702"
                }
            },
            
            "Sorting Algorithms": {
                "Bubble Sort": {
                    "Time": "O(n²)", "Space": "O(1)",
                    "Stable": "Yes", "Use Case": "Educational, small data"
                },
                "Selection Sort": {
                    "Time": "O(n²)", "Space": "O(1)",
                    "Stable": "No", "Use Case": "Memory constraints"
                },
                "Insertion Sort": {
                    "Time": "O(n²)", "Space": "O(1)",
                    "Stable": "Yes", "Use Case": "Small/nearly sorted data"
                },
                "Merge Sort": {
                    "Time": "O(n log n)", "Space": "O(n)",
                    "Stable": "Yes", "Use Case": "General purpose, external sorting"
                },
                "Quick Sort": {
                    "Time": "O(n log n)", "Space": "O(log n)",
                    "Stable": "No", "Use Case": "General purpose, in-place"
                },
                "Heap Sort": {
                    "Time": "O(n log n)", "Space": "O(1)",
                    "Stable": "No", "Use Case": "Guaranteed performance"
                },
                "Counting Sort": {
                    "Time": "O(n + k)", "Space": "O(k)",
                    "Stable": "Yes", "Use Case": "Small integer range"
                },
                "Radix Sort": {
                    "Time": "O(d(n + k))", "Space": "O(n + k)",
                    "Stable": "Yes", "Use Case": "Integer/string sorting"
                }
            }
        }
    
    def get_leetcode_problems_by_company(self):
        """LeetCode problems organized by company"""
        return {
            "Google": [
                "LC 23: Merge k Sorted Lists",
                "LC 295: Find Median from Data Stream",
                "LC 253: Meeting Rooms II",
                "LC 300: Longest Increasing Subsequence",
                "LC 1268: Search Suggestions System"
            ],
            
            "Facebook/Meta": [
                "LC 973: K Closest Points to Origin",
                "LC 680: Valid Palindrome II",
                "LC 560: Subarray Sum Equals K",
                "LC 56: Merge Intervals",
                "LC 347: Top K Frequent Elements"
            ],
            
            "Amazon": [
                "LC 1: Two Sum",
                "LC 15: 3Sum",
                "LC 16: 3Sum Closest",
                "LC 11: Container With Most Water",
                "LC 33: Search in Rotated Sorted Array"
            ],
            
            "Microsoft": [
                "LC 493: Reverse Pairs",
                "LC 373: Find K Pairs with Smallest Sums",
                "LC 215: Kth Largest Element",
                "LC 75: Sort Colors"
            ],
            
            "Apple": [
                "LC 164: Maximum Gap",
                "LC 34: Find First and Last Position",
                "LC 128: Longest Consecutive Sequence"
            ]
        }
    
    def get_study_roadmap(self):
        """Recommended study roadmap"""
        return {
            "Week 1 - Fundamentals": [
                "Master binary search template",
                "Understand O(n²) sorting algorithms",
                "Practice basic search problems"
            ],
            
            "Week 2 - Intermediate": [
                "Learn merge sort and quick sort",
                "Rotated array problems",
                "Two pointers technique"
            ],
            
            "Week 3 - Advanced": [
                "Heap sort and priority queues",
                "Non-comparison sorting",
                "String searching algorithms"
            ],
            
            "Week 4 - Expert": [
                "External sorting concepts",
                "Parallel algorithms",
                "Optimization techniques"
            ]
        }
    
    def get_interview_tips(self):
        """Tips for coding interviews"""
        return {
            "Binary Search": [
                "Always clarify array bounds",
                "Watch for integer overflow in mid calculation",
                "Consider edge cases: empty array, single element",
                "Practice both iterative and recursive versions"
            ],
            
            "Sorting": [
                "Know when to use each algorithm",
                "Understand stability requirements",
                "Consider space constraints",
                "Be ready to implement from scratch"
            ],
            
            "General Tips": [
                "Start with brute force, then optimize",
                "Clarify input constraints",
                "Test with edge cases",
                "Explain time/space complexity"
            ]
        }

# Main summary display
def display_summary():
    summary = SearchingSortingSummary()
    
    print("=== SEARCHING & SORTING - COMPLETE SUMMARY ===\n")
    
    print("📚 ALGORITHM OVERVIEW:")
    algorithms = summary.get_algorithm_summary()
    
    for category, algs in algorithms.items():
        print(f"\n{category}:")
        for name, details in algs.items():
            print(f"  • {name}: {details['Time']} time, {details['Space']} space")
    
    print("\n🏢 COMPANY PROBLEMS:")
    company_problems = summary.get_leetcode_problems_by_company()
    
    for company, problems in company_problems.items():
        print(f"\n{company}:")
        for problem in problems:
            print(f"  • {problem}")
    
    print("\n📅 STUDY ROADMAP:")
    roadmap = summary.get_study_roadmap()
    
    for week, topics in roadmap.items():
        print(f"\n{week}:")
        for topic in topics:
            print(f"  • {topic}")
    
    print("\n💡 INTERVIEW TIPS:")
    tips = summary.get_interview_tips()
    
    for category, tip_list in tips.items():
        print(f"\n{category}:")
        for tip in tip_list:
            print(f"  • {tip}")
    
    print("\n🎯 FILES IN THIS MODULE:")
    files = [
        "001_basic_searching_algorithms.py - Linear, Binary, Jump, Interpolation",
        "002_comparison_based_sorting.py - Bubble, Selection, Insertion, Merge, Quick, Heap",
        "003_non_comparison_sorting.py - Counting, Radix, Bucket, Pigeonhole",
        "004_advanced_searching_problems.py - 2D search, Peak finding, Binary search on answer",
        "005_specialized_sorting_techniques.py - Hybrid sorts, Topological sort",
        "006_interview_problems.py - Real company interview questions",
        "007_sorting_complexity_analysis.py - Performance analysis and benchmarking",
        "008_string_searching_algorithms.py - KMP, Rabin-Karp, Boyer-Moore",
        "009_optimization_techniques.py - Cache-friendly, memory-efficient algorithms",
        "010_summary_and_references.py - This comprehensive guide"
    ]
    
    for file_info in files:
        print(f"  • {file_info}")
    
    print("\n✅ TOTAL COVERAGE:")
    print("  • 50+ algorithms implemented")
    print("  • 100+ LeetCode problems referenced")
    print("  • All major company interview topics")
    print("  • Beginner to expert level content")
    print("  • Complete with time/space complexity analysis")

if __name__ == "__main__":
    display_summary() 