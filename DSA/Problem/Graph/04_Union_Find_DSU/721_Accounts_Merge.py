"""
721. Accounts Merge
Difficulty: Medium

Problem:
Given a list of accounts where each element accounts[i] is a list of strings, where the 
first element accounts[i][0] is a name, and the rest of the elements are emails representing 
emails of the account.

Now, we would like to merge these accounts. Two accounts definitely belong to the same person 
if there is some common email to both accounts. Note that even if two accounts have the same 
name, they may belong to different people as people could have the same name. A person can 
have any number of accounts initially, but all of their accounts definitely have the same name.

After merging the accounts, return the accounts in the following format: the first element 
of each account is the name, followed by a sorted list of emails. The accounts themselves 
can be returned in any order.

Examples:
Input: accounts = [["John","johnsmith@mail.com","john_newyork@mail.com"],["John","johnsmith@mail.com","john00@mail.com"],["Mary","mary@mail.com"],["John","johnnybravo@mail.com"]]
Output: [["John","john00@mail.com","john_newyork@mail.com","johnsmith@mail.com"],["Mary","mary@mail.com"],["John","johnnybravo@mail.com"]]

Input: accounts = [["Gabe","Gabe0@m.co","Gabe3@m.co","Gabe1@m.co"],["Kevin","Kevin3@m.co","Kevin5@m.co","Kevin0@m.co"],["Ethan","Ethan5@m.co","Ethan4@m.co","Ethan0@m.co"],["Hanzo","Hanzo3@m.co","Hanzo1@m.co","Hanzo0@m.co"],["Fern","Fern5@m.co","Fern1@m.co","Fern0@m.co"]]
Output: [["Ethan","Ethan0@m.co","Ethan4@m.co","Ethan5@m.co"],["Gabe","Gabe0@m.co","Gabe1@m.co","Gabe3@m.co"],["Hanzo","Hanzo0@m.co","Hanzo1@m.co","Hanzo3@m.co"],["Kevin","Kevin0@m.co","Kevin3@m.co","Kevin5@m.co"],["Fern","Fern0@m.co","Fern1@m.co","Fern5@m.co"]]

Constraints:
- 1 <= accounts.length <= 1000
- 2 <= accounts[i].length <= 10
- 1 <= accounts[i][j].length <= 30
- accounts[i][0] consists of English letters
- accounts[i][j] (for j > 0) is a valid email
"""

from typing import List
from collections import defaultdict

class UnionFind:
    """Union-Find for merging account groups"""
    
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        """Find with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """Union by rank"""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        return True

class Solution:
    def accountsMerge_approach1_union_find(self, accounts: List[List[str]]) -> List[List[str]]:
        """
        Approach 1: Union-Find (Optimal)
        
        Use Union-Find to group accounts that share emails.
        
        Time: O(N * M * α(N)) where N=accounts, M=max emails per account
        Space: O(N * M) for email mappings
        """
        email_to_account = {}  # Map email to first account index that has it
        email_to_name = {}     # Map email to account name
        
        # Build email mappings
        for i, account in enumerate(accounts):
            name = account[0]
            for email in account[1:]:
                email_to_name[email] = name
                if email in email_to_account:
                    # Email seen before - will need to merge accounts
                    continue
                else:
                    email_to_account[email] = i
        
        # Initialize Union-Find for accounts
        uf = UnionFind(len(accounts))
        
        # Union accounts that share emails
        for i, account in enumerate(accounts):
            for email in account[1:]:
                first_account = email_to_account[email]
                uf.union(i, first_account)
        
        # Group accounts by their root
        account_groups = defaultdict(list)
        for i in range(len(accounts)):
            root = uf.find(i)
            account_groups[root].extend(accounts[i][1:])  # Add emails only
        
        # Build result with sorted emails
        result = []
        for root, emails in account_groups.items():
            name = accounts[root][0]  # Get name from root account
            unique_emails = sorted(set(emails))  # Remove duplicates and sort
            result.append([name] + unique_emails)
        
        return result
    
    def accountsMerge_approach2_email_based_union_find(self, accounts: List[List[str]]) -> List[List[str]]:
        """
        Approach 2: Email-Based Union-Find
        
        Use emails as Union-Find elements instead of accounts.
        
        Time: O(Total_Emails * α(Total_Emails))
        Space: O(Total_Emails)
        """
        email_to_name = {}
        email_list = []
        email_to_id = {}
        
        # Collect all unique emails and map to names
        for account in accounts:
            name = account[0]
            for email in account[1:]:
                if email not in email_to_id:
                    email_to_id[email] = len(email_list)
                    email_list.append(email)
                email_to_name[email] = name
        
        # Initialize Union-Find for emails
        uf = UnionFind(len(email_list))
        
        # Union emails within each account
        for account in accounts:
            if len(account) > 1:  # Has emails
                first_email_id = email_to_id[account[1]]
                for email in account[2:]:
                    email_id = email_to_id[email]
                    uf.union(first_email_id, email_id)
        
        # Group emails by their root
        email_groups = defaultdict(list)
        for email, email_id in email_to_id.items():
            root = uf.find(email_id)
            email_groups[root].append(email)
        
        # Build result
        result = []
        for emails in email_groups.values():
            if emails:  # Should always be true
                name = email_to_name[emails[0]]
                sorted_emails = sorted(emails)
                result.append([name] + sorted_emails)
        
        return result
    
    def accountsMerge_approach3_dfs(self, accounts: List[List[str]]) -> List[List[str]]:
        """
        Approach 3: DFS Graph Traversal
        
        Build graph of email connections and use DFS to find components.
        
        Time: O(Total_Emails + Connections)
        Space: O(Total_Emails)
        """
        email_to_name = {}
        graph = defaultdict(set)
        
        # Build graph of email connections
        for account in accounts:
            name = account[0]
            emails = account[1:]
            
            # Map emails to name
            for email in emails:
                email_to_name[email] = name
            
            # Connect all emails in this account
            for i in range(len(emails)):
                for j in range(i + 1, len(emails)):
                    email1, email2 = emails[i], emails[j]
                    graph[email1].add(email2)
                    graph[email2].add(email1)
        
        visited = set()
        result = []
        
        def dfs(email, component):
            """DFS to collect all emails in connected component"""
            if email in visited:
                return
            
            visited.add(email)
            component.append(email)
            
            for neighbor in graph[email]:
                dfs(neighbor, component)
        
        # Find all connected components
        for email in email_to_name:
            if email not in visited:
                component = []
                dfs(email, component)
                
                if component:
                    name = email_to_name[component[0]]
                    sorted_emails = sorted(component)
                    result.append([name] + sorted_emails)
        
        return result
    
    def accountsMerge_approach4_optimized_union_find(self, accounts: List[List[str]]) -> List[List[str]]:
        """
        Approach 4: Optimized Union-Find with String Compression
        
        Optimize for better performance with string handling.
        
        Time: O(N * M * α(N))
        Space: O(N * M)
        """
        # Create mapping from email to account indices
        email_to_accounts = defaultdict(list)
        
        for i, account in enumerate(accounts):
            for email in account[1:]:
                email_to_accounts[email].append(i)
        
        # Initialize Union-Find
        uf = UnionFind(len(accounts))
        
        # Union all accounts that share any email
        for account_list in email_to_accounts.values():
            if len(account_list) > 1:
                for i in range(1, len(account_list)):
                    uf.union(account_list[0], account_list[i])
        
        # Group accounts by root and collect emails
        account_groups = defaultdict(set)
        for i, account in enumerate(accounts):
            root = uf.find(i)
            account_groups[root].update(account[1:])  # Use set for automatic deduplication
        
        # Build final result
        result = []
        for root, emails in account_groups.items():
            name = accounts[root][0]
            sorted_emails = sorted(emails)
            result.append([name] + sorted_emails)
        
        return result

def test_accounts_merge():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (accounts, expected_count)
        ([["John","johnsmith@mail.com","john_newyork@mail.com"],["John","johnsmith@mail.com","john00@mail.com"],["Mary","mary@mail.com"],["John","johnnybravo@mail.com"]], 3),
        ([["Gabe","Gabe0@m.co","Gabe3@m.co","Gabe1@m.co"],["Kevin","Kevin3@m.co","Kevin5@m.co","Kevin0@m.co"]], 2),
        ([["David","David0@m.co","David1@m.co"],["David","David3@m.co","David4@m.co"],["David","David4@m.co","David5@m.co"],["David","David2@m.co","David3@m.co"],["David","David1@m.co","David2@m.co"]], 1),
        ([["Alex","alex@email.com"]], 1),
    ]
    
    approaches = [
        ("Union-Find", solution.accountsMerge_approach1_union_find),
        ("Email-Based UF", solution.accountsMerge_approach2_email_based_union_find),
        ("DFS", solution.accountsMerge_approach3_dfs),
        ("Optimized UF", solution.accountsMerge_approach4_optimized_union_find),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (accounts, expected_count) in enumerate(test_cases):
            result = func(accounts[:])  # Copy to avoid modification
            result_count = len(result)
            status = "✓" if result_count == expected_count else "✗"
            print(f"Test {i+1}: {status} Accounts: {len(accounts)}, Expected groups: {expected_count}, Got: {result_count}")
            
            # Show first few results for verification
            if len(result) <= 3:
                for j, account_group in enumerate(result):
                    print(f"         Group {j+1}: {account_group}")

def demonstrate_union_find_process():
    """Demonstrate Union-Find process for account merging"""
    print("\n=== Union-Find Process Demo ===")
    
    accounts = [
        ["John","johnsmith@mail.com","john_newyork@mail.com"],
        ["John","johnsmith@mail.com","john00@mail.com"],
        ["Mary","mary@mail.com"],
        ["John","johnnybravo@mail.com"]
    ]
    
    print("Input accounts:")
    for i, account in enumerate(accounts):
        print(f"  Account {i}: {account}")
    
    # Build email to account mapping
    email_to_account = {}
    
    print(f"\nMapping emails to first occurrence:")
    for i, account in enumerate(accounts):
        for email in account[1:]:
            if email not in email_to_account:
                email_to_account[email] = i
                print(f"  {email} -> Account {i}")
            else:
                print(f"  {email} already seen in Account {email_to_account[email]} (will union with Account {i})")
    
    # Union-Find operations
    print(f"\nUnion-Find operations:")
    uf = UnionFind(len(accounts))
    
    for i, account in enumerate(accounts):
        print(f"\nProcessing Account {i}: {account[0]}")
        for email in account[1:]:
            first_account = email_to_account[email]
            if first_account != i:
                print(f"  Email {email}: Union({i}, {first_account})")
                uf.union(i, first_account)
            else:
                print(f"  Email {email}: First occurrence")
        
        # Show current components
        components = defaultdict(list)
        for j in range(len(accounts)):
            root = uf.find(j)
            components[root].append(j)
        
        print(f"  Current components: {dict(components)}")
    
    # Final grouping
    print(f"\nFinal account groups:")
    account_groups = defaultdict(list)
    for i in range(len(accounts)):
        root = uf.find(i)
        account_groups[root].extend(accounts[i][1:])
    
    for root, emails in account_groups.items():
        name = accounts[root][0]
        unique_emails = sorted(set(emails))
        print(f"  Group {root}: {name} - {unique_emails}")

def analyze_account_merging():
    """Analyze the account merging problem"""
    print("\n=== Account Merging Analysis ===")
    
    print("Problem Characteristics:")
    print("• Multiple accounts per person allowed")
    print("• Accounts belong to same person if they share emails")
    print("• Same name doesn't guarantee same person")
    print("• Transitive merging: A-B, B-C → A-B-C merged")
    
    print("\nKey Challenges:")
    print("1. **Email Deduplication:** Same email in multiple accounts")
    print("2. **Transitive Closure:** Chain of shared emails")
    print("3. **Name Preservation:** Keep original names")
    print("4. **Sorting Requirement:** Output emails must be sorted")
    
    print("\nUnion-Find Modeling:")
    print("• **Nodes:** Either accounts or emails")
    print("• **Edges:** Shared emails create connections")
    print("• **Components:** Groups of related accounts")
    print("• **Result:** One merged account per component")
    
    print("\nApproach Comparison:")
    print("• **Account-Based UF:** Union accounts that share emails")
    print("• **Email-Based UF:** Union emails within same account")
    print("• **DFS Graph:** Build email graph, find components")
    print("• **Optimized UF:** Reduce union operations")
    
    print("\nOptimization Strategies:")
    print("• Use sets for automatic email deduplication")
    print("• Path compression in Union-Find")
    print("• Union by rank for balanced trees")
    print("• Efficient email-to-account mapping")
    print("• Early termination when possible")

def demonstrate_edge_cases():
    """Demonstrate handling of edge cases"""
    print("\n=== Edge Cases Demo ===")
    
    edge_cases = [
        {
            "name": "Single Account",
            "accounts": [["John", "john@email.com"]],
            "description": "No merging needed"
        },
        {
            "name": "Same Name Different People", 
            "accounts": [["John", "john1@email.com"], ["John", "john2@email.com"]],
            "description": "Same name but no shared emails"
        },
        {
            "name": "Complex Chain",
            "accounts": [["A", "a@m.co", "b@m.co"], ["A", "b@m.co", "c@m.co"], ["A", "c@m.co", "d@m.co"]],
            "description": "Transitive merging through email chains"
        },
        {
            "name": "Duplicate Emails",
            "accounts": [["User", "email@m.co", "email@m.co", "other@m.co"]],
            "description": "Duplicate emails within same account"
        }
    ]
    
    solution = Solution()
    
    for case in edge_cases:
        print(f"\n{case['name']}:")
        print(f"  Description: {case['description']}")
        print(f"  Input: {case['accounts']}")
        
        result = solution.accountsMerge_approach1_union_find(case['accounts'])
        print(f"  Output: {result}")
        print(f"  Groups: {len(result)}")

def compare_implementation_strategies():
    """Compare different implementation strategies"""
    print("\n=== Implementation Strategies Comparison ===")
    
    print("1. **Account-Based Union-Find:**")
    print("   ✅ Natural problem modeling")
    print("   ✅ Fewer Union-Find nodes")
    print("   ✅ Direct account grouping")
    print("   ❌ Complex email tracking")
    
    print("\n2. **Email-Based Union-Find:**")
    print("   ✅ Direct email relationship modeling")
    print("   ✅ Clear transitivity handling")
    print("   ✅ Automatic email grouping")
    print("   ❌ More Union-Find nodes")
    print("   ❌ Name tracking complexity")
    
    print("\n3. **DFS Graph Approach:**")
    print("   ✅ Intuitive graph construction")
    print("   ✅ Standard connected components")
    print("   ✅ Easy to understand")
    print("   ❌ Extra space for adjacency lists")
    print("   ❌ Not as efficient as Union-Find")
    
    print("\nPerformance Analysis:")
    print("• **Time Complexity:** All approaches O(N*M*α(N)) or O(N*M)")
    print("• **Space Complexity:** O(N*M) for email storage")
    print("• **Practical Performance:** Union-Find generally fastest")
    print("• **Implementation Complexity:** DFS simplest, UF most optimal")
    
    print("\nReal-world Applications:")
    print("• **Identity Resolution:** Merging user profiles")
    print("• **Data Deduplication:** Combining duplicate records")
    print("• **Social Networks:** Finding connected users")
    print("• **Email Systems:** Organizing related contacts")
    print("• **Database Management:** Record linkage")

if __name__ == "__main__":
    test_accounts_merge()
    demonstrate_union_find_process()
    analyze_account_merging()
    demonstrate_edge_cases()
    compare_implementation_strategies()

"""
Union-Find Concepts:
1. Transitive Relationship Merging
2. Identity Resolution Problems
3. Connected Components in Bipartite Graphs
4. Data Deduplication and Aggregation

Key Problem Insights:
- Accounts belong to same person if they share emails
- Transitive merging: A↔B, B↔C → A↔B↔C
- Union-Find naturally handles transitivity
- Email sharing creates equivalence relations

Algorithm Strategy:
1. Map emails to account indices
2. Union accounts that share any email
3. Group emails by account components
4. Sort emails within each group

Union-Find Modeling Options:
- Account-based: Union accounts directly
- Email-based: Union emails within accounts
- Hybrid: Combine both approaches
- Graph-based: DFS on email connections

Real-world Applications:
- User identity resolution
- Customer data deduplication
- Social network analysis
- Contact management systems
- Database record linkage

This problem demonstrates Union-Find for
identity resolution and data merging tasks.
"""
