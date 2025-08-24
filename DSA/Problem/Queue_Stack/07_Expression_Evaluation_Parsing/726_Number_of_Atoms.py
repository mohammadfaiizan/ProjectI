"""
726. Number of Atoms - Multiple Approaches
Difficulty: Hard

Given a string formula representing a chemical formula, return the count of each atom.

The atomic element always starts with an uppercase character, then zero or more lowercase letters, representing the name.

One or more digits representing that element's count may follow if the count is greater than 1. If the count is 1, no digits will follow.

For example, "H2O" and "H2O2" are possible, but "H1O2" is not.

Two formulas are concatenated together to produce another formula.

For example, "H2O2He3Mg4" is also a formula.

A formula placed in parentheses, and a count (optionally added) is also a formula.

For example, "Mg(OH)2" and "Mg(OH2)2" are formulas.

Return the count of all elements as a string in the following form: the first name (in sorted order), followed by its count (if that count is more than 1), followed by the second name (in sorted order), followed by its count (if that count is more than 1), and so on.

The test cases are generated so that all the parentheses are properly matched.
"""

from typing import Dict, List
from collections import defaultdict, Counter

class NumberOfAtoms:
    """Multiple approaches to count atoms in chemical formula"""
    
    def countOfAtoms_stack_approach(self, formula: str) -> str:
        """
        Approach 1: Stack Approach (Optimal)
        
        Use stack to handle nested parentheses and multipliers.
        
        Time: O(n), Space: O(n)
        """
        stack = [defaultdict(int)]
        i = 0
        
        while i < len(formula):
            if formula[i] == '(':
                # Start new group
                stack.append(defaultdict(int))
                i += 1
            elif formula[i] == ')':
                # End current group and apply multiplier
                i += 1
                start = i
                
                # Parse multiplier
                while i < len(formula) and formula[i].isdigit():
                    i += 1
                
                multiplier = int(formula[start:i]) if i > start else 1
                
                # Pop current group and merge with previous
                current_group = stack.pop()
                for atom, count in current_group.items():
                    stack[-1][atom] += count * multiplier
            else:
                # Parse atom name
                start = i
                i += 1
                
                while i < len(formula) and formula[i].islower():
                    i += 1
                
                atom = formula[start:i]
                start = i
                
                # Parse count
                while i < len(formula) and formula[i].isdigit():
                    i += 1
                
                count = int(formula[start:i]) if i > start else 1
                stack[-1][atom] += count
        
        # Build result string
        atom_counts = stack[0]
        result = []
        
        for atom in sorted(atom_counts.keys()):
            count = atom_counts[atom]
            result.append(atom)
            if count > 1:
                result.append(str(count))
        
        return ''.join(result)
    
    def countOfAtoms_recursive(self, formula: str) -> str:
        """
        Approach 2: Recursive Approach
        
        Use recursion to handle nested structures.
        
        Time: O(n), Space: O(n)
        """
        def parse_formula(formula: str, index: int) -> tuple:
            """Parse formula and return (atom_counts, next_index)"""
            counts = defaultdict(int)
            
            while index < len(formula):
                if formula[index] == '(':
                    # Parse nested formula
                    nested_counts, index = parse_formula(formula, index + 1)
                    
                    # Parse multiplier after closing parenthesis
                    start = index
                    while index < len(formula) and formula[index].isdigit():
                        index += 1
                    
                    multiplier = int(formula[start:index]) if index > start else 1
                    
                    # Add nested counts with multiplier
                    for atom, count in nested_counts.items():
                        counts[atom] += count * multiplier
                
                elif formula[index] == ')':
                    # End of current group
                    return counts, index + 1
                
                else:
                    # Parse atom
                    atom, count, index = parse_atom(formula, index)
                    counts[atom] += count
            
            return counts, index
        
        def parse_atom(formula: str, index: int) -> tuple:
            """Parse single atom and return (atom, count, next_index)"""
            start = index
            index += 1
            
            # Parse atom name (uppercase + lowercase letters)
            while index < len(formula) and formula[index].islower():
                index += 1
            
            atom = formula[start:index]
            start = index
            
            # Parse count
            while index < len(formula) and formula[index].isdigit():
                index += 1
            
            count = int(formula[start:index]) if index > start else 1
            
            return atom, count, index
        
        atom_counts, _ = parse_formula(formula, 0)
        
        # Build result string
        result = []
        for atom in sorted(atom_counts.keys()):
            count = atom_counts[atom]
            result.append(atom)
            if count > 1:
                result.append(str(count))
        
        return ''.join(result)
    
    def countOfAtoms_regex_approach(self, formula: str) -> str:
        """
        Approach 3: Regex-based Parsing
        
        Use regular expressions to parse formula components.
        
        Time: O(n), Space: O(n)
        """
        import re
        
        def evaluate_formula(formula: str) -> Dict[str, int]:
            """Evaluate formula and return atom counts"""
            # Handle parentheses recursively
            while '(' in formula:
                # Find innermost parentheses
                match = re.search(r'\(([^()]*)\)(\d*)', formula)
                if not match:
                    break
                
                inner_formula = match.group(1)
                multiplier = int(match.group(2)) if match.group(2) else 1
                
                # Parse inner formula
                inner_counts = parse_simple_formula(inner_formula)
                
                # Apply multiplier
                expanded = ''
                for atom, count in inner_counts.items():
                    expanded += atom
                    total_count = count * multiplier
                    if total_count > 1:
                        expanded += str(total_count)
                
                # Replace in original formula
                formula = formula[:match.start()] + expanded + formula[match.end():]
            
            return parse_simple_formula(formula)
        
        def parse_simple_formula(formula: str) -> Dict[str, int]:
            """Parse formula without parentheses"""
            counts = defaultdict(int)
            
            # Find all atom-count pairs
            pattern = r'([A-Z][a-z]*)(\d*)'
            matches = re.findall(pattern, formula)
            
            for atom, count_str in matches:
                count = int(count_str) if count_str else 1
                counts[atom] += count
            
            return counts
        
        atom_counts = evaluate_formula(formula)
        
        # Build result string
        result = []
        for atom in sorted(atom_counts.keys()):
            count = atom_counts[atom]
            result.append(atom)
            if count > 1:
                result.append(str(count))
        
        return ''.join(result)
    
    def countOfAtoms_iterative_parsing(self, formula: str) -> str:
        """
        Approach 4: Iterative Parsing with State Machine
        
        Use state machine for parsing different components.
        
        Time: O(n), Space: O(n)
        """
        stack = [defaultdict(int)]
        i = 0
        
        while i < len(formula):
            char = formula[i]
            
            if char == '(':
                # Push new scope
                stack.append(defaultdict(int))
                i += 1
            
            elif char == ')':
                # Pop scope and apply multiplier
                i += 1
                
                # Parse multiplier
                num_start = i
                while i < len(formula) and formula[i].isdigit():
                    i += 1
                
                multiplier = int(formula[num_start:i]) if i > num_start else 1
                
                # Merge with parent scope
                current = stack.pop()
                for atom, count in current.items():
                    stack[-1][atom] += count * multiplier
            
            elif char.isupper():
                # Parse atom name
                atom_start = i
                i += 1
                
                while i < len(formula) and formula[i].islower():
                    i += 1
                
                atom = formula[atom_start:i]
                
                # Parse count
                count_start = i
                while i < len(formula) and formula[i].isdigit():
                    i += 1
                
                count = int(formula[count_start:i]) if i > count_start else 1
                stack[-1][atom] += count
            
            else:
                i += 1
        
        # Format result
        result = []
        for atom in sorted(stack[0].keys()):
            count = stack[0][atom]
            result.append(atom)
            if count > 1:
                result.append(str(count))
        
        return ''.join(result)


def test_number_of_atoms():
    """Test number of atoms algorithms"""
    solver = NumberOfAtoms()
    
    test_cases = [
        ("H2O", "H2O", "Example 1"),
        ("Mg(OH)2", "H2MgO2", "Example 2"),
        ("K4(ON(SO3)2)2", "K4N2O14S4", "Example 3"),
        ("Ca(OH)2", "CaH2O2", "Calcium hydroxide"),
        ("CaCl2", "CaCl2", "Simple compound"),
        ("H2SO4", "H2O4S", "Sulfuric acid"),
        ("Mg(NO3)2", "MgN2O6", "Magnesium nitrate"),
        ("Al2(SO4)3", "Al2O12S3", "Aluminum sulfate"),
        ("(NH4)2SO4", "H8N2O4S", "Ammonium sulfate"),
        ("Ca(C2H3O2)2", "C4CaH6O4", "Calcium acetate"),
    ]
    
    algorithms = [
        ("Stack Approach", solver.countOfAtoms_stack_approach),
        ("Recursive", solver.countOfAtoms_recursive),
        ("Regex Approach", solver.countOfAtoms_regex_approach),
        ("Iterative Parsing", solver.countOfAtoms_iterative_parsing),
    ]
    
    print("=== Testing Number of Atoms ===")
    
    for formula, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Formula: '{formula}'")
        print(f"Expected: '{expected}'")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(formula)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: '{result}'")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_stack_approach():
    """Demonstrate stack approach step by step"""
    print("\n=== Stack Approach Step-by-Step Demo ===")
    
    formula = "Mg(OH)2"
    print(f"Formula: '{formula}'")
    print("Strategy: Use stack to handle nested parentheses and multipliers")
    
    stack = [defaultdict(int)]
    i = 0
    
    print(f"\nStep-by-step processing:")
    
    while i < len(formula):
        char = formula[i]
        print(f"\nStep: Processing '{char}' at position {i}")
        print(f"  Stack before: {[dict(s) for s in stack]}")
        
        if char == '(':
            stack.append(defaultdict(int))
            print(f"  Started new group")
            i += 1
        elif char == ')':
            i += 1
            start = i
            
            while i < len(formula) and formula[i].isdigit():
                i += 1
            
            multiplier = int(formula[start:i]) if i > start else 1
            print(f"  Found multiplier: {multiplier}")
            
            current_group = stack.pop()
            print(f"  Popped group: {dict(current_group)}")
            
            for atom, count in current_group.items():
                stack[-1][atom] += count * multiplier
                print(f"    Added {atom}: {count} * {multiplier} = {count * multiplier}")
        
        else:
            # Parse atom
            start = i
            i += 1
            
            while i < len(formula) and formula[i].islower():
                i += 1
            
            atom = formula[start:i]
            start = i
            
            while i < len(formula) and formula[i].isdigit():
                i += 1
            
            count = int(formula[start:i]) if i > start else 1
            stack[-1][atom] += count
            
            print(f"  Added atom '{atom}' with count {count}")
        
        print(f"  Stack after: {[dict(s) for s in stack]}")
    
    # Build result
    atom_counts = stack[0]
    result = []
    
    for atom in sorted(atom_counts.keys()):
        count = atom_counts[atom]
        result.append(atom)
        if count > 1:
            result.append(str(count))
    
    final_result = ''.join(result)
    print(f"\nFinal atom counts: {dict(atom_counts)}")
    print(f"Formatted result: '{final_result}'")


def visualize_parsing_process():
    """Visualize parsing process for complex formula"""
    print("\n=== Parsing Process Visualization ===")
    
    formula = "Ca(C2H3O2)2"
    print(f"Formula: {formula}")
    print("Breaking down the parsing:")
    
    print(f"\n1. Structure Analysis:")
    print(f"   Ca - Calcium atom")
    print(f"   (C2H3O2) - Acetate group")
    print(f"   2 - Multiplier for acetate group")
    
    print(f"\n2. Parsing Steps:")
    print(f"   Step 1: Parse 'Ca' -> Ca: 1")
    print(f"   Step 2: Enter group '('")
    print(f"   Step 3: Parse 'C2' -> C: 2")
    print(f"   Step 4: Parse 'H3' -> H: 3")
    print(f"   Step 5: Parse 'O2' -> O: 2")
    print(f"   Step 6: Exit group ')' with multiplier 2")
    print(f"   Step 7: Apply multiplier: C: 2*2=4, H: 3*2=6, O: 2*2=4")
    
    print(f"\n3. Final Counts:")
    print(f"   Ca: 1, C: 4, H: 6, O: 4")
    print(f"   Sorted result: C4CaH6O4")
    
    # Verify with actual implementation
    solver = NumberOfAtoms()
    result = solver.countOfAtoms_stack_approach(formula)
    print(f"\nActual result: {result}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    solver = NumberOfAtoms()
    
    # Application 1: Chemical compound analysis
    print("1. Chemical Compound Analysis:")
    compounds = [
        ("H2SO4", "Sulfuric acid"),
        ("CaCO3", "Calcium carbonate (limestone)"),
        ("NaCl", "Sodium chloride (table salt)"),
        ("C6H12O6", "Glucose"),
        ("Ca(OH)2", "Calcium hydroxide (lime)"),
        ("Al2(SO4)3", "Aluminum sulfate"),
    ]
    
    for formula, name in compounds:
        result = solver.countOfAtoms_stack_approach(formula)
        print(f"  {name}: {formula} -> {result}")
    
    # Application 2: Molecular weight calculation preparation
    print(f"\n2. Molecular Weight Calculation Preparation:")
    
    # Atomic weights (simplified)
    atomic_weights = {
        'H': 1, 'C': 12, 'N': 14, 'O': 16, 'Na': 23, 'Mg': 24,
        'Al': 27, 'S': 32, 'Cl': 35.5, 'K': 39, 'Ca': 40
    }
    
    formulas = ["H2O", "NaCl", "CaCO3", "H2SO4"]
    
    for formula in formulas:
        atom_string = solver.countOfAtoms_stack_approach(formula)
        print(f"  {formula} -> {atom_string}")
        
        # Parse the result to calculate molecular weight
        i = 0
        total_weight = 0
        
        while i < len(atom_string):
            # Parse atom name
            atom_start = i
            i += 1
            while i < len(atom_string) and atom_string[i].islower():
                i += 1
            atom = atom_string[atom_start:i]
            
            # Parse count
            count_start = i
            while i < len(atom_string) and atom_string[i].isdigit():
                i += 1
            count = int(atom_string[count_start:i]) if i > count_start else 1
            
            if atom in atomic_weights:
                total_weight += atomic_weights[atom] * count
        
        print(f"    Molecular weight: {total_weight} g/mol")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Stack Approach", "O(n)", "O(n)", "Single pass with stack"),
        ("Recursive", "O(n)", "O(n)", "Recursion with parsing"),
        ("Regex Approach", "O(n²)", "O(n)", "Multiple regex operations"),
        ("Iterative Parsing", "O(n)", "O(n)", "State machine parsing"),
    ]
    
    print(f"{'Approach':<20} | {'Time':<8} | {'Space':<8} | {'Notes'}")
    print("-" * 60)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<20} | {time_comp:<8} | {space_comp:<8} | {notes}")
    
    print(f"\nStack and Iterative approaches are optimal")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = NumberOfAtoms()
    
    edge_cases = [
        ("H", "H", "Single atom"),
        ("H2", "H2", "Single atom with count"),
        ("HCl", "ClH", "Two different atoms"),
        ("((H))", "H", "Nested parentheses"),
        ("(H2O)1", "H2O", "Explicit count of 1"),
        ("Ca(OH)2", "CaH2O2", "Standard compound"),
        ("K4(ON(SO3)2)2", "K4N2O14S4", "Complex nesting"),
        ("Mg", "Mg", "Single element"),
        ("(H2)2", "H4", "Parentheses with multiplier"),
    ]
    
    for formula, expected, description in edge_cases:
        try:
            result = solver.countOfAtoms_stack_approach(formula)
            status = "✓" if result == expected else "✗"
            print(f"{description:25} | {status} | '{formula}' -> '{result}'")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")


if __name__ == "__main__":
    test_number_of_atoms()
    demonstrate_stack_approach()
    visualize_parsing_process()
    demonstrate_real_world_applications()
    analyze_time_complexity()
    test_edge_cases()

"""
Number of Atoms demonstrates advanced expression parsing for chemical
formulas, including stack-based parsing, recursive descent, and regex
approaches for handling nested parentheses and multipliers.
"""
