"""
Python Fundamentals: Practical Projects and Real-World Applications
Implementation-focused with minimal comments, maximum functionality coverage
"""

import json
import random
import time
import os
import tempfile
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional, Tuple
import re
import math

# Project 1: File-based Task Manager
class TaskManager:
    def __init__(self, filename="tasks.json"):
        self.filename = filename
        self.tasks = self.load_tasks()
    
    def load_tasks(self):
        try:
            with open(self.filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def save_tasks(self):
        with open(self.filename, 'w') as f:
            json.dump(self.tasks, f, indent=2)
    
    def add_task(self, title, description="", priority="medium"):
        task = {
            "id": len(self.tasks) + 1,
            "title": title,
            "description": description,
            "priority": priority,
            "completed": False,
            "created_at": time.time()
        }
        self.tasks.append(task)
        self.save_tasks()
        return task["id"]
    
    def complete_task(self, task_id):
        for task in self.tasks:
            if task["id"] == task_id:
                task["completed"] = True
                task["completed_at"] = time.time()
                self.save_tasks()
                return True
        return False
    
    def list_tasks(self, filter_completed=None):
        if filter_completed is None:
            return self.tasks
        return [task for task in self.tasks if task["completed"] == filter_completed]
    
    def get_statistics(self):
        total = len(self.tasks)
        completed = len([t for t in self.tasks if t["completed"]])
        by_priority = Counter(t["priority"] for t in self.tasks)
        
        return {
            "total_tasks": total,
            "completed_tasks": completed,
            "pending_tasks": total - completed,
            "completion_rate": completed / total * 100 if total > 0 else 0,
            "by_priority": dict(by_priority)
        }

def task_manager_demo():
    # Create temporary task manager
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    temp_file.close()
    
    try:
        tm = TaskManager(temp_file.name)
        
        # Add tasks
        task_ids = [
            tm.add_task("Learn Python", "Complete fundamentals course", "high"),
            tm.add_task("Build project", "Create task manager application", "high"),
            tm.add_task("Code review", "Review team's code", "medium"),
            tm.add_task("Documentation", "Write project documentation", "low"),
            tm.add_task("Testing", "Write unit tests", "medium")
        ]
        
        # Complete some tasks
        tm.complete_task(task_ids[0])
        tm.complete_task(task_ids[2])
        
        # Get results
        all_tasks = tm.list_tasks()
        pending_tasks = tm.list_tasks(filter_completed=False)
        stats = tm.get_statistics()
        
        return {
            "total_tasks": len(all_tasks),
            "pending_tasks": len(pending_tasks),
            "statistics": stats,
            "sample_task": all_tasks[0] if all_tasks else None
        }
    
    finally:
        os.unlink(temp_file.name)

# Project 2: Log File Analyzer
class LogAnalyzer:
    def __init__(self):
        self.log_pattern = re.compile(
            r'(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) \[(\w+)\] (.+)'
        )
    
    def parse_log_line(self, line):
        match = self.log_pattern.match(line.strip())
        if match:
            return {
                'date': match.group(1),
                'time': match.group(2),
                'level': match.group(3),
                'message': match.group(4)
            }
        return None
    
    def analyze_logs(self, log_content):
        lines = log_content.strip().split('\n')
        parsed_logs = [self.parse_log_line(line) for line in lines]
        valid_logs = [log for log in parsed_logs if log is not None]
        
        if not valid_logs:
            return {"error": "No valid log entries found"}
        
        # Analysis
        level_counts = Counter(log['level'] for log in valid_logs)
        dates = Counter(log['date'] for log in valid_logs)
        
        # Error analysis
        error_logs = [log for log in valid_logs if log['level'] == 'ERROR']
        error_patterns = Counter(log['message'][:50] for log in error_logs)
        
        # Time analysis
        hours = [int(log['time'][:2]) for log in valid_logs]
        peak_hour = Counter(hours).most_common(1)[0] if hours else (0, 0)
        
        return {
            "total_entries": len(valid_logs),
            "level_distribution": dict(level_counts),
            "dates_covered": dict(dates),
            "error_count": len(error_logs),
            "common_errors": dict(error_patterns.most_common(3)),
            "peak_hour": f"{peak_hour[0]}:00 ({peak_hour[1]} entries)"
        }

def log_analyzer_demo():
    # Generate sample log content
    sample_logs = """
2023-12-25 10:30:45 [INFO] Application started
2023-12-25 10:30:46 [INFO] Database connection established
2023-12-25 10:31:15 [WARNING] High memory usage detected
2023-12-25 10:32:00 [ERROR] Failed to connect to external API
2023-12-25 10:32:30 [INFO] User authentication successful
2023-12-25 10:33:00 [ERROR] Database timeout occurred
2023-12-25 10:33:15 [INFO] Retry mechanism activated
2023-12-25 10:34:00 [INFO] External API connection restored
2023-12-25 10:35:00 [WARNING] Disk space running low
2023-12-25 10:36:00 [ERROR] Failed to connect to external API
2023-12-25 11:00:00 [INFO] Scheduled backup completed
2023-12-25 11:15:00 [DEBUG] Cache cleared successfully
"""
    
    analyzer = LogAnalyzer()
    results = analyzer.analyze_logs(sample_logs)
    
    return results

# Project 3: Simple Banking System
class BankAccount:
    def __init__(self, account_number, initial_balance=0):
        self.account_number = account_number
        self.balance = initial_balance
        self.transaction_history = []
        self._add_transaction("OPEN", initial_balance, "Account opened")
    
    def _add_transaction(self, transaction_type, amount, description):
        transaction = {
            "timestamp": time.time(),
            "type": transaction_type,
            "amount": amount,
            "balance_after": self.balance,
            "description": description
        }
        self.transaction_history.append(transaction)
    
    def deposit(self, amount, description="Deposit"):
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        
        self.balance += amount
        self._add_transaction("DEPOSIT", amount, description)
        return self.balance
    
    def withdraw(self, amount, description="Withdrawal"):
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        
        self.balance -= amount
        self._add_transaction("WITHDRAWAL", -amount, description)
        return self.balance
    
    def get_statement(self, num_transactions=10):
        recent_transactions = self.transaction_history[-num_transactions:]
        return {
            "account_number": self.account_number,
            "current_balance": self.balance,
            "recent_transactions": recent_transactions
        }

class SimpleBankingSystem:
    def __init__(self):
        self.accounts = {}
        self.next_account_number = 1000
    
    def create_account(self, initial_balance=0):
        account_number = str(self.next_account_number)
        self.next_account_number += 1
        
        account = BankAccount(account_number, initial_balance)
        self.accounts[account_number] = account
        return account_number
    
    def get_account(self, account_number):
        return self.accounts.get(account_number)
    
    def transfer(self, from_account, to_account, amount, description="Transfer"):
        from_acc = self.get_account(from_account)
        to_acc = self.get_account(to_account)
        
        if not from_acc or not to_acc:
            raise ValueError("Invalid account number")
        
        # Withdraw from source
        from_acc.withdraw(amount, f"Transfer to {to_account}: {description}")
        # Deposit to destination
        to_acc.deposit(amount, f"Transfer from {from_account}: {description}")
        
        return {
            "from_balance": from_acc.balance,
            "to_balance": to_acc.balance
        }
    
    def get_bank_summary(self):
        total_accounts = len(self.accounts)
        total_balance = sum(acc.balance for acc in self.accounts.values())
        active_accounts = len([acc for acc in self.accounts.values() if acc.balance > 0])
        
        return {
            "total_accounts": total_accounts,
            "total_balance": total_balance,
            "active_accounts": active_accounts,
            "average_balance": total_balance / total_accounts if total_accounts > 0 else 0
        }

def banking_system_demo():
    bank = SimpleBankingSystem()
    
    # Create accounts
    acc1 = bank.create_account(1000)
    acc2 = bank.create_account(500)
    acc3 = bank.create_account(0)
    
    # Perform transactions
    account1 = bank.get_account(acc1)
    account2 = bank.get_account(acc2)
    account3 = bank.get_account(acc3)
    
    account1.deposit(200, "Salary deposit")
    account1.withdraw(150, "ATM withdrawal")
    account2.deposit(100, "Cash deposit")
    
    # Transfer money
    transfer_result = bank.transfer(acc1, acc3, 300, "Payment")
    
    # Get statements and summary
    statement1 = account1.get_statement()
    statement3 = account3.get_statement()
    bank_summary = bank.get_bank_summary()
    
    return {
        "accounts_created": [acc1, acc2, acc3],
        "transfer_result": transfer_result,
        "account1_balance": statement1["current_balance"],
        "account3_balance": statement3["current_balance"],
        "bank_summary": bank_summary
    }

# Project 4: Text Analysis Tool
class TextAnalyzer:
    def __init__(self):
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their'
        }
    
    def clean_text(self, text):
        # Remove punctuation and convert to lowercase
        cleaned = re.sub(r'[^\w\s]', '', text.lower())
        return cleaned
    
    def extract_words(self, text):
        cleaned = self.clean_text(text)
        words = cleaned.split()
        return [word for word in words if word and word not in self.stop_words]
    
    def analyze_text(self, text):
        if not text.strip():
            return {"error": "Empty text provided"}
        
        # Basic statistics
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = len(re.findall(r'[.!?]+', text))
        paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
        
        # Word analysis
        words = self.extract_words(text)
        word_frequencies = Counter(words)
        unique_words = len(set(words))
        
        # Reading metrics
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Reading time estimation (average 200 words per minute)
        reading_time_minutes = word_count / 200
        
        # Complexity analysis
        long_words = len([word for word in words if len(word) > 6])
        complexity_score = (long_words / len(words) * 100) if words else 0
        
        return {
            "basic_stats": {
                "characters": char_count,
                "words": word_count,
                "sentences": sentence_count,
                "paragraphs": paragraph_count,
                "unique_words": unique_words
            },
            "word_analysis": {
                "most_common": dict(word_frequencies.most_common(10)),
                "avg_word_length": round(avg_word_length, 2),
                "long_words_count": long_words,
                "vocabulary_richness": round(unique_words / word_count * 100, 2) if word_count > 0 else 0
            },
            "readability": {
                "avg_sentence_length": round(avg_sentence_length, 2),
                "complexity_score": round(complexity_score, 2),
                "reading_time_minutes": round(reading_time_minutes, 2)
            }
        }
    
    def compare_texts(self, text1, text2):
        analysis1 = self.analyze_text(text1)
        analysis2 = self.analyze_text(text2)
        
        if "error" in analysis1 or "error" in analysis2:
            return {"error": "Cannot compare - one or both texts are invalid"}
        
        # Extract common words
        words1 = set(self.extract_words(text1))
        words2 = set(self.extract_words(text2))
        
        common_words = words1 & words2
        unique_to_text1 = words1 - words2
        unique_to_text2 = words2 - words1
        
        similarity = len(common_words) / len(words1 | words2) * 100 if words1 | words2 else 0
        
        return {
            "text1_stats": analysis1["basic_stats"],
            "text2_stats": analysis2["basic_stats"],
            "comparison": {
                "common_words": len(common_words),
                "unique_to_text1": len(unique_to_text1),
                "unique_to_text2": len(unique_to_text2),
                "similarity_percentage": round(similarity, 2)
            }
        }

def text_analyzer_demo():
    analyzer = TextAnalyzer()
    
    sample_text = """
    Python is a high-level, interpreted programming language with dynamic semantics.
    Its high-level built-in data structures, combined with dynamic typing and dynamic binding,
    make it very attractive for Rapid Application Development, as well as for use as a
    scripting or glue language to connect existing components together.
    
    Python's simple, easy to learn syntax emphasizes readability and therefore reduces
    the cost of program maintenance. Python supports modules and packages, which encourages
    program modularity and code reuse. The Python interpreter and the extensive standard
    library are available in source or binary form without charge for all major platforms,
    and can be freely distributed.
    """
    
    # Analyze the text
    analysis = analyzer.analyze_text(sample_text)
    
    # Compare with a shorter text
    short_text = "Python is easy to learn and powerful to use."
    comparison = analyzer.compare_texts(sample_text, short_text)
    
    return {
        "text_analysis": analysis,
        "text_comparison": comparison
    }

# Project 5: Inventory Management System
class Product:
    def __init__(self, product_id, name, price, quantity=0):
        self.product_id = product_id
        self.name = name
        self.price = price
        self.quantity = quantity
        self.created_at = time.time()
    
    def to_dict(self):
        return {
            "product_id": self.product_id,
            "name": self.name,
            "price": self.price,
            "quantity": self.quantity,
            "created_at": self.created_at
        }

class InventoryManager:
    def __init__(self):
        self.products = {}
        self.transaction_log = []
    
    def add_product(self, name, price, quantity=0):
        product_id = f"PROD{len(self.products) + 1:04d}"
        product = Product(product_id, name, price, quantity)
        self.products[product_id] = product
        
        self._log_transaction("ADD_PRODUCT", product_id, quantity, f"Added {name}")
        return product_id
    
    def update_stock(self, product_id, quantity_change, reason="Stock update"):
        if product_id not in self.products:
            raise ValueError(f"Product {product_id} not found")
        
        product = self.products[product_id]
        old_quantity = product.quantity
        product.quantity += quantity_change
        
        if product.quantity < 0:
            product.quantity = old_quantity
            raise ValueError("Insufficient stock")
        
        transaction_type = "STOCK_IN" if quantity_change > 0 else "STOCK_OUT"
        self._log_transaction(transaction_type, product_id, quantity_change, reason)
        
        return product.quantity
    
    def _log_transaction(self, transaction_type, product_id, quantity, description):
        transaction = {
            "timestamp": time.time(),
            "type": transaction_type,
            "product_id": product_id,
            "quantity": quantity,
            "description": description
        }
        self.transaction_log.append(transaction)
    
    def get_product(self, product_id):
        return self.products.get(product_id)
    
    def search_products(self, query):
        query_lower = query.lower()
        return [
            product for product in self.products.values()
            if query_lower in product.name.lower()
        ]
    
    def get_low_stock_products(self, threshold=5):
        return [
            product for product in self.products.values()
            if product.quantity <= threshold
        ]
    
    def get_inventory_value(self):
        return sum(product.price * product.quantity for product in self.products.values())
    
    def generate_report(self):
        total_products = len(self.products)
        total_value = self.get_inventory_value()
        low_stock = self.get_low_stock_products()
        
        # Most valuable products
        valuable_products = sorted(
            self.products.values(),
            key=lambda p: p.price * p.quantity,
            reverse=True
        )[:5]
        
        # Recent transactions
        recent_transactions = self.transaction_log[-10:]
        
        return {
            "summary": {
                "total_products": total_products,
                "total_inventory_value": round(total_value, 2),
                "low_stock_items": len(low_stock),
                "total_transactions": len(self.transaction_log)
            },
            "low_stock_products": [p.to_dict() for p in low_stock],
            "most_valuable": [p.to_dict() for p in valuable_products],
            "recent_transactions": recent_transactions
        }

def inventory_management_demo():
    inventory = InventoryManager()
    
    # Add products
    products = [
        ("Laptop Computer", 999.99, 10),
        ("Wireless Mouse", 29.99, 50),
        ("USB Cable", 12.99, 100),
        ("Monitor", 299.99, 8),
        ("Keyboard", 79.99, 25),
        ("Webcam", 89.99, 3)  # Low stock item
    ]
    
    product_ids = []
    for name, price, quantity in products:
        product_id = inventory.add_product(name, price, quantity)
        product_ids.append(product_id)
    
    # Simulate some transactions
    inventory.update_stock(product_ids[0], -2, "Sale")
    inventory.update_stock(product_ids[1], -10, "Bulk sale")
    inventory.update_stock(product_ids[2], 50, "Restock")
    inventory.update_stock(product_ids[5], -1, "Sale")  # Make webcam even lower stock
    
    # Generate report
    report = inventory.generate_report()
    
    # Test search functionality
    search_results = inventory.search_products("USB")
    
    return {
        "products_added": len(product_ids),
        "inventory_report": report,
        "search_results": [p.to_dict() for p in search_results],
        "total_inventory_value": inventory.get_inventory_value()
    }

# Project 6: Data Processing Pipeline
class DataProcessor:
    def __init__(self):
        self.processors = []
    
    def add_processor(self, func, name=None):
        processor_name = name or func.__name__
        self.processors.append((processor_name, func))
    
    def process(self, data):
        result = data
        processing_log = []
        
        for name, processor in self.processors:
            try:
                start_time = time.time()
                result = processor(result)
                processing_time = time.time() - start_time
                
                processing_log.append({
                    "processor": name,
                    "processing_time_ms": round(processing_time * 1000, 2),
                    "status": "success",
                    "output_type": type(result).__name__
                })
            except Exception as e:
                processing_log.append({
                    "processor": name,
                    "status": "error",
                    "error": str(e)
                })
                break
        
        return result, processing_log

def data_processing_demo():
    # Create sample data
    raw_data = [
        {"name": "Alice", "age": "30", "salary": "75000", "department": "Engineering"},
        {"name": "Bob", "age": "25", "salary": "65000", "department": "Marketing"},
        {"name": "Charlie", "age": "35", "salary": "85000", "department": "Engineering"},
        {"name": "Diana", "age": "28", "salary": "70000", "department": "Sales"},
        {"name": "Eve", "age": "32", "salary": "80000", "department": "Engineering"}
    ]
    
    # Define processing functions
    def convert_types(data):
        """Convert string numbers to integers"""
        for record in data:
            record["age"] = int(record["age"])
            record["salary"] = int(record["salary"])
        return data
    
    def add_computed_fields(data):
        """Add computed fields"""
        for record in data:
            record["age_group"] = "young" if record["age"] < 30 else "experienced"
            record["salary_band"] = "high" if record["salary"] > 70000 else "standard"
        return data
    
    def filter_engineering(data):
        """Filter only engineering employees"""
        return [record for record in data if record["department"] == "Engineering"]
    
    def sort_by_salary(data):
        """Sort by salary descending"""
        return sorted(data, key=lambda x: x["salary"], reverse=True)
    
    def calculate_summary(data):
        """Calculate summary statistics"""
        if not data:
            return {"error": "No data to summarize"}
        
        total_employees = len(data)
        avg_age = sum(record["age"] for record in data) / total_employees
        avg_salary = sum(record["salary"] for record in data) / total_employees
        
        return {
            "summary": {
                "total_employees": total_employees,
                "average_age": round(avg_age, 1),
                "average_salary": round(avg_salary, 2),
                "salary_range": {
                    "min": min(record["salary"] for record in data),
                    "max": max(record["salary"] for record in data)
                }
            },
            "employees": data
        }
    
    # Create and configure processor
    processor = DataProcessor()
    processor.add_processor(convert_types, "Type Conversion")
    processor.add_processor(add_computed_fields, "Add Computed Fields")
    processor.add_processor(filter_engineering, "Filter Engineering")
    processor.add_processor(sort_by_salary, "Sort by Salary")
    processor.add_processor(calculate_summary, "Calculate Summary")
    
    # Process the data
    result, processing_log = processor.process(raw_data)
    
    return {
        "original_data_count": len(raw_data),
        "processed_result": result,
        "processing_log": processing_log
    }

# Comprehensive project demonstration
def run_all_projects():
    """Execute all practical projects"""
    projects = [
        ("task_manager", task_manager_demo),
        ("log_analyzer", log_analyzer_demo),
        ("banking_system", banking_system_demo),
        ("text_analyzer", text_analyzer_demo),
        ("inventory_management", inventory_management_demo),
        ("data_processing", data_processing_demo)
    ]
    
    results = {}
    for name, project_func in projects:
        try:
            start_time = time.time()
            result = project_func()
            execution_time = time.time() - start_time
            results[name] = {
                'result': result,
                'execution_time': f"{execution_time*1000:.2f}ms"
            }
        except Exception as e:
            results[name] = {'error': str(e)}
    
    return results

# Project integration example
def integrated_business_system():
    """Demonstrate integration of multiple project components"""
    
    # Initialize systems
    task_manager = TaskManager("business_tasks.json")
    inventory = InventoryManager()
    bank = SimpleBankingSystem()
    
    try:
        # Business scenario: Setting up a new product launch
        
        # 1. Create tasks for product launch
        task_ids = [
            task_manager.add_task("Product Development", "Develop new product line", "high"),
            task_manager.add_task("Inventory Setup", "Set up initial inventory", "high"),
            task_manager.add_task("Marketing Campaign", "Launch marketing campaign", "medium"),
            task_manager.add_task("Sales Training", "Train sales team", "medium")
        ]
        
        # 2. Set up inventory for new products
        product_ids = [
            inventory.add_product("Premium Widget", 199.99, 100),
            inventory.add_product("Standard Widget", 99.99, 200),
            inventory.add_product("Economy Widget", 49.99, 300)
        ]
        
        # 3. Set up business bank account
        business_account = bank.create_account(50000)  # Initial capital
        
        # 4. Simulate business operations
        
        # Complete product development task
        task_manager.complete_task(task_ids[0])
        
        # Make initial sales (reduce inventory)
        inventory.update_stock(product_ids[0], -10, "Initial sales")
        inventory.update_stock(product_ids[1], -25, "Initial sales")
        inventory.update_stock(product_ids[2], -50, "Initial sales")
        
        # Record revenue in bank account
        revenue = (10 * 199.99) + (25 * 99.99) + (50 * 49.99)
        bank_account = bank.get_account(business_account)
        bank_account.deposit(revenue, "Product sales revenue")
        
        # Complete inventory setup task
        task_manager.complete_task(task_ids[1])
        
        # Generate integrated report
        task_stats = task_manager.get_statistics()
        inventory_report = inventory.generate_report()
        bank_summary = bank.get_bank_summary()
        
        return {
            "business_scenario": "Product launch integration",
            "task_progress": task_stats,
            "inventory_status": inventory_report["summary"],
            "financial_status": {
                "account_balance": bank_account.balance,
                "revenue_generated": revenue
            },
            "integration_success": True
        }
    
    except Exception as e:
        return {"integration_error": str(e)}
    
    finally:
        # Cleanup
        try:
            os.unlink("business_tasks.json")
        except FileNotFoundError:
            pass

if __name__ == "__main__":
    print("=== Python Fundamentals: Practical Projects Demo ===")
    
    # Run all individual projects
    print("\n=== INDIVIDUAL PROJECTS ===")
    all_results = run_all_projects()
    
    for project_name, data in all_results.items():
        print(f"\n{project_name.upper().replace('_', ' ')}:")
        
        if 'error' in data:
            print(f"  Error: {data['error']}")
            continue
            
        result = data['result']
        print(f"  Execution time: {data['execution_time']}")
        
        # Display key results
        if isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, dict) and len(value) > 3:
                    print(f"  {key}: {dict(list(value.items())[:3])}... (truncated)")
                elif isinstance(value, list) and len(value) > 3:
                    print(f"  {key}: {value[:3]}... (showing first 3)")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  Result: {result}")
    
    # Run integrated system demo
    print("\n=== INTEGRATED BUSINESS SYSTEM ===")
    try:
        start_time = time.time()
        integration_result = integrated_business_system()
        integration_time = time.time() - start_time
        
        print(f"  Execution time: {integration_time*1000:.2f}ms")
        for key, value in integration_result.items():
            print(f"  {key}: {value}")
    
    except Exception as e:
        print(f"  Integration Error: {e}")
    
    print("\n=== PROJECT SUMMARY ===")
    
    project_descriptions = {
        "Task Manager": "File-based task tracking with JSON persistence",
        "Log Analyzer": "Log file parsing and analysis with regex patterns",
        "Banking System": "Account management with transaction history",
        "Text Analyzer": "Comprehensive text analysis and comparison",
        "Inventory Manager": "Product inventory with stock tracking",
        "Data Processor": "Configurable data processing pipeline"
    }
    
    for project, description in project_descriptions.items():
        print(f"  {project}: {description}")
    
    print("\n=== PERFORMANCE SUMMARY ===")
    total_time = sum(float(data.get('execution_time', '0ms')[:-2]) 
                    for data in all_results.values() 
                    if 'execution_time' in data)
    print(f"  Total execution time: {total_time:.2f}ms")
    print(f"  Projects executed: {len(all_results)}")
    print(f"  Average per project: {total_time/len(all_results):.2f}ms")
    
    print("\n=== SKILLS DEMONSTRATED ===")
    skills = [
        "File I/O and JSON handling",
        "Regular expressions and text processing",
        "Object-oriented programming",
        "Error handling and data validation",
        "Data structures and algorithms",
        "Modular code design",
        "System integration patterns"
    ]
    
    for skill in skills:
        print(f"  • {skill}")
    
    print("\n=== Module 1 Complete! ===")
    print("  All Python fundamentals covered with practical applications")
