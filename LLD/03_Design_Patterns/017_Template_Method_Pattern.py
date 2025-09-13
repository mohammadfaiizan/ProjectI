"""
TEMPLATE METHOD PATTERN - Behavioral Design Pattern
===================================================

Problem Statement:
Implement the Template Method pattern to define the skeleton of an algorithm
in a base class, letting subclasses override specific steps without changing
the algorithm's structure:
- Algorithm skeleton with customizable steps
- Invariant parts in base class, variant parts in subclasses
- Hook methods for optional customization
- Data processing pipelines with common structure
- Framework design with extension points

Learning Objectives:
- Understand Template Method vs Strategy pattern differences
- Implement algorithm skeletons with customizable steps
- Design hook methods for optional behavior
- Create extensible frameworks and pipelines
- Handle invariant and variant algorithm parts
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Callable
import time
import json
import csv
import xml.etree.ElementTree as ET
from datetime import datetime
from enum import Enum
import hashlib
import re


# ============================================================================
# TEMPLATE METHOD BASE CLASS
# ============================================================================

class AlgorithmTemplate(ABC):
    """Abstract base class defining template method pattern."""
    
    def template_method(self, *args, **kwargs) -> Any:
        """
        Template method defining the algorithm skeleton.
        This method should not be overridden by subclasses.
        """
        # Pre-processing hook
        self.pre_process(*args, **kwargs)
        
        # Required steps (must be implemented by subclasses)
        self.step_one(*args, **kwargs)
        self.step_two(*args, **kwargs)
        
        # Optional hook method
        if self.should_perform_optional_step(*args, **kwargs):
            self.optional_step(*args, **kwargs)
        
        # Final required step
        result = self.final_step(*args, **kwargs)
        
        # Post-processing hook
        self.post_process(result, *args, **kwargs)
        
        return result
    
    @abstractmethod
    def step_one(self, *args, **kwargs) -> None:
        """First required step - must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def step_two(self, *args, **kwargs) -> None:
        """Second required step - must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def final_step(self, *args, **kwargs) -> Any:
        """Final step that returns result - must be implemented by subclasses."""
        pass
    
    # Hook methods (optional to override)
    def pre_process(self, *args, **kwargs) -> None:
        """Hook method called before main algorithm steps."""
        pass
    
    def post_process(self, result: Any, *args, **kwargs) -> None:
        """Hook method called after main algorithm steps."""
        pass
    
    def should_perform_optional_step(self, *args, **kwargs) -> bool:
        """Hook method to determine if optional step should be performed."""
        return False
    
    def optional_step(self, *args, **kwargs) -> None:
        """Optional step - only called if should_perform_optional_step returns True."""
        pass
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get information about this algorithm implementation."""
        return {
            'algorithm_name': self.__class__.__name__,
            'template_method': 'template_method',
            'required_steps': ['step_one', 'step_two', 'final_step'],
            'hook_methods': ['pre_process', 'post_process', 'optional_step', 'should_perform_optional_step']
        }


# ============================================================================
# DATA PROCESSING PIPELINE
# ============================================================================

class DataProcessor(AlgorithmTemplate):
    """Abstract data processor using template method pattern."""
    
    def __init__(self):
        self.processed_records = 0
        self.errors = []
        self.start_time = None
        self.end_time = None
        self.metadata = {}
    
    def template_method(self, data_source: str, output_destination: str = None, **options) -> Dict[str, Any]:
        """Process data using template method."""
        self.start_time = datetime.now()
        self.processed_records = 0
        self.errors = []
        
        print(f"Starting data processing with {self.__class__.__name__}")
        
        # Call parent template method
        result = super().template_method(data_source, output_destination, **options)
        
        self.end_time = datetime.now()
        processing_time = (self.end_time - self.start_time).total_seconds()
        
        # Add processing statistics to result
        result.update({
            'processed_records': self.processed_records,
            'errors': len(self.errors),
            'processing_time_seconds': processing_time,
            'records_per_second': self.processed_records / processing_time if processing_time > 0 else 0
        })
        
        return result
    
    def pre_process(self, data_source: str, output_destination: str = None, **options) -> None:
        """Pre-processing hook - validate inputs and setup."""
        print(f"Pre-processing: Validating data source '{data_source}'")
        
        if not data_source:
            raise ValueError("Data source cannot be empty")
        
        # Store processing options
        self.metadata.update(options)
    
    def post_process(self, result: Dict[str, Any], data_source: str, output_destination: str = None, **options) -> None:
        """Post-processing hook - cleanup and reporting."""
        print(f"Post-processing: Processed {self.processed_records} records")
        
        if self.errors:
            print(f"Encountered {len(self.errors)} errors during processing")
            for error in self.errors[:3]:  # Show first 3 errors
                print(f"  - {error}")
        
        if output_destination:
            print(f"Results saved to: {output_destination}")


class CSVProcessor(DataProcessor):
    """CSV data processor implementation."""
    
    def __init__(self, delimiter: str = ',', has_header: bool = True):
        super().__init__()
        self.delimiter = delimiter
        self.has_header = has_header
        self.headers = []
        self.data_rows = []
    
    def step_one(self, data_source: str, output_destination: str = None, **options) -> None:
        """Step 1: Load and parse CSV data."""
        print(f"Step 1: Loading CSV data from '{data_source}'")
        
        # Simulate loading CSV data
        sample_data = [
            ['Name', 'Age', 'City', 'Salary'],
            ['Alice', '30', 'New York', '75000'],
            ['Bob', '25', 'San Francisco', '85000'],
            ['Charlie', '35', 'Chicago', '70000'],
            ['Diana', '28', 'Seattle', '80000']
        ]
        
        if self.has_header:
            self.headers = sample_data[0]
            self.data_rows = sample_data[1:]
        else:
            self.headers = [f'Column_{i}' for i in range(len(sample_data[0]))]
            self.data_rows = sample_data
        
        print(f"Loaded {len(self.data_rows)} data rows with headers: {self.headers}")
    
    def step_two(self, data_source: str, output_destination: str = None, **options) -> None:
        """Step 2: Validate and clean CSV data."""
        print("Step 2: Validating and cleaning CSV data")
        
        cleaned_rows = []
        
        for i, row in enumerate(self.data_rows):
            try:
                # Basic validation - ensure all fields are present
                if len(row) != len(self.headers):
                    raise ValueError(f"Row {i+1} has {len(row)} fields, expected {len(self.headers)}")
                
                # Clean data (trim whitespace, handle empty values)
                cleaned_row = [field.strip() if field else 'N/A' for field in row]
                cleaned_rows.append(cleaned_row)
                
            except Exception as e:
                self.errors.append(f"Row {i+1}: {str(e)}")
        
        self.data_rows = cleaned_rows
        print(f"Cleaned {len(self.data_rows)} rows, {len(self.errors)} errors found")
    
    def should_perform_optional_step(self, data_source: str, output_destination: str = None, **options) -> bool:
        """Check if data transformation should be performed."""
        return options.get('transform_data', False)
    
    def optional_step(self, data_source: str, output_destination: str = None, **options) -> None:
        """Optional step: Transform data based on business rules."""
        print("Optional Step: Transforming data")
        
        # Example transformation: convert salary to numeric and add bonus calculation
        if 'Salary' in self.headers:
            salary_index = self.headers.index('Salary')
            
            for row in self.data_rows:
                try:
                    salary = float(row[salary_index])
                    bonus = salary * 0.1  # 10% bonus
                    row.append(str(int(bonus)))
                except ValueError:
                    row.append('0')
            
            self.headers.append('Bonus')
            print("Added bonus calculation to data")
    
    def final_step(self, data_source: str, output_destination: str = None, **options) -> Dict[str, Any]:
        """Final step: Generate processing results."""
        print("Final Step: Generating CSV processing results")
        
        self.processed_records = len(self.data_rows)
        
        # Calculate statistics
        stats = {
            'total_rows': len(self.data_rows),
            'total_columns': len(self.headers),
            'headers': self.headers,
            'sample_data': self.data_rows[:3] if self.data_rows else [],
            'data_type': 'CSV'
        }
        
        # If salary column exists, calculate salary statistics
        if 'Salary' in self.headers:
            salary_index = self.headers.index('Salary')
            salaries = []
            
            for row in self.data_rows:
                try:
                    salaries.append(float(row[salary_index]))
                except ValueError:
                    pass
            
            if salaries:
                stats['salary_stats'] = {
                    'average': sum(salaries) / len(salaries),
                    'min': min(salaries),
                    'max': max(salaries),
                    'count': len(salaries)
                }
        
        return stats


class JSONProcessor(DataProcessor):
    """JSON data processor implementation."""
    
    def __init__(self):
        super().__init__()
        self.json_data = None
        self.processed_data = None
    
    def step_one(self, data_source: str, output_destination: str = None, **options) -> None:
        """Step 1: Load and parse JSON data."""
        print(f"Step 1: Loading JSON data from '{data_source}'")
        
        # Simulate loading JSON data
        sample_json = {
            "users": [
                {"id": 1, "name": "Alice", "email": "alice@example.com", "active": True},
                {"id": 2, "name": "Bob", "email": "bob@example.com", "active": False},
                {"id": 3, "name": "Charlie", "email": "charlie@example.com", "active": True}
            ],
            "metadata": {
                "version": "1.0",
                "created": "2024-01-01",
                "total_users": 3
            }
        }
        
        self.json_data = sample_json
        print(f"Loaded JSON data with {len(sample_json.get('users', []))} users")
    
    def step_two(self, data_source: str, output_destination: str = None, **options) -> None:
        """Step 2: Validate and normalize JSON data."""
        print("Step 2: Validating and normalizing JSON data")
        
        if not self.json_data:
            raise ValueError("No JSON data loaded")
        
        # Validate required fields
        users = self.json_data.get('users', [])
        validated_users = []
        
        for i, user in enumerate(users):
            try:
                # Validate required fields
                required_fields = ['id', 'name', 'email']
                for field in required_fields:
                    if field not in user:
                        raise ValueError(f"Missing required field: {field}")
                
                # Normalize data
                normalized_user = {
                    'id': int(user['id']),
                    'name': user['name'].strip(),
                    'email': user['email'].lower().strip(),
                    'active': user.get('active', True)
                }
                
                validated_users.append(normalized_user)
                
            except Exception as e:
                self.errors.append(f"User {i+1}: {str(e)}")
        
        self.json_data['users'] = validated_users
        print(f"Validated {len(validated_users)} users, {len(self.errors)} errors found")
    
    def should_perform_optional_step(self, data_source: str, output_destination: str = None, **options) -> bool:
        """Check if data enrichment should be performed."""
        return options.get('enrich_data', False)
    
    def optional_step(self, data_source: str, output_destination: str = None, **options) -> None:
        """Optional step: Enrich user data with additional information."""
        print("Optional Step: Enriching user data")
        
        users = self.json_data.get('users', [])
        
        for user in users:
            # Add domain from email
            email = user.get('email', '')
            domain = email.split('@')[1] if '@' in email else 'unknown'
            user['email_domain'] = domain
            
            # Add user hash for privacy
            user_string = f"{user['id']}{user['name']}{user['email']}"
            user['user_hash'] = hashlib.md5(user_string.encode()).hexdigest()[:8]
        
        print(f"Enriched {len(users)} user records")
    
    def final_step(self, data_source: str, output_destination: str = None, **options) -> Dict[str, Any]:
        """Final step: Generate JSON processing results."""
        print("Final Step: Generating JSON processing results")
        
        users = self.json_data.get('users', [])
        self.processed_records = len(users)
        
        # Calculate statistics
        active_users = sum(1 for user in users if user.get('active', False))
        email_domains = {}
        
        for user in users:
            domain = user.get('email_domain', 'unknown')
            email_domains[domain] = email_domains.get(domain, 0) + 1
        
        stats = {
            'total_users': len(users),
            'active_users': active_users,
            'inactive_users': len(users) - active_users,
            'email_domains': email_domains,
            'sample_users': users[:2] if users else [],
            'data_type': 'JSON'
        }
        
        return stats


class XMLProcessor(DataProcessor):
    """XML data processor implementation."""
    
    def __init__(self):
        super().__init__()
        self.xml_root = None
        self.processed_elements = []
    
    def step_one(self, data_source: str, output_destination: str = None, **options) -> None:
        """Step 1: Load and parse XML data."""
        print(f"Step 1: Loading XML data from '{data_source}'")
        
        # Simulate XML data
        xml_string = """<?xml version="1.0" encoding="UTF-8"?>
        <catalog>
            <book id="1">
                <title>Python Programming</title>
                <author>John Doe</author>
                <price>29.99</price>
                <category>Programming</category>
            </book>
            <book id="2">
                <title>Data Science Handbook</title>
                <author>Jane Smith</author>
                <price>39.99</price>
                <category>Data Science</category>
            </book>
            <book id="3">
                <title>Machine Learning Guide</title>
                <author>Bob Johnson</author>
                <price>49.99</price>
                <category>AI/ML</category>
            </book>
        </catalog>"""
        
        self.xml_root = ET.fromstring(xml_string)
        books = self.xml_root.findall('book')
        print(f"Loaded XML data with {len(books)} book elements")
    
    def step_two(self, data_source: str, output_destination: str = None, **options) -> None:
        """Step 2: Validate and extract XML data."""
        print("Step 2: Validating and extracting XML data")
        
        books = self.xml_root.findall('book')
        extracted_books = []
        
        for i, book in enumerate(books):
            try:
                # Extract book data
                book_data = {
                    'id': book.get('id'),
                    'title': book.find('title').text if book.find('title') is not None else '',
                    'author': book.find('author').text if book.find('author') is not None else '',
                    'price': book.find('price').text if book.find('price') is not None else '0',
                    'category': book.find('category').text if book.find('category') is not None else ''
                }
                
                # Validate required fields
                if not book_data['id'] or not book_data['title']:
                    raise ValueError("Missing required fields: id or title")
                
                # Convert price to float
                try:
                    book_data['price'] = float(book_data['price'])
                except ValueError:
                    book_data['price'] = 0.0
                
                extracted_books.append(book_data)
                
            except Exception as e:
                self.errors.append(f"Book {i+1}: {str(e)}")
        
        self.processed_elements = extracted_books
        print(f"Extracted {len(extracted_books)} valid books, {len(self.errors)} errors found")
    
    def should_perform_optional_step(self, data_source: str, output_destination: str = None, **options) -> bool:
        """Check if price analysis should be performed."""
        return options.get('analyze_prices', False)
    
    def optional_step(self, data_source: str, output_destination: str = None, **options) -> None:
        """Optional step: Analyze book prices and add price categories."""
        print("Optional Step: Analyzing book prices")
        
        for book in self.processed_elements:
            price = book.get('price', 0)
            
            if price < 30:
                price_category = 'Budget'
            elif price < 40:
                price_category = 'Standard'
            else:
                price_category = 'Premium'
            
            book['price_category'] = price_category
        
        print(f"Added price categories to {len(self.processed_elements)} books")
    
    def final_step(self, data_source: str, output_destination: str = None, **options) -> Dict[str, Any]:
        """Final step: Generate XML processing results."""
        print("Final Step: Generating XML processing results")
        
        self.processed_records = len(self.processed_elements)
        
        # Calculate statistics
        categories = {}
        total_price = 0
        
        for book in self.processed_elements:
            category = book.get('category', 'Unknown')
            categories[category] = categories.get(category, 0) + 1
            total_price += book.get('price', 0)
        
        stats = {
            'total_books': len(self.processed_elements),
            'categories': categories,
            'average_price': total_price / len(self.processed_elements) if self.processed_elements else 0,
            'total_value': total_price,
            'sample_books': self.processed_elements[:2] if self.processed_elements else [],
            'data_type': 'XML'
        }
        
        return stats


# ============================================================================
# REPORT GENERATION TEMPLATE
# ============================================================================

class ReportGenerator(AlgorithmTemplate):
    """Abstract report generator using template method pattern."""
    
    def __init__(self):
        self.report_data = {}
        self.report_content = []
        self.generation_time = None
    
    def template_method(self, data: Dict[str, Any], report_title: str = "Report", **options) -> str:
        """Generate report using template method."""
        self.generation_time = datetime.now()
        self.report_data = data
        self.report_content = []
        
        print(f"Generating {report_title} using {self.__class__.__name__}")
        
        # Call parent template method
        result = super().template_method(data, report_title, **options)
        
        return result
    
    def pre_process(self, data: Dict[str, Any], report_title: str = "Report", **options) -> None:
        """Pre-processing: Validate data and setup report."""
        print("Pre-processing: Validating report data")
        
        if not data:
            raise ValueError("Report data cannot be empty")
        
        # Add report header
        self.report_content.append(self.format_header(report_title))
        self.report_content.append(self.format_timestamp())
    
    def post_process(self, result: str, data: Dict[str, Any], report_title: str = "Report", **options) -> None:
        """Post-processing: Add footer and finalize report."""
        print("Post-processing: Finalizing report")
        
        # Add report footer
        self.report_content.append(self.format_footer())
    
    def should_perform_optional_step(self, data: Dict[str, Any], report_title: str = "Report", **options) -> bool:
        """Check if charts/graphs should be included."""
        return options.get('include_charts', False)
    
    def optional_step(self, data: Dict[str, Any], report_title: str = "Report", **options) -> None:
        """Optional step: Add charts and visualizations."""
        print("Optional Step: Adding charts and visualizations")
        self.add_visualizations(data)
    
    @abstractmethod
    def format_header(self, title: str) -> str:
        """Format report header - must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def format_timestamp(self) -> str:
        """Format timestamp - must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def format_footer(self) -> str:
        """Format report footer - must be implemented by subclasses."""
        pass
    
    def add_visualizations(self, data: Dict[str, Any]) -> None:
        """Add visualizations to report (hook method)."""
        pass


class HTMLReportGenerator(ReportGenerator):
    """HTML report generator implementation."""
    
    def step_one(self, data: Dict[str, Any], report_title: str = "Report", **options) -> None:
        """Step 1: Create HTML structure."""
        print("Step 1: Creating HTML report structure")
        
        self.report_content.append("<div class='report-body'>")
    
    def step_two(self, data: Dict[str, Any], report_title: str = "Report", **options) -> None:
        """Step 2: Add data sections to HTML."""
        print("Step 2: Adding data sections to HTML report")
        
        for section_name, section_data in data.items():
            self.report_content.append(f"<div class='section'>")
            self.report_content.append(f"<h2>{section_name.replace('_', ' ').title()}</h2>")
            
            if isinstance(section_data, dict):
                self.report_content.append("<ul>")
                for key, value in section_data.items():
                    self.report_content.append(f"<li><strong>{key}:</strong> {value}</li>")
                self.report_content.append("</ul>")
            elif isinstance(section_data, list):
                self.report_content.append(f"<p>Items: {len(section_data)}</p>")
                if section_data:
                    self.report_content.append("<ul>")
                    for item in section_data[:5]:  # Show first 5 items
                        self.report_content.append(f"<li>{item}</li>")
                    self.report_content.append("</ul>")
            else:
                self.report_content.append(f"<p>{section_data}</p>")
            
            self.report_content.append("</div>")
    
    def final_step(self, data: Dict[str, Any], report_title: str = "Report", **options) -> str:
        """Final step: Generate complete HTML report."""
        print("Final Step: Generating complete HTML report")
        
        self.report_content.append("</div>")  # Close report-body
        
        # Combine all content
        html_content = "\n".join(self.report_content)
        
        # Wrap in complete HTML document
        complete_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{report_title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .report-header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .timestamp {{ color: #666; font-size: 0.9em; }}
        .footer {{ margin-top: 30px; text-align: center; color: #888; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""
        
        return complete_html
    
    def format_header(self, title: str) -> str:
        """Format HTML header."""
        return f"<div class='report-header'><h1>{title}</h1></div>"
    
    def format_timestamp(self) -> str:
        """Format HTML timestamp."""
        return f"<p class='timestamp'>Generated on: {self.generation_time.strftime('%Y-%m-%d %H:%M:%S')}</p>"
    
    def format_footer(self) -> str:
        """Format HTML footer."""
        return "<div class='footer'><p>End of Report</p></div>"
    
    def add_visualizations(self, data: Dict[str, Any]) -> None:
        """Add HTML visualizations."""
        self.report_content.append("<div class='visualizations'>")
        self.report_content.append("<h2>Charts and Graphs</h2>")
        self.report_content.append("<p>[Chart placeholders would be inserted here]</p>")
        self.report_content.append("</div>")


class TextReportGenerator(ReportGenerator):
    """Plain text report generator implementation."""
    
    def step_one(self, data: Dict[str, Any], report_title: str = "Report", **options) -> None:
        """Step 1: Create text structure."""
        print("Step 1: Creating text report structure")
        
        self.report_content.append("=" * 60)
    
    def step_two(self, data: Dict[str, Any], report_title: str = "Report", **options) -> None:
        """Step 2: Add data sections to text."""
        print("Step 2: Adding data sections to text report")
        
        for section_name, section_data in data.items():
            self.report_content.append(f"\n{section_name.replace('_', ' ').upper()}")
            self.report_content.append("-" * 40)
            
            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    self.report_content.append(f"{key}: {value}")
            elif isinstance(section_data, list):
                self.report_content.append(f"Total items: {len(section_data)}")
                for i, item in enumerate(section_data[:5], 1):  # Show first 5 items
                    self.report_content.append(f"  {i}. {item}")
            else:
                self.report_content.append(str(section_data))
    
    def final_step(self, data: Dict[str, Any], report_title: str = "Report", **options) -> str:
        """Final step: Generate complete text report."""
        print("Final Step: Generating complete text report")
        
        # Combine all content
        return "\n".join(self.report_content)
    
    def format_header(self, title: str) -> str:
        """Format text header."""
        return f"{title.upper()}\n{'=' * len(title)}"
    
    def format_timestamp(self) -> str:
        """Format text timestamp."""
        return f"Generated: {self.generation_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    def format_footer(self) -> str:
        """Format text footer."""
        return f"\n{'=' * 60}\nEnd of Report"
    
    def add_visualizations(self, data: Dict[str, Any]) -> None:
        """Add text-based visualizations."""
        self.report_content.append("\nCHARTS AND VISUALIZATIONS")
        self.report_content.append("-" * 40)
        self.report_content.append("[ASCII charts would be generated here]")


# ============================================================================
# SORTING ALGORITHM TEMPLATE
# ============================================================================

class SortingAlgorithm(AlgorithmTemplate):
    """Abstract sorting algorithm using template method pattern."""
    
    def __init__(self):
        self.comparisons = 0
        self.swaps = 0
        self.data_size = 0
        self.sort_time = 0
    
    def template_method(self, data: List[Any], key_func: Callable = None, reverse: bool = False) -> List[Any]:
        """Sort data using template method."""
        self.comparisons = 0
        self.swaps = 0
        self.data_size = len(data)
        
        start_time = time.time()
        
        # Call parent template method
        result = super().template_method(data.copy(), key_func, reverse)
        
        self.sort_time = time.time() - start_time
        
        return result
    
    def pre_process(self, data: List[Any], key_func: Callable = None, reverse: bool = False) -> None:
        """Pre-processing: Validate input data."""
        if not isinstance(data, list):
            raise TypeError("Data must be a list")
        
        print(f"Pre-processing: Sorting {len(data)} items using {self.__class__.__name__}")
    
    def post_process(self, result: List[Any], data: List[Any], key_func: Callable = None, reverse: bool = False) -> None:
        """Post-processing: Report sorting statistics."""
        print(f"Post-processing: Completed in {self.sort_time:.4f}s, "
              f"{self.comparisons} comparisons, {self.swaps} swaps")
    
    def step_one(self, data: List[Any], key_func: Callable = None, reverse: bool = False) -> None:
        """Step 1: Initialize sorting algorithm."""
        print(f"Step 1: Initializing {self.__class__.__name__} algorithm")
    
    def step_two(self, data: List[Any], key_func: Callable = None, reverse: bool = False) -> None:
        """Step 2: Perform main sorting logic."""
        self.perform_sort(data, key_func, reverse)
    
    def final_step(self, data: List[Any], key_func: Callable = None, reverse: bool = False) -> List[Any]:
        """Final step: Return sorted data."""
        return data
    
    @abstractmethod
    def perform_sort(self, data: List[Any], key_func: Callable, reverse: bool) -> None:
        """Perform the actual sorting - must be implemented by subclasses."""
        pass
    
    def compare_elements(self, a: Any, b: Any, key_func: Callable, reverse: bool) -> bool:
        """Compare two elements and increment comparison counter."""
        self.comparisons += 1
        
        val_a = key_func(a) if key_func else a
        val_b = key_func(b) if key_func else b
        
        if reverse:
            return val_a > val_b
        else:
            return val_a < val_b
    
    def swap_elements(self, data: List[Any], i: int, j: int) -> None:
        """Swap two elements and increment swap counter."""
        self.swaps += 1
        data[i], data[j] = data[j], data[i]


class BubbleSort(SortingAlgorithm):
    """Bubble sort implementation using template method."""
    
    def perform_sort(self, data: List[Any], key_func: Callable, reverse: bool) -> None:
        """Perform bubble sort algorithm."""
        n = len(data)
        
        for i in range(n):
            swapped = False
            
            for j in range(0, n - i - 1):
                if not self.compare_elements(data[j], data[j + 1], key_func, reverse):
                    self.swap_elements(data, j, j + 1)
                    swapped = True
            
            # If no swapping occurred, array is sorted
            if not swapped:
                break


class SelectionSort(SortingAlgorithm):
    """Selection sort implementation using template method."""
    
    def perform_sort(self, data: List[Any], key_func: Callable, reverse: bool) -> None:
        """Perform selection sort algorithm."""
        n = len(data)
        
        for i in range(n):
            # Find the minimum/maximum element in remaining unsorted array
            extreme_idx = i
            
            for j in range(i + 1, n):
                if not self.compare_elements(data[extreme_idx], data[j], key_func, reverse):
                    extreme_idx = j
            
            # Swap the found minimum/maximum element with the first element
            if extreme_idx != i:
                self.swap_elements(data, i, extreme_idx)


# ============================================================================
# TEMPLATE METHOD FACTORY
# ============================================================================

class TemplateMethodFactory:
    """Factory for creating template method implementations."""
    
    @staticmethod
    def create_data_processor(processor_type: str) -> DataProcessor:
        """Create data processor by type."""
        processors = {
            'csv': CSVProcessor,
            'json': JSONProcessor,
            'xml': XMLProcessor
        }
        
        if processor_type not in processors:
            raise ValueError(f"Unknown processor type: {processor_type}")
        
        return processors[processor_type]()
    
    @staticmethod
    def create_report_generator(generator_type: str) -> ReportGenerator:
        """Create report generator by type."""
        generators = {
            'html': HTMLReportGenerator,
            'text': TextReportGenerator
        }
        
        if generator_type not in generators:
            raise ValueError(f"Unknown generator type: {generator_type}")
        
        return generators[generator_type]()
    
    @staticmethod
    def create_sorting_algorithm(algorithm_type: str) -> SortingAlgorithm:
        """Create sorting algorithm by type."""
        algorithms = {
            'bubble': BubbleSort,
            'selection': SelectionSort
        }
        
        if algorithm_type not in algorithms:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}")
        
        return algorithms[algorithm_type]()
    
    @staticmethod
    def get_available_implementations() -> Dict[str, List[str]]:
        """Get available template method implementations."""
        return {
            'data_processors': ['csv', 'json', 'xml'],
            'report_generators': ['html', 'text'],
            'sorting_algorithms': ['bubble', 'selection']
        }


def demonstrate_template_method_pattern():
    """
    Demonstrate Template Method pattern implementations.
    """
    print("=== TEMPLATE METHOD PATTERN DEMONSTRATION ===\n")
    
    # 1. Data Processing Pipeline
    print("1. DATA PROCESSING PIPELINE:")
    
    # Create different data processors
    csv_processor = CSVProcessor()
    json_processor = JSONProcessor()
    xml_processor = XMLProcessor()
    
    processors = [
        ("CSV", csv_processor, {"transform_data": True}),
        ("JSON", json_processor, {"enrich_data": True}),
        ("XML", xml_processor, {"analyze_prices": True})
    ]
    
    print("   Testing different data processors:")
    
    for name, processor, options in processors:
        print(f"\n   {name} Processor:")
        print("   " + "=" * 40)
        
        result = processor.template_method(f"sample_{name.lower()}_data.{name.lower()}", 
                                         f"output_{name.lower()}.processed", 
                                         **options)
        
        print(f"   Results: {json.dumps(result, indent=2)}")
        print()
    
    # 2. Report Generation
    print("2. REPORT GENERATION:")
    
    # Sample data for reports
    sample_data = {
        'summary': {
            'total_records': 1000,
            'processed_successfully': 950,
            'errors': 50,
            'processing_time': '2.5 seconds'
        },
        'categories': {
            'Category A': 300,
            'Category B': 450,
            'Category C': 200
        },
        'top_items': [
            'Item 1 - High Priority',
            'Item 2 - Medium Priority',
            'Item 3 - Low Priority'
        ]
    }
    
    # Create different report generators
    html_generator = HTMLReportGenerator()
    text_generator = TextReportGenerator()
    
    generators = [
        ("HTML", html_generator, {"include_charts": True}),
        ("Text", text_generator, {"include_charts": False})
    ]
    
    print("   Testing different report generators:")
    
    for name, generator, options in generators:
        print(f"\n   {name} Report Generator:")
        print("   " + "=" * 40)
        
        report = generator.template_method(sample_data, f"Sample {name} Report", **options)
        
        # Show first 500 characters of report
        print(f"   Generated Report (first 500 chars):")
        print(f"   {report[:500]}...")
        print()
    
    # 3. Sorting Algorithms
    print("3. SORTING ALGORITHMS:")
    
    # Test data
    test_data = [64, 34, 25, 12, 22, 11, 90, 5, 77, 30]
    
    # Create different sorting algorithms
    bubble_sort = BubbleSort()
    selection_sort = SelectionSort()
    
    algorithms = [
        ("Bubble Sort", bubble_sort),
        ("Selection Sort", selection_sort)
    ]
    
    print(f"   Original data: {test_data}")
    print()
    
    for name, algorithm in algorithms:
        print(f"   {name}:")
        print("   " + "-" * 30)
        
        sorted_data = algorithm.template_method(test_data.copy())
        
        print(f"   Sorted data: {sorted_data}")
        print(f"   Statistics: {algorithm.comparisons} comparisons, "
              f"{algorithm.swaps} swaps, {algorithm.sort_time:.4f}s")
        print()
    
    # Test sorting with custom key function
    print("   Sorting strings by length:")
    string_data = ['apple', 'pie', 'washington', 'book', 'python', 'a']
    
    bubble_sort_str = BubbleSort()
    sorted_strings = bubble_sort_str.template_method(string_data, key_func=len)
    
    print(f"   Original: {string_data}")
    print(f"   Sorted by length: {sorted_strings}")
    print()
    
    # 4. Template Method Factory
    print("4. TEMPLATE METHOD FACTORY:")
    
    factory = TemplateMethodFactory()
    available = factory.get_available_implementations()
    
    print("   Available implementations:")
    for category, implementations in available.items():
        print(f"     {category}: {implementations}")
    
    print()
    
    # Create instances using factory
    print("   Creating instances using factory:")
    
    csv_proc = factory.create_data_processor('csv')
    html_gen = factory.create_report_generator('html')
    bubble_alg = factory.create_sorting_algorithm('bubble')
    
    print(f"   Created CSV processor: {csv_proc.__class__.__name__}")
    print(f"   Created HTML generator: {html_gen.__class__.__name__}")
    print(f"   Created Bubble sort: {bubble_alg.__class__.__name__}")
    
    print()
    
    # 5. Algorithm Information
    print("5. ALGORITHM INFORMATION:")
    
    # Show algorithm information for different implementations
    implementations = [
        csv_processor,
        html_generator,
        bubble_sort
    ]
    
    print("   Algorithm information:")
    for impl in implementations:
        info = impl.get_algorithm_info()
        print(f"   {info['algorithm_name']}:")
        print(f"     Template method: {info['template_method']}")
        print(f"     Required steps: {info['required_steps']}")
        print(f"     Hook methods: {info['hook_methods']}")
        print()
    
    # 6. Custom Template Method Implementation
    print("6. CUSTOM TEMPLATE METHOD IMPLEMENTATION:")
    
    class CustomDataValidator(AlgorithmTemplate):
        """Custom data validator using template method."""
        
        def __init__(self):
            self.validation_errors = []
            self.validated_records = []
        
        def step_one(self, data: List[Dict], validation_rules: Dict, **options) -> None:
            """Step 1: Initialize validation."""
            print("Step 1: Initializing data validation")
            self.validation_errors = []
            self.validated_records = []
        
        def step_two(self, data: List[Dict], validation_rules: Dict, **options) -> None:
            """Step 2: Validate each record."""
            print("Step 2: Validating records")
            
            for i, record in enumerate(data):
                errors = []
                
                for field, rules in validation_rules.items():
                    if field not in record:
                        if rules.get('required', False):
                            errors.append(f"Missing required field: {field}")
                        continue
                    
                    value = record[field]
                    
                    # Type validation
                    if 'type' in rules and not isinstance(value, rules['type']):
                        errors.append(f"Field {field} must be of type {rules['type'].__name__}")
                    
                    # Range validation for numbers
                    if 'min' in rules and isinstance(value, (int, float)) and value < rules['min']:
                        errors.append(f"Field {field} must be >= {rules['min']}")
                    
                    if 'max' in rules and isinstance(value, (int, float)) and value > rules['max']:
                        errors.append(f"Field {field} must be <= {rules['max']}")
                
                if errors:
                    self.validation_errors.append(f"Record {i+1}: {'; '.join(errors)}")
                else:
                    self.validated_records.append(record)
        
        def should_perform_optional_step(self, data: List[Dict], validation_rules: Dict, **options) -> bool:
            """Check if data cleaning should be performed."""
            return options.get('clean_data', False) and self.validation_errors
        
        def optional_step(self, data: List[Dict], validation_rules: Dict, **options) -> None:
            """Optional step: Clean invalid data."""
            print("Optional Step: Cleaning invalid data")
            # In a real implementation, this would attempt to fix validation errors
            print(f"Would attempt to clean {len(self.validation_errors)} validation errors")
        
        def final_step(self, data: List[Dict], validation_rules: Dict, **options) -> Dict[str, Any]:
            """Final step: Return validation results."""
            return {
                'total_records': len(data),
                'valid_records': len(self.validated_records),
                'invalid_records': len(self.validation_errors),
                'validation_errors': self.validation_errors[:5],  # Show first 5 errors
                'success_rate': len(self.validated_records) / len(data) * 100 if data else 0
            }
    
    # Test custom validator
    validator = CustomDataValidator()
    
    test_data = [
        {'name': 'Alice', 'age': 30, 'email': 'alice@example.com'},
        {'name': 'Bob', 'age': -5, 'email': 'invalid-email'},  # Invalid age
        {'name': '', 'age': 25},  # Missing email, empty name
        {'name': 'Charlie', 'age': 35, 'email': 'charlie@example.com'}
    ]
    
    validation_rules = {
        'name': {'required': True, 'type': str},
        'age': {'required': True, 'type': int, 'min': 0, 'max': 120},
        'email': {'required': True, 'type': str}
    }
    
    print("   Testing custom data validator:")
    result = validator.template_method(test_data, validation_rules, clean_data=True)
    
    print(f"   Validation Results:")
    for key, value in result.items():
        print(f"     {key}: {value}")
    
    print()
    
    # 7. Template Method Pattern Benefits
    print("7. TEMPLATE METHOD PATTERN BENEFITS:")
    print("   ✓ Code Reuse: Common algorithm structure is reused across implementations")
    print("   ✓ Consistency: All implementations follow the same algorithm skeleton")
    print("   ✓ Flexibility: Subclasses can customize specific steps without changing overall structure")
    print("   ✓ Maintainability: Changes to algorithm structure are centralized in base class")
    print("   ✓ Extension Points: Hook methods provide optional customization points")
    print("   ✓ Inversion of Control: Framework controls the algorithm flow, subclasses provide details")
    print("   ✓ Open/Closed Principle: Open for extension (new implementations), closed for modification")
    print("   ✓ Template Enforcement: Ensures all implementations follow the required structure")
    print("   ✓ Documentation: Algorithm steps are clearly defined and documented")
    print("   ✓ Testing: Each step can be tested independently")
    print()
    
    print("=== TEMPLATE METHOD PATTERN DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_template_method_pattern()
