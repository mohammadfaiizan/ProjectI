"""
Main module for Document Processing System.

This module provides the main entry point and high-level functions for
setting up and running the document processing system.
"""

import os
from typing import List, Dict, Any, Optional

from Config import LLM_Config, Processing_Config, Classification_Config, Storage_Config
from Tools import Document_Store
from Agent import Document_Processing_Graph


# ============================================================================
# System Setup
# ============================================================================

def Setup_Processing_System(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    output_directory: str = "./processed_documents"
) -> Document_Processing_Graph:
    """
    Setup and initialize the document processing system.
    
    Args:
        model_name: Name of the OpenAI model to use
        temperature: Temperature for model generation
        output_directory: Directory for storing processed documents
        
    Returns:
        Configured Document_Processing_Graph instance
        
    Raises:
        ValueError: If OpenAI API key is not set
    """
    # Initialize configurations
    llm_config = LLM_Config(
        model_name=model_name,
        temperature=temperature
    )
    
    processing_config = Processing_Config()
    classification_config = Classification_Config()
    storage_config = Storage_Config(output_directory=output_directory)
    
    # Initialize document store
    document_store = Document_Store(
        storage_directory=storage_config.Get_Output_Directory()
    )
    
    # Create processing graph
    processing_graph = Document_Processing_Graph(
        llm_config=llm_config,
        classification_config=classification_config,
        document_store=document_store
    )
    
    print("Document Processing System initialized successfully")
    print(f"Model: {model_name}")
    print(f"Output directory: {output_directory}")
    print(f"Supported document types: {', '.join(processing_config.Get_Supported_Types())}")
    print(f"Document categories: {', '.join(classification_config.Get_Categories())}")
    
    return processing_graph


# ============================================================================
# Document Processing Functions
# ============================================================================

def Process_Document(
    text: str,
    filename: str,
    processing_graph: Optional[Document_Processing_Graph] = None
) -> Dict[str, Any]:
    """
    Process a single document and return results.
    
    Args:
        text: Raw document text content
        filename: Name of the source file
        processing_graph: Optional pre-initialized processing graph
        
    Returns:
        Dictionary containing processing results with keys:
        - doc_type: Classified document type
        - extracted_entities: Extracted structured entities
        - validation_result: Validation results
        - is_valid: Boolean indicating if extraction is valid
        - output: Summary output string
        - processing_history: List of processing steps
    """
    if processing_graph is None:
        processing_graph = Setup_Processing_System()
    
    print(f"\n{'='*60}")
    print(f"Processing document: {filename}")
    print(f"{'='*60}")
    
    # Process document
    results = processing_graph.Process_Document(text=text, filename=filename)
    
    # Print results
    print(f"\nDocument Type: {results.get('doc_type', 'Unknown')}")
    print(f"Validation Status: {'Valid' if results.get('is_valid') else 'Invalid'}")
    
    if results.get('extracted_entities'):
        print(f"\nExtracted Entities:")
        entities = results['extracted_entities']
        for key, value in entities.items():
            if isinstance(value, list) and len(value) > 0:
                print(f"  {key}: {len(value)} items")
            elif value:
                print(f"  {key}: {value}")
    
    if results.get('validation_result'):
        validation = results['validation_result']
        print(f"\nValidation Details:")
        print(f"  Completeness Score: {validation.get('completeness_score', 0):.2%}")
        if validation.get('missing_fields'):
            print(f"  Missing Fields: {', '.join(validation['missing_fields'])}")
        if validation.get('inconsistencies'):
            print(f"  Inconsistencies: {', '.join(validation['inconsistencies'])}")
    
    if results.get('output'):
        print(f"\nOutput Summary:")
        print(results['output'])
    
    return results


def Process_Batch(
    documents: List[Dict[str, str]],
    processing_graph: Optional[Document_Processing_Graph] = None
) -> List[Dict[str, Any]]:
    """
    Process multiple documents in batch.
    
    Args:
        documents: List of dictionaries with 'text' and 'filename' keys
        processing_graph: Optional pre-initialized processing graph
        
    Returns:
        List of processing result dictionaries
    """
    if processing_graph is None:
        processing_graph = Setup_Processing_System()
    
    print(f"\n{'='*60}")
    print(f"Processing batch of {len(documents)} documents")
    print(f"{'='*60}\n")
    
    results = []
    for i, doc in enumerate(documents, 1):
        print(f"\n[{i}/{len(documents)}] Processing: {doc.get('filename', 'unknown')}")
        result = processing_graph.Process_Document(
            text=doc['text'],
            filename=doc.get('filename', f'document_{i}.txt')
        )
        results.append(result)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Batch Processing Summary")
    print(f"{'='*60}")
    
    doc_types = {}
    valid_count = 0
    for result in results:
        doc_type = result.get('doc_type', 'unknown')
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        if result.get('is_valid'):
            valid_count += 1
    
    print(f"Total documents processed: {len(results)}")
    print(f"Valid extractions: {valid_count}/{len(results)}")
    print(f"\nDocument type distribution:")
    for doc_type, count in doc_types.items():
        print(f"  {doc_type}: {count}")
    
    return results


# ============================================================================
# Demo Function
# ============================================================================

def Run_Demo():
    """
    Run a demonstration of the document processing system.
    """
    print("Document Processing System - Demo")
    print("="*60)
    
    # Setup system
    processing_graph = Setup_Processing_System()
    
    # Sample documents for demo
    sample_documents = [
        {
            "filename": "sample_invoice.txt",
            "text": """INVOICE
Invoice Number: INV-2024-001
Date: 2024-01-15
Bill To: ABC Corporation, 123 Business St, City, State 12345
Ship To: Same as Bill To

Item Description          Quantity    Unit Price    Total
----------------------------------------------------------
Software License         2           $500.00       $1,000.00
Support Package          1           $200.00       $200.00
Training Session         1           $300.00       $300.00

Subtotal: $1,500.00
Tax (8%): $120.00
Total Due: $1,620.00

Payment Terms: Net 30
Thank you for your business!"""
        },
        {
            "filename": "sample_resume.txt",
            "text": """JOHN DOE
Email: john.doe@email.com
Phone: (555) 123-4567

PROFESSIONAL SUMMARY
Experienced software engineer with 5+ years in full-stack development.

WORK EXPERIENCE
Senior Software Engineer | Tech Corp | 2020-Present
- Developed microservices using Python and Django
- Led team of 3 developers

Software Engineer | Startup Inc | 2018-2020
- Built REST APIs with Flask
- Implemented CI/CD pipelines

EDUCATION
Bachelor of Science in Computer Science
State University | 2018

SKILLS
Python, JavaScript, React, Django, Flask, PostgreSQL, AWS, Docker"""
        }
    ]
    
    # Process sample documents
    results = Process_Batch(sample_documents, processing_graph)
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        print("Please set it before running the script.")
        exit(1)
    
    # Run demo
    Run_Demo()
