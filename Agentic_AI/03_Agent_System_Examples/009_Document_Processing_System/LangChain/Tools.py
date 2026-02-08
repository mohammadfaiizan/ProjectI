"""
Tools module for Document Processing System.
Contains tool functions for document ingestion, classification, entity extraction,
validation, and document storage.
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
import os
import re
from datetime import datetime


# ============================================================================
# Pydantic Models for Structured Extraction
# ============================================================================

class Invoice_Entities(BaseModel):
    """Structured entities extracted from invoice documents."""
    invoice_number: str = Field(description="Invoice number or ID")
    date: str = Field(description="Invoice date")
    vendor: str = Field(description="Vendor or seller name")
    items: List[Dict[str, Any]] = Field(description="List of invoice line items with description, quantity, price")
    subtotal: float = Field(description="Subtotal amount before tax")
    tax: float = Field(description="Tax amount")
    total: float = Field(description="Total amount including tax")


class Resume_Entities(BaseModel):
    """Structured entities extracted from resume documents."""
    name: str = Field(description="Full name of the candidate")
    email: str = Field(description="Email address")
    phone: str = Field(description="Phone number")
    skills: List[str] = Field(description="List of technical and professional skills")
    experience: List[Dict[str, Any]] = Field(description="List of work experience entries with company, role, duration")
    education: List[Dict[str, Any]] = Field(description="List of education entries with institution, degree, year")


class Contract_Entities(BaseModel):
    """Structured entities extracted from contract documents."""
    parties: List[str] = Field(description="List of parties involved in the contract")
    effective_date: str = Field(description="Contract effective date")
    terms: List[str] = Field(description="Key terms and conditions")
    value: Optional[float] = Field(description="Contract value or payment amount", default=None)
    duration: str = Field(description="Contract duration or term")


class Letter_Entities(BaseModel):
    """Structured entities extracted from letter documents."""
    sender: str = Field(description="Sender name and address")
    recipient: str = Field(description="Recipient name and address")
    date: str = Field(description="Letter date")
    subject: str = Field(description="Letter subject or topic")
    body_summary: str = Field(description="Summary of letter body content")


class Report_Entities(BaseModel):
    """Structured entities extracted from report documents."""
    title: str = Field(description="Report title")
    author: str = Field(description="Report author name")
    date: str = Field(description="Report date")
    sections: List[str] = Field(description="List of report section headings")
    summary: str = Field(description="Executive summary or overview")
    findings: List[str] = Field(description="Key findings or conclusions")


# ============================================================================
# Tool Functions
# ============================================================================

@tool
def Ingest_Document(text: str, filename: str) -> Dict[str, Any]:
    """
    Parse and ingest document text, detecting format and cleaning content.
    
    Args:
        text: Raw document text content
        filename: Name of the source file
        
    Returns:
        Dictionary containing parsed document data with fields:
        - cleaned_text: Normalized and cleaned text
        - format: Detected document format
        - word_count: Number of words
        - character_count: Number of characters
        - detected_language: Detected language (if applicable)
    """
    # Clean and normalize text
    cleaned_text = text.strip()
    
    # Remove excessive whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
    
    # Detect format based on content patterns
    format_type = "plain_text"
    if re.search(r'<html|<body|<div', cleaned_text, re.IGNORECASE):
        format_type = "html"
    elif re.search(r'^#+\s|^\*\s|^-\s', cleaned_text, re.MULTILINE):
        format_type = "markdown"
    elif re.search(r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}', cleaned_text):
        format_type = "structured"
    
    # Basic language detection (simplified)
    detected_language = "en"
    if re.search(r'[àáâãäåæçèéêë]', cleaned_text, re.IGNORECASE):
        detected_language = "fr"
    elif re.search(r'[äöüß]', cleaned_text, re.IGNORECASE):
        detected_language = "de"
    elif re.search(r'[ñáéíóúü]', cleaned_text, re.IGNORECASE):
        detected_language = "es"
    
    word_count = len(cleaned_text.split())
    character_count = len(cleaned_text)
    
    return {
        "cleaned_text": cleaned_text,
        "format": format_type,
        "word_count": word_count,
        "character_count": character_count,
        "detected_language": detected_language,
        "filename": filename,
        "ingestion_timestamp": datetime.now().isoformat()
    }


@tool
def Classify_Document_Type(text: str) -> str:
    """
    Classify document into predefined categories using pattern matching.
    
    Categories: invoice, resume, contract, letter, report
    
    Args:
        text: Document text content
        
    Returns:
        Document category name (invoice, resume, contract, letter, report)
    """
    text_lower = text.lower()
    
    # Invoice patterns
    invoice_keywords = ["invoice", "invoice number", "bill to", "ship to", "subtotal", "total due", "payment terms"]
    if any(keyword in text_lower for keyword in invoice_keywords):
        if "invoice" in text_lower or "bill" in text_lower:
            return "invoice"
    
    # Resume patterns
    resume_keywords = ["resume", "curriculum vitae", "cv", "work experience", "education", "skills", "objective", "summary"]
    if any(keyword in text_lower for keyword in resume_keywords):
        if "resume" in text_lower or "curriculum" in text_lower or "work experience" in text_lower:
            return "resume"
    
    # Contract patterns
    contract_keywords = ["contract", "agreement", "terms and conditions", "party", "effective date", "whereas", "now therefore"]
    if any(keyword in text_lower for keyword in contract_keywords):
        if "contract" in text_lower or "agreement" in text_lower:
            return "contract"
    
    # Letter patterns
    letter_keywords = ["dear", "sincerely", "yours truly", "regards", "to whom it may concern", "subject:"]
    if any(keyword in text_lower for keyword in letter_keywords):
        if "dear" in text_lower or "sincerely" in text_lower:
            return "letter"
    
    # Report patterns
    report_keywords = ["report", "executive summary", "findings", "recommendations", "conclusion", "analysis"]
    if any(keyword in text_lower for keyword in report_keywords):
        if "report" in text_lower or "executive summary" in text_lower:
            return "report"
    
    # Default classification based on structure
    if len(text.split()) < 100:
        return "letter"
    elif "section" in text_lower or "chapter" in text_lower:
        return "report"
    else:
        return "report"


@tool
def Extract_Entities_From_Text(text: str, entity_types: List[str]) -> Dict[str, Any]:
    """
    Extract named entities from text based on specified entity types.
    
    Args:
        text: Document text content
        entity_types: List of entity types to extract (e.g., ["PERSON", "DATE", "MONEY"])
        
    Returns:
        Dictionary mapping entity types to lists of extracted entities
    """
    extracted_entities = {entity_type: [] for entity_type in entity_types}
    
    # Extract dates
    if "DATE" in entity_types:
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',
            r'\d{2}/\d{2}/\d{4}',
            r'\d{2}-\d{2}-\d{4}',
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}'
        ]
        for pattern in date_patterns:
            dates = re.findall(pattern, text, re.IGNORECASE)
            extracted_entities["DATE"].extend(dates)
        extracted_entities["DATE"] = list(set(extracted_entities["DATE"]))
    
    # Extract money amounts
    if "MONEY" in entity_types:
        money_patterns = [
            r'\$[\d,]+\.?\d*',
            r'[\d,]+\.?\d*\s*(?:dollars|USD|EUR|GBP)',
            r'[\d,]+\.?\d*\s*(?:dollars|USD|EUR|GBP)'
        ]
        for pattern in money_patterns:
            amounts = re.findall(pattern, text, re.IGNORECASE)
            extracted_entities["MONEY"].extend(amounts)
        extracted_entities["MONEY"] = list(set(extracted_entities["MONEY"]))
    
    # Extract email addresses
    if "EMAIL" in entity_types:
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        extracted_entities["EMAIL"] = list(set(emails))
    
    # Extract phone numbers
    if "PHONE" in entity_types:
        phone_patterns = [
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}'
        ]
        for pattern in phone_patterns:
            phones = re.findall(pattern, text)
            extracted_entities["PHONE"].extend(phones)
        extracted_entities["PHONE"] = list(set(extracted_entities["PHONE"]))
    
    # Extract organization names (simplified)
    if "ORG" in entity_types:
        org_keywords = ["Inc.", "LLC", "Corp", "Corporation", "Ltd", "Company", "Co."]
        orgs = []
        for keyword in org_keywords:
            pattern = r'\b[A-Z][A-Za-z\s]+' + re.escape(keyword)
            matches = re.findall(pattern, text)
            orgs.extend(matches)
        extracted_entities["ORG"] = list(set(orgs))
    
    return extracted_entities


@tool
def Validate_Extracted_Data(data: Dict[str, Any], doc_type: str) -> Dict[str, Any]:
    """
    Validate completeness and consistency of extracted data based on document type.
    
    Args:
        data: Dictionary containing extracted entities
        doc_type: Document type (invoice, resume, contract, letter, report)
        
    Returns:
        Dictionary with validation results:
        - is_valid: Boolean indicating if data is valid
        - missing_fields: List of required fields that are missing
        - inconsistencies: List of detected inconsistencies
        - completeness_score: Float between 0 and 1
    """
    missing_fields = []
    inconsistencies = []
    
    doc_type_lower = doc_type.lower()
    
    if doc_type_lower == "invoice":
        required_fields = ["invoice_number", "date", "vendor", "items", "total"]
        for field in required_fields:
            if field not in data or not data[field]:
                missing_fields.append(field)
        
        # Check for consistency
        if "subtotal" in data and "tax" in data and "total" in data:
            calculated_total = data.get("subtotal", 0) + data.get("tax", 0)
            if abs(calculated_total - data.get("total", 0)) > 0.01:
                inconsistencies.append("Total does not match subtotal + tax")
        
        if "items" in data and isinstance(data["items"], list):
            if len(data["items"]) == 0:
                inconsistencies.append("Invoice has no line items")
    
    elif doc_type_lower == "resume":
        required_fields = ["name", "email", "skills"]
        for field in required_fields:
            if field not in data or not data[field]:
                missing_fields.append(field)
        
        # Validate email format
        if "email" in data and data["email"]:
            email_pattern = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$'
            if not re.match(email_pattern, data["email"]):
                inconsistencies.append("Invalid email format")
    
    elif doc_type_lower == "contract":
        required_fields = ["parties", "effective_date", "terms"]
        for field in required_fields:
            if field not in data or not data[field]:
                missing_fields.append(field)
        
        if "parties" in data and isinstance(data["parties"], list):
            if len(data["parties"]) < 2:
                inconsistencies.append("Contract should have at least 2 parties")
    
    elif doc_type_lower == "letter":
        required_fields = ["sender", "recipient", "date", "subject"]
        for field in required_fields:
            if field not in data or not data[field]:
                missing_fields.append(field)
    
    elif doc_type_lower == "report":
        required_fields = ["title", "author", "date", "sections"]
        for field in required_fields:
            if field not in data or not data[field]:
                missing_fields.append(field)
    
    # Calculate completeness score
    all_fields = list(data.keys())
    if doc_type_lower == "invoice":
        expected_fields = 7
    elif doc_type_lower == "resume":
        expected_fields = 6
    elif doc_type_lower == "contract":
        expected_fields = 5
    elif doc_type_lower == "letter":
        expected_fields = 5
    elif doc_type_lower == "report":
        expected_fields = 6
    else:
        expected_fields = len(all_fields)
    
    completeness_score = max(0.0, min(1.0, len(all_fields) / max(expected_fields, 1)))
    
    is_valid = len(missing_fields) == 0 and len(inconsistencies) == 0
    
    return {
        "is_valid": is_valid,
        "missing_fields": missing_fields,
        "inconsistencies": inconsistencies,
        "completeness_score": completeness_score,
        "validation_timestamp": datetime.now().isoformat()
    }


# ============================================================================
# Document Store Class
# ============================================================================

class Document_Store:
    """Class for storing and retrieving processed documents."""
    
    def __init__(self, storage_directory: str = "./processed_documents"):
        """
        Initialize document store.
        
        Args:
            storage_directory: Directory path for storing documents
        """
        self.storage_directory = storage_directory
        os.makedirs(storage_directory, exist_ok=True)
    
    def Store_Document(
        self,
        document_id: str,
        filename: str,
        doc_type: str,
        extracted_entities: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store processed document with metadata.
        
        Args:
            document_id: Unique identifier for the document
            filename: Original filename
            doc_type: Document type classification
            extracted_entities: Extracted structured entities
            metadata: Additional metadata dictionary
            
        Returns:
            Path to stored document file
        """
        document_data = {
            "document_id": document_id,
            "filename": filename,
            "doc_type": doc_type,
            "extracted_entities": extracted_entities,
            "metadata": metadata or {},
            "stored_at": datetime.now().isoformat()
        }
        
        # Create filename based on document_id
        storage_filename = f"{document_id}_{doc_type}.json"
        storage_path = os.path.join(self.storage_directory, storage_filename)
        
        # Write to JSON file
        with open(storage_path, 'w', encoding='utf-8') as f:
            json.dump(document_data, f, indent=2, ensure_ascii=False)
        
        return storage_path
    
    def Retrieve_Document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve stored document by ID.
        
        Args:
            document_id: Unique identifier for the document
            
        Returns:
            Document data dictionary or None if not found
        """
        # Search for document file
        for filename in os.listdir(self.storage_directory):
            if filename.startswith(document_id):
                file_path = os.path.join(self.storage_directory, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        return None
    
    def List_Documents(self, doc_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all stored documents, optionally filtered by type.
        
        Args:
            doc_type: Optional document type filter
            
        Returns:
            List of document metadata dictionaries
        """
        documents = []
        
        for filename in os.listdir(self.storage_directory):
            if filename.endswith('.json'):
                file_path = os.path.join(self.storage_directory, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        doc_data = json.load(f)
                        if doc_type is None or doc_data.get("doc_type") == doc_type:
                            documents.append({
                                "document_id": doc_data.get("document_id"),
                                "filename": doc_data.get("filename"),
                                "doc_type": doc_data.get("doc_type"),
                                "stored_at": doc_data.get("stored_at")
                            })
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
                    continue
        
        return documents
