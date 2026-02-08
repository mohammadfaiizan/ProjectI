"""
Document Processing System Implementation
A complete document processing system that ingests documents, classifies them,
extracts structured data, validates the data, and routes documents for processing.
"""

import os
import json
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from openai import OpenAI


@dataclass
class Document:
    """Data model for documents."""
    content: str
    type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    validation_status: str = "pending"
    routing_target: Optional[str] = None
    document_id: str = field(default_factory=lambda: f"doc_{datetime.now().timestamp()}")


class Document_Ingester:
    """Handles document ingestion from various sources."""
    
    def __init__(self):
        self.supported_formats = ["txt", "pdf", "email", "image"]
    
    def read_text_file(self, file_path: str) -> str:
        """Read content from a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
    
    def parse_pdf(self, file_path: str) -> str:
        """Simulate PDF parsing - in production, use pdfplumber or PyPDF2."""
        # Simulated PDF extraction
        return f"[PDF Content from {file_path}] This is simulated PDF content. In production, use a PDF parsing library."
    
    def parse_email(self, email_content: str) -> Dict[str, Any]:
        """Parse email content and extract text."""
        # Simple email parsing simulation
        lines = email_content.split('\n')
        subject = ""
        body = []
        in_body = False
        
        for line in lines:
            if line.startswith('Subject:'):
                subject = line.replace('Subject:', '').strip()
            elif line.startswith('Body:') or in_body:
                in_body = True
                if not line.startswith('Body:'):
                    body.append(line)
        
        return {
            'subject': subject,
            'body': '\n'.join(body),
            'full_text': email_content
        }
    
    def parse_image(self, file_path: str) -> str:
        """Simulate OCR from images - in production, use Tesseract or cloud OCR."""
        return f"[OCR Content from {file_path}] This is simulated OCR content. In production, use Tesseract or cloud OCR services."
    
    def normalize_content(self, content: str) -> str:
        """Normalize document content."""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        # Remove special characters but keep basic punctuation
        content = content.strip()
        return content
    
    def ingest_document(self, source: str, format_type: str = "txt") -> Document:
        """Ingest a document from a source."""
        if format_type == "txt":
            content = self.read_text_file(source)
        elif format_type == "pdf":
            content = self.parse_pdf(source)
        elif format_type == "email":
            parsed = self.parse_email(source)
            content = parsed['full_text']
        elif format_type == "image":
            content = self.parse_image(source)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        content = self.normalize_content(content)
        
        metadata = {
            'source': source,
            'format': format_type,
            'ingestion_timestamp': datetime.now().isoformat(),
            'content_length': len(content)
        }
        
        return Document(content=content, metadata=metadata)


class Document_Classifier:
    """Classifies documents using LLM-based classification."""
    
    def __init__(self, client: OpenAI):
        self.client = client
        self.supported_types = ["invoice", "contract", "resume", "letter", "report", "form", "other"]
    
    def classify_document(self, document: Document) -> Dict[str, Any]:
        """Classify a document into one of the supported types."""
        classification_prompt = f"""Classify the following document into one of these categories: invoice, contract, resume, letter, report, form, or other.

Document Content:
{document.content[:2000]}

Respond with JSON containing:
- "type": one of the categories above
- "confidence": a number between 0 and 1 indicating confidence
- "reasoning": brief explanation of the classification

JSON Response:"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a document classification expert. Always respond with valid JSON."},
                    {"role": "user", "content": classification_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            document.type = result.get("type", "other")
            
            return {
                "type": document.type,
                "confidence": result.get("confidence", 0.5),
                "reasoning": result.get("reasoning", "")
            }
        except Exception as e:
            document.type = "other"
            return {
                "type": "other",
                "confidence": 0.0,
                "reasoning": f"Classification error: {str(e)}"
            }


class Entity_Extractor:
    """Extracts structured data from documents based on type."""
    
    def __init__(self, client: OpenAI):
        self.client = client
    
    def extract_entities(self, document: Document) -> Dict[str, Any]:
        """Extract entities based on document type."""
        if not document.type:
            return {}
        
        if document.type == "invoice":
            return self._extract_invoice_entities(document)
        elif document.type == "resume":
            return self._extract_resume_entities(document)
        elif document.type == "contract":
            return self._extract_contract_entities(document)
        elif document.type == "letter":
            return self._extract_letter_entities(document)
        elif document.type == "report":
            return self._extract_report_entities(document)
        else:
            return {}
    
    def _extract_invoice_entities(self, document: Document) -> Dict[str, Any]:
        """Extract invoice-specific entities."""
        extraction_prompt = f"""Extract structured data from this invoice document. Extract the following fields:
- vendor: vendor name
- invoice_number: invoice number or ID
- date: invoice date
- due_date: payment due date
- line_items: list of items with description, quantity, unit_price, total
- subtotal: subtotal amount before tax
- tax: tax amount
- total: total amount
- currency: currency code (USD, EUR, etc.)

Document Content:
{document.content[:3000]}

Respond with JSON containing all extracted fields. Use null for missing fields.

JSON Response:"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at extracting structured data from invoices. Always respond with valid JSON."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": str(e)}
    
    def _extract_resume_entities(self, document: Document) -> Dict[str, Any]:
        """Extract resume-specific entities."""
        extraction_prompt = f"""Extract structured data from this resume document. Extract the following fields:
- name: candidate full name
- email: email address
- phone: phone number
- skills: list of skills
- experience: list of work experience entries with company, role, duration, responsibilities
- education: list of education entries with institution, degree, year
- certifications: list of certifications

Document Content:
{document.content[:3000]}

Respond with JSON containing all extracted fields. Use null for missing fields.

JSON Response:"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at extracting structured data from resumes. Always respond with valid JSON."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": str(e)}
    
    def _extract_contract_entities(self, document: Document) -> Dict[str, Any]:
        """Extract contract-specific entities."""
        extraction_prompt = f"""Extract structured data from this contract document. Extract the following fields:
- parties: list of contracting parties
- effective_date: contract effective date
- expiration_date: contract expiration date
- terms: key terms and conditions
- obligations: obligations for each party
- signatures_required: whether signatures are required
- renewal_clauses: renewal or termination clauses

Document Content:
{document.content[:3000]}

Respond with JSON containing all extracted fields. Use null for missing fields.

JSON Response:"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at extracting structured data from contracts. Always respond with valid JSON."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": str(e)}
    
    def _extract_letter_entities(self, document: Document) -> Dict[str, Any]:
        """Extract letter-specific entities."""
        extraction_prompt = f"""Extract structured data from this letter document. Extract the following fields:
- sender: sender name and information
- recipient: recipient name and information
- date: letter date
- subject: subject line
- key_points: main points or action items
- tone: tone of the letter (formal, informal, etc.)

Document Content:
{document.content[:3000]}

Respond with JSON containing all extracted fields. Use null for missing fields.

JSON Response:"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at extracting structured data from letters. Always respond with valid JSON."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": str(e)}
    
    def _extract_report_entities(self, document: Document) -> Dict[str, Any]:
        """Extract report-specific entities."""
        extraction_prompt = f"""Extract structured data from this report document. Extract the following fields:
- title: report title
- author: author name
- date: report date
- executive_summary: executive summary or abstract
- sections: list of main sections
- conclusions: key conclusions
- recommendations: recommendations if any

Document Content:
{document.content[:3000]}

Respond with JSON containing all extracted fields. Use null for missing fields.

JSON Response:"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at extracting structured data from reports. Always respond with valid JSON."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": str(e)}


class Data_Validator:
    """Validates extracted data for accuracy and completeness."""
    
    def __init__(self):
        self.required_fields = {
            "invoice": ["vendor", "invoice_number", "date", "total"],
            "resume": ["name", "email"],
            "contract": ["parties", "effective_date"],
            "letter": ["sender", "recipient", "date"],
            "report": ["title", "author", "date"]
        }
    
    def validate_data(self, document: Document) -> Dict[str, Any]:
        """Validate extracted data."""
        if not document.type or document.type not in self.required_fields:
            return {
                "status": "skipped",
                "errors": [],
                "warnings": ["Unknown document type"]
            }
        
        errors = []
        warnings = []
        
        # Check required fields
        required = self.required_fields.get(document.type, [])
        for field in required:
            if field not in document.extracted_data or document.extracted_data[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Type-specific validation
        if document.type == "invoice":
            errors.extend(self._validate_invoice(document.extracted_data))
        elif document.type == "resume":
            errors.extend(self._validate_resume(document.extracted_data))
        elif document.type == "contract":
            errors.extend(self._validate_contract(document.extracted_data))
        
        # Format validation
        warnings.extend(self._validate_formats(document.extracted_data, document.type))
        
        status = "valid" if len(errors) == 0 else "invalid"
        document.validation_status = status
        
        return {
            "status": status,
            "errors": errors,
            "warnings": warnings
        }
    
    def _validate_invoice(self, data: Dict[str, Any]) -> List[str]:
        """Validate invoice-specific fields."""
        errors = []
        
        # Validate amounts are numeric
        for field in ["subtotal", "tax", "total"]:
            if field in data and data[field] is not None:
                try:
                    float(str(data[field]).replace('$', '').replace(',', ''))
                except (ValueError, AttributeError):
                    errors.append(f"Invalid {field} format: {data[field]}")
        
        # Validate dates
        if "date" in data and data["date"]:
            if not self._is_valid_date(data["date"]):
                errors.append(f"Invalid date format: {data['date']}")
        
        # Validate line items structure
        if "line_items" in data and isinstance(data["line_items"], list):
            for item in data["line_items"]:
                if not isinstance(item, dict):
                    errors.append("Invalid line_items structure")
                    break
        
        return errors
    
    def _validate_resume(self, data: Dict[str, Any]) -> List[str]:
        """Validate resume-specific fields."""
        errors = []
        
        # Validate email format
        if "email" in data and data["email"]:
            if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', str(data["email"])):
                errors.append(f"Invalid email format: {data['email']}")
        
        # Validate phone format (basic check)
        if "phone" in data and data["phone"]:
            phone = re.sub(r'[^\d]', '', str(data["phone"]))
            if len(phone) < 10:
                errors.append(f"Invalid phone format: {data['phone']}")
        
        return errors
    
    def _validate_contract(self, data: Dict[str, Any]) -> List[str]:
        """Validate contract-specific fields."""
        errors = []
        
        # Validate parties is a list
        if "parties" in data:
            if not isinstance(data["parties"], list) or len(data["parties"]) < 2:
                errors.append("Contract must have at least two parties")
        
        # Validate dates
        for field in ["effective_date", "expiration_date"]:
            if field in data and data[field]:
                if not self._is_valid_date(data[field]):
                    errors.append(f"Invalid {field} format: {data[field]}")
        
        return errors
    
    def _validate_formats(self, data: Dict[str, Any], doc_type: str) -> List[str]:
        """Validate data formats."""
        warnings = []
        
        # Check for common format issues
        for key, value in data.items():
            if value is None:
                continue
            
            if "date" in key.lower():
                if not self._is_valid_date(str(value)):
                    warnings.append(f"Potential date format issue for {key}: {value}")
            
            if "email" in key.lower():
                if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', str(value)):
                    warnings.append(f"Potential email format issue for {key}: {value}")
        
        return warnings
    
    def _is_valid_date(self, date_str: str) -> bool:
        """Basic date validation."""
        # Try common date formats
        date_formats = [
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%B %d, %Y",
            "%d %B %Y"
        ]
        
        for fmt in date_formats:
            try:
                datetime.strptime(date_str, fmt)
                return True
            except ValueError:
                continue
        
        # Check if it's a reasonable date-like string
        if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
            return True
        
        return False


class Routing_Engine:
    """Routes documents to appropriate handlers based on type and data."""
    
    def __init__(self):
        self.routing_rules = {
            "invoice": "accounts_payable_handler",
            "contract": "legal_review_handler",
            "resume": "hr_recruitment_handler",
            "letter": "correspondence_handler",
            "report": "documentation_handler",
            "form": "form_processing_handler",
            "other": "general_handler"
        }
    
    def route_document(self, document: Document) -> Dict[str, Any]:
        """Route a document to appropriate handler."""
        if not document.type:
            target = "general_handler"
            reason = "Document type not classified"
        else:
            target = self.routing_rules.get(document.type, "general_handler")
            reason = f"Routed based on document type: {document.type}"
        
        # Additional routing logic based on extracted data
        if document.type == "invoice":
            # Route high-value invoices to manager approval
            if "total" in document.extracted_data:
                try:
                    total = float(str(document.extracted_data["total"]).replace('$', '').replace(',', ''))
                    if total > 10000:
                        target = "manager_approval_handler"
                        reason = f"High-value invoice (${total:,.2f}) requires manager approval"
                except (ValueError, TypeError):
                    pass
        
        # Route invalid documents to review queue
        if document.validation_status == "invalid":
            target = "review_queue_handler"
            reason = "Document validation failed - requires manual review"
        
        document.routing_target = target
        
        return {
            "target": target,
            "reason": reason
        }


class Storage_Manager:
    """Manages storage of processed documents and extracted data."""
    
    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.extracted_data_index: Dict[str, List[str]] = {}
        self.processing_logs: List[Dict[str, Any]] = []
    
    def store_document(self, document: Document) -> str:
        """Store a document."""
        self.documents[document.document_id] = document
        
        # Index extracted data
        if document.extracted_data:
            for key, value in document.extracted_data.items():
                if key not in self.extracted_data_index:
                    self.extracted_data_index[key] = []
                self.extracted_data_index[key].append(document.document_id)
        
        return document.document_id
    
    def store_extracted_data(self, document_id: str, extracted_data: Dict[str, Any]):
        """Store extracted data for a document."""
        if document_id in self.documents:
            self.documents[document_id].extracted_data = extracted_data
    
    def store_processing_log(self, document_id: str, step: str, result: Dict[str, Any]):
        """Store processing log entry."""
        log_entry = {
            "document_id": document_id,
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
        self.processing_logs.append(log_entry)
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Retrieve a document by ID."""
        return self.documents.get(document_id)
    
    def query_documents(self, filters: Dict[str, Any]) -> List[Document]:
        """Query documents based on filters."""
        results = []
        for doc in self.documents.values():
            match = True
            
            if "type" in filters and doc.type != filters["type"]:
                match = False
            
            if "validation_status" in filters and doc.validation_status != filters["validation_status"]:
                match = False
            
            if "routing_target" in filters and doc.routing_target != filters["routing_target"]:
                match = False
            
            if match:
                results.append(doc)
        
        return results


class Processing_Pipeline:
    """Orchestrates the complete document processing pipeline."""
    
    def __init__(self, client: OpenAI):
        self.ingester = Document_Ingester()
        self.classifier = Document_Classifier(client)
        self.extractor = Entity_Extractor(client)
        self.validator = Data_Validator()
        self.router = Routing_Engine()
        self.storage = Storage_Manager()
    
    def process_document(self, source: str, format_type: str = "txt") -> Document:
        """Process a single document through the full pipeline."""
        # Step 1: Ingest document
        document = self.ingester.ingest_document(source, format_type)
        self.storage.store_processing_log(document.document_id, "ingestion", {"status": "success"})
        
        # Step 2: Classify document
        classification_result = self.classifier.classify_document(document)
        self.storage.store_processing_log(document.document_id, "classification", classification_result)
        
        # Step 3: Extract entities
        extracted_data = self.extractor.extract_entities(document)
        document.extracted_data = extracted_data
        self.storage.store_extracted_data(document.document_id, extracted_data)
        self.storage.store_processing_log(document.document_id, "extraction", {"fields_extracted": len(extracted_data)})
        
        # Step 4: Validate data
        validation_result = self.validator.validate_data(document)
        self.storage.store_processing_log(document.document_id, "validation", validation_result)
        
        # Step 5: Route document
        routing_result = self.router.route_document(document)
        self.storage.store_processing_log(document.document_id, "routing", routing_result)
        
        # Step 6: Store document
        self.storage.store_document(document)
        
        return document
    
    def process_batch(self, documents: List[Dict[str, str]]) -> List[Document]:
        """Process multiple documents."""
        results = []
        for doc_info in documents:
            try:
                document = self.process_document(
                    doc_info["source"],
                    doc_info.get("format", "txt")
                )
                results.append(document)
            except Exception as e:
                print(f"Error processing {doc_info.get('source', 'unknown')}: {str(e)}")
        
        return results
    
    def generate_summary(self, documents: List[Document]) -> Dict[str, Any]:
        """Generate summary of processed documents."""
        summary = {
            "total_documents": len(documents),
            "by_type": {},
            "by_status": {},
            "by_routing": {},
            "validation_stats": {
                "valid": 0,
                "invalid": 0,
                "pending": 0
            }
        }
        
        for doc in documents:
            # Count by type
            doc_type = doc.type or "unknown"
            summary["by_type"][doc_type] = summary["by_type"].get(doc_type, 0) + 1
            
            # Count by validation status
            status = doc.validation_status
            summary["by_status"][status] = summary["by_status"].get(status, 0) + 1
            summary["validation_stats"][status] = summary["validation_stats"].get(status, 0) + 1
            
            # Count by routing target
            target = doc.routing_target or "unknown"
            summary["by_routing"][target] = summary["by_routing"].get(target, 0) + 1
        
        return summary


def main():
    """Main function demonstrating the document processing system."""
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    client = OpenAI(api_key=api_key)
    pipeline = Processing_Pipeline(client)
    
    # Sample documents
    sample_documents = [
        {
            "source": """
Invoice Number: INV-2024-001
Vendor: Acme Corporation
Date: 2024-01-15
Due Date: 2024-02-15

Line Items:
1. Software License - Quantity: 10, Unit Price: $500, Total: $5,000
2. Support Services - Quantity: 1, Unit Price: $1,200, Total: $1,200

Subtotal: $6,200
Tax (8%): $496
Total: $6,696

Payment Terms: Net 30
            """,
            "format": "txt",
            "description": "Sample Invoice"
        },
        {
            "source": """
John Doe
Email: john.doe@email.com
Phone: (555) 123-4567

Skills: Python, Machine Learning, Data Analysis, SQL

Experience:
- Senior Data Scientist at Tech Corp (2020-2024)
  Responsibilities: Developed ML models, analyzed large datasets
- Data Analyst at Analytics Inc (2018-2020)
  Responsibilities: Created dashboards, performed statistical analysis

Education:
- Master of Science in Computer Science, State University (2018)
- Bachelor of Science in Mathematics, State University (2016)

Certifications:
- AWS Certified Machine Learning Specialist
- Google Cloud Professional Data Engineer
            """,
            "format": "txt",
            "description": "Sample Resume"
        },
        {
            "source": """
SERVICE AGREEMENT

This Service Agreement ("Agreement") is entered into on January 1, 2024, between:

Party A: ABC Company, Inc.
Party B: XYZ Services, LLC

Effective Date: January 1, 2024
Expiration Date: December 31, 2024

TERMS AND CONDITIONS:
1. Party A agrees to provide consulting services to Party B.
2. Party B agrees to pay Party A $10,000 per month.
3. This agreement may be renewed for additional one-year terms upon mutual consent.

OBLIGATIONS:
- Party A: Provide monthly consulting reports and attend quarterly meetings.
- Party B: Make timely payments and provide necessary access to systems.

SIGNATURES REQUIRED: Yes

This agreement shall be governed by the laws of the State of California.
            """,
            "format": "txt",
            "description": "Sample Contract"
        },
        {
            "source": """
Subject: Quarterly Business Review

Dear Board of Directors,

I am writing to provide an update on our Q4 2023 performance.

Key Points:
- Revenue increased by 15% compared to Q3
- Customer acquisition exceeded targets by 20%
- New product launch scheduled for Q2 2024

We look forward to discussing these results in detail at the upcoming board meeting.

Best regards,
Jane Smith
CEO
            """,
            "format": "txt",
            "description": "Sample Letter"
        }
    ]
    
    print("=" * 80)
    print("Document Processing System - Processing Sample Documents")
    print("=" * 80)
    print()
    
    processed_documents = []
    
    for i, doc_info in enumerate(sample_documents, 1):
        print(f"Processing Document {i}: {doc_info['description']}")
        print("-" * 80)
        
        # Create temporary file for processing
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(doc_info["source"])
            temp_path = f.name
        
        try:
            document = pipeline.process_document(temp_path, doc_info["format"])
            processed_documents.append(document)
            
            print(f"Document ID: {document.document_id}")
            print(f"Type: {document.type}")
            print(f"Validation Status: {document.validation_status}")
            print(f"Routing Target: {document.routing_target}")
            print(f"Extracted Data Fields: {len(document.extracted_data)}")
            
            if document.extracted_data:
                print("\nExtracted Data:")
                for key, value in list(document.extracted_data.items())[:5]:
                    if isinstance(value, list):
                        print(f"  {key}: {len(value)} items")
                    elif isinstance(value, dict):
                        print(f"  {key}: {len(value)} fields")
                    else:
                        display_value = str(value)[:50]
                        print(f"  {key}: {display_value}")
            
            print()
        except Exception as e:
            print(f"Error: {str(e)}\n")
        finally:
            os.unlink(temp_path)
    
    # Generate summary
    print("=" * 80)
    print("Processing Summary")
    print("=" * 80)
    summary = pipeline.generate_summary(processed_documents)
    print(json.dumps(summary, indent=2))
    print()
    
    # Query examples
    print("=" * 80)
    print("Query Examples")
    print("=" * 80)
    
    invoices = pipeline.storage.query_documents({"type": "invoice"})
    print(f"Found {len(invoices)} invoice documents")
    
    valid_docs = pipeline.storage.query_documents({"validation_status": "valid"})
    print(f"Found {len(valid_docs)} valid documents")
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
