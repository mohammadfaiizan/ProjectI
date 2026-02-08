"""
Agent module for Document Processing System.

This module contains the LangGraph-based document processing agent with stateful
workflow for ingesting, classifying, extracting entities, validating, and storing
documents.
"""

from typing import TypedDict, Annotated, Literal, Dict, Any, Optional, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

from Config import LLM_Config, Classification_Config
from Tools import (
    Ingest_Document,
    Classify_Document_Type,
    Extract_Entities_From_Text,
    Validate_Extracted_Data,
    Document_Store,
    Invoice_Entities,
    Resume_Entities,
    Contract_Entities,
    Letter_Entities,
    Report_Entities
)


# ============================================================================
# State Definition
# ============================================================================

class Processing_State(TypedDict):
    """
    State schema for the document processing agent.
    
    Tracks document text, filename, classification, extracted entities,
    validation results, and processing history throughout the workflow.
    """
    raw_text: str
    filename: str
    doc_type: Optional[str]
    extracted_entities: Optional[Dict[str, Any]]
    validation_result: Optional[Dict[str, Any]]
    is_valid: bool
    processing_history: List[str]
    output: Optional[str]
    retry_count: int


# ============================================================================
# Document Processing Graph
# ============================================================================

class Document_Processing_Graph:
    """
    LangGraph-based document processing system.
    
    Processes documents through stages:
    1. Ingest: Clean and normalize document text
    2. Classify: Determine document type using LLM
    3. Extract_Entities: Extract structured data based on doc_type
    4. Validate: Check completeness and consistency
    5. Route_Output: Route to appropriate handler
    6. Store_Result: Save processed document
    """
    
    def __init__(
        self,
        llm_config: LLM_Config,
        classification_config: Classification_Config,
        document_store: Document_Store
    ):
        """
        Initialize document processing graph.
        
        Args:
            llm_config: LLM configuration instance
            classification_config: Classification configuration instance
            document_store: Document store instance
        """
        self.llm = llm_config.Get_LLM()
        self.classification_config = classification_config
        self.document_store = document_store
        
        # Initialize output parsers for each document type
        self.invoice_parser = PydanticOutputParser(pydantic_object=Invoice_Entities)
        self.resume_parser = PydanticOutputParser(pydantic_object=Resume_Entities)
        self.contract_parser = PydanticOutputParser(pydantic_object=Contract_Entities)
        self.letter_parser = PydanticOutputParser(pydantic_object=Letter_Entities)
        self.report_parser = PydanticOutputParser(pydantic_object=Report_Entities)
        
        # Build the graph
        self.graph = self.Build_Graph()
    
    def Build_Graph(self) -> StateGraph:
        """
        Build and compile the LangGraph workflow.
        
        Returns:
            Compiled StateGraph instance
        """
        workflow = StateGraph(Processing_State)
        
        # Add nodes
        workflow.add_node("Ingest", self.Ingest_Node)
        workflow.add_node("Classify", self.Classify_Node)
        workflow.add_node("Extract_Entities", self.Extract_Entities_Node)
        workflow.add_node("Validate", self.Validate_Node)
        workflow.add_node("Re_Extract", self.Re_Extract_Node)
        workflow.add_node("Route_Output", self.Route_Output_Node)
        workflow.add_node("Store_Result", self.Store_Result_Node)
        
        # Set entry point
        workflow.set_entry_point("Ingest")
        
        # Add edges
        workflow.add_edge("Ingest", "Classify")
        workflow.add_edge("Classify", "Extract_Entities")
        workflow.add_edge("Extract_Entities", "Validate")
        
        # Conditional edge from Validate
        workflow.add_conditional_edges(
            "Validate",
            self.Should_Re_Extract,
            {
                "valid": "Route_Output",
                "invalid": "Re_Extract"
            }
        )
        
        workflow.add_edge("Route_Output", "Store_Result")
        workflow.add_edge("Re_Extract", "Extract_Entities")
        workflow.add_edge("Store_Result", END)
        
        return workflow.compile()
    
    def Ingest_Node(self, state: Processing_State) -> Processing_State:
        """
        Ingest and clean document text.
        
        Args:
            state: Current processing state
            
        Returns:
            Updated state with cleaned text
        """
        history = state.get("processing_history", [])
        history.append("Starting document ingestion")
        
        # Use Ingest_Document tool
        ingestion_result = Ingest_Document.invoke({
            "text": state["raw_text"],
            "filename": state["filename"]
        })
        
        # Update state with cleaned text
        updated_state = {
            **state,
            "raw_text": ingestion_result["cleaned_text"],
            "processing_history": history
        }
        
        history.append(f"Ingested document: {ingestion_result['word_count']} words, format: {ingestion_result['format']}")
        
        return updated_state
    
    def Classify_Node(self, state: Processing_State) -> Processing_State:
        """
        Classify document type using LLM.
        
        Args:
            state: Current processing state
            
        Returns:
            Updated state with document type classification
        """
        history = state.get("processing_history", [])
        history.append("Classifying document type")
        
        # First try pattern-based classification
        doc_type = Classify_Document_Type.invoke({"text": state["raw_text"]})
        
        # Use LLM for more accurate classification if needed
        categories = self.classification_config.Get_Categories()
        categories_str = ", ".join(categories)
        
        classification_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a document classification expert. Classify the given document 
            into one of these categories: {categories_str}.
            
            Return only the category name, nothing else."""),
            ("human", "Document text:\n\n{text}\n\nCategory:")
        ])
        
        classification_chain = classification_prompt | self.llm
        llm_response = classification_chain.invoke({"text": state["raw_text"]})
        
        # Extract category from LLM response
        llm_category = llm_response.content.strip().lower()
        
        # Validate and use LLM result if it's a valid category, otherwise use pattern result
        if self.classification_config.Is_Valid_Category(llm_category):
            doc_type = llm_category
        elif self.classification_config.Is_Valid_Category(doc_type):
            pass  # Use pattern-based result
        else:
            doc_type = "report"  # Default fallback
        
        history.append(f"Classified as: {doc_type}")
        
        return {
            **state,
            "doc_type": doc_type,
            "processing_history": history
        }
    
    def Extract_Entities_Node(self, state: Processing_State) -> Processing_State:
        """
        Extract structured entities based on document type.
        
        Args:
            state: Current processing state
            
        Returns:
            Updated state with extracted entities
        """
        history = state.get("processing_history", [])
        doc_type = state.get("doc_type", "report")
        history.append(f"Extracting entities for {doc_type}")
        
        # Select appropriate parser and prompt based on document type
        if doc_type == "invoice":
            parser = self.invoice_parser
            extraction_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert at extracting structured information from invoices.
                Extract all relevant invoice information including invoice number, date, vendor,
                line items with quantities and prices, subtotal, tax, and total amount.
                
                {format_instructions}"""),
                ("human", "Invoice text:\n\n{text}")
            ])
        elif doc_type == "resume":
            parser = self.resume_parser
            extraction_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert at extracting structured information from resumes.
                Extract candidate name, contact information (email, phone), skills, work experience,
                and education details.
                
                {format_instructions}"""),
                ("human", "Resume text:\n\n{text}")
            ])
        elif doc_type == "contract":
            parser = self.contract_parser
            extraction_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert at extracting structured information from contracts.
                Extract parties involved, effective date, key terms and conditions, contract value,
                and duration.
                
                {format_instructions}"""),
                ("human", "Contract text:\n\n{text}")
            ])
        elif doc_type == "letter":
            parser = self.letter_parser
            extraction_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert at extracting structured information from letters.
                Extract sender information, recipient information, date, subject, and body summary.
                
                {format_instructions}"""),
                ("human", "Letter text:\n\n{text}")
            ])
        else:  # report
            parser = self.report_parser
            extraction_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert at extracting structured information from reports.
                Extract report title, author, date, section headings, executive summary, and key findings.
                
                {format_instructions}"""),
                ("human", "Report text:\n\n{text}")
            ])
        
        # Create extraction chain
        extraction_chain = extraction_prompt | self.llm | parser
        
        try:
            extracted_data = extraction_chain.invoke({
                "text": state["raw_text"],
                "format_instructions": parser.get_format_instructions()
            })
            
            # Convert Pydantic model to dictionary
            if hasattr(extracted_data, 'model_dump'):
                entities_dict = extracted_data.model_dump()
            else:
                entities_dict = extracted_data.dict()
            
            history.append(f"Successfully extracted {len(entities_dict)} entity fields")
            
        except Exception as e:
            history.append(f"Extraction error: {str(e)}")
            entities_dict = {}
        
        return {
            **state,
            "extracted_entities": entities_dict,
            "processing_history": history
        }
    
    def Validate_Node(self, state: Processing_State) -> Processing_State:
        """
        Validate extracted data for completeness and consistency.
        
        Args:
            state: Current processing state
            
        Returns:
            Updated state with validation results
        """
        history = state.get("processing_history", [])
        history.append("Validating extracted data")
        
        doc_type = state.get("doc_type", "report")
        extracted_entities = state.get("extracted_entities", {})
        
        validation_result = Validate_Extracted_Data.invoke({
            "data": extracted_entities,
            "doc_type": doc_type
        })
        
        is_valid = validation_result.get("is_valid", False)
        missing_fields = validation_result.get("missing_fields", [])
        inconsistencies = validation_result.get("inconsistencies", [])
        
        if is_valid:
            history.append("Validation passed")
        else:
            history.append(f"Validation failed: missing fields: {missing_fields}, inconsistencies: {inconsistencies}")
        
        return {
            **state,
            "validation_result": validation_result,
            "is_valid": is_valid,
            "processing_history": history
        }
    
    def Re_Extract_Node(self, state: Processing_State) -> Processing_State:
        """
        Re-extract entities with more specific prompt when validation fails.
        
        Args:
            state: Current processing state
            
        Returns:
            Updated state with re-extracted entities
        """
        history = state.get("processing_history", [])
        retry_count = state.get("retry_count", 0)
        
        if retry_count >= 2:
            history.append("Max retries reached, proceeding with current extraction")
            return {
                **state,
                "is_valid": True,  # Force proceed
                "processing_history": history
            }
        
        history.append(f"Re-extracting entities (attempt {retry_count + 1})")
        
        doc_type = state.get("doc_type", "report")
        validation_result = state.get("validation_result", {})
        missing_fields = validation_result.get("missing_fields", [])
        
        # Create enhanced prompt with missing fields highlighted
        if doc_type == "invoice":
            parser = self.invoice_parser
            enhanced_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are an expert at extracting structured information from invoices.
                Pay special attention to these fields that were missing: {', '.join(missing_fields)}.
                Extract all relevant invoice information including invoice number, date, vendor,
                line items with quantities and prices, subtotal, tax, and total amount.
                
                {parser.get_format_instructions()}"""),
                ("human", "Invoice text:\n\n{text}\n\nPlease ensure all fields are extracted, especially: {missing_fields}")
            ])
        elif doc_type == "resume":
            parser = self.resume_parser
            enhanced_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are an expert at extracting structured information from resumes.
                Pay special attention to these fields that were missing: {', '.join(missing_fields)}.
                Extract candidate name, contact information (email, phone), skills, work experience,
                and education details.
                
                {parser.get_format_instructions()}"""),
                ("human", "Resume text:\n\n{text}\n\nPlease ensure all fields are extracted, especially: {missing_fields}")
            ])
        else:
            # For other types, use similar pattern
            return self.Extract_Entities_Node(state)
        
        extraction_chain = enhanced_prompt | self.llm | parser
        
        try:
            extracted_data = extraction_chain.invoke({
                "text": state["raw_text"],
                "missing_fields": ', '.join(missing_fields)
            })
            
            if hasattr(extracted_data, 'model_dump'):
                entities_dict = extracted_data.model_dump()
            else:
                entities_dict = extracted_data.dict()
            
            history.append("Re-extraction completed")
            
        except Exception as e:
            history.append(f"Re-extraction error: {str(e)}")
            entities_dict = state.get("extracted_entities", {})
        
        return {
            **state,
            "extracted_entities": entities_dict,
            "retry_count": retry_count + 1,
            "processing_history": history
        }
    
    def Route_Output_Node(self, state: Processing_State) -> Processing_State:
        """
        Route output to appropriate handler based on document type.
        
        Args:
            state: Current processing state
            
        Returns:
            Updated state with routing information
        """
        history = state.get("processing_history", [])
        doc_type = state.get("doc_type", "report")
        
        history.append(f"Routing output for {doc_type} document")
        
        # Create output summary
        extracted_entities = state.get("extracted_entities", {})
        output_summary = f"Processed {doc_type} document: {state['filename']}\n"
        output_summary += f"Extracted {len(extracted_entities)} entity fields\n"
        
        # Add key fields based on document type
        if doc_type == "invoice" and "total" in extracted_entities:
            output_summary += f"Total amount: ${extracted_entities.get('total', 0)}\n"
        elif doc_type == "resume" and "name" in extracted_entities:
            output_summary += f"Candidate: {extracted_entities.get('name', 'N/A')}\n"
        elif doc_type == "contract" and "parties" in extracted_entities:
            output_summary += f"Parties: {', '.join(extracted_entities.get('parties', []))}\n"
        
        return {
            **state,
            "output": output_summary,
            "processing_history": history
        }
    
    def Store_Result_Node(self, state: Processing_State) -> Processing_State:
        """
        Store processed document with metadata.
        
        Args:
            state: Current processing state
            
        Returns:
            Updated state with storage information
        """
        history = state.get("processing_history", [])
        history.append("Storing processed document")
        
        # Generate document ID
        import hashlib
        doc_id = hashlib.md5(
            f"{state['filename']}_{state.get('doc_type', 'unknown')}".encode()
        ).hexdigest()[:12]
        
        # Store document
        metadata = {
            "processing_history": history,
            "validation_result": state.get("validation_result"),
            "is_valid": state.get("is_valid", False)
        }
        
        storage_path = self.document_store.Store_Document(
            document_id=doc_id,
            filename=state["filename"],
            doc_type=state.get("doc_type", "unknown"),
            extracted_entities=state.get("extracted_entities", {}),
            metadata=metadata
        )
        
        history.append(f"Document stored at: {storage_path}")
        
        return {
            **state,
            "processing_history": history
        }
    
    def Should_Re_Extract(self, state: Processing_State) -> Literal["valid", "invalid"]:
        """
        Determine if re-extraction is needed based on validation results.
        
        Args:
            state: Current processing state
            
        Returns:
            "valid" if data is valid, "invalid" if re-extraction needed
        """
        is_valid = state.get("is_valid", False)
        retry_count = state.get("retry_count", 0)
        
        if is_valid or retry_count >= 2:
            return "valid"
        else:
            return "invalid"
    
    def Process_Document(self, text: str, filename: str) -> Dict[str, Any]:
        """
        Process a document through the complete workflow.
        
        Args:
            text: Raw document text content
            filename: Name of the source file
            
        Returns:
            Dictionary containing processing results
        """
        initial_state: Processing_State = {
            "raw_text": text,
            "filename": filename,
            "doc_type": None,
            "extracted_entities": None,
            "validation_result": None,
            "is_valid": False,
            "processing_history": [],
            "output": None,
            "retry_count": 0
        }
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        return {
            "doc_type": final_state.get("doc_type"),
            "extracted_entities": final_state.get("extracted_entities"),
            "validation_result": final_state.get("validation_result"),
            "is_valid": final_state.get("is_valid"),
            "output": final_state.get("output"),
            "processing_history": final_state.get("processing_history", [])
        }
