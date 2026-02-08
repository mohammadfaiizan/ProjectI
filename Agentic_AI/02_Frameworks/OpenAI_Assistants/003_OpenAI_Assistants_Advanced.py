"""
OpenAI Assistants API Advanced Examples
========================================

This module demonstrates advanced usage of OpenAI's Assistants API including
assistant creation, thread management, code interpreter, file search, streaming,
and multi-tool assistants. Each section is standalone and can be run independently.

Sections:
1. Create Assistant
2. Thread Management
3. Code Interpreter
4. File Search
5. Streaming Runs
6. Multi-Tool Assistant
7. Run Management
8. Cleanup
"""

import json
import time
import os
from typing import List, Dict, Any, Optional
from openai import OpenAI


# Initialize OpenAI client
client = OpenAI(api_key="your-api-key-here")  # Replace with your API key


# ============================================================================
# SECTION 1: Create Assistant
# ============================================================================

def create_assistant():
    """
    Demonstrates creating an assistant with various configurations.
    """
    print("=" * 70)
    print("SECTION 1: Create Assistant")
    print("=" * 70)
    
    # Create a basic assistant
    assistant = client.beta.assistants.create(
        name="Data Analyst Assistant",
        instructions="You are a helpful data analyst assistant. You can analyze data, "
                    "create visualizations, and answer questions about datasets.",
        model="gpt-4-turbo-preview",
        tools=[{"type": "code_interpreter"}],
        temperature=0.7
    )
    
    print(f"\nAssistant created:")
    print(f"  ID: {assistant.id}")
    print(f"  Name: {assistant.name}")
    print(f"  Model: {assistant.model}")
    print(f"  Tools: {[tool.type for tool in assistant.tools]}")
    print(f"  Created: {assistant.created_at}")
    
    # Create assistant with function calling
    assistant_with_functions = client.beta.assistants.create(
        name="Multi-Tool Assistant",
        instructions="You are a versatile assistant with access to multiple tools.",
        model="gpt-4-turbo-preview",
        tools=[
            {"type": "code_interpreter"},
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ],
        temperature=0.5
    )
    
    print(f"\nMulti-tool assistant created:")
    print(f"  ID: {assistant_with_functions.id}")
    print(f"  Tools: {len(assistant_with_functions.tools)} tools")
    
    return assistant, assistant_with_functions


# ============================================================================
# SECTION 2: Thread Management
# ============================================================================

def thread_management():
    """
    Demonstrates creating and managing conversation threads.
    """
    print("\n" + "=" * 70)
    print("SECTION 2: Thread Management")
    print("=" * 70)
    
    # Create an assistant for this example
    assistant = client.beta.assistants.create(
        name="Conversation Assistant",
        instructions="You are a helpful assistant that maintains conversations.",
        model="gpt-4-turbo-preview"
    )
    
    # Create empty thread
    thread_empty = client.beta.threads.create()
    print(f"\nEmpty thread created: {thread_empty.id}")
    
    # Create thread with initial messages
    thread_with_messages = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": "Hello! I'm starting a conversation."
            }
        ]
    )
    print(f"Thread with initial message created: {thread_with_messages.id}")
    
    # Add message to thread
    message = client.beta.threads.messages.create(
        thread_id=thread_empty.id,
        role="user",
        content="What is 2 + 2?"
    )
    print(f"\nMessage added to thread:")
    print(f"  Message ID: {message.id}")
    print(f"  Content: {message.content[0].text.value}")
    
    # Retrieve thread messages
    messages = client.beta.threads.messages.list(thread_id=thread_empty.id)
    print(f"\nThread has {len(messages.data)} messages:")
    for msg in messages.data:
        print(f"  [{msg.role}]: {msg.content[0].text.value}")
    
    # Create a run
    run = client.beta.threads.runs.create(
        thread_id=thread_empty.id,
        assistant_id=assistant.id
    )
    print(f"\nRun created: {run.id}")
    print(f"  Status: {run.status}")
    
    # Wait for completion
    while run.status in ["queued", "in_progress"]:
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_empty.id,
            run_id=run.id
        )
        print(f"  Status: {run.status}")
    
    if run.status == "completed":
        # Retrieve updated messages
        messages = client.beta.threads.messages.list(thread_id=thread_empty.id)
        print(f"\nFinal messages:")
        for msg in messages.data:
            print(f"  [{msg.role}]: {msg.content[0].text.value}")
    
    return thread_empty, assistant


# ============================================================================
# SECTION 3: Code Interpreter
# ============================================================================

def code_interpreter_example():
    """
    Demonstrates using Code Interpreter tool for data analysis and visualization.
    """
    print("\n" + "=" * 70)
    print("SECTION 3: Code Interpreter")
    print("=" * 70)
    
    # Create assistant with Code Interpreter
    assistant = client.beta.assistants.create(
        name="Code Interpreter Assistant",
        instructions="You are a data analyst. When asked to analyze data or create "
                    "visualizations, use the code interpreter to write and execute Python code.",
        model="gpt-4-turbo-preview",
        tools=[{"type": "code_interpreter"}]
    )
    
    print(f"\nAssistant created: {assistant.id}")
    
    # Create thread
    thread = client.beta.threads.create()
    
    # Add message requesting data analysis
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content="Create a simple line chart showing y = x^2 for x from 0 to 10. "
               "Use matplotlib and save it as a file."
    )
    
    print(f"\nUser message: {message.content[0].text.value}")
    
    # Create run
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )
    
    print(f"\nRun started: {run.id}")
    
    # Poll for completion
    while run.status in ["queued", "in_progress"]:
        time.sleep(2)
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
        print(f"  Status: {run.status}")
    
    if run.status == "completed":
        # Retrieve run steps to see code execution
        steps = client.beta.threads.runs.steps.list(
            thread_id=thread.id,
            run_id=run.id
        )
        
        print(f"\nRun steps ({len(steps.data)}):")
        for step in steps.data:
            if step.type == "tool_calls":
                for tool_call in step.step_details.tool_calls:
                    if tool_call.type == "code_interpreter":
                        print(f"\n  Code executed:")
                        print(f"    Input: {tool_call.code_interpreter.input}")
                        if hasattr(tool_call.code_interpreter, 'outputs'):
                            for output in tool_call.code_interpreter.outputs:
                                if output.type == "logs":
                                    print(f"    Output: {output.logs}")
        
        # Get assistant response
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        assistant_message = messages.data[0]
        print(f"\nAssistant response:")
        print(f"  {assistant_message.content[0].text.value}")
        
        # Check for file outputs
        if hasattr(assistant_message.content[0].text, 'annotations'):
            for annotation in assistant_message.content[0].text.annotations:
                if annotation.type == "file_path":
                    print(f"\n  File created: {annotation.file_path.file_id}")
    
    return assistant, thread


# ============================================================================
# SECTION 4: File Search
# ============================================================================

def file_search_example():
    """
    Demonstrates uploading files, creating vector stores, and using file search.
    """
    print("\n" + "=" * 70)
    print("SECTION 4: File Search")
    print("=" * 70)
    
    # Create a sample file to upload
    sample_content = """
    OpenAI Assistants API Documentation
    
    The Assistants API allows you to build AI assistants that can:
    1. Use Code Interpreter to write and run Python code
    2. Search through uploaded files using File Search
    3. Call custom functions you define
    
    File Search uses vector stores to enable semantic search over your documents.
    Files are automatically chunked and embedded for efficient retrieval.
    """
    
    # Save sample file
    sample_file_path = "sample_document.txt"
    with open(sample_file_path, "w") as f:
        f.write(sample_content)
    
    print(f"\nCreated sample file: {sample_file_path}")
    
    # Upload file
    with open(sample_file_path, "rb") as f:
        uploaded_file = client.files.create(
            file=f,
            purpose="assistants"
        )
    
    print(f"File uploaded: {uploaded_file.id}")
    
    # Create vector store
    vector_store = client.beta.vector_stores.create(
        name="Documentation Store"
    )
    print(f"Vector store created: {vector_store.id}")
    
    # Add file to vector store
    file_batch = client.beta.vector_stores.files.create(
        vector_store_id=vector_store.id,
        file_id=uploaded_file.id
    )
    print(f"File added to vector store")
    
    # Wait for vector store to be ready
    print("\nWaiting for vector store to process...")
    while True:
        vs_status = client.beta.vector_stores.retrieve(vector_store.id)
        if vs_status.status == "completed":
            break
        time.sleep(2)
    
    print("Vector store ready")
    
    # Create assistant with file search
    assistant = client.beta.assistants.create(
        name="File Search Assistant",
        instructions="You are a helpful assistant that can search through uploaded "
                    "documents to answer questions. Use the file search tool to find "
                    "relevant information.",
        model="gpt-4-turbo-preview",
        tools=[{"type": "file_search"}],
        tool_resources={
            "file_search": {
                "vector_store_ids": [vector_store.id]
            }
        }
    )
    
    print(f"\nAssistant created: {assistant.id}")
    
    # Create thread and ask question
    thread = client.beta.threads.create()
    
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content="What can the Assistants API do? List the main capabilities."
    )
    
    print(f"\nUser question: {message.content[0].text.value}")
    
    # Create run
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )
    
    # Wait for completion
    while run.status in ["queued", "in_progress"]:
        time.sleep(2)
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
    
    if run.status == "completed":
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        assistant_message = messages.data[0]
        print(f"\nAssistant response:")
        print(f"  {assistant_message.content[0].text.value}")
    
    # Cleanup
    os.remove(sample_file_path)
    
    return assistant, thread, uploaded_file, vector_store


# ============================================================================
# SECTION 5: Streaming Runs
# ============================================================================

def streaming_runs():
    """
    Demonstrates streaming run events for real-time updates.
    """
    print("\n" + "=" * 70)
    print("SECTION 5: Streaming Runs")
    print("=" * 70)
    
    # Create assistant
    assistant = client.beta.assistants.create(
        name="Streaming Assistant",
        instructions="You are a helpful assistant. Provide detailed explanations.",
        model="gpt-4-turbo-preview"
    )
    
    # Create thread
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": "Explain how neural networks work in detail. "
                          "Cover the basic concepts, how they learn, and their applications."
            }
        ]
    )
    
    print("\nStarting streaming run...")
    
    # Create streaming run
    stream = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        stream=True
    )
    
    current_message = ""
    events_received = []
    
    # Process stream events
    for event in stream:
        event_type = event.event
        
        if event_type == "thread.run.created":
            print(f"\n[Event] Run created: {event.data.id}")
            events_received.append("run.created")
        
        elif event_type == "thread.run.queued":
            print(f"[Event] Run queued")
            events_received.append("run.queued")
        
        elif event_type == "thread.run.in_progress":
            print(f"[Event] Run in progress")
            events_received.append("run.in_progress")
        
        elif event_type == "thread.message.delta":
            if hasattr(event.data, 'delta') and hasattr(event.data.delta, 'content'):
                for content in event.data.delta.content:
                    if hasattr(content, 'text') and hasattr(content.text, 'value'):
                        delta_text = content.text.value
                        current_message += delta_text
                        print(delta_text, end="", flush=True)
        
        elif event_type == "thread.message.completed":
            print(f"\n[Event] Message completed")
            events_received.append("message.completed")
        
        elif event_type == "thread.run.completed":
            print(f"\n[Event] Run completed")
            events_received.append("run.completed")
            break
        
        elif event_type == "thread.run.requires_action":
            print(f"\n[Event] Run requires action")
            events_received.append("requires_action")
            break
        
        elif event_type == "error":
            print(f"\n[Event] Error: {event.data}")
            events_received.append("error")
            break
    
    print(f"\n\nTotal events received: {len(events_received)}")
    print(f"Events: {', '.join(events_received)}")
    
    return assistant, thread


# ============================================================================
# SECTION 6: Multi-Tool Assistant
# ============================================================================

def get_weather_function(location: str) -> str:
    """Mock weather function."""
    return json.dumps({
        "location": location,
        "temperature": 22,
        "condition": "sunny"
    })


def multi_tool_assistant():
    """
    Demonstrates an assistant with multiple tools: code_interpreter, file_search, and functions.
    """
    print("\n" + "=" * 70)
    print("SECTION 6: Multi-Tool Assistant")
    print("=" * 70)
    
    # Create sample file
    sample_content = "Python is a programming language. It's used for data science, web development, and AI."
    with open("python_info.txt", "w") as f:
        f.write(sample_content)
    
    # Upload file
    with open("python_info.txt", "rb") as f:
        uploaded_file = client.files.create(file=f, purpose="assistants")
    
    # Create vector store
    vector_store = client.beta.vector_stores.create(name="Info Store")
    client.beta.vector_stores.files.create(
        vector_store_id=vector_store.id,
        file_id=uploaded_file.id
    )
    
    # Wait for vector store
    while True:
        vs = client.beta.vector_stores.retrieve(vector_store.id)
        if vs.status == "completed":
            break
        time.sleep(2)
    
    # Create multi-tool assistant
    assistant = client.beta.assistants.create(
        name="Multi-Tool Assistant",
        instructions="You are a versatile assistant with access to code execution, "
                    "file search, and custom functions. Use the appropriate tool for each task.",
        model="gpt-4-turbo-preview",
        tools=[
            {"type": "code_interpreter"},
            {"type": "file_search"},
            {
                "type": "function",
                "function": {
                    "name": "get_weather_function",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ],
        tool_resources={
            "file_search": {
                "vector_store_ids": [vector_store.id]
            }
        }
    )
    
    print(f"\nMulti-tool assistant created: {assistant.id}")
    
    # Create thread with multi-part query
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": "1. Search the files for information about Python. "
                          "2. Calculate 15 * 23 using code. "
                          "3. Get the weather for New York."
            }
        ]
    )
    
    print("\nUser query submitted")
    
    # Create run
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )
    
    # Poll and handle requires_action
    while True:
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
        
        if run.status == "requires_action":
            print("\nRun requires action - processing tool calls...")
            
            tool_outputs = []
            for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                tool_id = tool_call.id
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                print(f"  Tool call: {function_name} with args: {arguments}")
                
                # Execute function
                if function_name == "get_weather_function":
                    result = get_weather_function(**arguments)
                else:
                    result = json.dumps({"error": "Unknown function"})
                
                tool_outputs.append({
                    "tool_call_id": tool_id,
                    "output": result
                })
            
            # Submit tool outputs
            run = client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread.id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )
        
        elif run.status == "completed":
            print("\nRun completed")
            break
        
        elif run.status in ["queued", "in_progress"]:
            time.sleep(2)
        
        else:
            print(f"\nRun status: {run.status}")
            break
    
    if run.status == "completed":
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        print(f"\nAssistant response:")
        print(f"  {messages.data[0].content[0].text.value}")
    
    # Cleanup
    os.remove("python_info.txt")
    
    return assistant, thread, uploaded_file, vector_store


# ============================================================================
# SECTION 7: Run Management
# ============================================================================

def run_management():
    """
    Demonstrates advanced run management: polling, cancellation, status checking.
    """
    print("\n" + "=" * 70)
    print("SECTION 7: Run Management")
    print("=" * 70)
    
    # Create assistant
    assistant = client.beta.assistants.create(
        name="Managed Assistant",
        instructions="You are a helpful assistant.",
        model="gpt-4-turbo-preview",
        tools=[{"type": "code_interpreter"}]
    )
    
    # Create thread
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": "Write a Python script that calculates fibonacci numbers up to 100."
            }
        ]
    )
    
    # Create run
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )
    
    print(f"\nRun created: {run.id}")
    print(f"Initial status: {run.status}")
    
    # Poll with timeout
    max_wait_time = 60
    start_time = time.time()
    poll_interval = 2
    
    while True:
        elapsed = time.time() - start_time
        
        if elapsed > max_wait_time:
            print(f"\nTimeout reached ({max_wait_time}s)")
            # Cancel the run
            try:
                cancelled_run = client.beta.threads.runs.cancel(
                    thread_id=thread.id,
                    run_id=run.id
                )
                print(f"Run cancelled: {cancelled_run.status}")
            except Exception as e:
                print(f"Error cancelling: {e}")
            break
        
        # Retrieve run status
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
        
        print(f"  Status: {run.status} (elapsed: {elapsed:.1f}s)")
        
        if run.status == "completed":
            print("\nRun completed successfully")
            
            # Get run steps
            steps = client.beta.threads.runs.steps.list(
                thread_id=thread.id,
                run_id=run.id
            )
            print(f"Total steps: {len(steps.data)}")
            
            break
        
        elif run.status == "failed":
            print(f"\nRun failed: {run.last_error}")
            break
        
        elif run.status == "cancelled":
            print("\nRun was cancelled")
            break
        
        elif run.status == "requires_action":
            print("\nRun requires action")
            # Handle tool calls (simplified)
            break
        
        time.sleep(poll_interval)
    
    # Get final messages
    if run.status == "completed":
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        print(f"\nFinal message count: {len(messages.data)}")
    
    return assistant, thread, run


# ============================================================================
# SECTION 8: Cleanup
# ============================================================================

def cleanup_resources(assistant_ids: List[str] = None,
                     thread_ids: List[str] = None,
                     file_ids: List[str] = None,
                     vector_store_ids: List[str] = None):
    """
    Demonstrates cleaning up created resources.
    """
    print("\n" + "=" * 70)
    print("SECTION 8: Cleanup")
    print("=" * 70)
    
    deleted_count = {"assistants": 0, "threads": 0, "files": 0, "vector_stores": 0}
    
    # Delete assistants
    if assistant_ids:
        for assistant_id in assistant_ids:
            try:
                client.beta.assistants.delete(assistant_id)
                deleted_count["assistants"] += 1
                print(f"Deleted assistant: {assistant_id}")
            except Exception as e:
                print(f"Error deleting assistant {assistant_id}: {e}")
    
    # Delete threads
    if thread_ids:
        for thread_id in thread_ids:
            try:
                client.beta.threads.delete(thread_id)
                deleted_count["threads"] += 1
                print(f"Deleted thread: {thread_id}")
            except Exception as e:
                print(f"Error deleting thread {thread_id}: {e}")
    
    # Delete files
    if file_ids:
        for file_id in file_ids:
            try:
                client.files.delete(file_id)
                deleted_count["files"] += 1
                print(f"Deleted file: {file_id}")
            except Exception as e:
                print(f"Error deleting file {file_id}: {e}")
    
    # Delete vector stores
    if vector_store_ids:
        for vs_id in vector_store_ids:
            try:
                # First delete files from vector store
                vs_files = client.beta.vector_stores.files.list(vector_store_id=vs_id)
                for vs_file in vs_files.data:
                    client.beta.vector_stores.files.delete(
                        vector_store_id=vs_id,
                        file_id=vs_file.id
                    )
                
                # Then delete vector store
                client.beta.vector_stores.delete(vs_id)
                deleted_count["vector_stores"] += 1
                print(f"Deleted vector store: {vs_id}")
            except Exception as e:
                print(f"Error deleting vector store {vs_id}: {e}")
    
    print(f"\nCleanup summary:")
    print(f"  Assistants: {deleted_count['assistants']}")
    print(f"  Threads: {deleted_count['threads']}")
    print(f"  Files: {deleted_count['files']}")
    print(f"  Vector stores: {deleted_count['vector_stores']}")


def list_all_resources():
    """
    Helper function to list all resources for cleanup.
    """
    print("\nListing all resources...")
    
    # List assistants
    assistants = client.beta.assistants.list()
    print(f"\nAssistants ({len(assistants.data)}):")
    for asst in assistants.data[:5]:  # Show first 5
        print(f"  {asst.id} - {asst.name}")
    
    # List files
    files = client.files.list()
    print(f"\nFiles ({len(files.data)}):")
    for file in files.data[:5]:  # Show first 5
        print(f"  {file.id} - {file.filename}")
    
    # List vector stores
    vector_stores = client.beta.vector_stores.list()
    print(f"\nVector stores ({len(vector_stores.data)}):")
    for vs in vector_stores.data[:5]:  # Show first 5
        print(f"  {vs.id} - {vs.name}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function to run all examples.
    Uncomment the sections you want to run.
    """
    print("\n" + "=" * 70)
    print("OpenAI Assistants API Advanced Examples")
    print("=" * 70)
    print("\nNote: Make sure to set your OpenAI API key in the client initialization.")
    print("Each section is standalone and can be run independently.\n")
    
    # Track created resources for cleanup
    created_assistants = []
    created_threads = []
    created_files = []
    created_vector_stores = []
    
    try:
        # Uncomment sections to run:
        
        # Section 1: Create Assistant
        # assistant1, assistant2 = create_assistant()
        # created_assistants.extend([assistant1.id, assistant2.id])
        
        # Section 2: Thread Management
        # thread2, assistant2 = thread_management()
        # created_threads.append(thread2.id)
        # created_assistants.append(assistant2.id)
        
        # Section 3: Code Interpreter
        # assistant3, thread3 = code_interpreter_example()
        # created_assistants.append(assistant3.id)
        # created_threads.append(thread3.id)
        
        # Section 4: File Search
        # assistant4, thread4, file4, vs4 = file_search_example()
        # created_assistants.append(assistant4.id)
        # created_threads.append(thread4.id)
        # created_files.append(file4.id)
        # created_vector_stores.append(vs4.id)
        
        # Section 5: Streaming Runs
        # assistant5, thread5 = streaming_runs()
        # created_assistants.append(assistant5.id)
        # created_threads.append(thread5.id)
        
        # Section 6: Multi-Tool Assistant
        # assistant6, thread6, file6, vs6 = multi_tool_assistant()
        # created_assistants.append(assistant6.id)
        # created_threads.append(thread6.id)
        # created_files.append(file6.id)
        # created_vector_stores.append(vs6.id)
        
        # Section 7: Run Management
        # assistant7, thread7, run7 = run_management()
        # created_assistants.append(assistant7.id)
        # created_threads.append(thread7.id)
        
        # Section 8: Cleanup (uncomment to clean up resources)
        # cleanup_resources(
        #     assistant_ids=created_assistants,
        #     thread_ids=created_threads,
        #     file_ids=created_files,
        #     vector_store_ids=created_vector_stores
        # )
        
        # List all resources
        # list_all_resources()
        
        print("\n" + "=" * 70)
        print("Examples completed. Uncomment sections in main() to run them.")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Make sure your API key is set correctly.")


if __name__ == "__main__":
    main()
