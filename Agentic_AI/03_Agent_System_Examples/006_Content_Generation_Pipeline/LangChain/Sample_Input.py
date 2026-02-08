"""
Sample Input module for Content Generation Pipeline.
Contains predefined content briefs and function to run sample generations.
"""

from Main import Setup_Pipeline, Generate_Content
from typing import List, Dict, Any


CONTENT_BRIEFS: List[Dict[str, Any]] = [
    {
        "topic": "Getting Started with AI Agents in 2025",
        "audience": "developers",
        "tone": "technical",
        "target_word_count": 1800,
        "target_keywords": [
            "AI agents",
            "LangChain",
            "LangGraph",
            "agent development",
            "autonomous systems"
        ]
    },
    {
        "topic": "How RAG is Transforming Enterprise Search",
        "audience": "business leaders",
        "tone": "professional",
        "target_word_count": 2000,
        "target_keywords": [
            "RAG",
            "retrieval augmented generation",
            "enterprise search",
            "knowledge management",
            "vector databases"
        ]
    },
    {
        "topic": "Building Your First LangChain Application",
        "audience": "beginners",
        "tone": "friendly",
        "target_word_count": 1500,
        "target_keywords": [
            "LangChain",
            "LLM applications",
            "Python",
            "getting started",
            "tutorial"
        ]
    }
]


def Run_Samples() -> None:
    """
    Generate content for all sample briefs and print results with quality metrics.
    """
    print("=" * 70)
    print("Content Generation Pipeline - Sample Inputs")
    print("=" * 70)
    print(f"\nRunning {len(CONTENT_BRIEFS)} sample content generations...\n")
    
    try:
        pipeline = Setup_Pipeline()
        print("Pipeline initialized successfully.\n")
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        print("Please ensure OPENAI_API_KEY is set in your environment.")
        return
    
    results = []
    
    for i, brief in enumerate(CONTENT_BRIEFS, 1):
        print("-" * 70)
        print(f"Sample {i}/{len(CONTENT_BRIEFS)}")
        print("-" * 70)
        print(f"Topic: {brief['topic']}")
        print(f"Audience: {brief['audience']}")
        print(f"Tone: {brief['tone']}")
        print(f"Target Word Count: {brief['target_word_count']}")
        print(f"Target Keywords: {', '.join(brief['target_keywords'])}")
        print("\nGenerating content...")
        
        try:
            result = Generate_Content(
                topic=brief["topic"],
                audience=brief["audience"],
                tone=brief["tone"],
                pipeline=pipeline,
                target_keywords=brief["target_keywords"],
                target_word_count=brief["target_word_count"]
            )
            
            results.append({
                "brief": brief,
                "result": result
            })
            
            print("\n" + "=" * 70)
            print(f"Sample {i} - Generation Complete")
            print("=" * 70)
            print(f"\nQuality Metrics:")
            print(f"  Quality Score: {result['quality_score']:.3f}")
            print(f"  Revision Count: {result['revision_count']}")
            print(f"  Stages Completed: {', '.join(result['stage_history'])}")
            
            word_count = len(result["final_content"].split())
            print(f"  Final Word Count: {word_count}")
            print(f"  Target Word Count: {brief['target_word_count']}")
            
            if word_count <= brief['target_word_count'] * 1.1:
                print("  Word Count Status: Within target range")
            else:
                print("  Word Count Status: Exceeds target range")
            
            print("\n" + "-" * 70)
            print("Generated Content Preview (first 500 characters):")
            print("-" * 70)
            preview = result["final_content"][:500]
            print(preview)
            if len(result["final_content"]) > 500:
                print("...")
            
            print("\n" + "=" * 70 + "\n")
        
        except Exception as e:
            print(f"\nError generating content for sample {i}: {e}")
            print("Skipping to next sample...\n")
            continue
    
    print("\n" + "=" * 70)
    print("Summary of All Samples")
    print("=" * 70)
    
    if not results:
        print("\nNo content was successfully generated.")
        return
    
    print(f"\nTotal Samples Processed: {len(results)}/{len(CONTENT_BRIEFS)}")
    print("\nQuality Metrics Summary:")
    print("-" * 70)
    
    avg_quality_score = sum(r["result"]["quality_score"] for r in results) / len(results)
    avg_revision_count = sum(r["result"]["revision_count"] for r in results) / len(results)
    total_stages = sum(len(r["result"]["stage_history"]) for r in results)
    avg_stages = total_stages / len(results)
    
    print(f"Average Quality Score: {avg_quality_score:.3f}")
    print(f"Average Revision Count: {avg_revision_count:.2f}")
    print(f"Average Stages per Generation: {avg_stages:.2f}")
    
    print("\n" + "-" * 70)
    print("Individual Sample Results:")
    print("-" * 70)
    
    for i, item in enumerate(results, 1):
        brief = item["brief"]
        result = item["result"]
        
        print(f"\nSample {i}: {brief['topic']}")
        print(f"  Quality Score: {result['quality_score']:.3f}")
        print(f"  Revisions: {result['revision_count']}")
        print(f"  Stages: {len(result['stage_history'])}")
        print(f"  Stage Flow: {' -> '.join(result['stage_history'])}")
    
    print("\n" + "=" * 70)
    
    save_all = input("\nDo you want to save all generated content to files? (y/n): ").strip().lower()
    
    if save_all == "y":
        for i, item in enumerate(results, 1):
            brief = item["brief"]
            result = item["result"]
            
            filename = f"sample_{i}_{brief['topic'].lower().replace(' ', '_')[:30]}.md"
            filename = "".join(c for c in filename if c.isalnum() or c in "._-")
            
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(f"# {brief['topic']}\n\n")
                    f.write(f"**Audience:** {brief['audience']}\n")
                    f.write(f"**Tone:** {brief['tone']}\n")
                    f.write(f"**Quality Score:** {result['quality_score']:.3f}\n")
                    f.write(f"**Revision Count:** {result['revision_count']}\n\n")
                    f.write("---\n\n")
                    f.write(result["final_content"])
                
                print(f"Saved: {filename}")
            except Exception as e:
                print(f"Error saving {filename}: {e}")
        
        print("\nAll files saved successfully!")
    
    print("\n" + "=" * 70)
    print("Sample generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    Run_Samples()
