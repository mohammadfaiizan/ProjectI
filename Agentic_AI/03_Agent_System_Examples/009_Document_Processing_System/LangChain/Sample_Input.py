"""
Sample Input module for Document Processing System.

This module contains sample documents for testing and demonstration purposes.
Includes realistic examples of invoices, resumes, contracts, letters, and reports.
"""

from typing import Dict
from Main import Setup_Processing_System, Process_Document


# ============================================================================
# Sample Documents
# ============================================================================

SAMPLE_DOCUMENTS: Dict[str, str] = {
    "invoice_001.txt": """
INVOICE

Invoice Number: INV-2024-001234
Invoice Date: March 15, 2024
Due Date: April 14, 2024

BILL TO:
Acme Manufacturing Corporation
1234 Industrial Boulevard
Springfield, IL 62701
United States
Contact: John Smith
Phone: (217) 555-0123
Email: accounts.payable@acme.com

SHIP TO:
Acme Manufacturing Corporation
1234 Industrial Boulevard
Springfield, IL 62701
United States

SERVICE PROVIDER:
Tech Solutions Inc.
5678 Business Park Drive
Chicago, IL 60601
United States
Phone: (312) 555-9876
Email: billing@techsolutions.com
Tax ID: 12-3456789

DESCRIPTION OF SERVICES:

Item #    Description                          Quantity    Unit Price    Amount
---------------------------------------------------------------------------
001       Software Development Services          120 hrs    $150.00       $18,000.00
002       System Integration Services            40 hrs    $175.00        $7,000.00
003       Quality Assurance Testing               30 hrs    $125.00        $3,750.00
004       Technical Documentation                 20 hrs    $100.00        $2,000.00
005       Project Management                      25 hrs    $200.00        $5,000.00

SUBTOTAL:                                                              $35,750.00
Sales Tax (8.5%):                                                       $3,038.75
TOTAL DUE:                                                             $38,788.75

PAYMENT TERMS:
Net 30 days. Payment is due within 30 days of invoice date.
Late payments may be subject to a 1.5% monthly finance charge.

PAYMENT METHODS ACCEPTED:
- Check (mail to address above)
- Wire Transfer (contact for banking details)
- ACH Transfer
- Credit Card (2.5% processing fee applies)

Thank you for your business. If you have any questions regarding this invoice,
please contact our billing department at billing@techsolutions.com or call
(312) 555-9876.

This invoice is payable to Tech Solutions Inc. All amounts are in USD.
""",
    
    "resume_john_doe.txt": """
JOHN MICHAEL DOE
Email: john.m.doe@email.com
Phone: (555) 234-5678
LinkedIn: linkedin.com/in/johndoe
Location: San Francisco, CA

PROFESSIONAL SUMMARY
Results-driven software engineer with over 8 years of experience in full-stack
development, specializing in Python, JavaScript, and cloud technologies. Proven
track record of leading cross-functional teams to deliver scalable web applications
and microservices. Strong expertise in agile methodologies, DevOps practices, and
system architecture design. Passionate about writing clean, maintainable code and
mentoring junior developers.

TECHNICAL SKILLS
Programming Languages: Python, JavaScript, TypeScript, Java, Go, SQL
Frameworks & Libraries: Django, Flask, React, Node.js, Express, FastAPI
Databases: PostgreSQL, MongoDB, Redis, MySQL
Cloud & DevOps: AWS (EC2, S3, Lambda, RDS), Docker, Kubernetes, CI/CD, Terraform
Tools & Technologies: Git, Jenkins, GitHub Actions, Elasticsearch, RabbitMQ
Testing: pytest, Jest, Selenium, Postman, Unit Testing, Integration Testing

PROFESSIONAL EXPERIENCE

Senior Software Engineer | Tech Innovations Inc. | San Francisco, CA
January 2020 - Present
- Lead development of microservices architecture serving 2M+ daily active users
- Architected and implemented RESTful APIs using Python/Django, reducing API
  response time by 40% through optimization and caching strategies
- Mentored team of 5 junior developers, conducting code reviews and technical
  training sessions
- Collaborated with product managers and designers to define technical requirements
  and deliver features on schedule
- Implemented CI/CD pipelines using Jenkins and Docker, reducing deployment time
  from 2 hours to 15 minutes
- Designed and developed real-time notification system using WebSockets and Redis,
  handling 10K+ concurrent connections
- Reduced system downtime by 60% through proactive monitoring and alerting systems

Software Engineer | Startup Solutions LLC | San Francisco, CA
June 2017 - December 2019
- Developed full-stack web applications using React frontend and Python/Flask backend
- Built and maintained PostgreSQL databases, optimizing queries to improve
  performance by 50%
- Implemented authentication and authorization systems using JWT tokens and OAuth2
- Created automated testing suites achieving 85% code coverage
- Participated in agile sprint planning and daily standups
- Deployed applications to AWS infrastructure, managing EC2 instances and S3 storage

Junior Software Developer | Web Services Co. | Oakland, CA
August 2015 - May 2017
- Developed and maintained web applications using JavaScript, HTML, and CSS
- Collaborated with senior developers to implement new features and fix bugs
- Participated in code reviews and learned best practices for software development
- Wrote unit tests and integration tests for new features
- Assisted in database design and optimization tasks

EDUCATION

Bachelor of Science in Computer Science
University of California, Berkeley | Berkeley, CA
Graduated: May 2015
GPA: 3.7/4.0
Relevant Coursework: Data Structures, Algorithms, Database Systems, Software
Engineering, Operating Systems, Computer Networks

CERTIFICATIONS
- AWS Certified Solutions Architect - Associate (2021)
- Certified Kubernetes Administrator (2022)
- Python Professional Certification (2019)

PROJECTS
- E-Commerce Platform: Built scalable e-commerce platform using Django and React,
  handling 100K+ products and 50K+ daily transactions
- Real-Time Chat Application: Developed WebSocket-based chat application with
  message persistence and file sharing capabilities
- Data Analytics Dashboard: Created interactive dashboard using React and D3.js
  for visualizing business metrics

REFERENCES
Available upon request.
""",
    
    "contract_service.txt": """
SERVICE AGREEMENT

This Service Agreement ("Agreement") is entered into on this 1st day of April, 2024,
by and between:

PARTY A (Service Provider):
Global Tech Services LLC
789 Corporate Plaza
New York, NY 10001
United States
Tax ID: 98-7654321
Represented by: Jane Smith, Chief Executive Officer

PARTY B (Client):
Innovation Labs Inc.
456 Research Avenue
Boston, MA 02115
United States
Tax ID: 12-9876543
Represented by: Robert Johnson, Chief Technology Officer

RECITALS
WHEREAS, Party A is engaged in the business of providing software development and
consulting services; and
WHEREAS, Party B desires to engage Party A to provide certain services as described
herein; and
WHEREAS, both parties desire to set forth the terms and conditions under which such
services will be provided.

NOW, THEREFORE, in consideration of the mutual covenants and agreements contained
herein, the parties agree as follows:

1. SCOPE OF SERVICES
Party A agrees to provide the following services to Party B:
a) Custom software development for web and mobile applications
b) System integration and API development
c) Technical consulting and architecture design
d) Quality assurance and testing services
e) Ongoing maintenance and support as specified in Section 4

2. TERM AND EFFECTIVE DATE
This Agreement shall commence on April 1, 2024 (the "Effective Date") and shall
continue for a period of twelve (12) months, unless earlier terminated in accordance
with the provisions of this Agreement. The Agreement may be renewed for additional
twelve-month periods upon mutual written agreement of both parties.

3. COMPENSATION AND PAYMENT TERMS
a) Total Contract Value: Party B agrees to pay Party A a total amount of
   $250,000.00 (Two Hundred Fifty Thousand Dollars) for the services rendered
   under this Agreement.
b) Payment Schedule: Payments shall be made monthly in installments of $20,833.33,
   due on the first day of each month, commencing May 1, 2024.
c) Late Payment: Any payment not received within 15 days of the due date shall
   incur a late fee of 1.5% per month on the outstanding balance.
d) All amounts are in United States Dollars (USD).

4. MAINTENANCE AND SUPPORT
Party A shall provide maintenance and support services for a period of six (6)
months following the completion of the initial development phase. Support includes:
- Bug fixes and critical issue resolution
- Security updates and patches
- Technical support via email and phone during business hours (9 AM - 5 PM EST)
- Monthly system health checks and performance reports

5. INTELLECTUAL PROPERTY
All intellectual property rights, including but not limited to copyrights, patents,
and trade secrets, in any work product developed under this Agreement shall be
owned by Party B upon full payment of all amounts due hereunder. Party A retains
the right to use general methodologies and techniques developed during the
performance of services.

6. CONFIDENTIALITY
Both parties agree to maintain the confidentiality of all proprietary information
disclosed during the term of this Agreement and for a period of three (3) years
thereafter. Confidential information includes but is not limited to business plans,
technical specifications, customer data, and financial information.

7. TERMINATION
Either party may terminate this Agreement with thirty (30) days written notice.
Upon termination, Party B shall pay Party A for all services rendered and expenses
incurred up to the date of termination. Party A shall deliver all work product and
materials to Party B within ten (10) days of termination.

8. LIMITATION OF LIABILITY
Party A's total liability under this Agreement shall not exceed the total amount
paid by Party B hereunder. Neither party shall be liable for indirect, incidental,
or consequential damages.

9. GOVERNING LAW
This Agreement shall be governed by and construed in accordance with the laws of
the State of New York, without regard to its conflict of law principles.

10. ENTIRE AGREEMENT
This Agreement constitutes the entire agreement between the parties and supersedes
all prior negotiations, representations, or agreements, whether oral or written.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first
written above.

PARTY A: Global Tech Services LLC          PARTY B: Innovation Labs Inc.

_________________________                  _________________________
Jane Smith                                 Robert Johnson
Chief Executive Officer                    Chief Technology Officer

Date: April 1, 2024                        Date: April 1, 2024
""",
    
    "letter_complaint.txt": """
[Your Name]
[Your Address]
[City, State ZIP Code]
[Your Email]
[Your Phone Number]
[Date: March 20, 2024]

[Recipient Name]
Customer Service Department
ABC Electronics Corporation
1234 Commerce Street
Los Angeles, CA 90001

SUBJECT: Formal Complaint Regarding Defective Product and Poor Customer Service

Dear Customer Service Department,

I am writing to file a formal complaint regarding a defective product I purchased
from your company and the unsatisfactory customer service I received when attempting
to resolve this issue.

On February 15, 2024, I purchased a Model XYZ-5000 Wireless Headphones (Serial
Number: XYZ5000-2024-001234) from your online store for $299.99, plus $24.99
shipping and handling, for a total of $324.98. The order confirmation number is
ORD-2024-567890.

Upon receiving the product on February 20, 2024, I immediately noticed several
issues. First, the left earcup produced a constant static noise that made the
headphones unusable. Second, the battery life was significantly shorter than
advertised - the headphones lasted only 4 hours instead of the promised 30 hours.
Third, the Bluetooth connectivity was intermittent, causing frequent disconnections
even when the device was within the specified range.

I contacted your customer service department on February 22, 2024, via phone call
(reference number: CS-2024-789012). The representative I spoke with was dismissive
and unhelpful, suggesting that the issues were due to user error rather than
product defects. When I requested a replacement or refund, I was told that the
product was outside the 7-day return window, which is incorrect as I had only
received it 2 days prior.

I followed up with an email on February 25, 2024, to customer.service@abcelectronics.com,
but received no response. I made another phone call on March 5, 2024, and was
placed on hold for over 45 minutes before being disconnected.

This experience has been extremely frustrating and disappointing. As a loyal
customer who has purchased multiple products from ABC Electronics over the past
three years, I expected better service and support. The product I received is
clearly defective and does not meet the quality standards or specifications
advertised on your website.

I am requesting the following resolution:
1. A full refund of $324.98, including shipping costs
2. A prepaid return shipping label to return the defective product
3. An apology for the poor customer service experience
4. Assurance that this issue will be addressed to prevent similar problems
   for other customers

I expect a response within 10 business days. If I do not receive a satisfactory
resolution, I will be forced to take further action, including filing a complaint
with the Better Business Bureau and disputing the charge with my credit card
company.

I have attached copies of my order confirmation, receipt, and previous
correspondence for your reference. I hope that we can resolve this matter
promptly and amicably.

Thank you for your attention to this matter.

Sincerely,

[Your Signature]

[Your Printed Name]
""",
    
    "report_quarterly.txt": """
QUARTERLY BUSINESS REPORT
Q1 2024 Performance Analysis

Prepared by: Sarah Williams, Director of Business Analytics
Date: April 10, 2024
Company: Global Solutions Inc.
Report Period: January 1, 2024 - March 31, 2024

EXECUTIVE SUMMARY

This report presents a comprehensive analysis of Global Solutions Inc.'s performance
during the first quarter of 2024. Overall, the company demonstrated strong growth
across key metrics, with revenue increasing by 18% compared to Q1 2023, and
operating margins improving by 3.2 percentage points. The quarter was marked by
successful product launches, expansion into new markets, and strategic partnerships
that positioned the company for continued growth in the coming quarters.

Key highlights include:
- Total revenue of $12.5 million, representing 18% year-over-year growth
- Net profit margin of 15.2%, up from 12.0% in Q1 2023
- Customer acquisition increased by 25% compared to the previous quarter
- Employee headcount grew to 145, with 12 new hires across engineering and sales
- Launched two new product lines, contributing $2.1 million in new revenue

FINANCIAL PERFORMANCE

Revenue Analysis
Total revenue for Q1 2024 reached $12,500,000, compared to $10,593,220 in Q1
2023, representing an 18% increase. Revenue growth was driven primarily by:
- Product sales: $8,750,000 (70% of total revenue), up 15% YoY
- Service revenue: $2,500,000 (20% of total revenue), up 22% YoY
- Subscription revenue: $1,250,000 (10% of total revenue), up 35% YoY

The subscription revenue growth is particularly noteworthy, reflecting successful
customer retention strategies and the introduction of premium subscription tiers.

Expense Management
Total operating expenses for the quarter were $8,750,000, representing 70% of
revenue, down from 73% in Q1 2023. Key expense categories:
- Salaries and benefits: $4,500,000 (36% of revenue)
- Research and development: $1,875,000 (15% of revenue)
- Sales and marketing: $1,500,000 (12% of revenue)
- General and administrative: $875,000 (7% of revenue)

Profitability
Net income for Q1 2024 was $1,900,000, resulting in a net profit margin of 15.2%,
significantly improved from 12.0% in Q1 2023. This improvement reflects both
revenue growth and effective cost management initiatives implemented throughout
2023.

OPERATIONAL METRICS

Customer Metrics
- Total active customers: 2,450 (up 25% from Q4 2023)
- New customer acquisitions: 612 (up 25% QoQ)
- Customer retention rate: 92% (maintained from previous quarter)
- Average customer lifetime value: $8,500 (up 8% YoY)
- Customer acquisition cost: $450 (down 10% from Q4 2023)

Product Performance
The company launched two new product lines during Q1 2024:
1. Enterprise Solution Suite: Generated $1.4 million in revenue with 45 enterprise
   customers acquired
2. Professional Tools Package: Generated $700,000 in revenue with 180 customers

Existing product lines continued to perform well:
- Core Platform: $5.2 million revenue, 15% growth YoY
- Advanced Features: $2.1 million revenue, 22% growth YoY
- Integration Services: $1.45 million revenue, 18% growth YoY

MARKET ANALYSIS

Market Position
Global Solutions Inc. maintained its position as a market leader in the enterprise
software solutions sector. Market share increased from 12% to 14% during the quarter,
driven by successful marketing campaigns and strategic partnerships.

Competitive Landscape
The competitive environment remained intense, with three major competitors launching
similar products. However, our focus on customer service and product innovation
allowed us to maintain competitive advantage. Customer satisfaction scores remained
high at 4.6 out of 5.0.

STRATEGIC INITIATIVES

Product Development
The R&D team made significant progress on several key initiatives:
- Completed development of AI-powered analytics module (scheduled for Q2 launch)
- Advanced work on mobile application platform (75% complete)
- Initiated research into blockchain integration capabilities

Market Expansion
The company successfully entered two new geographic markets:
- European Union: Established operations in Germany and France
- Asia-Pacific: Launched services in Singapore and Australia

These expansions contributed $850,000 in new revenue and are expected to drive
significant growth in subsequent quarters.

RISKS AND CHALLENGES

Market Risks
- Increased competition from established players entering our market segment
- Potential economic downturn affecting enterprise spending
- Regulatory changes in key markets requiring compliance adjustments

Operational Challenges
- Scaling customer support to maintain service quality with growing customer base
- Recruiting and retaining top talent in competitive job market
- Managing supply chain disruptions affecting hardware components

RECOMMENDATIONS

Based on the analysis presented in this report, the following recommendations are
proposed:

1. Continue investment in R&D to maintain product innovation leadership
2. Expand customer success team to support growing customer base
3. Accelerate international expansion, particularly in high-growth markets
4. Implement advanced analytics tools to improve decision-making capabilities
5. Strengthen partnerships with key technology vendors and integrators

CONCLUSION

Q1 2024 was a strong quarter for Global Solutions Inc., with solid financial
performance, successful product launches, and strategic market expansion. The
company is well-positioned to continue its growth trajectory in Q2 2024 and
beyond. Continued focus on customer satisfaction, product innovation, and
operational efficiency will be critical to maintaining competitive advantage
and achieving long-term strategic objectives.

The management team remains confident in the company's ability to execute its
strategic plan and deliver value to shareholders, customers, and employees.

APPENDICES
- Detailed financial statements
- Customer satisfaction survey results
- Product performance breakdowns
- Market research data
- Employee engagement metrics
"""
}


# ============================================================================
# Sample Processing Function
# ============================================================================

def Run_Samples():
    """
    Process all sample documents and print extracted entities and classifications.
    """
    print("Document Processing System - Sample Document Processing")
    print("="*70)
    
    # Setup processing system
    processing_graph = Setup_Processing_System()
    
    # Process each sample document
    results_summary = []
    
    for filename, text in SAMPLE_DOCUMENTS.items():
        print(f"\n{'='*70}")
        print(f"Processing: {filename}")
        print(f"{'='*70}\n")
        
        # Process document
        result = Process_Document(
            text=text,
            filename=filename,
            processing_graph=processing_graph
        )
        
        # Store summary
        results_summary.append({
            "filename": filename,
            "doc_type": result.get("doc_type"),
            "is_valid": result.get("is_valid"),
            "entities_count": len(result.get("extracted_entities", {}))
        })
        
        # Print extracted entities in detail
        if result.get("extracted_entities"):
            print(f"\nDetailed Extracted Entities:")
            entities = result["extracted_entities"]
            for key, value in entities.items():
                if isinstance(value, list):
                    print(f"  {key}:")
                    for item in value[:3]:  # Show first 3 items
                        if isinstance(item, dict):
                            print(f"    - {item}")
                        else:
                            print(f"    - {item}")
                    if len(value) > 3:
                        print(f"    ... and {len(value) - 3} more items")
                elif isinstance(value, dict):
                    print(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key}: {sub_value}")
                else:
                    print(f"  {key}: {value}")
    
    # Print overall summary
    print(f"\n{'='*70}")
    print("Processing Summary")
    print(f"{'='*70}\n")
    
    print(f"Total documents processed: {len(results_summary)}")
    print(f"Valid extractions: {sum(1 for r in results_summary if r['is_valid'])}/{len(results_summary)}")
    
    print(f"\nDocument Type Distribution:")
    doc_types = {}
    for result in results_summary:
        doc_type = result["doc_type"] or "unknown"
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
    
    for doc_type, count in doc_types.items():
        print(f"  {doc_type}: {count}")
    
    print(f"\n{'='*70}")
    print("Sample processing completed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    import os
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        print("Please set it before running the script.")
        exit(1)
    
    # Run samples
    Run_Samples()
