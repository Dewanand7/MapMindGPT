import random

knowledge = {
    "EDI": [
        ("What is EDI?", "EDI stands for Electronic Data Interchange, used for exchanging structured business documents."),
        ("What is X12?", "ANSI X12 is a common EDI standard used mainly in North America."),
        ("What is EDIFACT?", "EDIFACT is an international EDI standard."),
        ("What is AS2?", "AS2 is a protocol for securely exchanging EDI documents over HTTP."),
        ("What is an 850?", "EDI 850 is a purchase order transaction."),
        ("What is an 810?", "EDI 810 is an invoice transaction."),
        ("What is an 856?", "EDI 856 is an advance ship notice."),
        ("What is a 997?", "EDI 997 is a functional acknowledgement."),
        ("What is ISA segment?", "ISA is the interchange control header in X12."),
        ("What is GS segment?", "GS is the functional group header."),
        ("What is ST segment?", "ST identifies the transaction set."),
        ("What is partner onboarding?", "Partner onboarding means configuring a trading partner for EDI exchange."),
    ],

    "Oracle": [
        ("What is Oracle SOA Suite?", "Oracle SOA Suite is middleware for enterprise integration."),
        ("What is OIC?", "Oracle Integration Cloud is Oracle's cloud integration platform."),
        ("What is OSB?", "Oracle Service Bus is used for service virtualization and message routing."),
        ("What is BPEL?", "BPEL is used for process orchestration in Oracle SOA."),
        ("What is Mediator?", "Mediator performs routing and transformation."),
        ("What is dehydration?", "Dehydration stores process state in the database."),
        ("What is MDS?", "Metadata Services stores shared Oracle SOA artifacts."),
        ("What is Oracle AQ?", "Oracle Advanced Queuing provides message queuing."),
        ("What is JMS?", "JMS is Java Messaging Service for asynchronous messaging."),
        ("What is fault handling?", "Fault handling manages exceptions in integration flows."),
        ("What is composite?", "A composite bundles Oracle SOA components into one deployable unit."),
    ],

    "SEEBURGER": [
        ("What is SEEBURGER?", "SEEBURGER is a B2B integration and EDI platform."),
        ("What is BIS?", "Business Integration Suite is SEEBURGER's integration platform."),
        ("What is mapping designer?", "Mapping designer is used for data transformation."),
        ("What is routing?", "Routing determines message destination and processing flow."),
        ("What is message monitoring?", "Monitoring tracks message processing and errors."),
    ],

    "XML": [
        ("What is XML?", "XML is a markup language for structured hierarchical data."),
        ("What is XSD?", "XSD defines XML schema rules."),
        ("What is XPath?", "XPath selects XML nodes using expressions."),
        ("What is XSLT?", "XSLT transforms XML documents."),
        ("What is XSLT 1.0?", "XSLT 1.0 is the original XML transformation language version."),
        ("What are namespaces?", "Namespaces prevent XML element naming conflicts."),
        ("What is template matching?", "Template matching applies XSLT rules to nodes."),
        ("What is recursion in XSLT?", "Recursion is used instead of loops for repeated processing."),
    ],

    "AI": [
        ("What is AI?", "Artificial intelligence enables machines to simulate intelligent behavior."),
        ("What is machine learning?", "Machine learning learns patterns from data."),
        ("What is transformer model?", "Transformers use self-attention to process sequences."),
        ("What is tokenization?", "Tokenization converts text into smaller units."),
        ("What are embeddings?", "Embeddings represent tokens as vectors."),
        ("What is inference?", "Inference is generating predictions using a trained model."),
    ]
}

samples = []

for _ in range(100000):
    domain = random.choice(list(knowledge.keys()))
    q, a = random.choice(knowledge[domain])

    variation = random.choice([
        f"User: {q}\nAssistant: {a}\n",
        f"User: Explain {q.lower()}\nAssistant: {a}\n",
        f"User: Can you tell me {q.lower()}\nAssistant: {a}\n",
        f"User: Help me understand {q.lower()}\nAssistant: {a}\n"
    ])

    samples.append(variation)

with open("data/corpus.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(samples))

print("Dataset generated successfully.")