# Multimodal Systems — Full Course Module

## Module Overview
This module takes learners from text-only systems to multimodal workflows that process images, documents, and audio alongside text. The emphasis is on vision-language models, document intelligence pipelines, structured extraction, and optional speech workflows. Learners will build systems that see, read, and hear.

## Target Audience
- Developers and technical professionals
- Comfortable with Python, APIs, and Month 1–3 foundations (LLMs, RAG, fine-tuning)

## Learning Objectives
By the end of this module, learners will be able to:
- Understand how vision-language models process images and text together
- Build an OCR + summarization pipeline for document images
- Extract structured data from invoices, contracts, and forms
- Implement speech-to-text and voice agent workflows
- Evaluate multimodal system quality with appropriate metrics

---

## Prerequisites
- Month 1–3: LLM architecture, prompting, RAG, fine-tuning fundamentals
- Python 3.10+
- Access to a multimodal LLM API (OpenAI GPT-4o, Anthropic Claude 3.5 Sonnet, or local via Ollama with LLaVA)
- Basic familiarity with image processing concepts (PIL, OpenCV basics)
- Microphone access for audio labs (optional)

---

## Module Structure

| Module | Topic | Lab |
|--------|-------|-----|
| 4.1 | Vision-Language Foundations | OCR + summarization flow |
| 4.2 | Document Intelligence | Invoice/contract extraction pipeline |
| 4.3 | Optional Audio Workflows | Meeting notes generator |
| Mini-Project | Multimodal workflow with structured outputs | End-to-end system |

---

# Module 4.1: Vision-Language Foundations

## Core Concepts

### 1. What Are Vision-Language Models?

Vision-Language Models (VLMs) extend LLMs to process both images and text. Unlike text-only LLMs that can only read tokens, VLMs can "see" images by converting them into representations the model can reason about.

**The big idea:** A VLM takes two inputs — an image and a text prompt — and produces text output that can reference visual content. The model can describe what it sees, answer questions about the image, extract text from it, and reason about visual relationships.

**How it works — step by step:**

```
User: "What's in this image?"  +  [image of a cat]
                    │
                    ▼
         ┌─────────────────────┐
         │  Vision Encoder     │         Step 1: Convert the image into numbers
         │  (image → visual    │         (a sequence of "visual tokens")
         │   embeddings)       │
         └─────────┬───────────┘
                   │
                   ▼
         ┌─────────────────────┐
         │  Projection Layer   │         Step 2: Translate those numbers into
         │  (aligns visual     │         the same "language" the LLM uses
         │   embeds with text  │
         │   embedding space)  │
         └─────────┬───────────┘
                   │
                   ▼
         ┌─────────────────────┐
         │  LLM Decoder        │         Step 3: The LLM reads both the image
         │  (text + visual     │         tokens and your text prompt, then
         │   tokens → output)  │         generates a response
         └─────────┬───────────┘
                   │
                   ▼
         "This is a photograph of an orange tabby cat sitting on a windowsill."
```

**How VLMs work under the hood:**

The key architectural insight is that VLMs don't need to be trained from scratch. They combine a pre-trained vision encoder (like CLIP or SigLIP) with a pre-trained LLM, connected by a small projection layer:

1. **Vision Encoder**: A model like CLIP's ViT (Vision Transformer) converts the image into a sequence of visual patch embeddings. An image is split into 16×16 or 14×14 pixel patches, each patch becomes one vector.

2. **Projection Layer**: A small neural network (often a single linear layer or small MLP) maps vision embeddings into the LLM's text embedding space. This is the only new component — it's what gets trained to align the two modalities.

3. **LLM Decoder**: The LLM receives the visual tokens (projected) plus the text tokens as a unified sequence. It autoregressively generates text, attending to both visual and textual context.

**Analogy:** Think of the vision encoder as a translator who converts images into a language the LLM can understand. The projection layer is the dictionary that maps visual concepts to their text equivalents. The LLM is the thinker that uses both sources of information to produce the final answer.

**Three architectural approaches to VLMs:**

| Approach | Example Models | How It Works | Strengths |
|----------|---------------|--------------|-----------|
| **Cross-attention** | Flamingo, Kosmos-1 | Vision features attend into LLM layers via cross-attention | Efficient, separate visual stream |
| **Pre-fusion (projected)** | LLaVA, GPT-4o, Claude 3 | Vision tokens projected into LLM embedding space, concatenated with text tokens | Simple, effective, most common |
| **Unified embedding** | Fuyu-8B | Images are treated as special tokens in a unified vocabulary | No separate encoder, simpler architecture |

The **pre-fusion** approach is the most widely adopted today. Models like LLaVA show that you can train the projection layer on just a few hundred image-text pairs and get surprisingly good visual reasoning.

### 2. Image Embeddings vs Text Embeddings

Images and text are fundamentally different data types, but VLMs need to combine them in a shared space. Understanding the differences helps you debug multimodal systems:

```
Text Embedding:
"a cat sitting on a windowsill" → [0.23, -0.45, 0.89, ..., 0.12]  (1D sequence of tokens)

Image Embedding:
[image of a cat] → 256 patch embeddings × 1024 dims each → [patch_0, patch_1, ..., patch_255]
```

| Property | Text Embeddings | Image Embeddings |
|----------|----------------|------------------|
| **Granularity** | One vector per token (~word level) | One vector per image patch (~16×16 pixel region) |
| **Sequence length** | Variable (number of tokens) | Fixed for a given model (e.g., 256 patches for 224×224 image) |
| **Semantic density** | High — each token carries meaning | Low — individual patches have no semantic meaning; meaning emerges from full set |
| **Position information** | Positional encoding added explicitly | Positional encoding from patch grid layout |
| **Scale sensitivity** | None — text is scale-invariant | High — image resolution determines patch count and detail |

**Why this matters for building systems:**
- Image quality directly affects VLM performance — low-res images lose detail that the model needs for OCR or fine-grained recognition
- A VLM might "see" text in an image but miss small print or low-contrast elements
- The number of image patches consumes context window tokens — a 224×224 image might use 256 tokens, a 1024×1024 image might use 4096+ tokens

### 3. OCR — Optical Character Recognition

OCR is the process of extracting text from images. In the multimodal era, there are two approaches:

**Approach 1: Traditional OCR engine**

Traditional OCR follows a multi-step pipeline, like an assembly line:

```
  [scanned document image]           A photo or scan of a document
            │                         containing text
            ▼
  ┌─────────────────────┐
  │  Preprocessing      │           Clean up the image: convert to black/white,
  │  (grayscale,        │           fix brightness, straighten rotated text
  │   binarize,         │
  │   deskew)           │
  └─────────┬───────────┘
            │
            ▼
  ┌─────────────────────┐
  │  Text Detection     │           Find WHERE text exists in the image
  │  (find text regions)│           (draw boxes around text areas)
  └─────────┬───────────┘
            │
            ▼
  ┌─────────────────────┐
  │  Text Recognition   │           Read EACH text region and figure out
  │  (character/word    │           what characters/words it contains
  │   recognition)      │
  └─────────┬───────────┘
            │
            ▼
  ┌─────────────────────┐
  │  Postprocessing     │           Fix common errors, spell-check,
  │  (spell check,      │           reconstruct the original layout
  │   layout rebuild)   │           and reading order
  └─────────┬───────────┘
            │
            ▼
      "Extracted text with bounding boxes"
```

**What each step does:**
- **Preprocessing**: Converts color image to grayscale, then to pure black/white (binarization). Rotates slightly tilted images (deskew). This makes text easier to read.
- **Text Detection**: Scans the image to find rectangular regions containing text. Returns bounding boxes like: "Text found at position (x=50, y=100, width=200, height=30)".
- **Text Recognition**: For each detected text region, reads the pixels and determines which character each one represents. This is where machine learning happens — the model has been trained on millions of text images.
- **Postprocessing**: Fixes common OCR mistakes (e.g., "0" vs "O", "1" vs "l"), applies spell checking, and arranges the text in the correct reading order.

Tools: Tesseract, EasyOCR, PaddleOCR, AWS Textract, Google Document AI

**Approach 2: VLM-based OCR**

VLM-based OCR skips all the pipeline steps — it just shows the image to a vision-language model and asks it to read:

```
  [image with text]                    Any image containing text
            │
            ▼
  ┌─────────────────────┐
  │  VLM                │           The VLM "looks" at the image
  │                     │           like a human would, identifying
  │  "Read all the text │           text by its visual appearance
  │   in this image     │           and context, not by character
  │   and return it     │           recognition
  │   exactly."         │
  └─────────┬───────────┘
            │
            ▼
      "Extracted text in natural reading order"
```

**Why VLMs can do OCR:** VLMs were trained on millions of images that include text — signs, documents, webpages, books. They learned to recognize text visually, similar to how humans read. They don't need the multi-step pipeline because they process the whole image at once.

Tools: GPT-4o, Claude 3.5 Sonnet, Qwen-VL, LLaVA

**Comparing the approaches:**

| Factor | Traditional OCR | VLM-based OCR |
|--------|----------------|---------------|
| **Accuracy** | Very high for clean docs | Good but may hallucinate |
| **Speed** | Fast (ms per page) | Slower (1-5s per page) |
| **Cost** | Low (open source) | Higher (API calls) |
| **Layout preservation** | Bounding boxes + reading order | Natural text flow |
| **Handwriting** | Poor | Good to excellent |
| **Complex layouts** | Requires configuration | Handles naturally |
| **Hallucination risk** | None (deterministic) | May add/change text |
| **Integration complexity** | Medium (pipeline) | Low (single API call) |

**Best practice:** Use traditional OCR for high-volume, clean document extraction where accuracy is critical. Use VLM-based OCR for complex layouts, handwriting, or when you need understanding (not just extraction).

### 4. Captioning and Visual Question Answering

**Image captioning** generates a natural language description of an image. **Visual QA** answers specific questions about an image. These are the two core capabilities of VLMs.

```
Captioning:
Input:  [image] + "Describe this image in detail."
Output: "A modern office with three people sitting at a wooden table, 
         laptops open, whiteboard with diagrams in the background."

Visual QA:
Input:  [image of a chart] + "What was the revenue in Q3 2024?"
Output: "The revenue in Q3 2024 was $12.4 billion, representing a 
         15% increase from Q2 2024."
```

**What VLMs are good at (and bad at):**

| Task | VLM Performance | Notes |
|------|----------------|-------|
| Object recognition | ✅ Excellent | "Is there a dog in this image?" |
| Scene description | ✅ Excellent | "Describe the setting" |
| Text extraction (printed) | ✅ Good | "Read the sign in the background" |
| Text extraction (handwriting) | ⚠️ Variable | Depends on model and clarity |
| Counting objects | ⚠️ Fair | "How many people are in this photo?" — often wrong |
| Spatial relationships | ⚠️ Fair | "Is the cup to the left of the laptop?" |
| Fine-grained visual detail | ⚠️ Fair | Small text, subtle color differences |
| Mathematical reasoning on charts | ⚠️ Variable | Can misinterpret axes or scales |
| Hallucination of visual details | ❌ Can invent | Model may "see" things not present |
| Face recognition | ❌ Blocked | Safety filters in most API models |

**Prompting strategies for VLMs:**

```
Good: "Describe the contents of this image in detail, including any text you see."
Bad:  "What's in this image?"

Better for extraction:
"Read all the text in this image and return it exactly as written, 
preserving the original formatting and line breaks."

Better for analysis:
"Look at this chart. Tell me: (1) What type of chart is this? 
(2) What are the axes? (3) What is the trend? (4) What is the 
most important data point?"

Better for structured output:
"Extract the following fields from this document image and return them as JSON:
{
  "document_type": "...",
  "date": "...",
  "total_amount": ...,
  "vendor_name": "...",
  "line_items": [{"description": "...", "amount": ...}]
}"
```

---

## Lab 4.1: Build an OCR + Summarization Flow

### Goal
Build a pipeline that takes an image of a document, extracts text using both traditional OCR and VLM-based OCR, then summarizes the content. Compare the two approaches.

### Steps
1. Find or create 3 document images: a printed invoice, a handwritten note, and a newspaper clipping
2. Extract text using Tesseract (traditional OCR):
   - Install: `apt install tesseract-ocr` and `pip install pytesseract Pillow`
   - Preprocess images (grayscale, threshold, deskew)
   - Run OCR and record the raw output
3. Extract text using a VLM (GPT-4o or LLaVA):
   - Send the image with prompt: "Read all text in this image exactly as written"
   - Record the output
4. Compare the two extraction results:
   - Which approach made fewer errors?
   - Which approach handled handwriting better?
   - Which approach was faster?
5. Feed both extraction results into an LLM for summarization:
   - "Summarize the key information from this document in 3 bullet points"
6. Compare the summaries — does OCR quality affect the summary quality?

### Expected Observations
- Tesseract will be accurate on printed text but fail on handwriting
- VLM will handle handwriting but may hallucinate or miss small text
- Summaries from poor OCR will miss or distort key information
- The combination of OCR + LLM summarization is a powerful pattern

### Deliverable
A Python script that takes an image path, runs both OCR methods, and produces a summary with a comparison note.

---

## Exercises

1. **VLM Prompt Engineering**: Take a single image of a busy street scene. Write 5 different prompts asking the VLM to describe it. Compare: which prompt produces the most useful description? Which produces the most accurate? Which hallucinates the most?

2. **OCR Quality Test**: Take a screenshot of a webpage, a photo of a receipt, and a scanned form. Run Tesseract on all three. For each, count: total words in ground truth, correctly extracted words, wrongly extracted words, missed words. Calculate Word Error Rate (WER) = (substitutions + deletions + insertions) / total words.

3. **VLM Hallucination Audit**: Give a VLM a blank white image and ask "Describe this image in detail." Then give it an image with a single object and ask it to count the objects. Document any hallucinations you observe.

---

# Module 4.2: Document Intelligence

## Core Concepts

### 1. What is Document Intelligence?

Document intelligence is the application of AI to extract, understand, and structure information from documents. Unlike simple OCR (which just extracts text), document intelligence understands the *structure* and *semantics* of the document.

**The hierarchy of document understanding:**

Documents exist at three levels of understanding. Each level builds on the previous one:

```
Level 1: Raw text extraction (OCR)
  "Invoice #INV-2024-0891 Date: 2024-03-15 Item: Consulting Services..."
  
  → This is like reading individual words without understanding what they mean.
     You have the text, but you don't know which part is the header, which
     is the total, or what the document is about.

Level 2: Layout understanding
  ┌─────────────────────────────────────────────┐
  │                  HEADER                     │
  │  Company Logo    Invoice #INV-2024-0891     │
  │  Acme Corp       Date: 2024-03-15          │
  ├─────────────────────────────────────────────┤
  │               LINE ITEMS                    │
  │  Qty  Description          Unit Price  Total│
  │  2    Consulting Services  $150/hr     $600 │
  │  1    Software License     $1,200      $1,200│
  ├─────────────────────────────────────────────┤
  │                TOTALS                       │
  │  Subtotal: $1,800  Tax: $180  Total: $1,980│
  └─────────────────────────────────────────────┘
  
  → Now you understand the STRUCTURE. You know there's a header section,
     a table of line items, and a totals section. But you still need to
     map this structure to your application's needs.

Level 3: Semantic extraction (Document Intelligence)
  {
    "document_type": "invoice",
    "vendor": "Acme Corp",
    "invoice_number": "INV-2024-0891",
    "date": "2024-03-15",
    "line_items": [
      {"description": "Consulting Services", "quantity": 2, "unit_price": 150, "total": 600},
      {"description": "Software License", "quantity": 1, "unit_price": 1200, "total": 1200}
    ],
    "subtotal": 1800,
    "tax": 180,
    "total": 1980
  }
  
  → This is the GOAL: structured, validated data that your application
     can actually use. You know what each piece of information MEANS
     (vendor, date, line items) and where it belongs in your data model.
```

**Analogy:** 
- **OCR** is like being able to read individual words on a page — you can read "Acme Corp" but you don't know if it's the sender, receiver, or just a random name.
- **Layout understanding** is like knowing the document has a header, a table, and a footer — you can identify the structure but still need to figure out what each part means.
- **Document intelligence** is like an expert accounts payable clerk who looks at the document and immediately knows: "This is an invoice from Acme Corp for $1,980, dated March 15, 2024, with two line items." It requires both reading AND comprehension.

### 2. Layout Understanding and Document Parsing

Documents have visual structure — headers, tables, lists, paragraphs, signatures. Understanding this layout is essential for accurate extraction.

**Common document layouts:**

| Layout Type | Structure | Example Documents |
|-------------|-----------|-------------------|
| **Form-like** | Label-value pairs, fields | Invoices, applications, tax forms |
| **Tabular** | Column-row structure | Financial reports, bills of materials |
| **Narrative** | Paragraphs, headings, sections | Contracts, reports, articles |
| **Mixed** | Combination of above | Annual reports, whitepapers |

**Three approaches to document parsing:**

**Approach 1: Template-based**
```python
# Define regex patterns for each field
import re

def parse_invoice_template(text):
    invoice_num = re.search(r'Invoice\s*#?\s*:\s*(\S+)', text)
    date = re.search(r'Date\s*:\s*(\d{4}-\d{2}-\d{2})', text)
    total = re.search(r'Total\s*:\s*\$\s*([\d,]+\.?\d*)', text)
    return {
        "invoice_number": invoice_num.group(1) if invoice_num else None,
        "date": date.group(1) if date else None,
        "total": total.group(1) if total else None,
    }
```
Pros: Fast, deterministic, no API calls
Cons: Brittle — breaks on format variations, needs per-template coding

**Approach 2: VLM-based extraction**
```python
def parse_invoice_vlm(image_path):
    prompt = """Extract the following fields from this invoice image.
Return ONLY valid JSON with these fields:
{
    "invoice_number": "...",
    "date": "...",
    "vendor_name": "...",
    "vendor_address": "...",
    "customer_name": "...",
    "line_items": [
        {"description": "...", "quantity": 0, "unit_price": 0.0, "total": 0.0}
    ],
    "subtotal": 0.0,
    "tax": 0.0,
    "total": 0.0,
    "currency": "..."
}"""
    response = call_vlm(image_path, prompt)
    return json.loads(extract_json(response))
```
Pros: Handles layout variations, no template coding
Cons: Slower, costs per call, may hallucinate field values

**Approach 3: Hybrid (OCR + layout analysis + LLM)**
```python
def parse_invoice_hybrid(image_path):
    # Step 1: OCR with positional data
    ocr_results = tesseract_with_boxes(image_path)
    # Returns: [{"text": "Invoice #123", "bbox": [x1,y1,x2,y2], "page": 1}, ...]
    
    # Step 2: Group by layout regions
    regions = group_by_spatial_proximity(ocr_results)
    # Returns: {"header": [...], "line_items_table": [...], "totals": [...]}
    
    # Step 3: LLM extracts structured data from structured input
    prompt = f"""Given this invoice text organized by regions, extract the structured data.

HEADER:
{regions['header']}

LINE ITEMS TABLE:
{regions['line_items_table']}

TOTALS:
{regions['totals']}

Return as JSON with fields: invoice_number, date, vendor, line_items, totals."""
    
    return llm_extract(prompt)
```
Pros: Best accuracy, deterministic OCR + flexible LLM
Cons: More complex pipeline, two-stage processing

### 3. Table Extraction

Tables are the hardest structure to extract from documents. They come in countless formats:
- With or without borders (many tables have no visible lines)
- Merged cells (e.g., a header spanning multiple columns)
- Multi-line cells (text that wraps within a cell)
- Nested tables (a table inside another table's cell)
- Rotated or split tables (table continues on next page)

**Why tables are hard:** Unlike text which flows in one direction, tables are 2D structures. You need to understand both the content AND the spatial relationships between cells. A cell says "$1,500" — but is that the unit price, the total, or the tax? You need the column header to know.

**Table extraction approaches:**

| Approach | Method | Best For | Limitations |
|----------|--------|----------|-------------|
| **Detection + recognition** | Detect table boundaries, recognize cell structure | Bordered tables | Fails on borderless tables |
| **VLM direct extraction** | "Read this table as JSON" | Complex layouts | May hallucinate or misalign columns |
| **Camelot/Tabula** | PDF table extraction libraries | Digital PDFs | Only works on PDFs, not images |
| **OCR + row grouping** | Detect rows by Y-coordinate proximity | Scanned tables | Requires careful threshold tuning |

```python
# Simple column-based table extraction from OCR output
def extract_table_from_ocr(ocr_items):
    """Group OCR text items into table rows by Y-position.
    
    The idea: items that share roughly the same vertical position
    are on the same row. Items sorted by horizontal position give
    the column order within that row.
    """
    # Sort by vertical position (Y coordinate), then horizontal (X)
    sorted_items = sorted(ocr_items, key=lambda x: (x['bbox'][1], x['bbox'][0]))
    
    # Group items that are on the same row (Y within threshold)
    rows = []
    current_row = []
    current_y = None
    y_threshold = 10  # pixels — items within 10px vertically are on same row
    
    for item in sorted_items:
        y = item['bbox'][1]  # top edge of the text bounding box
        if current_y is None or abs(y - current_y) <= y_threshold:
            current_row.append(item)
            current_y = y
        else:
            # New row — save the current row (sorted left-to-right)
            rows.append(sorted(current_row, key=lambda x: x['bbox'][0]))
            current_row = [item]
            current_y = y
    if current_row:
        rows.append(sorted(current_row, key=lambda x: x['bbox'][0]))
    
    # Convert to text table
    return [[item['text'] for item in row] for row in rows]
```

**Visual example of how row grouping works:**

```
OCR output (each box is a text item with position):
┌─────────┐  ┌──────────┐  ┌───────────┐
│ Product │  │ Quantity │  │    Price  │    ← Y ≈ 50px (same row)
└─────────┘  └──────────┘  └───────────┘
┌─────────┐  ┌──────────┐  ┌───────────┐
│  Apple  │  │    5     │  │   $0.80   │    ← Y ≈ 100px (same row)
└─────────┘  └──────────┘  └───────────┘
┌─────────┐  ┌──────────┐  ┌───────────┐
│ Banana  │  │   12     │  │   $0.30   │    ← Y ≈ 150px (same row)
└─────────┘  └──────────┘  └───────────┘

After grouping by Y-position:
[["Product", "Quantity", "Price"],
 ["Apple", "5", "$0.80"],
 ["Banana", "12", "$0.30"]]
```

### 4. Schema Mapping and Validation

Once you extract fields from a document, you need to map them to a target schema and validate the results.

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import date

class InvoiceLineItem(BaseModel):
    description: str = Field(..., min_length=1)
    quantity: float = Field(..., gt=0)
    unit_price: float = Field(..., ge=0)
    total: float = Field(..., ge=0)

    @field_validator('total')
    @classmethod
    def check_total(cls, v, info):
        """Validate that total ≈ quantity × unit_price (within rounding)."""
        values = info.data
        expected = values.get('quantity', 0) * values.get('unit_price', 0)
        if abs(v - expected) > 0.02:
            raise ValueError(f'Total {v} does not match qty×price = {expected}')
        return v

class Invoice(BaseModel):
    invoice_number: str = Field(..., pattern=r'^INV-\d{4}-\d+$')
    date: date
    vendor_name: str = Field(..., min_length=1)
    vendor_address: Optional[str] = None
    customer_name: str = Field(..., min_length=1)
    line_items: list[InvoiceLineItem] = Field(..., min_length=1)
    subtotal: float = Field(..., ge=0)
    tax: float = Field(..., ge=0)
    total: float = Field(..., ge=0)
    currency: str = Field(default="USD", pattern=r'^[A-Z]{3}$')

    @field_validator('total')
    @classmethod
    def check_grand_total(cls, v, info):
        """Validate that total ≈ subtotal + tax (within rounding)."""
        values = info.data
        expected = values.get('subtotal', 0) + values.get('tax', 0)
        if abs(v - expected) > 0.02:
            raise ValueError(f'Total {v} does not match subtotal+tax = {expected}')
        return v
```

**Why schema validation matters in production:**
- Catches extraction errors before they reach downstream systems
- Ensures consistent output format for databases and APIs
- Provides structured error messages for debugging
- Acts as documentation for what the extraction system should produce

### 5. Document Classification Before Extraction

Not all documents are the same type. In production, you often need to route documents to the right extraction pipeline — you wouldn't try to extract an invoice number from a contract, or a patient name from a tax form. Each document type has its own specific fields and layout.

**Why classify first?**

Think of it like a post office: before delivering mail, you need to figure out if it's a letter, a package, or a bill — each goes to a different department. Similarly, an invoice needs fields like "vendor", "line items", and "total", while a contract needs fields like "parties", "terms", and "effective date".

```
                    [Document Image]
                           │
                           ▼
                 ┌─────────────────────┐
                 │  Document           │        The classifier looks at the image
                 │  Classifier         │        and determines: "This is an invoice"
                 └─────────┬───────────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              ▼              ▼
     ┌────────────┐ ┌────────────┐ ┌────────────┐
     │ Invoice    │ │ Contract   │ │ Receipt    │        Each document type goes to
     │ Extractor  │ │ Extractor  │ │ Extractor  │        its own extraction pipeline
     │            │ │            │ │            │        with the right prompts and schema
     └────────────┘ └────────────┘ └────────────┘
```

```python
def classify_document(image):
    """Classify document type using a VLM."""
    prompt = """Classify this document into one of these types:
- invoice
- contract
- receipt
- report
- form
- letter
- other

Return only the type name, nothing else."""
    return call_vlm(image, prompt).strip().lower()
```

**How the classifier works:** A VLM looks at the document image and identifies visual patterns that distinguish document types:
- **Invoices**: Usually have tables with line items, totals, vendor/customer sections
- **Contracts**: Dense text, signatures at the bottom, clause numbering
- **Receipts**: Short, itemized list, total at bottom, store name at top
- **Forms**: Filled-in fields, checkboxes, structured layout

The classifier doesn't need to read every word — it just needs to recognize the overall layout pattern, similar to how you can tell a document type at a glance before reading it.

---

## Lab 4.2: Invoice/Contract Extraction Pipeline

### Goal
Build a complete document intelligence pipeline that takes a document image and extracts structured data into a validated schema.

### Steps
1. Find or create 3 document images of the same type (e.g., 3 different invoices from different vendors)
2. Build a VLM-based extraction function with a structured prompt
3. Implement pydantic schema validation on the extracted fields
4. Run the pipeline on all 3 documents
5. For each document, manually verify the extracted fields for accuracy
6. Build a fallback mechanism: if validation fails, retry with a more specific prompt
7. Add a document classifier upfront so the pipeline can handle multiple document types

### Expected Observations
- Schema validation catches extraction errors (wrong field types, missing values)
- Different vendor layouts require different extraction strategies
- The VLM may get field names wrong or invent values for missing fields
- Retry with specific prompts improves accuracy
- Document classification is critical for multi-type pipelines

### Deliverable
A Python script that takes an image path, classifies the document, extracts structured data, validates it, and returns clean JSON.

---

## Exercises

1. **Schema Design**: Design a pydantic schema for a medical prescription form. Include fields like patient name, medication name, dosage, frequency, prescriber, date, pharmacy info. Add at least 3 field validators.

2. **Table Extraction**: Take a screenshot of a table (bank statement, stock prices, sports scores). Extract it as a list of dictionaries using a VLM. Compare the VLM output with ground truth. Where does the VLM make mistakes?

3. **Multi-Document Pipeline**: Create a pipeline that handles invoices, receipts, and contracts using one VLM with different prompts per type. Test it on 2 documents of each type. Report accuracy per type.

---

# Module 4.3: Optional Audio Workflows

## Core Concepts

### 1. Speech-to-Text (ASR)

Automatic Speech Recognition (ASR) converts audio speech into text. Modern ASR systems use deep learning models that can handle multiple languages, accents, and acoustic conditions.

**How ASR works at a high level:**

The process of turning sound waves into written words happens in four stages. Let me walk through each one:

**Stage 1: Audio Waveform — What IS sound?**

Sound is a vibration that travels through air as a pressure wave. When you record audio, your microphone captures these vibrations as a digital signal — a series of numbers representing the air pressure at each moment in time. This raw recording is called a **waveform**.

```
What you hear:  "Hello"
                 │
What the mic records:  ∿∿∿∿∿∿∿∿∿∿∿∿∿  (a squiggly line of numbers)
                       │
What a computer sees:  [0.0, 0.1, 0.3, 0.8, 0.9, 0.6, 0.2, -0.1, -0.3, ...]
                       (thousands of numbers per second)
```

The waveform is just raw electrical signals — it doesn't "know" anything about language. It's like a photograph of sound: it captures the shape, but needs processing to understand what it means.

**Stage 2: Feature Extraction — Making sound readable**

Raw waveforms contain too much data and too much noise. Feature extraction converts the waveform into a more compact representation that highlights the important parts.

The most common representation is a **mel spectrogram**:

1. **Split the audio into tiny chunks** (called "frames"), each about 25 milliseconds long
2. **For each chunk, measure the frequencies present** — like a musical analysis showing which notes (frequencies) are playing at that moment
3. **Apply the "mel scale"** — this adjusts the frequency measurements to match how human ears actually hear. We're much better at distinguishing between low pitches (e.g., 100Hz vs 200Hz) than high pitches (e.g., 8000Hz vs 8100Hz)
4. **Create a 2D image** where the x-axis is time, the y-axis is frequency, and the color/brightness shows how loud each frequency is at each moment

```
Raw waveform:        ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿
                           │
                           ▼
Mel spectrogram:     ████░░████░░░░████
                     ██░░░░████░░░░██░░
                     █░░░░░░███░░░░██░░    (time →)
                     ░░░░░░░░░░░░░░░░░
                       frequency (low → high)
```

**Why this matters:** The mel spectrogram is like a "visual" representation of speech. A VLM or neural network can "read" it almost like reading an image of text. The spectrogram preserves the important phonetic information (what sounds are being made) while discarding irrelevant details (background hum, echo, recording quality).

**Stage 3: Acoustic Model — From sounds to speech units**

The acoustic model's job is to listen to the spectrogram and figure out what speech sounds (or units) are present. This is the hardest part because:

- The same word sounds different when spoken quickly vs slowly
- Different people pronounce the same word differently
- Background noise muddles the signal
- Words blend together (coarticulation) — the "t" in "at" sounds different from the "t" in "top"

**What are phonemes and subword units?**

- **Phonemes** are the smallest meaningful sound units in a language. English has about 40 phonemes. For example, the word "cat" is made of 3 phonemes: /k/ /æ/ /t/. The phoneme /p/ in "pin" is aspirated (with a puff of air), but the /p/ in "spin" is not — same letter, different phoneme realization.

- **Subword units** (also called "tokens" or "BPE units") are chunks of text that the model uses as its building blocks. Modern models like Whisper don't predict individual phonemes — they predict subword chunks. For example: "un", "##believ", "##able" might be three subword units that together spell "unbelievable".

The acoustic model takes the spectrogram (a sequence of frequency patterns) and outputs a sequence of these units. Think of it as: "given this pattern of frequencies over time, what sequence of speech sounds was most likely produced?"

```
Mel spectrogram:  ████░░████░░░░████
                  ██░░░░████░░░░██░░
                       │
                       ▼
Acoustic model outputs: "h" "eh" "l" "ow"  (phonemes)
           OR:          "hel" "lo"           (subword units)
```

**Stage 4: Language Model — Making it read like English**

The acoustic model might output: "hel lo" or "hello" — but it might also produce errors like "hell oh" or "helow". The language model uses its knowledge of how words typically appear in sentences to fix these errors.

For example:
- If the acoustic model says "I want to go to the store", the LM keeps it
- If the acoustic model says "I want to go to the sore", the LM might correct "sore" → "store" because "go to the store" is a much more common phrase
- If the acoustic model says "I wall ant to go", the LM corrects it to "I want to go"

**Why you need both models:** The acoustic model is good at matching sounds to units, but it can make mistakes. The language model knows what makes sense in context. By combining them, you get the best of both worlds: sound matching + language knowledge.

```
Acoustic model output:  "I wall ant to go to the sore"
                              │
                              ▼
Language model corrects: "I want to go to the store"
                              │
                              ▼
Final transcription:     "I want to go to the store"
```

**The complete pipeline:**

```
[Audio waveform]                    (raw recording: numbers representing air pressure)
       │
       ▼
┌─────────────────────┐
│  Feature Extraction │           (convert raw audio into a "picture" of frequencies)
│  (mel spectrogram   │           that highlights the important speech patterns
│   or raw waveform)  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Acoustic Model     │           (read the "picture" to identify what speech sounds
│  (audio → phonemes  │           or subword chunks are present, despite noise,
│   or subword units) │           different speakers, and accent variations)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Language Model     │           (use knowledge of English to fix errors, resolve
│  (sequence of units │           ambiguities, and produce grammatically correct text
│   → text)           │           that matches what the speaker actually meant)
└─────────┬───────────┘
          │
          ▼
     "Transcribed text"           (final output: readable, accurate text)
```

**Modern ASR models:**

Today's best models (like Whisper) combine all these stages into a single neural network trained end-to-end. You don't need to build each stage separately — the model learns to do feature extraction, acoustic modeling, and language modeling all at once from millions of hours of transcribed audio.

| Model | Type | Languages | Strengths |
|-------|------|-----------|-----------|
| **Whisper (OpenAI)** | End-to-end transformer | 100+ | Excellent accuracy, punctuation, multilingual |
| **DeepSpeech (Mozilla)** | RNN-based | ~30 | Open source, lightweight |
| **Wav2Vec 2.0 (Meta)** | Self-supervised | ~50 | Good with limited labeled data |
| **Speech-to-Text (Google)** | API | 125+ | Cloud-scale, optimized |

**Whisper is the current standard for developers.** It's open source, accurate, and supports dozens of languages directly.

```python
import whisper

# Load model (tiny, base, small, medium, large)
model = whisper.load_model("base")

# Transcribe audio file
result = model.transcribe("meeting_recording.mp3")
print(result["text"])

# With language specification
result = model.transcribe("meeting_recording.mp3", language="en")

# With speaker segments (if available)
for segment in result["segments"]:
    print(f"[{segment['start']:.1f}s - {segment['end']:.1f}s]: {segment['text']}")
```

**Whisper model size vs accuracy vs speed tradeoff:**

| Model | Parameters | Relative Speed | WER (English) | VRAM |
|-------|-----------|----------------|---------------|------|
| tiny | 39M | ~10x | ~12% | ~1GB |
| base | 74M | ~7x | ~9% | ~1GB |
| small | 244M | ~4x | ~6% | ~2GB |
| medium | 769M | ~2x | ~4% | ~5GB |
| large | 1550M | 1x | ~3% | ~10GB |

### 2. Speaker Diarization

Speaker diarization answers the question: "Who spoke when?" It partitions an audio stream into homogeneous segments based on speaker identity.

**Why is this needed?** When you transcribe a meeting, you get one long block of text. But meetings are conversations — knowing who said what is crucial for understanding decisions, action items, and context. Diarization labels each segment of speech with the speaker's identity.

**The problem is hard because:**
- The model doesn't know how many speakers there are in advance
- Speakers may have similar voices
- People interrupt each other
- Background noise can confuse the model

**How diarization works (embedding clustering approach):**

1. **Split audio into short segments** (1-3 seconds) — short enough that each segment likely contains one speaker
2. **Extract a "speaker embedding" from each segment** — a numerical fingerprint (vector) that captures the unique characteristics of a person's voice (pitch, resonance, speaking style). Think of it like a voice "fingerprint"
3. **Cluster embeddings** — group similar voice fingerprints together. If two segments have very similar embeddings, they're likely the same speaker
4. **Assign labels** — name the clusters "Speaker A", "Speaker B", etc.

```
Audio recording:  [Alice speaks] [Bob speaks] [Alice speaks again] [Bob responds]
                      │               │              │                    │
                      ▼               ▼              ▼                    ▼
Voice embeddings:   [vec_A1]       [vec_B1]       [vec_A2]            [vec_B2]
                      │               │              │                    │
                      ▼               ▼              ▼                    ▼
Clustering:        Speaker A      Speaker B      Speaker A           Speaker B
```

**What makes a good speaker embedding?**
- It should be the same for the same person regardless of what they're saying
- It should be different for different people
- It should be robust to background noise
- The model that produces these embeddings was trained on thousands of hours of speech from many speakers

```python
# Simplified diarization pipeline
from pyannote.audio import Pipeline

# Load pre-trained diarization pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.0",
    use_auth_token="your_hf_token"  # Requires Hugging Face token
)

# Run diarization
diarization = pipeline("meeting_recording.mp3")

# Iterate over speaker turns
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"[{turn.start:.1f}s - {turn.end:.1f}s] {speaker}: ...")
```

**Combining diarization with transcription (the full pipeline):**

Whisper gives you WHAT was said (text), but not WHO said it. PyAnnote gives you WHO spoke WHEN, but not WHAT they said. By combining both, you get a complete picture.

```
Audio file (meeting recording)
       │
       ├──────────────────────┐
       ▼                      ▼
┌─────────────────────┐ ┌─────────────────────┐
│  Whisper (ASR)      │ │  PyAnnote           │
│  (what was said)    │ │  (who spoke when)   │
│  → text + timestamps│ │  → speaker segments  │
└─────────┬───────────┘ └───────────┬──────────┘
          │                          │
          └──────────┬───────────────┘
                     ▼
          ┌─────────────────────┐
          │  Alignment          │
          │                     │
          │  Match text         │
          │  timestamps with    │
          │  speaker segments   │
          │  to produce:        │
          │  "Speaker A: text"  │
          └─────────────────────┘
                     │
                     ▼
          "Speaker A: Hello, I'm Alice"
          "Speaker B: Hi Alice, this is Bob"
```

### 3. Audio Summarization and Meeting Notes

Once audio is transcribed and diarized, you can apply LLM-based summarization to extract key information.

```python
def summarize_meeting(transcript, speaker_diarization=None):
    """Generate meeting notes from a transcript."""
    
    if speaker_diarization:
        formatted = format_with_speakers(transcript, speaker_diarization)
    else:
        formatted = transcript
    
    prompt = f"""Below is a meeting transcript. Generate structured meeting notes:

1. **Meeting Title**: (short descriptive title)
2. **Date**: (infer from context or note as "not specified")
3. **Attendees**: (list of speakers mentioned)
4. **Key Discussion Points**: (3-5 bullet points)
5. **Decisions Made**: (list any decisions)
6. **Action Items**: (who does what by when)
7. **Follow-up**: (next meeting date, pending items)

TRANSCRIPT:
{formatted}

Return as structured markdown."""

    return call_llm(prompt)

def format_with_speakers(transcript, diarization):
    """Merge transcript text with speaker labels."""
    # Simplified — real implementation aligns timestamps
    formatted = []
    for segment_text, speaker in diarization.align(transcript):
        formatted.append(f"[{speaker}]: {segment_text}")
    return "\n".join(formatted)
```

### 4. Voice Agents

A voice agent combines ASR, LLM reasoning, and Text-to-Speech (TTS) to create a conversational experience over voice. Instead of typing to a chatbot, you speak to it and it speaks back.

**How a voice agent works — step by step:**

1. **You speak** — the microphone captures your voice
2. **ASR (Whisper)** — converts your speech to text. Now the system knows what you asked
3. **LLM (GPT-4o, etc.)** — processes the text and decides what to do. It might:
   - Answer the question directly
   - Call a tool (weather API, database lookup, calendar)
   - Ask for clarification
4. **TTS (Text-to-Speech)** — converts the LLM's text response back into speech
5. **You hear the response** — through your speakers or headphones

```
User speaks: "What's the weather in Nairobi?"
       │
       ▼
┌─────────────────────┐
│  ASR (Whisper)      │           The microphone captures your voice,
│  → "What's the      │           and Whisper converts it to the text:
│     weather in      │           "What's the weather in Nairobi?"
│     Nairobi?"       │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  LLM + Tools        │           The LLM reads the question, decides
│                     │           it needs to check a weather API,
│  → Calls weather    │           retrieves the data, and formulates
│    API → "22°C..."  │           a natural response.
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  TTS (ElevenLabs,   │           The TTS model converts the text
│   OpenAI TTS)       │           response into natural-sounding speech.
│  → Audio response   │           The voice sounds human-like.
└─────────┬───────────┘
          │
          ▼
User hears: "The weather in Nairobi is 22°C and partly cloudy."
```

**Key voice agent challenges:**

| Challenge | Why It's Hard | Current Solutions |
|-----------|---------------|-------------------|
| **Latency** | People expect <500ms responses in conversation. Each step (ASR + LLM + TTS) adds delay | Streaming ASR (transcribe while speaking), streaming TTS (start speaking before full response), edge models |
| **Turn detection** | When has the user stopped speaking? Silence doesn't always mean "done" | VAD (Voice Activity Detection), end-of-turn prediction models, barge-in detection |
| **Interruption** | User might interrupt mid-response. Agent needs to stop and listen | Audio level monitoring, interruption detection, graceful response cancellation |
| **Background noise** | Microphone picks up everything — music, other people, TV | Noise reduction preprocessing, directional microphones, VAD filtering |
| **Accents** | ASR may misrecognize accented speech | Whisper handles 100+ languages, but accent-specific fine-tuning helps |

```python
import whisper
import openai
from openai import OpenAI

# Simplified voice agent loop
def voice_agent_loop():
    model = whisper.load_model("base")
    client = OpenAI()
    
    while True:
        # 1. Record audio (simplified — use sounddevice or pyaudio)
        audio = record_audio(duration=5, silence_threshold=0.5)
        
        # 2. Transcribe — convert speech to text
        result = model.transcribe(audio)
        user_text = result["text"]
        print(f"User: {user_text}")
        
        if "exit" in user_text.lower():
            break
        
        # 3. LLM response — decide what to say back
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": user_text}]
        )
        agent_text = response.choices[0].message.content
        print(f"Agent: {agent_text}")
        
        # 4. Text-to-Speech — convert response to audio
        tts_response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",        # "alloy", "echo", "fable", "onyx", "nova", "shimmer"
            input=agent_text
        )
        play_audio(tts_response.content)  # play the audio
```

---

## Lab 4.3: Meeting Notes Generator

### Goal
Build a pipeline that takes an audio recording, transcribes it with speaker diarization, and generates structured meeting notes.

### Steps
1. Record or find a short meeting-style audio (2-5 minutes, 2-3 speakers)
2. Transcribe using Whisper (base or small model)
3. (Optional) Add speaker diarization using pyannote-audio
4. Generate structured meeting notes using an LLM
5. Test the pipeline with:
   - Clean audio (quiet room, clear speakers)
   - Noisy audio (background sounds, overlapping speech)
   - Different accents
6. Compare note quality across audio conditions

### Expected Observations
- Clean audio produces accurate transcripts and good notes
- Background noise degrades ASR accuracy significantly
- Overlapping speech is poorly handled by both ASR and diarization
- LLM can produce reasonable notes even from imperfect transcripts
- Speaker labels add significant value to meeting notes

### Deliverable
A Python script that takes an audio file and produces structured meeting notes with a quality assessment.

---

## Exercises

1. **ASR Comparison**: Record yourself reading the same paragraph in 3 conditions: quiet room, with background music, and with another person talking nearby. Run all 3 through Whisper. Calculate WER for each. Plot the relationship between noise level and accuracy.

2. **Prompt for Summarization**: Take the same meeting transcript and try 3 different summarization prompts. Compare: which produces the most actionable notes? Which preserves the most detail? Which is shortest?

3. **Voice Agent Prototype**: Build a simple voice agent that answers questions about a document. The user speaks a question, the agent retrieves from a RAG system, and speaks the answer. Test it with 5 questions and measure end-to-end latency.

---

## Mini-Project: Multimodal Workflow with Structured Outputs

### Goal
Build an end-to-end multimodal system that takes a mixed input (document image + optional audio note) and produces structured, validated output with a clear evaluation of quality.

### Requirements
1. **Input**: Accept at least two modalities — document image (required) and audio note (optional)
2. **Processing Pipeline**:
   - Extract text from image (OCR or VLM)
   - If audio provided, transcribe it
   - Combine and analyze both sources
   - Extract structured data into a validated schema
3. **Output**: Structured JSON validated against a pydantic schema
4. **Evaluation**: Measure extraction accuracy against ground truth for at least 5 test documents
5. **Fallback handling**: If VLM extraction fails validation, retry with a traditional OCR approach

### Suggested Scenarios
- **Expense report system**: Take a photo of a receipt, record a voice note explaining the expense, output structured expense entry
- **Medical intake form**: Scan a patient form, record doctor's notes, output structured patient record
- **Field inspection report**: Take photos of equipment, record observations, output structured inspection report with findings and action items
- **Legal document review**: Scan a contract page, record lawyer's commentary, output structured clause analysis

### Deliverable
A GitHub-ready directory containing:
- `pipeline.py` — multimodal processing pipeline
- `schemas.py` — pydantic schemas with validators
- `evaluate.py` — evaluation script against ground truth
- `test_data/` — sample images and audio (or instructions to generate)
- `report.md` — results summary with accuracy metrics

### Rubric (100 points)
- **Pipeline design (25 points)**: Clean architecture, proper error handling, fallback mechanisms
- **Extraction accuracy (25 points)**: Schema-validated extraction with high field-level accuracy
- **Multimodal integration (25 points)**: Effective combination of image and audio inputs
- **Evaluation (25 points)**: Rigorous accuracy measurement, error analysis, documented tradeoffs

---

## Assessment: Quick Quiz (5 Questions)

1. **What are the three main architectural components of a vision-language model?**
   A vision encoder (e.g., CLIP ViT) that converts images to patch embeddings, a projection layer that maps those embeddings into the LLM's text embedding space, and the LLM decoder that processes both visual and text tokens to generate output.

2. **When would you choose traditional OCR over VLM-based OCR?**
   Traditional OCR is preferred for high-volume, clean document extraction where accuracy is critical and hallucination is unacceptable. VLM-based OCR is better for complex layouts, handwriting, and when understanding (not just extraction) is needed.

3. **What is the difference between OCR and document intelligence?**
   OCR extracts raw text from images. Document intelligence understands the structure and semantics of the document — it knows which text is a header, which is a table cell, and which number is the total. Document intelligence = OCR + layout understanding + semantic extraction.

4. **Why is schema validation important in document extraction pipelines?**
   Schema validation catches extraction errors before they reach downstream systems, ensures consistent output format for databases/APIs, provides structured error messages for debugging, and acts as living documentation of what the extraction system should produce.

5. **What are the main challenges in building a voice agent?**
   Low latency requirements (<500ms for natural conversation), accurate turn detection (knowing when the user has finished speaking), handling interruptions, maintaining context across turns, dealing with background noise and different accents, and choosing an appropriate TTS voice.

---

## Common Pitfalls and How to Address Them

- **Assuming VLMs are perfectly accurate at reading text**
  VLMs can hallucinate text in images, especially small, rotated, or low-contrast text. *Solution*: Always validate extracted text against the image, use traditional OCR as a fallback, and implement human review for critical fields.

- **Ignoring image resolution**
  Low-resolution images lose detail that VLMs need for accurate OCR and recognition. *Solution*: Ensure minimum resolution (300 DPI for scanned documents), preprocess images (sharpen, contrast adjust), and test your pipeline with worst-case image quality.

- **Treating all documents as the same type**
  A single extraction prompt won't work well for invoices, contracts, and forms. *Solution*: Implement document classification first, then route to type-specific extraction pipelines.

- **Underestimating audio quality requirements**
  Background noise, overlapping speech, and accents significantly degrade ASR accuracy. *Solution*: Test with real-world audio conditions, use noise reduction preprocessing, and design your system to handle imperfect transcripts.

- **No fallback for extraction failures**
  VLM extraction can fail silently — returning plausible-looking but wrong data. *Solution*: Implement schema validation with retry logic, compare VLM output with traditional OCR for consistency, and log all extraction confidence scores.

---

## Resources

- **Papers**: "Visual Instruction Tuning" (LLaVA, 2023), "CLIP: Learning Transferable Visual Models" (Radford et al., 2021), "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper, 2022)
- **Models**: LLaVA (open source VLM), CLIP (vision encoder), Whisper (ASR), pyannote-audio (diarization)
- **Tools**: Tesseract (OCR), Camelot (PDF table extraction), PaddleOCR (layout-aware OCR), LangChain (multimodal chains)
- **Libraries**: `transformers`, `pillow`, `opencv-python`, `pytesseract`, `whisper`, `pyannote.audio`, `pydantic`
- **APIs**: OpenAI GPT-4o (vision), Anthropic Claude 3.5 Sonnet (vision), ElevenLabs (TTS), Google Cloud Speech-to-Text

---

## Code Examples

### Basic VLM Call with OpenAI

```python
from openai import OpenAI
import base64

client = OpenAI()

def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def analyze_image(image_path, prompt):
    b64_image = image_to_base64(image_path)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content

# Usage
result = analyze_image("invoice.jpg", "Extract the total amount and date from this invoice.")
print(result)
```

### OCR with Tesseract

```python
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

def preprocess_image(image_path):
    """Preprocess image for better OCR accuracy."""
    img = Image.open(image_path)
    
    # Convert to grayscale
    img = img.convert("L")
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    
    # Sharpen
    img = img.filter(ImageFilter.SHARPEN)
    
    # Binarize (threshold)
    img = img.point(lambda x: 255 if x > 128 else 0)
    
    return img

def ocr_with_boxes(image_path):
    """Extract text with bounding box information."""
    img = preprocess_image(image_path)
    
    # Get detailed OCR data
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    
    results = []
    for i in range(len(data["text"])):
        if data["text"][i].strip():
            results.append({
                "text": data["text"][i],
                "confidence": data["conf"][i],
                "bbox": [
                    data["left"][i],
                    data["top"][i],
                    data["left"][i] + data["width"][i],
                    data["top"][i] + data["height"][i]
                ]
            })
    
    return results

def ocr_text(image_path):
    """Simple text extraction."""
    img = preprocess_image(image_path)
    text = pytesseract.image_to_string(img)
    return text.strip()
```

### Whisper Transcription

```python
import whisper

def transcribe_audio(audio_path, model_size="base"):
    """Transcribe audio file to text."""
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    return result

def transcribe_with_timestamps(audio_path, model_size="base"):
    """Transcribe with word-level timestamps."""
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, word_timestamps=True)
    return result

# Example usage
result = transcribe_with_timestamps("meeting.mp3")
print(f"Detected language: {result['language']}")
print(f"Full text: {result['text']}")
for segment in result["segments"]:
    print(f"[{segment['start']:.1f}s - {segment['end']:.1f}s]: {segment['text']}")
```

### Complete Document Extraction Pipeline

```python
import json
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
import pytesseract
from PIL import Image
from typing import Optional

client = OpenAI()

class ExtractedDocument(BaseModel):
    document_type: str = Field(..., pattern=r"^(invoice|receipt|contract|form|letter|other)$")
    date: Optional[str] = None
    document_number: Optional[str] = None
    vendor_or_sender: Optional[str] = None
    total_amount: Optional[float] = None
    currency: Optional[str] = None
    summary: str = Field(..., min_length=5)

def extract_with_vlm(image_path):
    """Extract using VLM."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    
    prompt = """Extract the following from this document image.
Return ONLY valid JSON with these fields:
{
    "document_type": "invoice|receipt|contract|form|letter|other",
    "date": "YYYY-MM-DD or null",
    "document_number": "identifier or null",
    "vendor_or_sender": "name or null",
    "total_amount": 0.0 or null,
    "currency": "USD/EUR/etc or null",
    "summary": "2-3 sentence summary of the document"
}"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def extract_with_ocr_fallback(image_path):
    """Fallback: extract using OCR + LLM."""
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    
    prompt = f"""Extract structured data from this OCR'd document text.
Return ONLY valid JSON matching this schema:
{{
    "document_type": "...",
    "date": "... or null",
    "document_number": "... or null",
    "vendor_or_sender": "... or null",
    "total_amount": 0.0 or null,
    "currency": "... or null",
    "summary": "..."
}}

TEXT:
{text}

Note: OCR may have errors. Do your best to extract accurate information."""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def extract_document(image_path):
    """Main extraction function with fallback."""
    # Try VLM first
    try:
        data = extract_with_vlm(image_path)
        validated = ExtractedDocument(**data)
        print("✓ VLM extraction successful")
        return validated.model_dump()
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"VLM extraction failed: {e}")
    
    # Fallback to OCR
    try:
        data = extract_with_ocr_fallback(image_path)
        validated = ExtractedDocument(**data)
        print("✓ OCR fallback successful")
        return validated.model_dump()
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"OCR fallback also failed: {e}")
        return None
```

---

## Summary

Multimodal systems extend LLM capabilities to process images, documents, and audio. Vision-language models combine vision encoders with LLMs via projection layers, enabling image understanding, OCR, and structured extraction. Document intelligence pipelines add layout understanding and schema validation for production-grade extraction. Audio workflows add speech-to-text, diarization, and voice agent capabilities. The key to building robust multimodal systems is combining the flexibility of VLMs with the reliability of traditional tools (OCR engines, ASR models) through fallback architectures and schema validation.

## Key Takeaways

- **VLMs combine vision encoders + projection layers + LLMs** to process images and text together
- **OCR approaches**: Traditional OCR (deterministic, fast) vs VLM-based (flexible, handles complex layouts)
- **Document intelligence = OCR + layout understanding + semantic extraction**
- **Schema validation** catches extraction errors and ensures consistent output
- **Audio workflows** add ASR (Whisper), diarization, summarization, and voice agent capabilities
- **Hybrid pipelines** with fallbacks are more robust than pure VLM or pure traditional approaches
- **Multimodal evaluation** needs task-specific metrics (WER for OCR, field accuracy for extraction, WER for ASR)
