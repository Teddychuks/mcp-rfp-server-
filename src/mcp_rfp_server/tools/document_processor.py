"""
Document Processing Tool for MCP-RFP Server
"""
import asyncio
import base64
import io
import logging
import re
from typing import Dict, List, Any
from typing import TYPE_CHECKING

import docx
import fitz
import spacy

from ..config import ServerConfig
from ..schemas.rfp_schemas import Requirement, DocumentMetadata

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document processing operations"""

    def __init__(self, config: ServerConfig, orchestrator: "GeminiOrchestrator" = None):
        self.config = config
        self.orchestrator = orchestrator
        self.nlp = None
        self._initialized = False

        # RFP requirement detection keywords (still useful for initial filtering)
        self.requirement_keywords = {
            "shall", "must", "will provide", "is required to", "is required",
            "are required", "responsible for", "contractor shall", "contractor must",
            "solution must", "solution shall", "system must", "system shall",
            "the contractor will", "the offeror shall", "deliver", "capable of",
            "support", "compliant with", "provide", "required to", "should",
            "will be responsible", "needs to", "has to", "obligated to"
        }

    async def initialize(self):
        """Initialize spaCy model"""
        if self._initialized:
            return

        try:
            self.nlp = spacy.load(self.config.spacy_model)
            self._initialized = True
            logger.info("Document processor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize document processor: {str(e)}")
            raise

    async def extract_requirements(
            self,
            document_content: str,
            document_type: str,
            filters: List[str] = None
    ) -> Dict[str, Any]:
        """Extract requirements from RFP document"""

        await self.initialize()

        try:
            # Decode and extract text
            document_bytes = base64.b64decode(document_content)

            if document_type == "pdf":
                text = self._extract_text_from_pdf(document_bytes)
            elif document_type == "docx":
                text = self._extract_text_from_docx(document_bytes)
            else:
                raise ValueError(f"Unsupported document type: {document_type}")

            if not text.strip():
                return {
                    "success": False,
                    "error": "No text extracted from document",
                    "requirements": [],
                    "metadata": {}
                }

            # Extract requirements
            requirements = await self._find_requirements(text, filters or [])

            # Generate metadata
            metadata = self._generate_document_metadata(text, document_type)

            return {
                "success": True,
                "requirements": [req.dict() for req in requirements],
                "total_count": len(requirements),
                "text_length": len(text),
                "metadata": metadata.dict(),
                "filters_applied": filters or []
            }

        except Exception as e:
            logger.error(f"Requirement extraction failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "requirements": [],
                "metadata": {}
            }

    def _extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract clean text from PDF"""
        try:
            text = ""
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text()

            return self._clean_extracted_text(text)
        except Exception as e:
            logger.error(f"PDF text extraction failed: {str(e)}")
            return ""

    def _extract_text_from_docx(self, docx_bytes: bytes) -> str:
        """Extract clean text from DOCX"""
        try:
            doc = docx.Document(io.BytesIO(docx_bytes))
            text = "\n".join([para.text for para in doc.paragraphs])
            return self._clean_extracted_text(text)
        except Exception as e:
            logger.error(f"DOCX text extraction failed: {str(e)}")
            return ""

    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove hyphenated line breaks
        text = re.sub(r'-\n', '', text)

        # Normalize multiple line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Fix sentence boundaries
        text = re.sub(r'(?<=\.)\s*\n(?=[A-Z])', '\n', text)

        # Convert single line breaks to spaces (except paragraph breaks)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    async def _find_requirements(self, text: str, filters: List[str]) -> List[Requirement]:
        """Extract requirements from text using NLP"""

        if not text.strip():
            return []

        requirements = []
        doc = self.nlp(text)

        candidate_sentences = []
        for sent_idx, sentence in enumerate(doc.sents):
            sent_text = sentence.text.strip()
            if len(sent_text.split()) <= 3:
                continue
            if any(keyword in sent_text.lower() for keyword in self.requirement_keywords):
                if not filters or any(filter_term.lower() in sent_text.lower() for filter_term in filters):
                    candidate_sentences.append((sent_idx, sentence))

        # Batch process classifications and priorities with the LLM
        classification_tasks = [self._classify_requirement(s.text) for _, s in candidate_sentences]
        priority_tasks = [self._extract_priority(s.text) for _, s in candidate_sentences]

        classifications = await asyncio.gather(*classification_tasks)
        priorities = await asyncio.gather(*priority_tasks)

        for i, (sent_idx, sentence) in enumerate(candidate_sentences):
            sent_text = sentence.text.strip()

            requirement = Requirement(
                id=f"temp_id_{sent_idx}", # Temporary ID
                text=sent_text,
                type=classifications[i],
                priority=priorities[i],
                section=self._identify_section(sentence, doc),
                keywords=self._extract_keywords(sent_text),
                confidence=self._calculate_confidence(sent_text),
                metadata={
                    "sentence_index": sent_idx,
                    "word_count": len(sent_text.split()),
                    "character_count": len(sent_text)
                }
            )
            requirements.append(requirement)

        # Sort by confidence score (highest first)
        requirements.sort(key=lambda r: r.confidence, reverse=True)

        # Limit to max requirements per document
        max_requirements = self.config.max_requirements_per_document
        if len(requirements) > max_requirements:
            logger.warning(
                f"Found {len(requirements)} requirements, "
                f"limiting to {max_requirements}"
            )
            requirements = requirements[:max_requirements]

        # Re-assign sequential IDs after sorting and truncation
        for i, req in enumerate(requirements):
            req.id = f"req_{i + 1:03d}"


        return requirements

    async def _classify_requirement(self, text: str) -> str:
        """Classify requirement into categories using the LLM."""
        if not self.orchestrator:
            return "functional" # Fallback

        prompt = f"""
        Classify the following requirement into ONE of the following categories:
        'security', 'performance', 'integration', 'compliance', 'technical', 'functional'.
        
        Requirement: "{text}"
        
        Respond with only the category name.
        """
        category = await self.orchestrator._call_gemini(prompt)
        return category.strip().lower()

    async def _extract_priority(self, text: str) -> str:
        """Extract priority level from requirement text using the LLM."""
        if not self.orchestrator:
            return "medium" # Fallback

        prompt = f"""
        Analyze the following requirement and determine its priority level.
        Choose ONE of: 'high', 'medium', 'low'.
        
        - 'high' for words like 'must', 'shall', 'critical', 'mandatory'.
        - 'medium' for words like 'should', 'preferred', 'important'.
        - 'low' for words like 'may', 'could', 'optional'.

        Requirement: "{text}"
        
        Respond with only the priority level.
        """
        priority = await self.orchestrator._call_gemini(prompt)
        return priority.strip().lower()

    def _identify_section(self, sentence, doc) -> str:
        """Identify document section for this requirement"""
        # Look backwards for section headers (e.g., "1.1 Introduction", "Section 2:")
        header_pattern = re.compile(r"^(?:\d+(?:\.\d+)*\s*|\w+\s*:|Section\s+\d+)")
        sent_start_char = sentence.start_char

        # Search the text before the current sentence for a header
        text_before = doc.text[:sent_start_char]
        lines_before = text_before.split('\n')

        for line in reversed(lines_before):
            line = line.strip()
            if header_pattern.match(line) and len(line.split()) < 10:  # Headers are usually short
                return line

        return "general"

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from requirement text"""
        if not self.nlp:
            return []

        doc = self.nlp(text)
        keywords = []

        for token in doc:
            # Extract nouns, proper nouns, and important adjectives
            if (token.pos_ in ["NOUN", "PROPN", "ADJ"] and
                    not token.is_stop and
                    len(token.text) > 2 and
                    token.text.isalnum()):
                keywords.append(token.text.lower())

        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)

        return unique_keywords[:10]  # Limit to top 10 keywords

    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence score for requirement detection"""
        score = 0.0

        # Keyword presence boost
        keyword_count = sum(
            1 for keyword in self.requirement_keywords
            if keyword in text.lower()
        )
        score += min(keyword_count * 0.15, 0.6)  # Max 0.6 from keywords

        # Strong requirement words give extra boost
        if any(word in text.lower() for word in ["shall", "must", "required"]):
            score += 0.25

        # Sentence structure indicators
        if any(word in text.lower() for word in ["contractor", "system", "solution"]):
            score += 0.1

        # Length appropriateness (not too short, not too long)
        word_count = len(text.split())
        if 5 <= word_count <= 40:
            score += 0.05
        elif word_count > 100:
            score -= 0.1  # Very long sentences might be less clear requirements

        return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1

    def _generate_document_metadata(self, text: str, document_type: str) -> DocumentMetadata:
        """Generate document metadata"""

        word_count = len(text.split())
        char_count = len(text)
        sentence_count = len(list(self.nlp(text).sents)) if self.nlp else 0

        return DocumentMetadata(
            document_type=document_type,
            word_count=word_count,
            character_count=char_count,
            sentence_count=sentence_count,
            processing_model=self.config.spacy_model,
            extraction_method="spacy_nlp",
            confidence_threshold=self.config.confidence_threshold
        )