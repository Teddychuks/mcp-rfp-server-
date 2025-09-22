"""
Text Processing Utilities for MCP-RFP Server
"""
import re
import string
import logging
from typing import List, Dict, Any, Set
from collections import Counter
import spacy
from spacy.tokens import Doc, Span

logger = logging.getLogger(__name__)


class TextProcessor:
    """Advanced text processing utilities"""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.nlp = None
        self.spacy_model = spacy_model
        self._initialized = False

    async def initialize(self):
        """Initialize spaCy model"""
        if self._initialized:
            return

        try:
            self.nlp = spacy.load(self.spacy_model)
            self._initialized = True
            logger.info(f"Text processor initialized with {self.spacy_model}")
        except Exception as e:
            logger.error(f"Failed to initialize text processor: {e}")
            raise

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Fix hyphenated line breaks
        text = re.sub(r'-\n\s*', '', text)

        # Normalize line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Fix sentence boundaries
        text = re.sub(r'(?<=\.)\s*\n(?=[A-Z])', ' ', text)

        # Convert single line breaks to spaces (except paragraph breaks)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

        return text.strip()

    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        if not self._initialized:
            raise RuntimeError("Text processor not initialized")

        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def extract_keywords(self, text: str, max_keywords: int = 20) -> List[str]:
        """Extract important keywords from text"""
        if not self._initialized:
            raise RuntimeError("Text processor not initialized")

        doc = self.nlp(text)

        # Extract tokens that are likely keywords
        keywords = []
        for token in doc:
            if (token.pos_ in ["NOUN", "PROPN", "ADJ"] and
                    not token.is_stop and
                    not token.is_punct and
                    len(token.text) > 2 and
                    token.text.isalnum()):
                keywords.append(token.lemma_.lower())

        # Count occurrences and return most frequent
        keyword_counts = Counter(keywords)
        return [kw for kw, _ in keyword_counts.most_common(max_keywords)]

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        if not self._initialized:
            raise RuntimeError("Text processor not initialized")

        doc = self.nlp(text)
        entities = {}

        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)

        # Remove duplicates while preserving order
        for label in entities:
            entities[label] = list(dict.fromkeys(entities[label]))

        return entities

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        if not text:
            return []

        words = text.split()
        if len(words) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)

            if end >= len(words):
                break

            start += chunk_size - overlap

        return chunks

    def extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical terms and acronyms"""
        # Pattern for acronyms (2+ uppercase letters)
        acronym_pattern = r'\b[A-Z]{2,}\b'
        acronyms = re.findall(acronym_pattern, text)

        # Pattern for technical terms (CamelCase, hyphenated, etc.)
        technical_pattern = r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b|\b\w+(?:-\w+)+\b'
        technical_terms = re.findall(technical_pattern, text)

        # Combine and deduplicate
        all_terms = list(set(acronyms + technical_terms))

        # Filter out common words
        common_words = {"The", "This", "That", "And", "But", "For", "Not", "With"}
        technical_terms = [term for term in all_terms if term not in common_words]

        return technical_terms

    def calculate_readability_score(self, text: str) -> Dict[str, float]:
        """Calculate basic readability metrics"""
        if not text:
            return {"sentences": 0, "words": 0, "avg_words_per_sentence": 0}

        sentences = self.extract_sentences(text)
        words = text.split()

        avg_words_per_sentence = len(words) / len(sentences) if sentences else 0

        # Count syllables (rough approximation)
        syllable_count = sum(self._count_syllables(word) for word in words)
        avg_syllables_per_word = syllable_count / len(words) if words else 0

        return {
            "sentences": len(sentences),
            "words": len(words),
            "syllables": syllable_count,
            "avg_words_per_sentence": round(avg_words_per_sentence, 2),
            "avg_syllables_per_word": round(avg_syllables_per_word, 2)
        }

    def _count_syllables(self, word: str) -> int:
        """Rough syllable counting"""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        previous_was_vowel = False

        for char in word:
            if char in vowels:
                if not previous_was_vowel:
                    syllable_count += 1
                previous_was_vowel = True
            else:
                previous_was_vowel = False

        # Handle silent e
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1

        return max(1, syllable_count)

    def find_similar_sentences(self, target_sentence: str, text_corpus: str, threshold: float = 0.7) -> List[
        Dict[str, Any]]:
        """Find sentences similar to target in corpus"""
        if not self._initialized:
            raise RuntimeError("Text processor not initialized")

        corpus_sentences = self.extract_sentences(text_corpus)
        target_doc = self.nlp(target_sentence)

        similar_sentences = []

        for sentence in corpus_sentences:
            sentence_doc = self.nlp(sentence)
            similarity = target_doc.similarity(sentence_doc)

            if similarity >= threshold:
                similar_sentences.append({
                    "sentence": sentence,
                    "similarity": similarity
                })

        # Sort by similarity
        similar_sentences.sort(key=lambda x: x["similarity"], reverse=True)
        return similar_sentences

    def extract_requirements_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extract requirement-like patterns with context"""
        if not self._initialized:
            raise RuntimeError("Text processor not initialized")

        requirement_patterns = [
            r'\b(?:shall|must|will|should|required to|responsible for)\b.*?[.!?]',
            r'\b(?:contractor|system|solution|platform)\s+(?:shall|must|will)\b.*?[.!?]',
            r'\b(?:provide|deliver|support|implement|maintain)\b.*?[.!?]'
        ]

        requirements = []

        for pattern in requirement_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                requirement_text = match.group().strip()

                # Skip very short matches
                if len(requirement_text.split()) < 5:
                    continue

                # Extract context (surrounding sentences)
                start_pos = max(0, match.start() - 200)
                end_pos = min(len(text), match.end() + 200)
                context = text[start_pos:end_pos].strip()

                requirements.append({
                    "text": requirement_text,
                    "context": context,
                    "start_position": match.start(),
                    "end_position": match.end(),
                    "pattern_type": pattern
                })

        return requirements