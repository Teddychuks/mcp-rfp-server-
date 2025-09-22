"""
Proposal Generator Tool for MCP-RFP Server
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..config import ServerConfig
from ..schemas.rfp_schemas import ProposalSection

logger = logging.getLogger(__name__)


class ProposalGenerator:
    """Generates proposal responses using Gemini integration"""

    def __init__(self, config: ServerConfig, gemini_orchestrator: "GeminiOrchestrator"):
        self.config = config
        self.gemini_orchestrator = gemini_orchestrator
        self._initialized = False

        # Response templates for different styles
        self.response_templates = {
            "formal": {
                "structure": "executive_summary,technical_approach,past_experience,team_qualifications,timeline",
                "tone": "professional and authoritative",
                "length": "detailed with comprehensive coverage"
            },
            "technical": {
                "structure": "technical_solution,architecture,implementation_details,testing_approach,maintenance",
                "tone": "technical and precise",
                "length": "detailed technical specifications"
            },
            "executive": {
                "structure": "business_value,strategic_approach,competitive_advantages,risk_mitigation",
                "tone": "strategic and business-focused",
                "length": "concise with high-level overview"
            }
        }

    async def initialize(self):
        """Initialize proposal generator"""
        if self._initialized:
            return

        # Ensure Gemini orchestrator is initialized
        await self.gemini_orchestrator.initialize()

        self._initialized = True
        logger.info("Proposal generator initialized successfully")

    async def generate_with_gemini(
            self,
            requirement: str,
            context: List[str] = None,
            style: str = "formal"
    ) -> Dict[str, Any]:
        """Generate proposal response using Gemini"""

        await self.initialize()

        try:
            # Build comprehensive prompt
            prompt = self._build_generation_prompt(requirement, context or [], style)

            # Call Gemini to generate response
            response = await self.gemini_orchestrator._call_gemini(prompt)

            # Process and structure the response
            processed_response = self._process_gemini_response(response, requirement, context, style)

            return {
                "success": True,
                "requirement": requirement,
                "response": processed_response["response"],
                "metadata": {
                    "style": style,
                    "context_sources": len(context or []),
                    "word_count": processed_response["word_count"],
                    "confidence_score": processed_response["confidence_score"],
                    "generated_at": datetime.utcnow().isoformat(),
                    "gemini_model": self.config.gemini_model
                }
            }

        except Exception as e:
            logger.error(f"Proposal generation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "requirement": requirement,
                "response": "",
                "metadata": {}
            }

    async def generate_multiple_sections(
            self,
            requirements: List[str],
            contexts: List[List[str]] = None,
            style: str = "formal"
    ) -> Dict[str, Any]:
        """Generate multiple proposal sections efficiently"""

        await self.initialize()

        if not requirements:
            return {
                "success": False,
                "error": "No requirements provided",
                "sections": []
            }

        try:
            # Prepare contexts
            if contexts is None:
                contexts = [[] for _ in requirements]
            elif len(contexts) != len(requirements):
                # Pad contexts list to match requirements
                contexts.extend([[] for _ in range(len(requirements) - len(contexts))])

            # Generate all sections
            sections = []
            total_word_count = 0

            for i, (requirement, context) in enumerate(zip(requirements, contexts)):
                logger.info(f"Generating section {i + 1}/{len(requirements)}")

                section_result = await self.generate_with_gemini(requirement, context, style)

                if section_result["success"]:
                    sections.append(section_result)
                    total_word_count += section_result["metadata"].get("word_count", 0)
                else:
                    logger.error(f"Failed to generate section {i + 1}: {section_result.get('error')}")
                    # Add error placeholder
                    sections.append({
                        "success": False,
                        "requirement": requirement,
                        "response": f"Error generating response: {section_result.get('error', 'Unknown error')}",
                        "metadata": {"error": True}
                    })

            return {
                "success": True,
                "sections": sections,
                "summary": {
                    "total_requirements": len(requirements),
                    "successful_generations": sum(1 for s in sections if s["success"]),
                    "failed_generations": sum(1 for s in sections if not s["success"]),
                    "total_word_count": total_word_count,
                    "style_used": style
                }
            }

        except Exception as e:
            logger.error(f"Multiple section generation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "sections": []
            }

    def _build_generation_prompt(
            self,
            requirement: str,
            context: List[str],
            style: str
    ) -> str:
        """Build comprehensive prompt for Gemini"""

        template = self.response_templates.get(style, self.response_templates["formal"])

        prompt = f"""
You are an expert proposal writer for Innovate Solutions Inc., a leading technology company specializing in secure data analytics and AI-driven platforms. You are responding to a specific RFP requirement.

COMPANY CONTEXT:
- Innovate Solutions Inc. specializes in secure, scalable, AI-driven platforms
- We serve government and commercial clients
- Our flagship product is QuantumGrid (AI-powered data fusion platform)
- We have extensive experience with security compliance (FedRAMP, NIST, HIPAA)
- We use Agile methodology and have strong past performance

RFP REQUIREMENT TO ADDRESS:
{requirement}

RELEVANT COMPANY KNOWLEDGE:
"""

        # Add context information
        if context:
            for i, ctx in enumerate(context[:5], 1):
                prompt += f"\nContext {i}:\n{ctx}\n"
        else:
            prompt += "\nNo specific context provided - use general company capabilities.\n"

        prompt += f"""

RESPONSE REQUIREMENTS:
- Style: {template["tone"]}
- Structure: {template["structure"]}
- Length: {template["length"]}
- Must directly address the specific requirement
- Use company knowledge and past experience where relevant
- Be specific and demonstrate understanding of the requirement
- Include concrete examples when possible
- Maintain professional, confident tone
- Show how our solution meets or exceeds the requirement

IMPORTANT:
- Write as Innovate Solutions Inc.
- Reference our actual capabilities and experience
- Be specific, not generic. Avoid vague marketing language like "world-class" or "best-in-class".
- Address the requirement completely
- Use technical details where appropriate
- Show understanding of the client's needs

Generate a comprehensive, professional response that directly addresses this RFP requirement:
"""

        return prompt

    def _process_gemini_response(
            self,
            response: str,
            requirement: str,
            context: List[str],
            style: str
    ) -> Dict[str, Any]:
        """Process and validate Gemini response"""

        # Clean up response
        cleaned_response = response.strip()

        # Calculate metrics
        word_count = len(cleaned_response.split())

        # Calculate confidence score based on various factors
        confidence_score = self._calculate_response_confidence(
            cleaned_response, requirement, context
        )

        # Validate response quality
        quality_issues = self._validate_response_quality(cleaned_response, requirement)

        return {
            "response": cleaned_response,
            "word_count": word_count,
            "confidence_score": confidence_score,
            "quality_issues": quality_issues,
            "processing_notes": self._generate_processing_notes(
                cleaned_response, requirement, style
            )
        }

    def _calculate_response_confidence(
            self,
            response: str,
            requirement: str,
            context: List[str]
    ) -> float:
        """Calculate confidence score for generated response"""

        score = 0.5  # Start from a baseline score

        # Length appropriateness (reward for being in a good range)
        word_count = len(response.split())
        if 75 <= word_count <= 1000:
            score += 0.2
        elif word_count < 50:
            score -= 0.3 # Penalize very short responses

        # Check for specific company references (indicates customization)
        company_terms = ["innovate solutions", "quantumgrid"]
        if any(term in response.lower() for term in company_terms):
            score += 0.2

        # Context utilization (simple check for now)
        if context and len(context) > 0:
            score += 0.1

        # Check for placeholder language
        if "to be determined" in response.lower() or "tbd" in response.lower():
            score -= 0.4


        return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1

    def _validate_response_quality(self, response: str, requirement: str) -> List[str]:
        """Validate response quality and identify issues"""

        issues = []

        # Check minimum length
        if len(response.split()) < 50:
            issues.append("Response too short")

        # Check maximum length
        if len(response.split()) > 1500:
            issues.append("Response too long")

        # Check for generic content
        generic_phrases = [
            "we are a leading company",
            "we provide best-in-class",
            "industry-leading solution",
            "world-class service"
        ]
        if any(phrase in response.lower() for phrase in generic_phrases):
            issues.append("Contains generic marketing language")

        # Check for requirement addressing
        req_key_words = [word for word in requirement.split() if len(word) > 4][:5]
        if req_key_words:
            addressed = sum(1 for word in req_key_words if word.lower() in response.lower())
            if addressed < len(req_key_words) * 0.5:
                issues.append("May not fully address requirement")

        # Check for company-specific content
        if "innovate solutions" not in response.lower():
            issues.append("Missing company identification")

        return issues

    def _generate_processing_notes(
            self,
            response: str,
            requirement: str,
            style: str
    ) -> List[str]:
        """Generate notes about the response processing"""

        notes = []

        notes.append(f"Generated using {style} style")
        notes.append(f"Response length: {len(response.split())} words")

        # Style-specific notes
        if style == "technical":
            if any(term in response.lower() for term in ["architecture", "implementation", "technical"]):
                notes.append("Contains technical content as expected")
            else:
                notes.append("May need more technical details")

        elif style == "executive":
            if any(term in response.lower() for term in ["strategic", "business", "value"]):
                notes.append("Contains executive-level content")
            else:
                notes.append("May need more strategic focus")

        return notes

    async def refine_response(
            self,
            original_response: str,
            requirement: str,
            feedback: str,
            context: List[str] = None
    ) -> Dict[str, Any]:
        """Refine existing response based on feedback"""

        await self.initialize()

        try:
            refine_prompt = f"""
You are refining a proposal response based on feedback. Here is the context:

ORIGINAL REQUIREMENT:
{requirement}

ORIGINAL RESPONSE:
{original_response}

FEEDBACK FOR IMPROVEMENT:
{feedback}

RELEVANT CONTEXT:
{chr(10).join(context or [])}

Please provide an improved response that:
1. Addresses the feedback provided
2. Maintains the professional tone and company positioning
3. Better addresses the original requirement
4. Incorporates any relevant context

Generate the refined response:
"""

            refined_response = await self.gemini_orchestrator._call_gemini(refine_prompt)

            return {
                "success": True,
                "original_response": original_response,
                "refined_response": refined_response.strip(),
                "improvement_notes": self._analyze_improvements(
                    original_response, refined_response, feedback
                ),
                "metadata": {
                    "refinement_applied": feedback,
                    "generated_at": datetime.utcnow().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"Response refinement failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "original_response": original_response,
                "refined_response": original_response
            }

    def _analyze_improvements(
            self,
            original: str,
            refined: str,
            feedback: str
    ) -> List[str]:
        """Analyze what improvements were made"""

        improvements = []

        # Length comparison
        orig_words = len(original.split())
        refined_words = len(refined.split())

        if refined_words > orig_words * 1.1:
            improvements.append("Expanded content")
        elif refined_words < orig_words * 0.9:
            improvements.append("Condensed content")

        # Check if feedback keywords appear in refined version
        feedback_words = [word.lower() for word in feedback.split() if len(word) > 4]
        refined_lower = refined.lower()

        for word in feedback_words:
            if word in refined_lower and word not in original.lower():
                improvements.append(f"Added focus on '{word}'")

        if not improvements:
            improvements.append("General improvements applied")

        return improvements