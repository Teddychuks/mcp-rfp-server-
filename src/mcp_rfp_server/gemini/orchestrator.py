"""
Gemini Orchestrator - AI reasoning engine for RFP processing
"""
import json
import logging
from typing import Dict, List, Any

import google.generativeai as genai

from ..config import ServerConfig
from ..tools.document_processor import DocumentProcessor
from ..tools.knowledge_search import KnowledgeSearcher
from ..utils.quality_assurance import QualityAssurance

logger = logging.getLogger(__name__)


class GeminiOrchestrator:
    """Orchestrates RFP processing using Gemini's reasoning capabilities"""

    def __init__(
        self,
        config: ServerConfig,
        document_processor: DocumentProcessor,
        knowledge_searcher: KnowledgeSearcher
    ):
        self.config = config
        self.document_processor = document_processor
        self.knowledge_searcher = knowledge_searcher
        self.proposal_generator = None  # Injected after initialization
        self.model = None
        self._initialized = False


    async def initialize(self):
        """Initialize Gemini client"""
        if self._initialized:
            return

        try:

            genai.configure(api_key=self.config.google_api_key)
            self.model = genai.GenerativeModel(self.config.gemini_model)
            self._initialized = True
            logger.info("Gemini orchestrator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {str(e)}")
            raise

    async def process_rfp_document(
            self,
            document_content: str,
            document_type: str,
            processing_options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Complete RFP processing workflow with AI orchestration"""

        await self.initialize()
        processing_options = processing_options or {}

        try:
            plan = await self._create_processing_plan(document_content, document_type, processing_options)

            execution_results = await self._execute_processing_plan(
                plan, document_content, document_type, processing_options
            )

            final_analysis = await self._generate_final_analysis(execution_results)

            return {
                "success": True,
                "processing_plan": plan,
                "execution_results": execution_results,
                "final_analysis": final_analysis,
                "metadata": {
                    "document_type": document_type,
                    "processing_options": processing_options,
                    "gemini_model": self.config.gemini_model
                }
            }

        except Exception as e:
            logger.error(f"RFP processing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "processing_plan": None,
                "execution_results": {},
                "final_analysis": None
            }

    async def analyze_requirements(
            self,
            requirements: List[str],
            analysis_type: str = "priority"
    ) -> Dict[str, Any]:
        """Intelligent requirement analysis"""

        await self.initialize()

        try:
            prompt = self._build_requirement_analysis_prompt(requirements, analysis_type)
            response = await self._call_gemini(prompt)

            # Parse Gemini response
            analysis = self._parse_requirement_analysis(response)

            return {
                "success": True,
                "analysis_type": analysis_type,
                "requirements_count": len(requirements),
                "analysis": analysis,
                "recommendations": analysis.get("recommendations", []),
                "priority_ranking": analysis.get("priority_ranking", [])
            }

        except Exception as e:
            logger.error(f"Requirement analysis failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "analysis": {}
            }

    async def _create_processing_plan(
            self,
            document_content: str, # Now takes document content
            document_type: str,
            options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a dynamic, AI-generated processing plan based on initial analysis."""

        # Perform a quick pre-analysis to inform the plan
        pre_analysis_prompt = f"""
        Briefly analyze the following RFP document's characteristics.
        Focus on: estimated length (short, medium, long), complexity (low, medium, high), and primary focus (e.g., technical, management, pricing).
        Respond with a short, one-sentence summary.
        
        Document Content Preview:
        {document_content[:2000]}
        """
        document_summary = await self._call_gemini(pre_analysis_prompt)


        prompt = f"""
        You are an expert RFP processing system planner. Create a detailed, custom processing plan for a {document_type} RFP document.

        Document Analysis Summary: {document_summary}

        Processing Options Provided:
        {json.dumps(options, indent=2)}

        Create a processing plan with these steps:
        1. Document analysis and requirement extraction
        2. Knowledge base search strategy
        3. Response generation approach
        4. Quality assurance steps

        Based on the analysis summary, tailor the parameters for each step. For example, a long, complex document might need a larger context window for generation.

        Return a JSON object with:
        {{
            "plan_id": "unique_id",
            "steps": [
                {{
                    "step_number": 1,
                    "step_name": "step_name",
                    "description": "what this step does",
                    "parameters": {{}},
                    "expected_output": "what we expect from this step"
                }}
            ],
            "success_criteria": ["criterion1", "criterion2"],
            "estimated_duration": "time estimate"
        }}

        Respond ONLY with valid JSON.
        """

        response = await self._call_gemini(prompt)
        return self._parse_json_response(response)

    async def _execute_processing_plan(
            self,
            plan: Dict[str, Any],
            document_content: str,
            document_type: str,
            options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the processing plan step by step"""

        execution_results = {}
        context = {
            "document_content": document_content,
            "document_type": document_type,
            "options": options
        }

        for step in plan.get("steps", []):
            step_name = step.get("step_name", "unknown_step")
            logger.info(f"Executing step: {step_name}")
            step_name_lower = step_name.lower()  # Prepare a lowercase version for matching

            try:
                # --- THIS IS THE CORRECTED LOGIC ---
                if "requirement extraction" in step_name_lower:
                    result = await self._execute_extraction_step(step, context)
                elif "knowledge base search" in step_name_lower:
                    result = await self._execute_search_step(step, context)
                elif "response generation" in step_name_lower:
                    result = await self._execute_generation_step(step, context)
                elif "quality assurance" in step_name_lower:
                    result = await self._execute_qa_step(step, context)
                else:
                    result = {"success": False, "error": f"Unknown step type: {step_name}"}

                execution_results[step_name] = result

                # Stop execution if a critical step fails
                if not result.get("success"):
                    logger.error(f"Stopping execution due to failure in step: {step_name}")
                    break

                # Add results to context for next steps
                context[f"result_{step_name}"] = result

            except Exception as e:
                logger.error(f"Step execution failed: {step_name} - {str(e)}")
                execution_results[step_name] = {
                    "success": False,
                    "error": str(e)
                }

        return execution_results

    async def _generate_final_analysis(
            self,
            execution_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate final analysis and recommendations"""

        prompt = f"""
        Analyze the RFP processing results and provide final recommendations.

        Processing Results:
        {json.dumps(execution_results, indent=2)}

        Provide analysis on:
        1. Processing success rate
        2. Requirements coverage
        3. Response quality indicators
        4. Areas needing attention
        5. Overall recommendations

        Return JSON with:
        {{
            "overall_success": true/false,
            "processing_summary": {{
                "requirements_found": number,
                "responses_generated": number,
                "success_rate": percentage
            }},
            "quality_indicators": {{
                "completeness": "high/medium/low",
                "relevance": "high/medium/low",
                "consistency": "high/medium/low"
            }},
            "recommendations": ["recommendation1", "recommendation2"],
            "next_steps": ["step1", "step2"]
        }}

        Respond ONLY with valid JSON.
        """

        response = await self._call_gemini(prompt)
        return self._parse_json_response(response)

    async def _execute_extraction_step(
            self,
            step: Dict[str, Any],
            context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute requirement extraction step"""
        extraction_result = await self.document_processor.extract_requirements(
            document_content=context["document_content"],
            document_type=context["document_type"],
            filters=context.get("options", {}).get("filters", [])
        )
        if extraction_result.get("success"):
            context["requirements"] = extraction_result.get("requirements", [])
        return extraction_result


    async def _execute_search_step(
            self,
            step: Dict[str, Any],
            context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute knowledge search step"""
        requirements = context.get("requirements", [])
        if not requirements:
            return {"success": True, "message": "No requirements to search for.", "contexts_found": 0}

        contexts = []
        for req in requirements:
            search_results = await self.knowledge_searcher.search(query=req["text"])
            contexts.append({
                "requirement_id": req["id"],
                "context": [res["content"] for res in search_results.get("results", [])]
            })

        context["contexts"] = contexts
        return {
            "success": True,
            "step_type": "knowledge_search",
            "message": "Knowledge base searched successfully",
            "contexts_found": len(contexts)
        }


    async def _execute_generation_step(
            self,
            step: Dict[str, Any],
            context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute response generation step"""
        requirements = context.get("requirements", [])
        contexts = context.get("contexts", [])
        style = context.get("options", {}).get("response_style", "formal")

        if not requirements:
            return {"success": True, "message": "No requirements for response generation.", "responses_count": 0}

        responses = await self.proposal_generator.generate_multiple_sections(
            requirements=[req["text"] for req in requirements],
            contexts=[ctx["context"] for ctx in contexts],
            style=style
        )
        context["generated_responses"] = responses.get("sections", [])
        return responses


    async def _execute_qa_step(
            self,
            step: Dict[str, Any],
            context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute quality assurance step"""
        generated_responses = context.get("generated_responses", [])
        if not generated_responses:
            return {"success": True, "message": "No responses to check.", "quality_score": None}

        qa_checker = QualityAssurance()
        qa_results = qa_checker.run_checks(generated_responses)
        return {
            "success": True,
            "step_type": "quality_assurance",
            "message": "Quality checks completed",
            "quality_score": qa_results["overall_quality_score"]
        }


    def _build_requirement_analysis_prompt(
            self,
            requirements: List[str],
            analysis_type: str
    ) -> str:
        """Build prompt for requirement analysis"""

        base_prompt = f"""
        Analyze the following RFP requirements with focus on {analysis_type}:

        Requirements:
        {json.dumps(requirements, indent=2)}

        Analysis Type: {analysis_type}
        """

        if analysis_type == "priority":
            prompt = base_prompt + """
            Rank requirements by priority considering:
            - Business criticality
            - Implementation complexity
            - Risk factors
            - Dependencies

            Return JSON with priority ranking and reasoning.
            """
        elif analysis_type == "complexity":
            prompt = base_prompt + """
            Analyze technical complexity considering:
            - Technical difficulty
            - Resource requirements
            - Timeline implications
            - Skill requirements

            Return JSON with complexity assessment.
            """
        elif analysis_type == "coverage":
            prompt = base_prompt + """
            Analyze requirement coverage considering:
            - Completeness of specifications
            - Missing details
            - Ambiguities
            - Clarification needs

            Return JSON with coverage analysis.
            """
        else:
            prompt = base_prompt + """
            Provide general analysis of the requirements.
            Return JSON with insights and recommendations.
            """

        return prompt + "\n\nRespond ONLY with valid JSON."

    async def _call_gemini(self, prompt: str) -> str:
        """Make API call to Gemini"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API call failed: {str(e)}")
            raise

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from Gemini"""
        try:
            # Clean response
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]

            return json.loads(cleaned.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            logger.error(f"Response content: {response}")
            return {"error": "Invalid JSON response from Gemini"}

    def _parse_requirement_analysis(self, response: str) -> Dict[str, Any]:
        """Parse requirement analysis response"""
        parsed = self._parse_json_response(response)

        # Ensure required fields exist
        if "recommendations" not in parsed:
            parsed["recommendations"] = []
        if "priority_ranking" not in parsed:
            parsed["priority_ranking"] = []

        return parsed