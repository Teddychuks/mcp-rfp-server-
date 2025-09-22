"""
Quality Assurance for MCP-RFP Server
"""
from typing import Dict, List, Any

class QualityAssurance:
    """Runs quality checks on generated proposal sections."""

    def run_checks(self, generated_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Runs a series of quality checks on the generated responses.
        """
        total_responses = len(generated_responses)
        if total_responses == 0:
            return {
                "success": True,
                "message": "No responses to check.",
                "overall_quality_score": None,
                "total_responses_checked": 0,
                "issues_found": 0,
            }

        issues_found = 0
        quality_scores = []

        for response in generated_responses:
            score = self.calculate_quality_score(response.get("response", ""))
            quality_scores.append(score)
            if score < 0.7:
                issues_found += 1

        overall_quality_score = sum(quality_scores) / total_responses

        return {
            "success": True,
            "message": "Quality checks completed",
            "overall_quality_score": overall_quality_score,
            "total_responses_checked": total_responses,
            "issues_found": issues_found,
        }

    def calculate_quality_score(self, response_text: str) -> float:
        """
        Calculates a quality score for a single response.
        """
        score = 1.0
        # Penalize short responses
        if len(response_text.split()) < 50:
            score -= 0.3

        # Penalize for placeholder-like text
        if "lorem ipsum" in response_text.lower() or "tbd" in response_text.lower():
            score -= 0.5

        # Reward for containing certain keywords
        if "our solution" in response_text.lower() or "we will provide" in response_text.lower():
            score += 0.1

        return max(0.0, min(1.0, score))