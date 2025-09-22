"""
Template Resource for MCP-RFP Server
"""
import logging
from typing import List, Dict, Any
from mcp.types import Resource, TextContent, ReadResourceResult

from ..config import ServerConfig

logger = logging.getLogger(__name__)


class TemplateResource:
    """Provides MCP resource access to response templates"""

    def __init__(self, config: ServerConfig):
        self.config = config
        self._initialized = False
        self._templates = {}

    async def initialize(self):
        """Initialize template resources"""
        if self._initialized:
            return

        try:
            self._create_default_templates()
            self._initialized = True
            logger.info(f"Template resource initialized with {len(self._templates)} templates")
        except Exception as e:
            logger.error(f"Failed to initialize template resource: {str(e)}")
            raise

    async def list_resources(self) -> List[Resource]:
        """List all available template resources"""
        await self.initialize()

        resources = []

        for template_id, template_data in self._templates.items():
            resource = Resource(
                uri=f"template://{template_data['category']}/{template_id}",
                name=template_data['name'],
                description=template_data['description'],
                mimeType="text/plain"
            )
            resources.append(resource)

        return resources

    async def read_resource(self, uri: str) -> ReadResourceResult:
        """Read content of a specific template resource"""
        await self.initialize()

        try:
            # Parse URI: template://category/template_id
            if not uri.startswith("template://"):
                raise ValueError(f"Invalid template URI: {uri}")

            path_part = uri[11:]  # Remove "template://" prefix
            parts = path_part.split('/')

            if len(parts) != 2:
                raise ValueError(f"Invalid template URI format: {uri}")

            category, template_id = parts

            # Find matching template
            if template_id not in self._templates:
                raise ValueError(f"Template not found: {template_id}")

            template_data = self._templates[template_id]

            # Return template content
            return ReadResourceResult(
                contents=[
                    TextContent(
                        type="text",
                        text=template_data['content']
                    )
                ]
            )

        except Exception as e:
            logger.error(f"Failed to read template {uri}: {str(e)}")
            return ReadResourceResult(
                contents=[
                    TextContent(
                        type="text",
                        text=f"Error reading template: {str(e)}"
                    )
                ]
            )

    def _create_default_templates(self):
        """Create default response templates"""

        self._templates = {
            "formal_response": {
                "name": "Formal Response Template",
                "category": "response_styles",
                "description": "Professional, formal response template for standard RFP requirements",
                "content": """
**Response Structure:**
1. Executive Summary
2. Technical Approach
3. Past Experience and Qualifications
4. Implementation Timeline
5. Risk Mitigation
6. Value Proposition

**Tone:** Professional, confident, and authoritative
**Length:** Comprehensive and detailed
**Focus:** Demonstrating capability and compliance
                """.strip()
            },

            "technical_response": {
                "name": "Technical Response Template",
                "category": "response_styles",
                "description": "Technical response template emphasizing implementation details",
                "content": """
**Response Structure:**
1. Technical Solution Overview
2. Architecture and Design
3. Implementation Methodology
4. Testing and Quality Assurance
5. Technical Team Qualifications
6. Technology Stack and Tools

**Tone:** Technical, precise, and detailed
**Length:** In-depth technical specifications
**Focus:** Technical excellence and implementation details
                """.strip()
            },

            "executive_response": {
                "name": "Executive Response Template",
                "category": "response_styles",
                "description": "Executive-level response focusing on business value",
                "content": """
**Response Structure:**
1. Business Value Proposition
2. Strategic Approach
3. Competitive Advantages
4. Executive Team Oversight
5. Success Metrics
6. Long-term Partnership Benefits

**Tone:** Strategic, business-focused, and visionary
**Length:** Concise with high-level overview
**Focus:** Business outcomes and strategic value
                """.strip()
            },

            "security_response": {
                "name": "Security Response Template",
                "category": "specialized",
                "description": "Template for security and compliance requirements",
                "content": """
**Response Structure:**
1. Security Framework Adherence
2. Compliance Certifications
3. Data Protection Measures
4. Access Control Implementation
5. Security Monitoring and Incident Response
6. Audit and Reporting Capabilities

**Key Elements:**
- Reference specific compliance standards (FedRAMP, NIST, HIPAA)
- Mention security certifications and clearances
- Detail technical security controls
- Emphasize continuous monitoring
                """.strip()
            },

            "past_performance": {
                "name": "Past Performance Template",
                "category": "specialized",
                "description": "Template for demonstrating relevant experience",
                "content": """
**Response Structure:**
1. Project Overview and Scope
2. Client and Contract Details
3. Technical Challenges Addressed
4. Key Achievements and Metrics
5. Lessons Learned and Applied
6. Client References and Testimonials

**Key Elements:**
- Quantifiable results and outcomes
- Specific technical accomplishments
- Client satisfaction metrics
- Relevance to current requirement
                """.strip()
            },

            "pricing_response": {
                "name": "Pricing Response Template",
                "category": "specialized",
                "description": "Template for cost and pricing sections",
                "content": """
**Response Structure:**
1. Cost Summary and Breakdown
2. Pricing Methodology
3. Value Justification
4. Cost-Benefit Analysis
5. Flexible Pricing Options
6. Total Cost of Ownership

**Key Elements:**
- Transparent pricing structure
- Clear value proposition
- Competitive positioning
- Flexibility and options
                """.strip()
            },

            "management_approach": {
                "name": "Management Approach Template",
                "category": "specialized",
                "description": "Template for project management and approach",
                "content": """
**Response Structure:**
1. Project Management Methodology (Agile/Scrum)
2. Team Structure and Roles
3. Communication Plan
4. Risk Management Strategy
5. Quality Assurance Process
6. Change Management Approach

**Key Elements:**
- Proven methodologies
- Clear governance structure
- Regular communication cadence
- Proactive risk management
                """.strip()
            }
        }

    def get_template(self, template_id: str) -> Dict[str, Any]:
        """Get template data by ID"""
        return self._templates.get(template_id)

    def get_templates_by_category(self, category: str) -> Dict[str, Any]:
        """Get all templates in a specific category"""
        return {
            tid: template for tid, template in self._templates.items()
            if template['category'] == category
        }

    def get_available_categories(self) -> List[str]:
        """Get list of available template categories"""
        return list(set(template['category'] for template in self._templates.values()))