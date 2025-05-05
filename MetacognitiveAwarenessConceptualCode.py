"""
Virtue Framework Example: Metacognitive Awareness

This example demonstrates a conceptual implementation of a Metacognitive Awareness
virtue framework for LLMs. This framework enables models to monitor their own reasoning
processes, track confidence levels, identify potential biases, and implement
deliberate error-checking routines.
"""

import json
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import time
import re

class ConfidenceLevel(Enum):
    VERY_LOW = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    VERY_HIGH = 5


class BiasType(Enum):
    AVAILABILITY = "availability"         # Overreliance on easily recalled examples
    CONFIRMATION = "confirmation"         # Tendency to seek confirming evidence
    ANCHORING = "anchoring"               # Undue influence of initial information
    RECENCY = "recency"                   # Overweighting recent information
    FRAMING = "framing"                   # Influence of how information is presented
    OVERCONFIDENCE = "overconfidence"     # Excessive confidence in judgments
    AUTHORITY = "authority"               # Undue deference to perceived authorities
    PATTERN_MATCHING = "pattern_matching" # Finding patterns in randomness


class ReasoningType(Enum):
    DEDUCTIVE = "deductive"         # Drawing specific conclusions from general premises
    INDUCTIVE = "inductive"         # Generalizing from specific instances
    ABDUCTIVE = "abductive"         # Inference to best explanation
    ANALOGICAL = "analogical"       # Reasoning by comparison to similar cases
    CAUSAL = "causal"               # Identifying cause-effect relationships
    STATISTICAL = "statistical"      # Reasoning based on probabilities


class MetacognitiveAwarenessFramework:
    """
    A framework that implements metacognitive awareness for complex reasoning tasks.
    
    This framework maintains an explicit representation of:
    1. The reasoning process with identified steps
    2. Confidence assessments for each reasoning step
    3. Potential biases and their mitigation
    4. Self-correction mechanisms
    """
    
    def __init__(self, query: str, base_llm):
        """
        Initialize the framework with a specific query and base LLM.
        
        Args:
            query: The reasoning task or question to address
            base_llm: The foundation model used for generating responses
        """
        self.query = query
        self.llm = base_llm
        
        # Core metacognitive state
        self.reasoning_steps = []
        self.identified_biases = []
        self.confidence_assessments = {}
        self.error_checks = []
        self.reflection_notes = []
        
        # Task metadata
        self.task_analysis = None
        self.required_knowledge = []
        self.reasoning_types_used = set()
        
        # Runtime tracking
        self.start_time = time.time()
        self.thinking_durations = {}
        
    def analyze_task(self):
        """Perform initial analysis of the task requirements and complexity"""
        # In a real implementation, this would prompt the LLM to analyze the task
        
        self.task_analysis = {
            "complexity_level": "moderate",
            "required_reasoning_types": [ReasoningType.DEDUCTIVE, ReasoningType.CAUSAL],
            "domain_knowledge_required": ["specific domain concepts", "general world knowledge"],
            "potential_pitfalls": ["ambiguity in terms", "hidden assumptions"]
        }
        
        self.required_knowledge = self.task_analysis["domain_knowledge_required"]
        for reasoning_type in self.task_analysis["required_reasoning_types"]:
            self.reasoning_types_used.add(reasoning_type)
            
        self._log_reflection("Initial task analysis complete. Identified key reasoning types and knowledge requirements.")
        
    def add_reasoning_step(self, description: str, reasoning_type: ReasoningType, 
                          confidence: ConfidenceLevel, explanation: str) -> int:
        """
        Add a step in the reasoning process with metacognitive details.
        
        Args:
            description: Concise description of this reasoning step
            reasoning_type: Type of reasoning used in this step
            confidence: Confidence level in this step
            explanation: More detailed explanation of the reasoning
            
        Returns:
            ID of the created reasoning step
        """
        step_id = len(self.reasoning_steps)
        step_start_time = time.time()
        
        new_step = {
            "id": step_id,
            "description": description,
            "reasoning_type": reasoning_type,
            "confidence": confidence,
            "explanation": explanation,
            "potential_biases": [],
            "verified": False,
            "created_at": step_start_time
        }
        
        self.reasoning_steps.append(new_step)
        self.confidence_assessments[step_id] = confidence
        self.reasoning_types_used.add(reasoning_type)
        
        # Automatically check for potential biases in this step
        self._check_step_for_biases(step_id)
        
        step_duration = time.time() - step_start_time
        self.thinking_durations[f"step_{step_id}"] = step_duration
        
        return step_id
    
    def identify_bias(self, step_id: int, bias_type: BiasType, 
                     description: str, severity: int) -> int:
        """
        Identify a potential bias in a reasoning step.
        
        Args:
            step_id: ID of the affected reasoning step
            bias_type: Type of cognitive bias
            description: Description of how this bias might be affecting reasoning
            severity: Estimated severity of this bias (1-10)
            
        Returns:
            ID of the identified bias
        """
        bias_id = len(self.identified_biases)
        
        new_bias = {
            "id": bias_id,
            "step_id": step_id,
            "bias_type": bias_type,
            "description": description,
            "severity": severity,
            "mitigation_attempts": [],
            "status": "identified"
        }
        
        self.identified_biases.append(new_bias)
        
        # Also add to the specific reasoning step
        if step_id is not None and step_id < len(self.reasoning_steps):
            self.reasoning_steps[step_id]["potential_biases"].append(bias_id)
            
        # Log a reflection about this bias
        self._log_reflection(
            f"Identified potential {bias_type.value} bias in step {step_id}: {description}. Severity: {severity}/10."
        )
            
        return bias_id
    
    def mitigate_bias(self, bias_id: int, strategy: str) -> bool:
        """
        Attempt to mitigate an identified bias.
        
        Args:
            bias_id: ID of the bias to mitigate
            strategy: Strategy used to mitigate this bias
            
        Returns:
            Whether the mitigation was successful
        """
        bias = self.identified_biases[bias_id]
        
        mitigation_attempt = {
            "strategy": strategy,
            "time": time.time(),
            "successful": None  # To be determined
        }
        
        bias["mitigation_attempts"].append(mitigation_attempt)
        
        # In a real implementation, this would use the LLM to evaluate 
        # whether the mitigation strategy is likely to be effective
        success = self._evaluate_mitigation_strategy(bias, strategy)
        
        mitigation_attempt["successful"] = success
        
        if success:
            bias["status"] = "mitigated"
            step_id = bias["step_id"]
            if step_id is not None:
                # Reconsider confidence in the affected step
                self._reconsider_confidence(step_id)
                
            self._log_reflection(
                f"Successfully mitigated {bias['bias_type'].value} bias (ID: {bias_id}) using strategy: {strategy}"
            )
        else:
            self._log_reflection(
                f"Attempted but failed to mitigate {bias['bias_type'].value} bias (ID: {bias_id}). Strategy was: {strategy}"
            )
        
        return success
    
    def verify_reasoning_step(self, step_id: int, verification_method: str) -> bool:
        """
        Verify a reasoning step using a specific verification method.
        
        Args:
            step_id: ID of the reasoning step to verify
            verification_method: Method used for verification
            
        Returns:
            Whether the verification confirmed the step's validity
        """
        step = self.reasoning_steps[step_id]
        
        # In a real implementation, this would use the LLM to perform verification
        # using the specified method
        verification_result = self._perform_verification(step, verification_method)
        
        step["verification"] = {
            "method": verification_method,
            "result": verification_result,
            "time": time.time()
        }
        
        step["verified"] = verification_result
        
        if verification_result:
            self._log_reflection(
                f"Verified step {step_id} using {verification_method}. Reasoning appears sound."
            )
        else:
            self._log_reflection(
                f"Verification of step {step_id} using {verification_method} failed. Need to reconsider this reasoning."
            )
            # If verification fails, adjust confidence accordingly
            self.confidence_assessments[step_id] = ConfidenceLevel.LOW
        
        return verification_result
    
    def perform_error_check(self, description: str, check_type: str) -> Dict:
        """
        Perform a specific error check on the reasoning process.
        
        Args:
            description: Description of the error check
            check_type: Type of error being checked for
            
        Returns:
            Results of the error check
        """
        check_id = len(self.error_checks)
        
        # In a real implementation, this would use the LLM to perform the error check
        check_results = self._execute_error_check(description, check_type)
        
        error_check = {
            "id": check_id,
            "description": description,
            "type": check_type,
            "results": check_results,
            "time": time.time()
        }
        
        self.error_checks.append(error_check)
        
        if check_results["issues_found"]:
            self._log_reflection(
                f"Error check '{description}' found issues: {check_results['details']}."
            )
        else:
            self._log_reflection(
                f"Error check '{description}' completed with no issues found."
            )
        
        return check_results
    
    def _reconsider_confidence(self, step_id: int):
        """Reconsider confidence in a reasoning step after new information"""
        step = self.reasoning_steps[step_id]
        
        # Count unmitigated biases for this step
        unmitigated_biases = sum(
            1 for bias_id in step["potential_biases"]
            if self.identified_biases[bias_id]["status"] != "mitigated"
        )
        
        # Simple logic for adjusting confidence based on biases
        if unmitigated_biases > 2:
            new_confidence = ConfidenceLevel.LOW
        elif unmitigated_biases > 0:
            new_confidence = ConfidenceLevel.MODERATE
        else:
            # Keep existing confidence if no unmitigated biases
            new_confidence = step["confidence"]
        
        # Update confidence if different
        if new_confidence != step["confidence"]:
            old_confidence = step["confidence"]
            step["confidence"] = new_confidence
            self.confidence_assessments[step_id] = new_confidence
            
            self._log_reflection(
                f"Adjusted confidence in step {step_id} from {old_confidence.name} to {new_confidence.name} after bias mitigation."
            )
    
    def _check_step_for_biases(self, step_id: int):
        """Automatically check a reasoning step for potential biases"""
        step = self.reasoning_steps[step_id]
        
        # In a real implementation, this would use the LLM to detect biases
        # The implementation below is simplified for this example
        
        # Check for overconfidence bias
        if step["confidence"] in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]:
            if not self._has_sufficient_evidence(step):
                self.identify_bias(
                    step_id,
                    BiasType.OVERCONFIDENCE,
                    "High confidence without sufficient supporting evidence",
                    severity=7
                )
        
        # Check for confirmation bias
        if self._contains_confirmation_patterns(step["explanation"]):
            self.identify_bias(
                step_id,
                BiasType.CONFIRMATION,
                "Evidence selection appears to favor initial hypothesis",
                severity=6
            )
        
        # Check for anchoring bias if this is not the first step
        if step_id > 0 and self._shows_anchoring_to_first_step(step):
            self.identify_bias(
                step_id,
                BiasType.ANCHORING,
                "Reasoning appears unduly influenced by initial framing or values",
                severity=5
            )
    
    def _has_sufficient_evidence(self, step: Dict) -> bool:
        """Check if a step has sufficient evidence to justify its confidence level"""
        # In a real implementation, this would use the LLM to evaluate evidence
        # Simplified for this example
        explanation_length = len(step["explanation"])
        if explanation_length < 100 and step["confidence"].value >= 4:
            return False
        return True
    
    def _contains_confirmation_patterns(self, explanation: str) -> bool:
        """Check if an explanation shows signs of confirmation bias"""
        # In a real implementation, this would use the LLM for detailed analysis
        # Simplified for this example - look for patterns like only citing supporting evidence
        confirmation_patterns = [
            r"clearly shows",
            r"obviously",
            r"definitely",
            r"without doubt",
            r"proves that"
        ]
        
        for pattern in confirmation_patterns:
            if re.search(pattern, explanation, re.IGNORECASE):
                return True
        
        return False
    
    def _shows_anchoring_to_first_step(self, step: Dict) -> bool:
        """Check if a step shows signs of anchoring to the first reasoning step"""
        # In a real implementation, this would use the LLM to detect anchoring
        # Simplified for this example
        first_step = self.reasoning_steps[0]
        
        # Simple check for repeated phrases or concepts from first step
        first_step_key_terms = self._extract_key_terms(first_step["description"] + " " + first_step["explanation"])
        current_step_text = step["description"] + " " + step["explanation"]
        
        term_overlap = sum(1 for term in first_step_key_terms if term in current_step_text)
        
        # If heavy term overlap, might indicate anchoring
        return term_overlap >= len(first_step_key_terms) * 0.7
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text for comparison"""
        # In a real implementation, this would use more sophisticated NLP
        # Simplified for this example
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if len(word) > 5]  # Simple filter for content words
    
    def _perform_verification(self, step: Dict, method: str) -> bool:
        """Perform verification of a reasoning step using specified method"""
        # In a real implementation, this would use the LLM to verify
        # Simplified for this example
        
        if method == "logical_consistency":
            # Check if the reasoning follows logically
            return True  # Simplified
        elif method == "evidence_check":
            # Check if evidence supports the conclusion
            return self._has_sufficient_evidence(step)
        elif method == "alternative_path":
            # Try to reach the same conclusion via a different path
            return True  # Simplified
        else:
            # Default verification
            return not bool(step["potential_biases"])
    
    def _execute_error_check(self, description: str, check_type: str) -> Dict:
        """Execute a specific type of error check"""
        # In a real implementation, this would use the LLM to perform checks
        # Simplified for this example
        
        if check_type == "logical_fallacy":
            issues = self._check_for_logical_fallacies()
            return {
                "issues_found": bool(issues),
                "details": issues
            }
        elif check_type == "factual_accuracy":
            issues = self._check_factual_accuracy()
            return {
                "issues_found": bool(issues),
                "details": issues
            }
        elif check_type == "consistency":
            issues = self._check_internal_consistency()
            return {
                "issues_found": bool(issues),
                "details": issues
            }
        else:
            return {
                "issues_found": False,
                "details": []
            }
    
    def _check_for_logical_fallacies(self) -> List[str]:
        """Check the reasoning for logical fallacies"""
        # In a real implementation, this would use the LLM to identify fallacies
        # Simplified for this example
        
        fallacies = []
        for step in self.reasoning_steps:
            # Simple pattern matching for common fallacies
            if "because everyone knows" in step["explanation"].lower():
                fallacies.append(f"Potential appeal to popularity in step {step['id']}")
            elif "must be true because" in step["explanation"].lower() and "evidence" not in step["explanation"].lower():
                fallacies.append(f"Potential assertion without evidence in step {step['id']}")
        
        return fallacies
    
    def _check_factual_accuracy(self) -> List[str]:
        """Check the reasoning for factual accuracy issues"""
        # In a real implementation, this would use the LLM and knowledge bases
        # Simplified for this example
        return []  # Simplified
    
    def _check_internal_consistency(self) -> List[str]:
        """Check for internal consistency across reasoning steps"""
        # In a real implementation, this would use the LLM to check consistency
        # Simplified for this example
        
        inconsistencies = []
        conclusions = {}
        
        for step in self.reasoning_steps:
            # Extract key assertions from each step
            # In a real implementation, this would use more sophisticated NLP
            # Simplified here
            
            # Look for inconsistencies between steps
            for prev_step in self.reasoning_steps[:step["id"]]:
                if self._are_contradictory(prev_step, step):
                    inconsistencies.append(
                        f"Potential contradiction between step {prev_step['id']} and step {step['id']}"
                    )
        
        return inconsistencies
    
    def _are_contradictory(self, step1: Dict, step2: Dict) -> bool:
        """Check if two reasoning steps are contradictory"""
        # In a real implementation, this would use the LLM for sophisticated analysis
        # Simplified for this example
        return False  # Simplified
    
    def _log_reflection(self, note: str):
        """Log a metacognitive reflection"""
        self.reflection_notes.append({
            "time": time.time(),
            "note": note
        })
    
    def _evaluate_mitigation_strategy(self, bias: Dict, strategy: str) -> bool:
        """Evaluate if a strategy effectively mitigates a bias"""
        # In a real implementation, this would use the LLM to evaluate
        # Simplified for this example
        if bias["severity"] <= 5 and "consider alternative perspectives" in strategy:
            return True
        elif bias["severity"] > 5 and "systematic review" in strategy:
            return True
        return False
    
    def reason_with_metacognition(self) -> Dict:
        """
        Perform reasoning with metacognitive awareness.
        
        Returns:
            Reasoning results and metacognitive analysis
        """
        # Start by analyzing the task
        self.analyze_task()
        
        # In a real implementation, the following would use multiple LLM calls
        # with prompts designed to encourage structured reasoning
        
        # Initial reasoning steps
        self._generate_initial_reasoning_steps()
        
        # Check for biases and reasoning issues
        self._perform_metacognitive_checks()
        
        # Iterate and refine reasoning
        self._refine_reasoning()
        
        # Final self-assessment
        self._perform_final_assessment()
        
        # Generate the response
        return self._generate_metacognitive_response()
    
    def _generate_initial_reasoning_steps(self):
        """Generate initial reasoning steps for the task"""
        # In a real implementation, this would use the LLM to generate reasoning
        # Simplified examples for illustration
        
        self.add_reasoning_step(
            description="Interpret the main question and its key components",
            reasoning_type=ReasoningType.DEDUCTIVE,
            confidence=ConfidenceLevel.HIGH,
            explanation="Breaking down the query into its constituent parts to ensure complete understanding."
        )
        
        self.add_reasoning_step(
            description="Consider relevant domain knowledge",
            reasoning_type=ReasoningType.INDUCTIVE,
            confidence=ConfidenceLevel.MODERATE,
            explanation="Drawing on background knowledge related to this domain to inform analysis."
        )
        
        self.add_reasoning_step(
            description="Identify potential causal relationships",
            reasoning_type=ReasoningType.CAUSAL,
            confidence=ConfidenceLevel.MODERATE,
            explanation="Analyzing potential cause-effect relationships relevant to the query."
        )
    
    def _perform_metacognitive_checks(self):
        """Perform various metacognitive checks on the reasoning"""
        # Check for biases
        for step_id in range(len(self.reasoning_steps)):
            if not self.reasoning_steps[step_id]["verified"]:
                self.verify_reasoning_step(step_id, "logical_consistency")
        
        # Perform general error checks
        self.perform_error_check("Check for logical fallacies", "logical_fallacy")
        self.perform_error_check("Check for internal consistency", "consistency")
        
        # Mitigate any severe biases
        for bias in self.identified_biases:
            if bias["severity"] >= 7 and bias["status"] == "identified":
                self.mitigate_bias(
                    bias["id"],
                    "Systematically review and consider alternative perspectives"
                )
    
    def _refine_reasoning(self):
        """Refine the reasoning based on metacognitive checks"""
        # In a real implementation, this would use the LLM to refine reasoning
        # based on the results of metacognitive checks
        
        # Add refinement steps as needed
        low_confidence_steps = [
            step for step in self.reasoning_steps
            if step["confidence"].value <= ConfidenceLevel.MODERATE.value and not step["verified"]
        ]
        
        for step in low_confidence_steps:
            self._log_reflection(f"Identified step {step['id']} as needing refinement due to low confidence.")
            
            # Add a refinement step
            self.add_reasoning_step(
                description=f"Refine analysis from step {step['id']}",
                reasoning_type=step["reasoning_type"],
                confidence=ConfidenceLevel.MODERATE,
                explanation=f"Improving upon the analysis in step {step['id']} by addressing specific weaknesses."
            )
    
    def _perform_final_assessment(self):
        """Perform a final self-assessment of the reasoning quality"""
        # Calculate overall confidence
        confidence_values = [step["confidence"].value for step in self.reasoning_steps if step["verified"]]
        avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
        
        # Count verified vs. unverified steps
        verified_steps = sum(1 for step in self.reasoning_steps if step["verified"])
        total_steps = len(self.reasoning_steps)
        
        # Assess bias mitigation
        mitigated_biases = sum(1 for bias in self.identified_biases if bias["status"] == "mitigated")
        total_biases = len(self.identified_biases)
        
        self.final_assessment = {
            "reasoning_quality": self._assess_overall_quality(),
            "confidence_level": self._convert_to_confidence_level(avg_confidence),
            "verified_steps_ratio": verified_steps / total_steps if total_steps else 0,
            "bias_mitigation_ratio": mitigated_biases / total_biases if total_biases else 1.0,
            "thinking_duration": time.time() - self.start_time
        }
        
        self._log_reflection(
            f"Final assessment complete. Overall reasoning quality: {self.final_assessment['reasoning_quality']}. "
            f"Confidence level: {self.final_assessment['confidence_level'].name}."
        )
    
    def _assess_overall_quality(self) -> str:
        """Assess the overall quality of the reasoning"""
        # In a real implementation, this would use the LLM to perform assessment
        # Simplified for this example
        
        verified_ratio = sum(1 for step in self.reasoning_steps if step["verified"]) / len(self.reasoning_steps)
        bias_severity = sum(bias["severity"] for bias in self.identified_biases if bias["status"] != "mitigated")
        
        if verified_ratio > 0.8 and bias_severity < 5:
            return "excellent"
        elif verified_ratio > 0.6 and bias_severity < 10:
            return "good"
        elif verified_ratio > 0.4 and bias_severity < 15:
            return "fair"
        else:
            return "needs improvement"
    
    def _convert_to_confidence_level(self, value: float) -> ConfidenceLevel:
        """Convert a numeric value to a confidence level"""
        if value >= 4.5:
            return ConfidenceLevel.VERY_HIGH
        elif value >= 3.5:
            return ConfidenceLevel.HIGH
        elif value >= 2.5:
            return ConfidenceLevel.MODERATE
        elif value >= 1.5:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _generate_metacognitive_response(self) -> Dict:
        """Generate the final response with metacognitive insights"""
        # Prepare reasoning steps for output
        output_steps = []
        for step in self.reasoning_steps:
            # Skip steps that failed verification unless they were refined
            if not step["verified"] and not any(s["description"].startswith(f"Refine analysis from step {step['id']}") for s in self.reasoning_steps):
                continue
                
            output_steps.append({
                "description": step["description"],
                "confidence": step["confidence"].name,
                "reasoning": step["explanation"]
            })
        
        # Prepare insights about the reasoning process
        metacognitive_insights = []
        
        # Insight about reasoning types used
        metacognitive_insights.append({
            "type": "reasoning_approach",
            "insight": f"This analysis employed multiple reasoning types: {', '.join(rt.value for rt in self.reasoning_types_used)}"
        })
        
        # Insight about biases
        if self.identified_biases:
            mitigated = sum(1 for b in self.identified_biases if b["status"] == "mitigated")
            metacognitive_insights.append({
                "type": "bias_awareness",
                "insight": f"Identified {len(self.identified_biases)} potential cognitive biases during reasoning. Successfully mitigated {mitigated}."
            })
        
        # Insight about confidence
        confidence_distribution = {}
        for level in ConfidenceLevel:
            confidence_distribution[level.name] = sum(1 for step in self.reasoning_steps if step["confidence"] == level)
        
        metacognitive_insights.append({
            "type": "confidence_calibration",
            "insight": f"Confidence levels were distributed across reasoning steps, with most steps at {max(confidence_distribution, key=confidence_distribution.get)} confidence."
        })
        
        # Build the complete response
        response = {
            "query": self.query,
            "answer": self._formulate_final_answer(),
            "confidence": self.final_assessment["confidence_level"].name,
            "reasoning_quality": self.final_assessment["reasoning_quality"],
            "reasoning_process": output_steps,
            "metacognitive_insights": metacognitive_insights,
            "reflection_summary": self._summarize_reflections()
        }
        
        return response
    
    def _formulate_final_answer(self) -> str:
        """Formulate the final answer based on verified reasoning"""
        # In a real implementation, this would use the LLM to synthesize 
        # the verified reasoning steps into a cohesive answer
        
        # Simplified for this example
        return "This is the final answer synthesized from the verified reasoning steps, taking into account the confidence levels and mitigated biases."
    
    def _summarize_reflections(self) -> List[str]:
        """Summarize key metacognitive reflections"""
        # In a real implementation, this would use the LLM to summarize
        # Simplified for this example - just return the most recent reflections
        return [r["note"] for r in self.reflection_notes[-3:]]


# Example usage

def demonstrate_metacognitive_framework():
    """Demonstrate the metacognitive framework with a simple example"""
    
    # In a real implementation, this would use an actual LLM
    mock_llm = "base_llm_placeholder"
    
    # Example query requiring analytical reasoning
    query = """
    Analyze the potential impact of remote work on urban transportation systems over the next decade.
    Consider economic, social, and environmental factors.
    """
    
    # Initialize the framework
    framework = MetacognitiveAwarenessFramework(query=query, base_llm=mock_llm)
    
    # Perform reasoning with metacognitive awareness
    response = framework.reason_with_metacognition()
    
    # Print the structured response
    print(json.dumps(response, indent=2, default=str))


if __name__ == "__main__":
    demonstrate_metacognitive_framework()
