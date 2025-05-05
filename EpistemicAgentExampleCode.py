class EpistemicAgent:
    """
    An agent that implements principles of epistemic humility:
    - Maintains confidence levels for beliefs rather than binary true/false
    - Updates beliefs based on new evidence
    - Represents uncertainty explicitly
    - Seeks diverse information sources
    - Recognizes limitations in its own knowledge
    """

    def __init__(self):
        # Dictionary to store beliefs with confidence levels (0.0 to 1.0)
        self.beliefs = {}
        
        # Track sources of evidence for each belief
        self.evidence_sources = {}
        
        # Keep track of known unknowns
        self.known_unknowns = set()
        
        # Threshold for considering a belief "certain enough" to act upon
        self.action_threshold = 0.7
        
        # Bias awareness - track potential cognitive biases
        self.potential_biases = {
            "confirmation_bias": 0.0,
            "recency_bias": 0.0,
            "overconfidence": 0.0
        }

    def add_belief(self, proposition, confidence, evidence_source):
        """Add a new belief with associated confidence and evidence source."""
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
            
        if proposition in self.beliefs:
            # Update existing belief using Bayesian-inspired approach
            self.update_belief(proposition, confidence, evidence_source)
        else:
            # Add new belief
            self.beliefs[proposition] = confidence
            self.evidence_sources[proposition] = [evidence_source]
            
        # Check for overconfidence bias
        if confidence > 0.95:
            self.potential_biases["overconfidence"] += 0.1
    
    def update_belief(self, proposition, new_confidence, evidence_source):
        """Update an existing belief with new evidence."""
        if proposition not in self.beliefs:
            self.add_belief(proposition, new_confidence, evidence_source)
            return
            
        # Add the new evidence source
        if evidence_source not in self.evidence_sources[proposition]:
            self.evidence_sources[proposition].append(evidence_source)
        
        # Simple weighted average for demonstration purposes
        # A more sophisticated approach would use Bayes' theorem
        current_confidence = self.beliefs[proposition]
        
        # Check for potential confirmation bias
        if abs(current_confidence - new_confidence) < 0.1:
            self.potential_biases["confirmation_bias"] += 0.05
            
        # Weight based on number of sources (more sources = more stability)
        source_count = len(self.evidence_sources[proposition])
        weight_existing = min(0.7, source_count / (source_count + 1))
        weight_new = 1 - weight_existing
        
        # Update confidence
        updated_confidence = (current_confidence * weight_existing) + (new_confidence * weight_new)
        self.beliefs[proposition] = updated_confidence
        
        # Track recency bias when heavily weighting new information
        if weight_new > 0.5:
            self.potential_biases["recency_bias"] += 0.1
    
    def acknowledge_uncertainty(self, topic):
        """Explicitly acknowledge a known unknown."""
        self.known_unknowns.add(topic)
        
    def can_act_on_belief(self, proposition):
        """Determine if a belief has sufficient confidence to act upon."""
        if proposition not in self.beliefs:
            return False
            
        # Consider the evidential diversity as well as confidence level
        confidence = self.beliefs[proposition]
        source_diversity = len(self.evidence_sources.get(proposition, []))
        
        # Adjust for potential biases
        adjusted_confidence = confidence
        for bias, level in self.potential_biases.items():
            adjusted_confidence -= level * 0.1  # Reduce confidence based on bias levels
            
        return adjusted_confidence >= self.action_threshold and source_diversity >= 2
    
    def seek_contrary_evidence(self, proposition):
        """
        Simulate actively seeking evidence that might contradict a current belief.
        Returns a list of topics to investigate further.
        """
        if proposition not in self.beliefs:
            return [f"Gather baseline information about {proposition}"]
            
        # Generate questions that might challenge the current belief
        confidence = self.beliefs[proposition]
        questions = []
        
        if confidence > 0.7:
            questions.append(f"What evidence would disprove {proposition}?")
            questions.append(f"What are alternative explanations for the evidence supporting {proposition}?")
        
        questions.append(f"What are the strongest arguments against {proposition}?")
        
        # Reduce overconfidence bias when seeking contrary evidence
        self.potential_biases["overconfidence"] = max(0.0, self.potential_biases["overconfidence"] - 0.1)
        
        return questions
    
    def summarize_belief(self, proposition):
        """Provide a summary of a belief with appropriate epistemic qualifiers."""
        if proposition not in self.beliefs:
            return f"I don't have any information about '{proposition}'."
            
        confidence = self.beliefs[proposition]
        source_count = len(self.evidence_sources.get(proposition, []))
        
        # Select epistemic qualifier based on confidence level
        qualifier = self._get_confidence_qualifier(confidence)
        
        # Include information about evidence sources
        source_info = f"based on {source_count} source{'s' if source_count > 1 else ''}"
        
        # Acknowledge potential biases if present
        bias_disclaimer = ""
        significant_biases = [bias for bias, level in self.potential_biases.items() if level > 0.2]
        if significant_biases:
            bias_list = ", ".join(significant_biases)
            bias_disclaimer = f" (Note: This assessment may be influenced by {bias_list}.)"
            
        return f"I {qualifier} that {proposition} {source_info}.{bias_disclaimer}"
    
    def _get_confidence_qualifier(self, confidence):
        """Convert a numerical confidence value to a verbal qualifier."""
        if confidence > 0.95:
            return "am virtually certain"
        elif confidence > 0.9:
            return "am highly confident"
        elif confidence > 0.75:
            return "believe"
        elif confidence > 0.6:
            return "think it's likely"
        elif confidence > 0.4:
            return "think it's possible"
        elif confidence > 0.25:
            return "am skeptical"
        else:
            return "doubt"

# Example usage
if __name__ == "__main__":
    # Create an agent with epistemic humility
    agent = EpistemicAgent()
    
    # Add some initial beliefs
    agent.add_belief("Regular exercise improves health", 0.9, "medical research")
    agent.add_belief("Climate change is human-caused", 0.85, "IPCC report")
    agent.add_belief("This medication will cure the disease", 0.6, "preliminary study")
    
    # Acknowledge uncertainties
    agent.acknowledge_uncertainty("Long-term effects of the medication")
    agent.acknowledge_uncertainty("Precise climate tipping points")
    
    # Update a belief with new evidence
    agent.update_belief("This medication will cure the disease", 0.75, "larger clinical trial")
    
    # Demonstrate seeking contrary evidence
    questions = agent.seek_contrary_evidence("Climate change is human-caused")
    
    # Check if we have sufficient confidence to act
    can_act = agent.can_act_on_belief("Regular exercise improves health")
    
    # Print out belief summaries
    print(agent.summarize_belief("Regular exercise improves health"))
    print(agent.summarize_belief("Climate change is human-caused"))
    print(agent.summarize_belief("This medication will cure the disease"))
    print(f"Questions to investigate: {questions}")
    print(f"Sufficient confidence to recommend exercise: {can_act}")
    print(f"Known unknowns: {agent.known_unknowns}")
