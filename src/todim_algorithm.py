# ============================================================
# TODIM Algorithm with Prospect Theory
# Part of: Disease Risk Prioritization System
# ============================================================
#STEP 1: Implementing the TODIM (Interactive Multi-criteria Decision Making (MCDM)) Algorithm
#this implementation includes prospect theory and loss aversion concepts
import numpy as np
class TODIM_Algorithm:
    """
    TODIM (Interactive Multi-criteria Decision Making) Algorithm Implementation
    
    This class implements the TODIM method based on Prospect Theory, which considers:
    - Loss aversion: People feel losses more intensely than equivalent gains
    - Reference point dependence: Outcomes are evaluated relative to a reference point
    - Psychological behavior: DMs tend to be risk-averse for gains and risk-seeking for losses
    """
    
    def __init__(self, theta=2.25, alpha=0.88, beta=0.88, lambda_param=2.25):
        """
        Initialize TODIM with Prospect Theory parameters
        
        Parameters:
        - theta: Loss aversion parameter (typically 2.25, range [1, 10])
        - alpha: Curvature parameter for gains (typically 0.88)
        - beta: Curvature parameter for losses (typically 0.88)
        - lambda_param: Loss aversion coefficient (typically 2.25)
        """
        self.theta = theta  # Loss aversion factor
        self.alpha = alpha  # Gains curvature
        self.beta = beta    # Losses curvature
        self.lambda_param = lambda_param  # Loss aversion coefficient
        
    def normalize_matrix(self, decision_matrix):
        """
        Normalize the decision matrix using min-max normalization
        
        Parameters:
        - decision_matrix: numpy array of alternatives x criteria
        
        Returns:
        - normalized_matrix: normalized decision matrix
        """
        normalized_matrix = np.zeros_like(decision_matrix)
        
        for j in range(decision_matrix.shape[1]):  # For each criterion
            col_min = np.min(decision_matrix[:, j])
            col_max = np.max(decision_matrix[:, j])
            
            if col_max != col_min:
                normalized_matrix[:, j] = (decision_matrix[:, j] - col_min) / (col_max - col_min)
            else:
                normalized_matrix[:, j] = decision_matrix[:, j]
                
        return normalized_matrix
    
    def calculate_relative_weights(self, weights):
        """
        Calculate relative weights with respect to the reference criterion
        The reference criterion is the one with the highest weight
        
        Parameters:
        - weights: array of criterion weights
        
        Returns:
        - relative_weights: array of relative weights
        """
        max_weight = np.max(weights)
        relative_weights = weights / max_weight
        return relative_weights
    
    def calculate_dominance_degree(self, alternative_i, alternative_j, criterion_k, 
                                 relative_weight_k, normalized_matrix):
        """
        Calculate the dominance degree of alternative i over alternative j 
        for criterion k using Prospect Theory
        
        This is the core of TODIM - it measures how much one alternative 
        dominates another for a specific criterion
        """
        # Getting performance values for the criterion
        perf_i = normalized_matrix[alternative_i, criterion_k]
        perf_j = normalized_matrix[alternative_j, criterion_k]
        
        # Calculating the difference
        diff = perf_i - perf_j
        
        if diff >= 0:  # Gain situation
            # Use square root for gains (as in classical TODIM)
            dominance = np.sqrt(relative_weight_k * diff)
        else:  # Loss situation
            # Apply loss aversion - losses are felt more intensely
            dominance = -1/self.theta * np.sqrt((-diff) / relative_weight_k)
            
        return dominance
    
    def calculate_overall_dominance(self, alternative_i, alternative_j, 
                                  relative_weights, normalized_matrix):
        """
        Calculate overall dominance of alternative i over alternative j
        across all criteria
        """
        total_dominance = 0
        
        for criterion_k in range(len(relative_weights)):
            dominance_k = self.calculate_dominance_degree(
                alternative_i, alternative_j, criterion_k, 
                relative_weights[criterion_k], normalized_matrix
            )
            total_dominance += dominance_k
            
        return total_dominance
    
    def calculate_perceived_value(self, alternative_i, normalized_matrix, relative_weights):
        """
        Calculate the perceived value (final score) for alternative i
        This represents how attractive the alternative is to the decision maker
        """
        n_alternatives = normalized_matrix.shape[0]
        total_dominance = 0
        
        for alternative_j in range(n_alternatives):
            if alternative_i != alternative_j:
                dominance_ij = self.calculate_overall_dominance(
                    alternative_i, alternative_j, relative_weights, normalized_matrix
                )
                total_dominance += dominance_ij
                
        return total_dominance
    
    def rank_alternatives(self, decision_matrix, weights, alternative_names=None):
        """
        Main method to rank alternatives using TODIM
        
        Parameters:
        - decision_matrix: numpy array (alternatives x criteria)
        - weights: array of criterion weights
        - alternative_names: list of alternative names (optional)
        
        Returns:
        - ranking_results: dictionary with rankings and scores
        """
        # Step 1: Normalizing the decision matrix
        normalized_matrix = self.normalize_matrix(decision_matrix)
        
        # Step 2: Calculating relative weights
        relative_weights = self.calculate_relative_weights(weights)
        
        # Step 3: Calculating perceived values for all alternatives
        n_alternatives = normalized_matrix.shape[0]
        perceived_values = []
        
        for i in range(n_alternatives):
            perceived_value = self.calculate_perceived_value(i, normalized_matrix, relative_weights)
            perceived_values.append(perceived_value)
        
        # Step 4: Normalizing perceived values to [0, 1]
        perceived_values = np.array(perceived_values)
        min_val = np.min(perceived_values)
        max_val = np.max(perceived_values)
        
        if max_val != min_val:
            normalized_scores = (perceived_values - min_val) / (max_val - min_val)
        else:
            normalized_scores = np.ones_like(perceived_values)
        
        # Step 5: Creating ranking
        ranking_indices = np.argsort(normalized_scores)[::-1]  # Descending order
        
        # Preparing results
        if alternative_names is None:
            alternative_names = [f"Alternative_{i+1}" for i in range(n_alternatives)]
            
        ranking_results = {
            'rankings': [(alternative_names[i], normalized_scores[i]) for i in ranking_indices],
            'scores': normalized_scores,
            'raw_perceived_values': perceived_values,
            'normalized_matrix': normalized_matrix,
            'relative_weights': relative_weights
        }
        
        return ranking_results
    