# ============================================================
# Integrated Disease Risk Prioritization System
# Combines TODIM + Fuzzy Logic + Machine Learning
# Part of: Disease Risk Prioritization System
# ============================================================

# STEP 4: Create the Integrated Disease Risk Prioritization System
import numpy as np
from src.todim_algorithm import TODIM_Algorithm
from src.fuzzy_logic import FuzzyLogicSystem
from src.ml_predictor import MLDiseaseRiskPredictor
class IntegratedDiseaseRiskSystem:
    """
    Integrated Disease Risk Prioritization System
    
    This system combines:
    1. TODIM (Multi-Criteria Decision Making)
    2. Fuzzy Logic (Uncertainty handling)
    3. Machine Learning (Pattern recognition)
    4. Trust and Risk factors
    """
    
    def __init__(self):
        self.todim = TODIM_Algorithm(theta=2.25)
        self.fuzzy_system = FuzzyLogicSystem()
        self.ml_system = MLDiseaseRiskPredictor()
        self.is_initialized = False
        
        # Disease categories for prioritization
        self.disease_categories = [
            'Cardiovascular Disease',
            'Type 2 Diabetes',
            'Hypertension',
            'Obesity-related Disorders',
            'Metabolic Syndrome'
        ]
        
        # Criteria weights for TODIM (these can be adjusted based on medical expertise)
        self.criteria_weights = np.array([
            0.25,  # ML Probability Score
            0.20,  # Fuzzy Risk Score
            0.15,  # Age Factor
            0.15,  # Clinical Biomarkers
            0.12,  # Lifestyle Factors
            0.08,  # Family History
            0.05   # Trust Factor
        ])
        
        self.criteria_names = [
            'ML_Probability',
            'Fuzzy_Risk',
            'Age_Factor',
            'Clinical_Biomarkers',
            'Lifestyle_Factors',
            'Family_History',
            'Trust_Factor'
        ]
    
    def initialize_system(self, n_training_samples=1000):
        """
        Initialize the integrated system with training data
        """
        print("Initializing Integrated Disease Risk System...")
        
        #generating training data
        print("Generating synthetic medical dataset...")
        training_data = self.ml_system.create_synthetic_dataset(n_training_samples)
        
        #training ML models
        print("Training machine learning models...")
        model_scores = self.ml_system.train_models(training_data)
        
        #initializing fuzzy system
        print("Initializing fuzzy logic system...")
        self.fuzzy_system.initialize_all_fuzzy_sets()
        
        self.is_initialized = True
        print("System initialization complete!")
        
        return training_data, model_scores
    
    def calculate_trust_factor(self, patient_data):
        """
        Calculating trust factor based on data completeness and reliability
        
        Parameters:
        - patient_data: dictionary with patient information
        
        Returns:
        - trust_score: value between 0 and 1
        """
        #counting available data points
        required_fields = ['age', 'bmi', 'systolic_bp', 'cholesterol', 
                          'smoking', 'family_history', 'physical_activity', 'diabetes']
        
        available_fields = sum(1 for field in required_fields 
                             if field in patient_data and patient_data[field] is not None)
        
        completeness_score = available_fields / len(required_fields)
        
        #adding data quality assessment (simplified)
        quality_score = 1.0  # Assume high quality for synthetic data
        
        #combining completeness and quality
        trust_score = (completeness_score * 0.7) + (quality_score * 0.3)
        
        return trust_score
    
    def calculate_age_factor(self, age):
        """
        Calculating age-based risk factor using a non-linear function
        """
        #age risk increases non-linearly, especially after 50
        if age < 30:
            return 0.1
        elif age < 50:
            return 0.2 + (age - 30) * 0.015
        else:
            return 0.5 + (age - 50) * 0.02
        
        return min(1.0, age_risk)
    
    def calculate_clinical_biomarkers_score(self, patient_data):  #biomarkers are clues in your body that tells doctors about your health
        """
        Calculating composite score from clinical biomarkers
        """
        #normalizing and combining key biomarkers
        bp_score = min(1.0, max(0.0, (patient_data['systolic_bp'] - 120) / 80))
        cholesterol_score = min(1.0, max(0.0, (patient_data['cholesterol'] - 200) / 150))
        bmi_score = min(1.0, max(0.0, (patient_data['bmi'] - 25) / 15))
        
        #weight the biomarkers
        composite_score = (bp_score * 0.4 + cholesterol_score * 0.3 + bmi_score * 0.3)
        
        return composite_score
    
    def calculate_lifestyle_factors_score(self, patient_data):
        """
        Calculate lifestyle risk score
        """
        smoking_score = patient_data['smoking'] * 0.8  # High impact
        activity_score = (5 - patient_data['physical_activity']) / 5  # Inverted
        diabetes_score = patient_data['diabetes'] * 0.9  # Very high impact
        
        # Weighted combination
        lifestyle_score = (smoking_score * 0.4 + activity_score * 0.3 + diabetes_score * 0.3)
        
        return min(1.0, lifestyle_score)
    
    def create_decision_matrix_for_patient(self, patient_data):
        """
        Create decision matrix for TODIM analysis for a single patient
        across multiple disease categories
        
        Parameters:
        - patient_data: dictionary with patient information
        
        Returns:
        - decision_matrix: numpy array (diseases x criteria)
        """
        if not self.is_initialized:
            raise ValueError("System must be initialized before creating decision matrix")
        
        #getting ML predictions for different disease risks (simulated)
        ml_prediction = self.ml_system.predict_individual_risk(patient_data)
        ml_probs = ml_prediction['average_probabilities']
        
        #getting fuzzy logic assessment
        fuzzy_score, _ = self.fuzzy_system.evaluate_patient_risk(
            patient_data['age'], 
            patient_data['bmi'], 
            patient_data['systolic_bp']
        )
        
        #calculating other factors
        age_factor = self.calculate_age_factor(patient_data['age'])
        clinical_score = self.calculate_clinical_biomarkers_score(patient_data)
        lifestyle_score = self.calculate_lifestyle_factors_score(patient_data)
        family_history_score = patient_data['family_history']
        trust_score = self.calculate_trust_factor(patient_data)
        
        #creating decision matrix (diseases x criteria)
        decision_matrix = np.zeros((len(self.disease_categories), len(self.criteria_names)))
        
        for i, disease in enumerate(self.disease_categories):
            # ML Probability (use High Risk probability as proxy)
            decision_matrix[i, 0] = ml_probs.get('High Risk', 0.5)
            
            # Fuzzy Risk Score
            decision_matrix[i, 1] = fuzzy_score
            
            # Age Factor
            decision_matrix[i, 2] = age_factor
            
            # Clinical Biomarkers
            decision_matrix[i, 3] = clinical_score
            
            # Lifestyle Factors
            decision_matrix[i, 4] = lifestyle_score
            
            # Family History
            decision_matrix[i, 5] = family_history_score
            
            # Trust Factor (same for all diseases for this patient)
            decision_matrix[i, 6] = trust_score
            
            # Add disease-specific adjustments
            if disease == 'Cardiovascular Disease':
                decision_matrix[i, 0] *= 1.2  # Higher weight on ML prediction
                decision_matrix[i, 3] *= 1.3  # Higher weight on clinical markers
            elif disease == 'Type 2 Diabetes':
                decision_matrix[i, 4] *= 1.4  # Higher weight on lifestyle
                if patient_data['diabetes']:
                    decision_matrix[i, 0] = 0.9  # Very high if already diabetic
            elif disease == 'Hypertension':
                decision_matrix[i, 1] *= 1.3  # Higher weight on fuzzy (includes BP)
                decision_matrix[i, 3] *= 1.2  # Higher weight on clinical
            elif disease == 'Obesity-related Disorders':
                if patient_data['bmi'] > 30:
                    decision_matrix[i, 0] *= 1.5  # Much higher if obese
            elif disease == 'Metabolic Syndrome':
                decision_matrix[i, 3] *= 1.2  # Composite of multiple biomarkers
                decision_matrix[i, 4] *= 1.2  # Lifestyle very important
        
        # Normalize to [0, 1] range
        decision_matrix = np.clip(decision_matrix, 0, 1)
        
        return decision_matrix
    
    def prioritize_disease_risks(self, patient_data):
        """
        Main method to prioritize disease risks for a patient
        
        Parameters:
        - patient_data: dictionary with patient information
        
        Returns:
        - prioritization_results: comprehensive analysis results
        """
        if not self.is_initialized:
            raise ValueError("System must be initialized before prioritization")
        
        print(f"Analyzing patient risk profile...")
        
        # Step 1: creating decision matrix
        decision_matrix = self.create_decision_matrix_for_patient(patient_data)
        
        # Step 2: applying TODIM ranking
        todim_results = self.todim.rank_alternatives(
            decision_matrix, 
            self.criteria_weights, 
            self.disease_categories
        )
        
        # Step 3: getting individual system assessments
        ml_prediction = self.ml_system.predict_individual_risk(patient_data)
        fuzzy_score, fuzzy_details = self.fuzzy_system.evaluate_patient_risk(
            patient_data['age'], 
            patient_data['bmi'], 
            patient_data['systolic_bp']
        )
        
        # Step 4: calculating additional metrics
        trust_score = self.calculate_trust_factor(patient_data)
        
        # Step 5: compiling comprehensive results
        prioritization_results = {
            'patient_profile': patient_data,
            'disease_prioritization': todim_results['rankings'],
            'todim_scores': todim_results['scores'],
            'decision_matrix': decision_matrix,
            'criteria_weights': self.criteria_weights,
            'ml_assessment': ml_prediction,
            'fuzzy_assessment': {
                'risk_score': fuzzy_score,
                'details': fuzzy_details
            },
            'trust_score': trust_score,
            'recommendations': self.generate_recommendations(todim_results, patient_data)
        }
        
        return prioritization_results
    
    def generate_recommendations(self, todim_results, patient_data):
        """
        Generate personalized recommendations based on risk analysis
        """
        recommendations = []
        
        #getting top 3 risk diseases
        top_risks = todim_results['rankings'][:3]
        
        for disease, score in top_risks:
            if score > 0.7:  # High risk
                recommendations.append({
                    'disease': disease,
                    'priority': 'HIGH',
                    'action': f'Immediate consultation with specialist for {disease.lower()}',
                    'timeline': 'Within 1-2 weeks'
                })
            elif score > 0.4:  # Medium risk
                recommendations.append({
                    'disease': disease,
                    'priority': 'MEDIUM',
                    'action': f'Regular monitoring and lifestyle modifications for {disease.lower()}',
                    'timeline': 'Within 1-3 months'
                })
            else:  # Lower risk
                recommendations.append({
                    'disease': disease,
                    'priority': 'LOW',
                    'action': f'Preventive measures and annual checkup for {disease.lower()}',
                    'timeline': 'Annual review'
                })
        
        #adding general lifestyle recommendations
        if patient_data['smoking']:
            recommendations.append({
                'disease': 'General Health',
                'priority': 'HIGH',
                'action': 'Smoking cessation program',
                'timeline': 'Immediate'
            })
        
        if patient_data['physical_activity'] < 2:
            recommendations.append({
                'disease': 'General Health',
                'priority': 'MEDIUM',
                'action': 'Increase physical activity to at least 150 minutes/week',
                'timeline': 'Within 2-4 weeks'
            })
        
        return recommendations
