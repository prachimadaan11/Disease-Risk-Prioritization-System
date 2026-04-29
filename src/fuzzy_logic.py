# ============================================================
# Fuzzy Logic System for Uncertainty Handling
# Part of: Disease Risk Prioritization System
# ============================================================

# #STEP 2: now here we are implementing Fuzzy Logic System for handling uncertainty in medical diagnosis

class FuzzyLogicSystem:
    """
    Fuzzy Logic System for handling uncertainty in medical diagnosis and risk assessment
    
    This system uses fuzzy sets to model linguistic variables like:
    - "Low Risk", "Medium Risk", "High Risk"
    - "Young", "Middle-aged", "Elderly"
    - "Normal", "Elevated", "High" (for biomarkers)
    """
    
    def __init__(self):
        self.fuzzy_sets = {}
        self.rules = []
        
    def triangular_membership(self, x, a, b, c):
        """
        Calculate triangular membership function
        
        Parameters:
        - x: input value
        - a, b, c: triangle parameters (left, peak, right)
        
        Returns:
        - membership degree (0 to 1)
        """
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x < c:
            return (c - x) / (c - b)
        else:
            return 0.0
    
    def trapezoidal_membership(self, x, a, b, c, d):
        """
        Calculate trapezoidal membership function
        
        Parameters:
        - x: input value
        - a, b, c, d: trapezoid parameters
        
        Returns:
        - membership degree (0 to 1)
        """
        if x <= a or x >= d:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x <= c:
            return 1.0
        elif c < x < d:
            return (d - x) / (d - a)
        else:
            return 0.0
    
    def define_age_fuzzy_sets(self):
        """
        Define fuzzy sets for age categories
        """
        def young_membership(age):
            return self.trapezoidal_membership(age, 0, 0, 25, 45)
        
        def middle_aged_membership(age):
            return self.triangular_membership(age, 30, 50, 70)
        
        def elderly_membership(age):
            return self.trapezoidal_membership(age, 55, 75, 100, 100)
        
        self.fuzzy_sets['age'] = {
            'young': young_membership,
            'middle_aged': middle_aged_membership,
            'elderly': elderly_membership
        }
    
    def define_bmi_fuzzy_sets(self):
        """
        Define fuzzy sets for BMI categories
        """
        def underweight_membership(bmi):
            return self.trapezoidal_membership(bmi, 0, 0, 16, 18.5)
        
        def normal_membership(bmi):
            return self.triangular_membership(bmi, 17, 21.5, 25)
        
        def overweight_membership(bmi):
            return self.triangular_membership(bmi, 23, 27.5, 32)
        
        def obese_membership(bmi):
            return self.trapezoidal_membership(bmi, 30, 35, 50, 50)
        
        self.fuzzy_sets['bmi'] = {
            'underweight': underweight_membership,
            'normal': normal_membership,
            'overweight': overweight_membership,
            'obese': obese_membership
        }
    
    def define_blood_pressure_fuzzy_sets(self):
        """
        Define fuzzy sets for blood pressure categories (systolic)
        """
        def normal_bp_membership(bp):
            return self.trapezoidal_membership(bp, 80, 80, 120, 130)
        
        def elevated_bp_membership(bp):
            return self.triangular_membership(bp, 120, 135, 150)
        
        def high_bp_membership(bp):
            return self.trapezoidal_membership(bp, 140, 160, 200, 200)
        
        self.fuzzy_sets['blood_pressure'] = {
            'normal': normal_bp_membership,
            'elevated': elevated_bp_membership,
            'high': high_bp_membership
        }
    
    def define_risk_output_fuzzy_sets(self):
        """
        Define output fuzzy sets for disease risk levels
        """
        def low_risk_membership(risk):
            return self.triangular_membership(risk, 0, 0, 0.4)
        
        def medium_risk_membership(risk):
            return self.triangular_membership(risk, 0.2, 0.5, 0.8)
        
        def high_risk_membership(risk):
            return self.triangular_membership(risk, 0.6, 1.0, 1.0)
        
        self.fuzzy_sets['risk_level'] = {
            'low': low_risk_membership,
            'medium': medium_risk_membership,
            'high': high_risk_membership
        }
    
    def initialize_all_fuzzy_sets(self):
        """
        Initialize all fuzzy sets for the medical diagnosis system
        """
        self.define_age_fuzzy_sets()
        self.define_bmi_fuzzy_sets()
        self.define_blood_pressure_fuzzy_sets()
        self.define_risk_output_fuzzy_sets()
    
    def fuzzify_input(self, variable_name, value):
        """
        Fuzzify an input value for a given variable
        
        Parameters:
        - variable_name: name of the fuzzy variable
        - value: crisp input value
        
        Returns:
        - dictionary of membership degrees for each fuzzy set
        """
        if variable_name not in self.fuzzy_sets:
            raise ValueError(f"Fuzzy variable '{variable_name}' not defined")
        
        memberships = {}
        for set_name, membership_func in self.fuzzy_sets[variable_name].items():
            memberships[set_name] = membership_func(value)
        
        return memberships
    
    def apply_fuzzy_rules(self, age_memberships, bmi_memberships, bp_memberships):
        """
        Apply fuzzy rules to determine disease risk
        
        Example rules:
        - IF age is elderly AND bmi is obese THEN risk is high
        - IF age is young AND bmi is normal AND bp is normal THEN risk is low
        """
        rules_output = []
        
        # Rule 1: High risk rules
        rule1_strength = min(age_memberships['elderly'], bmi_memberships['obese'])
        if rule1_strength > 0:
            rules_output.append(('high', rule1_strength))
        
        rule2_strength = min(bp_memberships['high'], 
                           max(bmi_memberships['overweight'], bmi_memberships['obese']))
        if rule2_strength > 0:
            rules_output.append(('high', rule2_strength))
        
        # Rule 2: Medium risk rules
        rule3_strength = min(age_memberships['middle_aged'], 
                           max(bmi_memberships['overweight'], bp_memberships['elevated']))
        if rule3_strength > 0:
            rules_output.append(('medium', rule3_strength))
        
        rule4_strength = min(age_memberships['young'], bp_memberships['high'])
        if rule4_strength > 0:
            rules_output.append(('medium', rule4_strength))
        
        # Rule 3: Low risk rules
        rule5_strength = min(min(age_memberships['young'], bmi_memberships['normal']),
                           bp_memberships['normal'])
        if rule5_strength > 0:
            rules_output.append(('low', rule5_strength))
        
        rule6_strength = min(age_memberships['middle_aged'], 
                           min(bmi_memberships['normal'], bp_memberships['normal']))
        if rule6_strength > 0:
            rules_output.append(('low', rule6_strength))
        
        return rules_output
    
    def defuzzify_output(self, rules_output):
        """
        Defuzzify the output using centroid method
        
        Parameters:
        - rules_output: list of (risk_level, strength) tuples
        
        Returns:
        - crisp risk score (0 to 1)
        """
        if not rules_output:
            return 0.5  # Default medium risk if no rules fire
        
        # Aggregate rules for each output class
        risk_aggregation = {'low': 0, 'medium': 0, 'high': 0}
        
        for risk_level, strength in rules_output:
            risk_aggregation[risk_level] = max(risk_aggregation[risk_level], strength)
        
        # Calculate weighted average (centroid)
        numerator = (risk_aggregation['low'] * 0.2 + 
                    risk_aggregation['medium'] * 0.5 + 
                    risk_aggregation['high'] * 0.8)
        
        denominator = sum(risk_aggregation.values())
        
        if denominator == 0:
            return 0.5
        
        return numerator / denominator
    
    def evaluate_patient_risk(self, age, bmi, systolic_bp):
        """
        Complete fuzzy evaluation for a patient
        
        Parameters:
        - age: patient age
        - bmi: patient BMI
        - systolic_bp: patient systolic blood pressure
        
        Returns:
        - fuzzy_risk_score: crisp risk score (0-1)
        - detailed_analysis: detailed breakdown of fuzzy analysis
        """
        #initializing fuzzy sets if not already done
        if not self.fuzzy_sets:
            self.initialize_all_fuzzy_sets()
        
        #fuzzifying inputs
        age_memberships = self.fuzzify_input('age', age)
        bmi_memberships = self.fuzzify_input('bmi', bmi)
        bp_memberships = self.fuzzify_input('blood_pressure', systolic_bp)
        
        #applying fuzzy rules here
        rules_output = self.apply_fuzzy_rules(age_memberships, bmi_memberships, bp_memberships)
        
        #defuzzifying output here
        fuzzy_risk_score = self.defuzzify_output(rules_output)
        
        #preparing detailed analysis
        detailed_analysis = {
            'age_memberships': age_memberships,
            'bmi_memberships': bmi_memberships,
            'bp_memberships': bp_memberships,
            'fired_rules': rules_output,
            'fuzzy_risk_score': fuzzy_risk_score
        }
        
        return fuzzy_risk_score, detailed_analysis

