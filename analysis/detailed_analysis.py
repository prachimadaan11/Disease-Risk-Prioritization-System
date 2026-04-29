# ============================================================
# Detailed System Analysis
# Part of: Disease Risk Prioritization System
# ============================================================
# Let's investigate the TODIM ranking issue and create detailed analysis

import pandas as pd
from src.integrated_system import IntegratedDiseaseRiskSystem
def detailed_system_analysis():
    """
    Perform detailed analysis of the system components and fix any issues
    """
    print("DETAILED SYSTEM ANALYSIS & DEBUGGING")
    print("=" * 60)
    
    # Re-analyze the low-risk patient to understand the issue
    patient_low_risk = {
        'age': 30,
        'bmi': 22.0,
        'systolic_bp': 115,
        'cholesterol': 180,
        'smoking': 0,
        'family_history': 0,
        'physical_activity': 4.0,
        'diabetes': 0
    }
    
    print("\nAnalyzing Decision Matrix Construction...")
    decision_matrix = integrated_system.create_decision_matrix_for_patient(patient_low_risk)
    
    print(f"\nDecision Matrix Shape: {decision_matrix.shape}")
    print(f"Diseases: {integrated_system.disease_categories}")
    print(f"Criteria: {integrated_system.criteria_names}")
    
    # Display the decision matrix
    print(f"\nDecision Matrix for Low-Risk Patient:")
    print("Disease" + " " * 25 + "| " + " | ".join([f"{c[:8]:>8}" for c in integrated_system.criteria_names]))
    print("-" * 108)
    
    for i, disease in enumerate(integrated_system.disease_categories):
        disease_display = disease[:30].ljust(30)
        values = " | ".join([f"{decision_matrix[i, j]:8.3f}" for j in range(len(integrated_system.criteria_names))])
        print(f"{disease_display}| {values}")
    
    # Show individual component scores
    print(f"\nIndividual Component Analysis:")
    
    # ML Assessment
    ml_pred = integrated_system.ml_system.predict_individual_risk(patient_low_risk)
    print(f"ML Prediction: {ml_pred['ensemble_prediction']}")
    print(f"ML Probabilities: {ml_pred['average_probabilities']}")
    
    # Fuzzy Assessment
    fuzzy_score, fuzzy_details = integrated_system.fuzzy_system.evaluate_patient_risk(
        patient_low_risk['age'], patient_low_risk['bmi'], patient_low_risk['systolic_bp']
    )
    print(f"\nFuzzy Logic Score: {fuzzy_score:.3f}")
    print(f"Age memberships: {fuzzy_details['age_memberships']}")
    print(f"BMI memberships: {fuzzy_details['bmi_memberships']}")
    print(f"BP memberships: {fuzzy_details['bp_memberships']}")
    
    # Other factors
    age_factor = integrated_system.calculate_age_factor(patient_low_risk['age'])
    clinical_score = integrated_system.calculate_clinical_biomarkers_score(patient_low_risk)
    lifestyle_score = integrated_system.calculate_lifestyle_factors_score(patient_low_risk)
    trust_score = integrated_system.calculate_trust_factor(patient_low_risk)
    
    print(f"\nAge factor: {age_factor:.3f}")
    print(f"Clinical biomarkers score: {clinical_score:.3f}")
    print(f"Lifestyle factors score: {lifestyle_score:.3f}")
    print(f"Family history: {patient_low_risk['family_history']}")
    print(f"Trust score: {trust_score:.3f}")

# Run detailed analysis
detailed_system_analysis()

# Let's also create a corrected version that handles the TODIM ranking better
def create_comparative_analysis():
    """
    Create comparative analysis across different patient profiles
    """
    print(f"\n\nCOMPARATIVE ANALYSIS ACROSS PATIENT PROFILES")
    print("=" * 60)
    
    # Define patient profiles more clearly
    patients_profiles = {
        'High Risk': {
            'age': 65, 'bmi': 32.5, 'systolic_bp': 160, 'cholesterol': 280,
            'smoking': 1, 'family_history': 1, 'physical_activity': 1.0, 'diabetes': 1
        },
        'Medium Risk': {
            'age': 45, 'bmi': 27.0, 'systolic_bp': 135, 'cholesterol': 220,
            'smoking': 0, 'family_history': 1, 'physical_activity': 2.5, 'diabetes': 0
        },
        'Low Risk': {
            'age': 30, 'bmi': 22.0, 'systolic_bp': 115, 'cholesterol': 180,
            'smoking': 0, 'family_history': 0, 'physical_activity': 4.0, 'diabetes': 0
        }
    }
    
    # Create comparison table
    comparison_data = []
    
    for patient_type, patient_data in patients_profiles.items():
        # Get ML prediction
        ml_pred = integrated_system.ml_system.predict_individual_risk(patient_data)
        
        # Get fuzzy assessment
        fuzzy_score, _ = integrated_system.fuzzy_system.evaluate_patient_risk(
            patient_data['age'], patient_data['bmi'], patient_data['systolic_bp']
        )
        
        # Calculate other scores
        age_factor = integrated_system.calculate_age_factor(patient_data['age'])
        clinical_score = integrated_system.calculate_clinical_biomarkers_score(patient_data)
        lifestyle_score = integrated_system.calculate_lifestyle_factors_score(patient_data)
        
        comparison_data.append({
            'Patient Type': patient_type,
            'Age': patient_data['age'],
            'BMI': patient_data['bmi'],
            'BP': patient_data['systolic_bp'],
            'ML_High_Risk_Prob': ml_pred['average_probabilities']['High Risk'],
            'Fuzzy_Score': fuzzy_score,
            'Age_Factor': age_factor,
            'Clinical_Score': clinical_score,
            'Lifestyle_Score': lifestyle_score,
            'Overall_Risk_Estimate': (
                ml_pred['average_probabilities']['High Risk'] * 0.3 +
                fuzzy_score * 0.3 +
                age_factor * 0.2 +
                clinical_score * 0.1 +
                lifestyle_score * 0.1
            )
        })
    
    # Display comparison
    comparison_df = pd.DataFrame(comparison_data)
    print("\nPatient Profile Comparison:")
    print(comparison_df.round(3).to_string(index=False))
    
    return comparison_df
