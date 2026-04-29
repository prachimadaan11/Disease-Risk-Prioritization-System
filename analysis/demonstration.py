# ============================================================
# Comprehensive System Demonstration
# Part of: Disease Risk Prioritization System
# ============================================================

# STEP 5: Comprehensive System Demonstration and Analysis
from src.integrated_system import IntegratedDiseaseRiskSystem
def demonstrate_complete_system():
    """
    demonstrating the complete integrated disease risk prioritization system
    """
    print("COMPREHENSIVE SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    # Initialize the integrated system
    print("\nPHASE 1: System Initialization")
    print("-" * 40)
    training_data, model_scores = integrated_system.initialize_system(n_training_samples=1500)
    
    print(f"\nTraining Dataset Statistics:")
    print(f"• Total samples: {len(training_data)}")
    print(f"• Features: {training_data.shape[1] - 2}")  # excluding the target columns
    print(f"• Risk distribution:")
    risk_dist = training_data['risk_level'].value_counts()
    for risk, count in risk_dist.items():
        print(f"  - {risk}: {count} ({count/len(training_data)*100:.1f}%)")
    
    # creating test patients with different risk profiles
    print("\nPHASE 2: Patient Risk Assessment")
    print("-" * 40)
    
    # Patient 1: High-risk profile
    patient_1 = {
        'age': 65,
        'bmi': 32.5,
        'systolic_bp': 160,
        'cholesterol': 280,
        'smoking': 1,
        'family_history': 1,
        'physical_activity': 1.0,
        'diabetes': 1
    }
    
    # Patient 2: Medium-risk profile
    patient_2 = {
        'age': 45,
        'bmi': 27.0,
        'systolic_bp': 135,
        'cholesterol': 220,
        'smoking': 0,
        'family_history': 1,
        'physical_activity': 2.5,
        'diabetes': 0
    }
    
    # Patient 3: Low-risk profile
    patient_3 = {
        'age': 30,
        'bmi': 22.0,
        'systolic_bp': 115,
        'cholesterol': 180,
        'smoking': 0,
        'family_history': 0,
        'physical_activity': 4.0,
        'diabetes': 0
    }
    
    patients = [
        ("High-Risk Patient", patient_1),
        ("Medium-Risk Patient", patient_2),
        ("Low-Risk Patient", patient_3)
    ]
    
    #analyzing each patient
    all_results = []
    
    for patient_name, patient_data in patients:
        print(f"\nAnalyzing {patient_name}")
        print("." * 50)
        
        #getting comprehensive risk analysis
        results = integrated_system.prioritize_disease_risks(patient_data)
        all_results.append((patient_name, results))
        
        #displaying key results
        print(f"\nRisk Prioritization Results:")
        for i, (disease, score) in enumerate(results['disease_prioritization'][:3], 1):
            risk_level = "HIGH" if score > 0.7 else "MEDIUM" if score > 0.4 else "LOW"
            print(f"  {i}. {disease}: {score:.3f} ({risk_level} RISK)")
        
        print(f"\nML Assessment: {results['ml_assessment']['ensemble_prediction']}")
        ml_probs = results['ml_assessment']['average_probabilities']
        for risk_cat, prob in ml_probs.items():
            print(f"  • {risk_cat}: {prob:.3f}")
        
        print(f"\nFuzzy Logic Score: {results['fuzzy_assessment']['risk_score']:.3f}")
        print(f"Trust Score: {results['trust_score']:.3f}")
        
        print(f"\nTop Recommendations:")
        for i, rec in enumerate(results['recommendations'][:3], 1):
            print(f"  {i}. [{rec['priority']}] {rec['action']} ({rec['timeline']})")
    
    return all_results, training_data