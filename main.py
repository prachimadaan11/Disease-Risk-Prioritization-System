# ============================================================
# Disease Risk Prioritization System
# Main Entry Point
# 
# Methodology: TODIM + Fuzzy Logic + Machine Learning Ensemble
# Author: Prachi Madaan
# Institution: IIT Delhi (Department of Management Studies)
# ============================================================

from src.integrated_system import IntegratedDiseaseRiskSystem

def main():
    print("=" * 60)
    print("  Disease Risk Prioritization System")
    print("  TODIM + Fuzzy Logic + ML Ensemble")
    print("=" * 60)

    # Initialize the system
    system = IntegratedDiseaseRiskSystem()
    system.initialize_system(n_training_samples=1500)

    # Example patient
    patient = {
        'age': 52,
        'bmi': 28.5,
        'systolic_bp': 142,
        'cholesterol': 235,
        'smoking': 1,
        'family_history': 1,
        'physical_activity': 1.5,
        'diabetes': 0
    }

    print("\nAnalyzing patient risk profile...")
    results = system.prioritize_disease_risks(patient)

    print("\nDisease Risk Prioritization (Top 3):")
    for i, (disease, score) in enumerate(results['disease_prioritization'][:3], 1):
        risk = "HIGH" if score > 0.7 else "MEDIUM" if score > 0.4 else "LOW"
        print(f"  {i}. {disease}: {score:.3f} ({risk})")

    print("\nML Assessment:", results['ml_assessment']['ensemble_prediction'])
    print("Fuzzy Score:", round(results['fuzzy_assessment']['risk_score'], 3))
    print("Trust Score:", round(results['trust_score'], 3))

    print("\nRecommendations:")
    for i, rec in enumerate(results['recommendations'][:3], 1):
        print(f"  {i}. [{rec['priority']}] {rec['action']}")

if __name__ == "__main__":
    main()