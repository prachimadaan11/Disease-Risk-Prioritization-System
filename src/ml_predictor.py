# ============================================================
# Machine Learning Pipeline for Disease Risk Prediction
# Part of: Disease Risk Prioritization System
# ============================================================

# STEP 3: THE BIG THING FOR ME.. implementing Machine Learning Pipeline for Disease Risk Prediction
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
class MLDiseaseRiskPredictor:
    """
    Machine Learning Pipeline for Disease Risk Prediction
    
    This class is integrating multiple ML algorithms to predict disease risks
    and provides probability estimates that can be used with TODIM
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.is_trained = False
        
        # Initialize multiple ML models for ensemble prediction
        self.initialize_models()
    
    def initialize_models(self):
        """
        initializing different ML models for ensemble learning
        """
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                max_depth=10,
                min_samples_split=5
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, 
                random_state=42,
                learning_rate=0.1,
                max_depth=6
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                solver='liblinear'
            ),
            'svm': SVC(     #support vector machine(the algo concept) and svc is the class to train an svm classifier
                kernel='rbf', #kernels allow svm to handle non linear data. sometimes data cannot be separated by a straight line.. so a kernel is used to transform the data into higher dimensions where it can be separated linearly
                probability=True, #rbf is a common svm kernel that is flexible and works well with non linear data and can handle complex boundaries
                random_state=42,
                C=1.0,
                gamma='scale'
            ),
            'naive_bayes': GaussianNB(),  #gaussian naive bayes assumes that the features follow a normal distribution
            'decision_tree': DecisionTreeClassifier(
                random_state=42,
                max_depth=8,
                min_samples_split=10
            )
        }
    
    def create_synthetic_dataset(self, n_samples=1000):
        """
        Create a synthetic medical dataset for demonstration
        
        This simulates real medical data with various risk factors
        """
        np.random.seed(42)
        
        # Generate patient demographics
        age = np.random.normal(50, 15, n_samples)
        age = np.clip(age, 18, 90)
        
        # BMI with some correlation to age
        bmi_base = np.random.normal(25, 4, n_samples)
        age_effect = (age - 40) * 0.05  # Older people tend to have higher BMI
        bmi = bmi_base + age_effect + np.random.normal(0, 1, n_samples)
        bmi = np.clip(bmi, 15, 45)
        
        # Blood pressure (systolic) correlated with age and BMI
        bp_base = 110 + (age - 30) * 0.8 + (bmi - 25) * 1.2
        systolic_bp = bp_base + np.random.normal(0, 15, n_samples)
        systolic_bp = np.clip(systolic_bp, 90, 200)
        
        # Cholesterol levels
        cholesterol = 180 + (age - 30) * 0.5 + (bmi - 25) * 2 + np.random.normal(0, 30, n_samples)
        cholesterol = np.clip(cholesterol, 120, 350)
        
        # Smoking status (binary)
        smoking_prob = 0.15 + (age < 60) * 0.1  # Younger people more likely to smoke
        smoking = np.random.binomial(1, smoking_prob, n_samples)
        
        # Family history (binary)
        family_history = np.random.binomial(1, 0.3, n_samples)
        
        # Physical activity (0-5 scale)
        activity_base = 3 - (age - 40) * 0.02  # Older people less active
        physical_activity = activity_base + np.random.normal(0, 1, n_samples)
        physical_activity = np.clip(physical_activity, 0, 5)
        
        # Diabetes status (binary)
        diabetes_prob = 0.08 + (bmi > 30) * 0.15 + (age > 50) * 0.1
        diabetes = np.random.binomial(1, diabetes_prob, n_samples)
        
        # Create target variable: Disease Risk Level
        # This is a complex function of all risk factors
        risk_score = (
            (age - 30) * 0.02 +  # Age effect
            (bmi - 25) * 0.05 +  # BMI effect
            (systolic_bp - 120) * 0.01 +  # BP effect
            (cholesterol - 200) * 0.002 +  # Cholesterol effect
            smoking * 0.3 +  # Smoking effect
            family_history * 0.2 +  # Family history effect
            diabetes * 0.4 -  # Diabetes effect
            physical_activity * 0.1 +  # Physical activity effect (negative)
            np.random.normal(0, 0.1, n_samples)  # Random noise
        )
        
        #converting to categorical risk levels
        risk_categories = np.where(risk_score < 0.3, 'Low Risk',
                          np.where(risk_score < 0.7, 'Medium Risk', 'High Risk'))
        
        #creating the DataFrame
        dataset = pd.DataFrame({
            'age': age,
            'bmi': bmi,
            'systolic_bp': systolic_bp,
            'cholesterol': cholesterol,
            'smoking': smoking,
            'family_history': family_history,
            'physical_activity': physical_activity,
            'diabetes': diabetes,
            'risk_level': risk_categories,
            'risk_score': risk_score
        })
        
        return dataset
    
    def preprocess_data(self, data, fit_transform=True):
        """
        Preprocessing the data for ML models
        
        Parameters:
        - data: pandas DataFrame with features
        - fit_transform: whether to fit the scaler (True for training, False for prediction)
        
        Returns:
        - processed_features: scaled feature matrix
        """
        #selecting feature columns (excluding the target variables)
        feature_columns = [col for col in data.columns 
                          if col not in ['risk_level', 'risk_score']]
        
        features = data[feature_columns].copy()
        
        if fit_transform:
            self.feature_names = feature_columns
            processed_features = self.scaler.fit_transform(features)
        else:
            processed_features = self.scaler.transform(features)
        
        return processed_features
    
    def train_models(self, training_data):
        """
        now training all ML models on the provided dataset
        
        Parameters:
        - training_data: pandas DataFrame with features and target
        """
        print("Training Machine Learning Models...")
        
        #preprocessing the features
        X = self.preprocess_data(training_data, fit_transform=True)
        
        #encoding target variable
        y = self.label_encoder.fit_transform(training_data['risk_level'])
        
        #splitting data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        #training each model and evaluating
        model_scores = {}
        
        for model_name, model in self.models.items():
            print(f"  Training {model_name}...")
            
            #for training the model
            model.fit(X_train, y_train)
            
            #now evaluating on validation set
            val_score = model.score(X_val, y_val)
            model_scores[model_name] = val_score
            
            print(f"    Validation Accuracy: {val_score:.4f}")
        
        self.is_trained = True
        
        print(f"\nAll models trained successfully!")
        print(f"Best performing model: {max(model_scores, key=model_scores.get)} "
              f"(Accuracy: {max(model_scores.values()):.4f})")
        
        return model_scores
    
    def predict_individual_risk(self, patient_data):
        """
        Predicting disease risk for an individual patient using ensemble of models
        
        Parameters:
        - patient_data: dictionary or pandas Series with patient features
        
        Returns:
        - ensemble_prediction: dictionary with predictions and probabilities
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        #converting to DataFrame if needed
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = pd.DataFrame([patient_data])
        
        #preprocessing
        X_patient = self.preprocess_data(patient_df, fit_transform=False)
        
        #getting predictions from all models
        predictions = {}
        probabilities = {}
        
        for model_name, model in self.models.items():
            #getting class prediction
            pred = model.predict(X_patient)[0]
            predictions[model_name] = self.label_encoder.inverse_transform([pred])[0]
            
            #getting probability predictions
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_patient)[0]
                probabilities[model_name] = {
                    class_label: prob for class_label, prob in 
                    zip(self.label_encoder.classes_, proba)
                }
        
        # Ensemble prediction (majority vote)
        pred_counts = {}
        for pred in predictions.values():
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
        
        ensemble_prediction = max(pred_counts, key=pred_counts.get)
        
        # Average probabilities across models
        avg_probabilities = {}
        for class_label in self.label_encoder.classes_:
            class_probs = [probabilities[model][class_label] 
                          for model in probabilities.keys() 
                          if class_label in probabilities[model]]
            avg_probabilities[class_label] = np.mean(class_probs) if class_probs else 0
        
        return {
            'ensemble_prediction': ensemble_prediction,
            'individual_predictions': predictions,
            'average_probabilities': avg_probabilities,
            'individual_probabilities': probabilities
        }
    
    def batch_predict_risks(self, patients_data):
        """
        Predicting risks for multiple patients
        
        Parameters:
        - patients_data: pandas DataFrame with patient features
        
        Returns:
        - batch_predictions: list of prediction dictionaries
        """
        batch_predictions = []
        
        for idx, patient in patients_data.iterrows():
            prediction = self.predict_individual_risk(patient)
            prediction['patient_id'] = idx
            batch_predictions.append(prediction)
        
        return batch_predictions
