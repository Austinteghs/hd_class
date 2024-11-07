
# Heart Disease Prediction Project

This project aims to predict whether a patient has heart disease based on various medical features using machine learning algorithms. The dataset contains information about patients' health metrics, and several classification models are evaluated to predict the presence of heart disease.

## Project Overview

- **Objective**: Predict the presence or absence of heart disease in patients based on health-related features.
- **Dataset**: The dataset includes columns like age, sex, chest pain type, blood pressure, cholesterol levels, and more.
- **Machine Learning Models Used**: 
  - Logistic Regression
  - Support Vector Classifier (SVC)
  - K-Nearest Neighbors (KNN)
  - Decision Tree Classifier
  - Random Forest Classifier
  - Naive Bayes
  
## Dataset Features

The dataset includes the following columns:

- **age**: Age in years
- **sex**: Gender (1 = male, 0 = female)
- **cp**: Chest pain type (0 to 3)
- **trestbps**: Resting blood pressure (in mm Hg)
- **chol**: Serum cholesterol (in mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **restecg**: Resting electrocardiographic results
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise-induced angina (1 = yes, 0 = no)
- **oldpeak**: ST depression induced by exercise
- **slope**: Slope of the peak exercise ST segment
- **ca**: Number of major vessels colored by fluoroscopy
- **thal**: Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)
- **target**: Heart disease status (1 = disease present, 0 = no disease)

## Key Steps in the Project

### 1. **Data Exploration and Preprocessing**
   - Data is imported and cleaned.
   - Missing values (if any) are handled.
   - Categorical variables are converted into dummy variables.
   - Data is split into training and test sets.

### 2. **Model Training**
   Various machine learning algorithms are trained on the dataset, including:
   - **Logistic Regression**
   - **Support Vector Classifier (SVC)**
   - **K-Nearest Neighbors (KNN)**
   - **Decision Tree Classifier**
   - **Random Forest Classifier**
   - **Naive Bayes**

### 3. **Model Evaluation**
   - The models are evaluated based on their accuracy using both training and testing datasets.
   - The performance of the models is compared to select the best-performing algorithm.

### 4. **Confusion Matrix and Classification Report**
   A confusion matrix is used to evaluate the performance of the model, and a classification report provides precision, recall, and F1-scores.

### 5. **Predictive Model**
   After evaluating the models, a predictive model is built using the Random Forest classifier to predict heart disease status for new input data.

## How to Use

1. Clone the repository or download the code.
2. Install the required libraries:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Ensure that the dataset (`heart.csv`) is in the working directory.
4. Run the code to see model training and prediction in action.

## Output

- The code prints the accuracy of each model trained.
- A confusion matrix heatmap is displayed.
- The model makes a prediction for a given set of input features (a patient's medical details).

## Conclusion

- **Best Model**: The **Random Forest** classifier provides the best performance, with high accuracy on both the training and testing datasets.
- **Predictions**: The trained model can predict whether a new patient has heart disease based on their medical details.
