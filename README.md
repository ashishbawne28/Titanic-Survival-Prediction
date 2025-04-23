# Titanic Survival Prediction Model

## Project Overview
This project develops a machine learning model to predict passenger survival on the Titanic. Using passenger data (age, gender, ticket class, etc.), we implement and compare multiple classification models to predict survival outcomes.

## Dataset Information
- **Source**: Titanic Dataset
- **Features**: Age, Sex, Pclass, SibSp, Parch, Fare, Embarked
- **Target Variable**: Survived (0 = No, 1 = Yes)
- **Dataset Size**: 418 records

## Data Preprocessing Steps
1. **Handling Missing Values**
   - Age: Imputed with median value
   - Fare: Imputed with median value
   - Embarked: Filled with most frequent value ('S')

2. **Feature Engineering**
   - Dropped low-importance columns: 'PassengerId', 'Name', 'Ticket', 'Cabin'
   - Encoded categorical variables:
     * Sex: Label encoded (male/female → 0/1)
     * Embarked: Label encoded (S/C/Q → 0/1/2)

3. **Feature Scaling**
   - StandardScaler applied to: Age, Fare, SibSp, Parch

## Model Implementation

### Models Tested
1. Logistic Regression
2. Random Forest Classifier

### Model Performance
#### Logistic Regression
- Accuracy: 100%
- Precision: 100%
- Recall: 100%
- F1 Score: 100%
- Cross-validation Score: 100%

#### Random Forest
- Accuracy: 100%
- Precision: 100%
- Recall: 100%
- F1 Score: 100%
- Cross-validation Score: 100%

### Feature Importance (Random Forest)
1. Sex (84.13%)
2. Fare (5.52%)
3. Age (3.90%)
4. Parch (2.29%)
5. SibSp (1.50%)
6. Embarked (1.46%)
7. Pclass (1.21%)

## Project Structure
```
titanic_survival_prediction/
│
├── data/
│   ├── tested.csv
│   └── processed/
│
├── notebooks/
│   └── model_development.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── utils.py
│
├── requirements.txt
└── README.md
```

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from src.preprocessing import preprocess_data
from src.model import train_model

# Load and preprocess data
X_processed, y = preprocess_data('data/tested.csv')

# Train model
model = train_model(X_processed, y)

# Make predictions
predictions = model.predict(X_test)
```

## Model Training Process
1. Data splitting: 80% training, 20% testing
2. Model initialization with default parameters
3. Training using preprocessed features
4. Performance evaluation using multiple metrics
5. Cross-validation for robust performance estimation

## Performance Analysis
- Both models achieved perfect scores on the test set
- Random Forest showed more robust feature importance analysis
- Sex is the most crucial feature for prediction
- The high scores suggest potential overfitting or data leakage
- Real-world performance might vary

## Future Improvements
1. Feature engineering:
   - Create new features from existing ones
   - Better handling of missing values
2. Model optimization:
   - Hyperparameter tuning
   - Ensemble methods
3. Cross-validation strategies:
   - K-fold cross-validation
   - Stratified sampling

## Contributing
Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License
MIT License

