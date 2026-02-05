
#best Model: XGBoost Classifier
# expected balanced accuracy: 0.7719
# Stanescu Ovidiu Gabriel


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# load data
train = pd.read_csv('job_change_train.csv')
test = pd.read_csv('job_change_test.csv')

# Preprocess
def preprocess(df):
    df = df.copy()
    ysc_map = {'never_changed': 0, '1': 1, '2': 2, '3': 3, '4': 4, '>4': 5, 'unknown': -1}
    df['years_since_job_change'] = df['years_since_job_change'].map(ysc_map)

    def parse_exp(x):
        if x == '>20': return 21
        if x == '<1': return 0
        if x == 'unknown': return -1
        return int(x)
    df['years_of_experience'] = df['years_of_experience'].apply(parse_exp)
    return df

train = preprocess(train)
test = preprocess(test)

# features
X = train.drop(['id', 'willing_to_change_job'], axis=1)
y = (train['willing_to_change_job'] == 'Yes').astype(int)
X_test = test.drop(['id'], axis=1)

numeric_features = ['age', 'relative_wage', 'years_since_job_change',
                   'years_of_experience', 'hours_of_training', 'is_certified']
categorical_features = ['gender', 'education', 'field_of_studies',
                       'is_studying', 'county', 'size_of_company', 'type_of_company']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])

# best model with tuned hyperparameters
model = Pipeline([
    ('prep', preprocessor),
    ('clf', XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    ))
])

# Train and predict
model.fit(X, y)
predictions = model.predict(X_test)

# Save predictions
submission = pd.DataFrame({
    'id': test['id'],
    'willing_to_change_job': predictions
})
submission.to_csv('StanescuOvidiuGabriel_predictions.csv', index=False)

print(f'Predictions saved: {len(predictions)} samples')
print(f'Distribution: Yes={sum(predictions)}, No={len(predictions)-sum(predictions)}')
