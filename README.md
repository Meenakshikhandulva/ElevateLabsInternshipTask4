Steps Performed
Data Loading & Preprocessing

Removed unnecessary columns (id, Unnamed: 32)

Converted diagnosis to numeric:

M → 1 (Malignant)

B → 0 (Benign)

Train/Test Split & Standardization

Split data into 80% train and 20% test sets

Standardized features using StandardScaler

Model Training

Used LogisticRegression from scikit-learn

Trained the model on standardized features

Evaluation Metrics

Confusion Matrix

Precision

Recall

ROC-AUC Score

ROC Curve Plot

Sigmoid Function Visualization

Plotted the sigmoid function to show probability mapping from (-∞, ∞) to (0, 1)
