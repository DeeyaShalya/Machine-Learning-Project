#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# In[2]:


# Step 2: Load Data
data = pd.read_csv('student_spending (1).csv')


# In[3]:


# Step 3: Data Preprocessing
# Encode categorical variables
X = pd.get_dummies(data.drop(columns=['preferred_payment_method']), drop_first=True)
y = data['preferred_payment_method']

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Training
model = GaussianNB()
model.fit(X_train, y_train)


# In[4]:


# Step 6: Model Evaluation
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[5]:


# Step 6: Model Evaluation
# Predict probabilities
probabilities = model.predict_proba(X_test)

# Step 6: Model Evaluation
# Predict probabilities
probabilities = model.predict_proba(X_test)

# Convert probabilities to DataFrame
prob_df = pd.DataFrame(probabilities, columns=model.classes_)

# Display probabilities
print("Probabilities for each class:")
print(prob_df)


# In[6]:


# Predict class labels
predicted_labels = model.predict(X_test)

# Convert predicted labels to DataFrame
result_df = pd.DataFrame({'Actual': y_test, 'Predicted': predicted_labels})

# Display prediction results
print("\nPredicted labels:")
print(result_df)


# In[7]:


# Display probabilities
print("Probabilities for each class:")
print(probabilities)


# In[8]:


# Predict class labels
predicted_labels = model.predict(X_test)

# Display prediction results
print("Predicted labels:")
print(predicted_labels)


# In[9]:


# Visualize Scatter Plot
# Choose features for visualization
payment_methods = data['preferred_payment_method'].unique()
colors = ['red', 'blue', 'green']  # You can define colors for each payment method if needed

plt.figure(figsize=(10, 6))
for i, method in enumerate(payment_methods):
    mask = data['preferred_payment_method'] == method
    plt.scatter(data[mask]['age'], data[mask]['monthly_income'], label=method, color=colors[i])

plt.xlabel('Age')
plt.ylabel('Monthly Income')
plt.title('Scatter Plot of Age vs Monthly Income')
plt.legend(title='Preferred Payment Method')
plt.grid(True)
plt.show()


# In[ ]:




