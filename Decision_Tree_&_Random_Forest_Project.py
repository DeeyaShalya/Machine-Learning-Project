#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# In[30]:


# Step 1: Load the dataset
df = pd.read_csv('student_spending (1).csv')


# In[31]:


df.head()


# In[32]:


# Step 2: Preprocess the data
final_data = pd.get_dummies(df, columns=["gender", "year_in_school", "major", "preferred_payment_method"], drop_first=True)


# In[33]:


# Step 3: Split the data into features and target variable
X = final_data.drop('financial_aid', axis=1)
y = final_data['financial_aid']


# In[34]:


# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[35]:


# Step 5: Train Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)


# In[36]:


# Step 6: Train Random Forest Classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)


# In[37]:


# Step 7: Make predictions on new training data
dt_pred = dt_classifier.predict(X_test)
rf_pred = rf_classifier.predict(X_test)


# In[38]:


# Step 8: Evaluate Decision Tree Classifier
print("Decision Tree Classifier:")
print(classification_report(y_test, dt_pred))  # Print classification report including precision, recall, F1-score, and support
print("Confusion Matrix:")
print(confusion_matrix(y_test, dt_pred))  # Print confusion matrix


# In[39]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Assuming rf_pred and y_test are your predicted and true labels, respectively
cm = confusion_matrix(y_test, rf_pred)

# Plotting the confusion matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[40]:


import matplotlib.pyplot as plt

# Assuming dt_pred and rf_pred are your predicted labels for Decision Tree and Random Forest models, respectively
# Plotting scatter plot for Decision Tree
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, dt_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Decision Tree: Actual vs Predicted')

# Plotting scatter plot for Random Forest
plt.subplot(1, 2, 2)
plt.scatter(y_test, rf_pred, color='red')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Random Forest: Actual vs Predicted')

plt.tight_layout()
plt.show()


# In[ ]:




