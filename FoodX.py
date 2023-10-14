#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis

# In[22]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[9]:


# Read a CSV file into a DataFrame
df = pd.read_csv('XTern 2024 Artificial Intelegence Data Set - Xtern_TrainData.csv')
df.head()


# In[30]:


# Check for missing values
print(df.isnull().sum())


# In[3]:


df.info()


# In[4]:


df.describe()


# In[26]:


sns.histplot(df['University'])
plt.xticks(rotation=90) 
plt.show()


# In[27]:


sns.histplot(df['Order'])
plt.xticks(rotation=90) 
plt.show()


# In[28]:


crosstab = pd.crosstab(df['University'], df['Order'])

# Display the crosstab
print(crosstab)


# In[43]:


# Create a heatmap of the crosstab
plt.figure(figsize=(10, 6))  
sns.heatmap(crosstab, annot=True, fmt="d", cmap="YlGnBu") 

plt.title('Crosstab: University vs. Order')
plt.xlabel('Order')
plt.ylabel('University')

plt.show()


# Must include items in the Menu based on the University
# 
# 
# Ball University
# --
# Breaded Pork Tenderloin Sandwich
# 
# Indiana Corn on the Cob (brushed with garlic butter)
# 
# Indiana Pork Chili
# 
# Sugar Cream Pie
# 
# Ultimate Grilled Cheese Sandwich
# 
# Indiana Buffalo Chicken Tacos (3 tacos)
# 
# Sweet Potato Fries
# 
# Butler University
# --
# Breaded Pork Tenderloin Sandwich
# 
# Indiana Corn on the Cob (brushed with garlic butter)
# 
# Indiana Pork Chili
# 
# Sugar Cream Pie
# 
# Cornbread Hush Puppies
# 
# Fried Catfish Basket
# 
# Hoosier BBQ Pulled Pork Sandwich
# 
# Indiana State University 
# ---
# Ultimate Grilled Cheese Sandwich
# 
# Cornbread Hush Puppies
# 
# Fried Catfish Basket
# 
# Hoosier BBQ Pulled Pork Sandwich
# 
# Indiana Buffalo Chicken Tacos (3 tacos)
# 
# Sweet Potato Fries
# 
# Sugar Cream Pie
# 
# Indiana University-Purdue University Indianapolis
# --
# Cornbread Hush Puppies
# 
# Fried Catfish Basket
# 
# Hoosier BBQ Pulled Pork Sandwich
# 
# Indiana Buffalo Chicken Tacos (3 tacos)
# 
# Sweet Potato Fries
# 
# University of Notre Dame
# ---
# Ultimate Grilled Cheese Sandwich
# 
# For other universities the most liked item has not yet been discovered as per the observation

# In[44]:


crosstab1 = pd.crosstab(df['Major'], df['Order'])

# Display the crosstab
print(crosstab1)


# In[45]:


# Create a heatmap of the crosstab
plt.figure(figsize=(10, 6))  
sns.heatmap(crosstab1, annot=True, fmt="d", cmap="viridis") 

plt.title('Crosstab: Major vs. Order')
plt.xlabel('Order')
plt.ylabel('Major')

plt.show()


# Based on Major there is a certain trend
# 
# Ultimate Grilled Cheese Sandwich (with bacon and tomato) -->Mostly liked by Mathematics,Anthropology and business people
# Similarly for other order items also there is a bit trend

# In[49]:


crosstab2 = pd.crosstab(df['Year'], df['Order'])

# Create a heatmap of the crosstab
plt.figure(figsize=(10, 6))  
sns.heatmap(crosstab2, annot=True, fmt="d", cmap="Reds") 

plt.title('Crosstab: Students Year vs. Order')
plt.xlabel('Order')
plt.ylabel('Year')

plt.show()


# Only the 2nd year and 3rd year students are major consumers

# # Implications of data collection, storage, and data biases

# When considering the implications of data collection, storage, and data biases in the context of the FoodX data, it's essential to address various aspects, including Data Ethics, Business Outcomes, and Technical Implications.
# 
# Ethical Implications:
# 
# Privacy and Consent: Collecting data from users on their college experience and order predictions raises privacy concerns. Ensure that users have given informed consent to share this information.
# 
# Bias and Fairness: Data collection and prediction processes may introduce bias, potentially affecting certain groups more than others. It's crucial to regularly assess the fairness of the model's predictions and ensure they don't unfairly discriminate against specific demographics, universities, or other groups.
# 
# Transparency: The model's prediction mechanisms should be transparent and interpretable. Users should have a clear understanding of how their data is used and how predictions are made. Lack of transparency can lead to mistrust.
# 
# Data Handling and Security: Proper data storage and security measures are essential to protect users' sensitive information. Data breaches can have significant ethical consequences.
# 
# Business Outcome Implications:
# 
# Customer Satisfaction: Biased or inaccurate predictions can lead to customer dissatisfaction if they consistently receive incorrect orders. This can impact brand reputation and customer retention.
# 
# Operational Efficiency: Accurate predictions can improve operational efficiency by reducing cooking and preparation time, minimizing food wastage, and streamlining staff workload.
# 
# Discount Costs: Offering a 10% discount for incorrect predictions can result in a financial burden if predictions are consistently inaccurate. Accurate predictions are essential to manage costs effectively.
# 
# Customer Engagement: Ethical and transparent data practices can enhance customer engagement and loyalty. Customers are more likely to trust and continue using the FoodX app if they believe their data is handled responsibly.
# 
# Technical Implications:
# 
# Data Quality: Ensure data quality and consistency in the collected data. Inaccurate or inconsistent data can lead to unreliable predictions and biases.
# 
# Bias Mitigation: Implement techniques to mitigate bias in prediction models, such as fairness-aware machine learning, bias audits, and careful feature selection.
# 
# Model Explainability: Use interpretable machine learning models and provide explanations for predictions to increase transparency.
# 
# Data Security: Implement robust data security measures to protect user data, such as encryption and access control. Data breaches can have severe technical and legal consequences.
# 
# Scalability: Ensure that the data storage and processing infrastructure can scale with the increasing number of users and data points as the business grows.
# 
# In summary, ethical data collection and handling practices are vital to maintain trust with users and avoid potential biases. Accurate predictions have direct impacts on customer satisfaction, operational efficiency, and costs. On the technical side, data quality, security, and the scalability of systems are essential considerations to ensure a successful and ethical deployment of the FoodX app. Addressing these factors can contribute to both ethical and profitable outcomes for the business.
# 

# # Building Prediction Model

# In[121]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the dataset (replace 'foodx_data.csv' with your dataset path)
data = pd.read_csv('XTern 2024 Artificial Intelegence Data Set - Xtern_TrainData.csv')

# Preprocessing: Convert string features into numeric
encoder = LabelEncoder()
data['University'] = encoder.fit_transform(data['University'])
data['Year'] = encoder.fit_transform(data['Year'])
data['Order'] = encoder.fit_transform(data['Order'])

# Define features (University,Time and Year) and target (Order)
X = data[['University', 'Year','Time']]
y = data['Order']


# In[122]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

# Model Selection: Decision Tree Classifier
model = DecisionTreeClassifier()

# Training the Model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Pickle the Model
with open('foodx_order_classifier.pkl', 'wb') as file:
    pickle.dump(model, file)


# In[123]:


from sklearn.metrics import confusion_matrix
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# Evaluating whether bringing a solution like the FoodX order prediction system to maturity is a suitable course of action involves various considerations. Here are key factors to assess:
# 
# Business Goals and Objectives:
# 
# Align the solution with the overarching business goals and objectives. Does it help in achieving the company's mission or strategic goals?
# Consider the potential impact on revenue, customer satisfaction, and operational efficiency.
# Return on Investment (ROI):
# 
# Calculate the expected ROI of developing and maintaining the system. Assess the potential financial benefits against the costs involved.
# Performance and Accuracy:
# 
# Analyze the model's performance and accuracy. Is it accurate enough to make meaningful order predictions? If not, can it be improved?
# Data Quality and Availability:
# 
# Ensure data quality and reliability. Reliable data is crucial for machine learning models to make accurate predictions.
# Ethical and Legal Considerations:
# 
# Address ethical concerns related to data privacy, bias, and fairness.
# Ensure compliance with relevant data protection laws and regulations (e.g., GDPR, CCPA).
# Technical Feasibility:
# 
# Assess the technical feasibility of implementing and maintaining the system, including infrastructure, scalability, and compatibility with existing systems.
# User Adoption:
# 
# Evaluate whether users, both customers and staff, will adopt and trust the system. User feedback and acceptance are critical.
# Operational Impact:
# 
# Consider the impact on day-to-day operations, workload, and staff training.
# Assess whether the system can streamline processes and reduce operational costs.
# Competition and Market Dynamics:
# 
# Analyze the competitive landscape. Are competitors offering similar solutions? How does the solution differentiate the business?
# Risk Assessment:
# 
# Identify potential risks, such as model inaccuracies, data breaches, or technical issues, and develop mitigation strategies.
# Scalability:
# 
# Ensure the solution can scale as the business grows. Consider the capacity to handle increased data volume and user traffic.
# Maintenance and Support:
# 
# Plan for ongoing maintenance, updates, and technical support. Machine learning models require regular monitoring and fine-tuning.
# Cost-Benefit Analysis:
# 
# Weigh the benefits against the total cost of ownership, including development, maintenance, and support costs.
# Pilot Testing:
# 
# Consider conducting a pilot test or proof of concept to evaluate the solution's performance and acceptance in a real-world setting.
# Feedback and Iteration:
# 
# Establish a feedback loop to continually improve the system based on user feedback and changing business needs.
# Alternative Solutions:
# 
# Explore alternative solutions, including both technical and non-technical approaches, to address the problem.
# Long-Term Viability:
# 
# Assess the long-term viability of the solution in the context of changing market trends and technologies.
# Stakeholder Alignment:
# 
# Ensure alignment among all stakeholders, from executives to end-users, regarding the value and goals of the solution.
# Regulatory and Compliance Changes:
# 
# Stay aware of potential regulatory changes that might affect the solution and be prepared to adapt.
# Ultimately, the decision to proceed with developing and maturing the FoodX order prediction system should be based on a holistic assessment of these factors, weighing the benefits against the associated risks and costs. It's crucial to have a clear understanding of the business context and the potential impact of the solution on both the customer experience and the bottom line

# In[ ]:




