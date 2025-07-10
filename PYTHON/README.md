# 📊 Grouping, Visualizing, and Describing Data in Python
*A Complete Data Analysis Module for Beginners*

---

## 🎯 Module Overview

Welcome to your journey into the heart of data analysis! Now that you've mastered cleaning and transforming datasets, it's time to unlock the real power of your data through grouping, visualization, and statistical analysis. 

**What You'll Master:**
- Transform raw data into meaningful insights through grouping
- Create compelling visualizations that tell stories
- Apply statistical methods to understand your data deeply
- Detect outliers and anomalies like a pro

**Real-World Focus:** We'll work with retail sales, student performance, and medical data throughout this module.

---

# 📊 Part A: Aggregating and Grouping with Pandas

## 🎯 Learning Objectives
By the end of this section, you'll be able to:
- Use `groupby()` to segment data meaningfully
- Apply multiple aggregation functions efficiently
- Create insightful pivot tables
- Handle multi-level grouping scenarios
- Sort and present grouped results professionally

## 📘 Introduction to GroupBy Operations

Think of `groupby()` as your data's personal organizer. Just like sorting your emails into folders, `groupby()` helps you organize your data into meaningful categories for analysis.

**The Magic Behind GroupBy:**
```python
# The pattern: Split → Apply → Combine
df.groupby('category')['sales'].sum()
```

This simple line:
1. **Splits** your data by category
2. **Applies** the sum function to sales in each group
3. **Combines** the results into a clean summary

## 💻 Hands-On Code: Retail Sales Analysis

Let's start with a realistic retail dataset:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create sample retail data
np.random.seed(42)
retail_data = pd.DataFrame({
    'region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books'], 1000),
    'sales_amount': np.random.normal(100, 30, 1000).round(2),
    'quantity_sold': np.random.randint(1, 10, 1000),
    'month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'], 1000)
})

# Ensure positive sales amounts
retail_data['sales_amount'] = retail_data['sales_amount'].abs()

print("Sample of our retail data:")
print(retail_data.head())
```

### 🔍 Basic Grouping Operations

```python
# 1. Simple grouping: Total sales by region
sales_by_region = retail_data.groupby('region')['sales_amount'].sum()
print("Total Sales by Region:")
print(sales_by_region)

# 2. Multiple statistics at once
region_stats = retail_data.groupby('region')['sales_amount'].agg(['sum', 'mean', 'count'])
print("\nDetailed Region Statistics:")
print(region_stats)

# 3. Grouping by multiple columns
region_category_sales = retail_data.groupby(['region', 'product_category'])['sales_amount'].sum()
print("\nSales by Region and Category:")
print(region_category_sales.head(10))
```

### 🎯 Advanced Aggregation with .agg()

```python
# Custom aggregation functions
def sales_range(x):
    return x.max() - x.min()

# Multiple aggregations with custom names
advanced_agg = retail_data.groupby('product_category').agg({
    'sales_amount': ['sum', 'mean', 'std', sales_range],
    'quantity_sold': ['sum', 'mean']
}).round(2)

print("Advanced Aggregation Results:")
print(advanced_agg)
```

### 📊 Creating Powerful Pivot Tables

```python
# Create a comprehensive pivot table
pivot_sales = retail_data.pivot_table(
    values='sales_amount',
    index='region',
    columns='product_category',
    aggfunc='sum',
    fill_value=0
)

print("Sales Pivot Table (Region × Product Category):")
print(pivot_sales)

# Multiple value columns
pivot_detailed = retail_data.pivot_table(
    values=['sales_amount', 'quantity_sold'],
    index='region',
    columns='product_category',
    aggfunc='mean',
    fill_value=0
).round(2)

print("\nDetailed Pivot Table:")
print(pivot_detailed)
```

### 🔄 Sorting and Ranking Results

```python
# Sort aggregated results
top_performing_regions = retail_data.groupby('region')['sales_amount'].sum().sort_values(ascending=False)
print("Regions Ranked by Total Sales:")
print(top_performing_regions)

# Rank within groups
retail_data['category_rank'] = retail_data.groupby('product_category')['sales_amount'].rank(ascending=False)
print("\nTop 5 sales with their category rankings:")
print(retail_data.nlargest(5, 'sales_amount')[['region', 'product_category', 'sales_amount', 'category_rank']])
```

## 🧪 Part A Exercises

### Exercise 1: Regional Performance Analysis
```python
# TODO: Create a grouped analysis that shows:
# 1. Average sales amount by region
# 2. Total quantity sold by region
# 3. Number of transactions by region
# Hint: Use .agg() with multiple functions
```

### Exercise 2: Monthly Trends
```python
# TODO: Create a pivot table showing:
# - Rows: Months
# - Columns: Product categories
# - Values: Average sales amount
# Which month and category combination performs best?
```

---

# 📈 Part B: Data Visualization with Matplotlib & Seaborn

## 🎯 Learning Objectives
By the end of this section, you'll be able to:
- Create professional charts with matplotlib
- Use seaborn for advanced statistical visualizations
- Choose the right chart type for your data story
- Customize visualizations for maximum impact
- Interpret complex visual patterns

## 📘 The Art of Data Storytelling

Visualizations are the language of data. A well-crafted chart can reveal insights that might take hours to discover through tables alone. Let's master this visual language!

## 💻 Matplotlib Fundamentals

### 📊 Basic Chart Types

```python
# Set up the plotting environment
plt.style.use('seaborn-v0_8')  # Modern, clean style
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Line Plot: Sales trends over time
monthly_sales = retail_data.groupby('month')['sales_amount'].mean()
axes[0, 0].plot(monthly_sales.index, monthly_sales.values, marker='o', linewidth=2)
axes[0, 0].set_title('Average Sales by Month', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Month')
axes[0, 0].set_ylabel('Average Sales ($)')
axes[0, 0].grid(True, alpha=0.3)

# 2. Bar Chart: Sales by region
region_sales = retail_data.groupby('region')['sales_amount'].sum()
bars = axes[0, 1].bar(region_sales.index, region_sales.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
axes[0, 1].set_title('Total Sales by Region', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Region')
axes[0, 1].set_ylabel('Total Sales ($)')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                   f'${height:,.0f}', ha='center', va='bottom')

# 3. Histogram: Distribution of sales amounts
axes[1, 0].hist(retail_data['sales_amount'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[1, 0].set_title('Distribution of Sales Amounts', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Sales Amount ($)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].axvline(retail_data['sales_amount'].mean(), color='red', linestyle='--', 
                  label=f'Mean: ${retail_data["sales_amount"].mean():.2f}')
axes[1, 0].legend()

# 4. Pie Chart: Market share by product category
category_sales = retail_data.groupby('product_category')['sales_amount'].sum()
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
axes[1, 1].pie(category_sales.values, labels=category_sales.index, autopct='%1.1f%%', 
              colors=colors, startangle=90)
axes[1, 1].set_title('Market Share by Product Category', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
```

### 🎨 Advanced Matplotlib Customization

```python
# Create a professional-looking chart
fig, ax = plt.subplots(figsize=(12, 8))

# Detailed regional analysis
region_category_pivot = retail_data.pivot_table(
    values='sales_amount', 
    index='region', 
    columns='product_category', 
    aggfunc='mean'
)

# Create a grouped bar chart
x = np.arange(len(region_category_pivot.index))
width = 0.2

for i, category in enumerate(region_category_pivot.columns):
    ax.bar(x + i*width, region_category_pivot[category], width, 
           label=category, alpha=0.8)

ax.set_xlabel('Region', fontsize=12)
ax.set_ylabel('Average Sales Amount ($)', fontsize=12)
ax.set_title('Average Sales by Region and Product Category', fontsize=16, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(region_category_pivot.index)
ax.legend(title='Product Category', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 🌟 Seaborn: Statistical Visualization Made Beautiful

### 📊 Essential Seaborn Plots

```python
# Create a comprehensive seaborn visualization dashboard
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Count Plot: Frequency of transactions by region
sns.countplot(data=retail_data, x='region', ax=axes[0, 0], palette='viridis')
axes[0, 0].set_title('Number of Transactions by Region', fontweight='bold')
axes[0, 0].tick_params(axis='x', rotation=45)

# 2. Box Plot: Sales distribution by product category
sns.boxplot(data=retail_data, x='product_category', y='sales_amount', ax=axes[0, 1], palette='Set2')
axes[0, 1].set_title('Sales Distribution by Product Category', fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. Histogram with KDE: Sales amount distribution
sns.histplot(data=retail_data, x='sales_amount', kde=True, ax=axes[0, 2], color='coral')
axes[0, 2].set_title('Sales Amount Distribution with Density', fontweight='bold')

# 4. Heatmap: Correlation between numerical variables
numerical_data = retail_data.select_dtypes(include=[np.number])
correlation_matrix = numerical_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
axes[1, 0].set_title('Correlation Heatmap', fontweight='bold')

# 5. Violin Plot: Advanced distribution comparison
sns.violinplot(data=retail_data, x='region', y='sales_amount', ax=axes[1, 1], palette='muted')
axes[1, 1].set_title('Sales Distribution Shape by Region', fontweight='bold')
axes[1, 1].tick_params(axis='x', rotation=45)

# 6. Scatter Plot: Relationship between quantity and sales
sns.scatterplot(data=retail_data, x='quantity_sold', y='sales_amount', 
               hue='product_category', ax=axes[1, 2], alpha=0.6)
axes[1, 2].set_title('Quantity vs Sales Amount', fontweight='bold')

plt.tight_layout()
plt.show()
```

### 💡 Student Performance Analysis Example

```python
# Create student performance dataset
np.random.seed(123)
student_data = pd.DataFrame({
    'student_id': range(1, 501),
    'math_score': np.random.normal(75, 15, 500),
    'science_score': np.random.normal(72, 12, 500),
    'english_score': np.random.normal(78, 10, 500),
    'study_hours': np.random.normal(5, 2, 500),
    'grade_level': np.random.choice(['9th', '10th', '11th', '12th'], 500)
})

# Ensure realistic score ranges
for col in ['math_score', 'science_score', 'english_score']:
    student_data[col] = student_data[col].clip(0, 100)

student_data['study_hours'] = student_data['study_hours'].clip(0, 12)

# Create comprehensive student analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Average scores by grade level
grade_scores = student_data.groupby('grade_level')[['math_score', 'science_score', 'english_score']].mean()
grade_scores.plot(kind='bar', ax=axes[0, 0], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[0, 0].set_title('Average Scores by Grade Level', fontweight='bold')
axes[0, 0].set_ylabel('Average Score')
axes[0, 0].legend(title='Subject')
axes[0, 0].tick_params(axis='x', rotation=45)

# 2. Study hours vs Math performance
sns.scatterplot(data=student_data, x='study_hours', y='math_score', 
               hue='grade_level', ax=axes[0, 1], alpha=0.6)
axes[0, 1].set_title('Study Hours vs Math Performance', fontweight='bold')

# 3. Score distribution comparison
score_cols = ['math_score', 'science_score', 'english_score']
student_data[score_cols].hist(bins=20, ax=axes[1, 0], alpha=0.7)
axes[1, 0].set_title('Score Distributions by Subject', fontweight='bold')

# 4. Performance heatmap by grade
performance_pivot = student_data.pivot_table(
    values=['math_score', 'science_score', 'english_score'],
    index='grade_level',
    aggfunc='mean'
)
sns.heatmap(performance_pivot.T, annot=True, cmap='YlOrRd', ax=axes[1, 1])
axes[1, 1].set_title('Performance Heatmap by Grade', fontweight='bold')

plt.tight_layout()
plt.show()
```

## 🧪 Part B Exercises

### Exercise 3: Create Your Visualization Story
```python
# TODO: Using the retail_data, create a 2x2 subplot showing:
# 1. A bar chart of total sales by month
# 2. A scatter plot of quantity_sold vs sales_amount
# 3. A box plot comparing sales across regions
# 4. A pie chart showing the proportion of sales by product category
# Add proper titles, labels, and styling
```

### Exercise 4: Advanced Seaborn Analysis
```python
# TODO: Create a comprehensive analysis of the student_data:
# 1. Use sns.pairplot() to show relationships between all score variables
# 2. Create a correlation heatmap
# 3. Use sns.boxplot() to compare study_hours across grade levels
# 4. Create a violin plot showing math_score distribution by grade_level
```

---

# 📐 Part C: Descriptive Statistics & Outlier Detection

## 🎯 Learning Objectives
By the end of this section, you'll be able to:
- Calculate and interpret key statistical measures
- Identify the center, spread, and shape of distributions
- Detect outliers using multiple methods
- Apply statistical insights to real-world problems
- Make data-driven recommendations based on statistical analysis

## 📘 The Foundation of Data Understanding

Statistics are the foundation of data analysis. They help us understand what our data is telling us, identify unusual patterns, and make informed decisions. Let's master the essential statistical toolkit!

## 💻 Comprehensive Statistical Analysis

### 📊 Central Tendency and Variability

```python
# Create a medical dataset for analysis
np.random.seed(456)
medical_data = pd.DataFrame({
    'patient_id': range(1, 1001),
    'age': np.random.normal(45, 15, 1000).clip(18, 90),
    'blood_pressure_systolic': np.random.normal(120, 20, 1000).clip(80, 200),
    'blood_pressure_diastolic': np.random.normal(80, 15, 1000).clip(50, 120),
    'cholesterol': np.random.normal(200, 40, 1000).clip(100, 350),
    'bmi': np.random.normal(25, 5, 1000).clip(15, 45),
    'heart_rate': np.random.normal(70, 12, 1000).clip(50, 120),
    'department': np.random.choice(['Cardiology', 'General', 'Emergency', 'Pediatrics'], 1000)
})

print("=== COMPREHENSIVE STATISTICAL SUMMARY ===")
print("\n1. BASIC DESCRIPTIVE STATISTICS")
print(medical_data.describe().round(2))

print("\n2. DETAILED STATISTICAL MEASURES")
for column in ['age', 'blood_pressure_systolic', 'cholesterol', 'bmi']:
    print(f"\n--- {column.upper()} ---")
    data = medical_data[column]
    
    print(f"Mean: {data.mean():.2f}")
    print(f"Median: {data.median():.2f}")
    print(f"Mode: {data.mode().iloc[0]:.2f}")
    print(f"Standard Deviation: {data.std():.2f}")
    print(f"Variance: {data.var():.2f}")
    print(f"Range: {data.max() - data.min():.2f}")
    print(f"Skewness: {data.skew():.2f}")
    print(f"Kurtosis: {data.kurtosis():.2f}")
```

### 🔍 Advanced Statistical Analysis

```python
# Percentiles and quartiles analysis
print("\n=== PERCENTILE ANALYSIS ===")
for column in ['age', 'blood_pressure_systolic', 'cholesterol']:
    print(f"\n{column.upper()}:")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = medical_data[column].quantile(p/100)
        print(f"  {p}th percentile: {value:.2f}")

# Unique values analysis
print("\n=== UNIQUE VALUES ANALYSIS ===")
for column in medical_data.columns:
    unique_count = medical_data[column].nunique()
    total_count = len(medical_data[column])
    print(f"{column}: {unique_count} unique values out of {total_count} total ({unique_count/total_count*100:.1f}%)")

# Value counts for categorical data
print("\n=== DEPARTMENT DISTRIBUTION ===")
dept_counts = medical_data['department'].value_counts()
dept_percent = medical_data['department'].value_counts(normalize=True) * 100
for dept in dept_counts.index:
    print(f"{dept}: {dept_counts[dept]} patients ({dept_percent[dept]:.1f}%)")
```

### 📊 Statistical Visualization

```python
# Create comprehensive statistical visualizations
fig, axes = plt.subplots(3, 2, figsize=(15, 18))

# 1. Distribution analysis
medical_data['age'].hist(bins=30, ax=axes[0, 0], alpha=0.7, color='lightblue', edgecolor='black')
axes[0, 0].axvline(medical_data['age'].mean(), color='red', linestyle='--', label=f'Mean: {medical_data["age"].mean():.1f}')
axes[0, 0].axvline(medical_data['age'].median(), color='green', linestyle='--', label=f'Median: {medical_data["age"].median():.1f}')
axes[0, 0].set_title('Age Distribution with Central Tendency', fontweight='bold')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()

# 2. Box plot for outlier detection
medical_data.boxplot(column='blood_pressure_systolic', by='department', ax=axes[0, 1])
axes[0, 1].set_title('Blood Pressure by Department', fontweight='bold')
axes[0, 1].set_xlabel('Department')
axes[0, 1].set_ylabel('Systolic BP')

# 3. Correlation analysis
correlation_matrix = medical_data.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
axes[1, 0].set_title('Correlation Matrix', fontweight='bold')

# 4. Scatter plot with statistical overlay
sns.scatterplot(data=medical_data, x='age', y='blood_pressure_systolic', 
               hue='department', ax=axes[1, 1], alpha=0.6)
axes[1, 1].set_title('Age vs Blood Pressure by Department', fontweight='bold')

# 5. Statistical summary by group
dept_stats = medical_data.groupby('department')[['age', 'blood_pressure_systolic', 'cholesterol']].mean()
dept_stats.plot(kind='bar', ax=axes[2, 0])
axes[2, 0].set_title('Average Values by Department', fontweight='bold')
axes[2, 0].tick_params(axis='x', rotation=45)
axes[2, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 6. Distribution comparison
medical_data['cholesterol'].hist(bins=30, ax=axes[2, 1], alpha=0.7, color='lightgreen', edgecolor='black')
axes[2, 1].set_title('Cholesterol Distribution', fontweight='bold')
axes[2, 1].set_xlabel('Cholesterol Level')
axes[2, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
```

## 🚨 Outlier Detection: The Detective Work

### 🔍 IQR Method (Interquartile Range)

```python
def detect_outliers_iqr(data, column):
    """
    Detect outliers using the IQR method
    Outliers are defined as values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    
    print(f"=== OUTLIER DETECTION FOR {column.upper()} ===")
    print(f"Q1 (25th percentile): {Q1:.2f}")
    print(f"Q3 (75th percentile): {Q3:.2f}")
    print(f"IQR: {IQR:.2f}")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")
    print(f"Number of outliers: {len(outliers)}")
    print(f"Percentage of outliers: {len(outliers)/len(data)*100:.2f}%")
    
    if len(outliers) > 0:
        print(f"Outlier values range: {outliers[column].min():.2f} to {outliers[column].max():.2f}")
    
    return outliers, lower_bound, upper_bound

# Detect outliers in multiple columns
outlier_columns = ['age', 'blood_pressure_systolic', 'cholesterol', 'bmi']
outlier_results = {}

for col in outlier_columns:
    outliers, lower, upper = detect_outliers_iqr(medical_data, col)
    outlier_results[col] = {'outliers': outliers, 'lower': lower, 'upper': upper}
    print("\n" + "="*50 + "\n")
```

### 📊 Visualizing Outliers

```python
# Create outlier visualization dashboard
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

for idx, column in enumerate(outlier_columns):
    row = idx // 2
    col = idx % 2
    
    # Box plot to show outliers
    sns.boxplot(data=medical_data, y=column, ax=axes[row, col])
    axes[row, col].set_title(f'{column.replace("_", " ").title()} - Outlier Detection', fontweight='bold')
    
    # Add outlier statistics
    outlier_info = outlier_results[column]
    outlier_count = len(outlier_info['outliers'])
    outlier_percent = outlier_count / len(medical_data) * 100
    
    axes[row, col].text(0.02, 0.98, f'Outliers: {outlier_count} ({outlier_percent:.1f}%)', 
                       transform=axes[row, col].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()
```

### 🎯 Advanced Outlier Analysis

```python
# Z-score method for outlier detection
from scipy import stats

def detect_outliers_zscore(data, column, threshold=3):
    """
    Detect outliers using Z-score method
    Outliers are defined as values with |Z-score| > threshold
    """
    z_scores = np.abs(stats.zscore(data[column]))
    outliers = data[z_scores > threshold]
    
    print(f"=== Z-SCORE OUTLIER DETECTION FOR {column.upper()} ===")
    print(f"Threshold: {threshold}")
    print(f"Number of outliers: {len(outliers)}")
    print(f"Percentage of outliers: {len(outliers)/len(data)*100:.2f}%")
    
    return outliers

# Compare IQR and Z-score methods
print("=== COMPARISON: IQR vs Z-SCORE METHODS ===")
for column in ['blood_pressure_systolic', 'cholesterol']:
    print(f"\n{column.upper()}:")
    
    # IQR method
    iqr_outliers = outlier_results[column]['outliers']
    print(f"IQR method: {len(iqr_outliers)} outliers")
    
    # Z-score method
    zscore_outliers = detect_outliers_zscore(medical_data, column)
    print(f"Z-score method: {len(zscore_outliers)} outliers")
    
    # Overlap analysis
    iqr_ids = set(iqr_outliers.index)
    zscore_ids = set(zscore_outliers.index)
    overlap = len(iqr_ids.intersection(zscore_ids))
    print(f"Overlap: {overlap} outliers detected by both methods")
```

### 🏥 Real-World Outlier Investigation

```python
# Investigate outliers in context
print("=== INVESTIGATING HIGH BLOOD PRESSURE OUTLIERS ===")
bp_outliers = outlier_results['blood_pressure_systolic']['outliers']

print(f"Analyzing {len(bp_outliers)} blood pressure outliers:")
print("\nOutlier characteristics:")
print(bp_outliers[['age', 'blood_pressure_systolic', 'cholesterol', 'bmi', 'department']].describe())

print("\nDepartment distribution of outliers:")
outlier_dept_dist = bp_outliers['department'].value_counts()
total_dept_dist = medical_data['department'].value_counts()

for dept in outlier_dept_dist.index:
    outlier_rate = outlier_dept_dist[dept] / total_dept_dist[dept] * 100
    print(f"{dept}: {outlier_dept_dist[dept]} outliers ({outlier_rate:.1f}% of {dept} patients)")

# Visual investigation
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Outliers by department
outlier_dept_percent = bp_outliers['department'].value_counts(normalize=True) * 100
normal_dept_percent = medical_data['department'].value_counts(normalize=True) * 100

comparison_df = pd.DataFrame({
    'Outliers': outlier_dept_percent,
    'Normal': normal_dept_percent
}).fillna(0)

comparison_df.plot(kind='bar', ax=axes[0])
axes[0].set_title('Department Distribution: Outliers vs Normal', fontweight='bold')
axes[0].set_ylabel('Percentage')
axes[0].tick_params(axis='x', rotation=45)

# Age distribution of outliers
axes[1].hist(medical_data['age'], bins=30, alpha=0.7, label='All patients', color='lightblue')
axes[1].hist(bp_outliers['age'], bins=15, alpha=0.7, label='BP outliers', color='red')
axes[1].set_title('Age Distribution: All Patients vs BP Outliers', fontweight='bold')
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Frequency')
axes[1].legend()

plt.tight_layout()
plt.show()
```

## 🧪 Part C Exercises

### Exercise 5: Complete Statistical Analysis
```python
# TODO: Using the medical_data, perform a comprehensive statistical analysis:
# 1. Calculate mean, median, mode, and standard deviation for all numerical columns
# 2. Identify which variables have the highest and lowest variability
# 3. Find the correlation between age and blood_pressure_systolic
# 4. Determine if there are significant differences in cholesterol levels between departments
# 5. Create a summary report with your findings
```

### Exercise 6: Outlier Investigation
```python
# TODO: Investigate outliers in the cholesterol column:
# 1. Use both IQR and Z-score methods to detect outliers
# 2. Compare the results - which method finds more outliers?
# 3. Create visualizations showing the outliers
# 4. Analyze the characteristics of cholesterol outliers (age, department, other health metrics)
# 5. Provide recommendations for handling these outliers
```

### Exercise 7: Department Performance Analysis
```python
# TODO: Create a comprehensive analysis comparing medical departments:
# 1. Calculate average values for all health metrics by department
# 2. Identify which department has the most variation in patient health
# 3. Create box plots comparing each health metric across departments
# 4. Determine if any department has unusually high outlier rates
# 5. Write a brief report on your findings
```

---

# 📂 Major Assignments

## 🎯 Assignment 1: Retail Sales Deep Dive
**Objective:** Build a comprehensive pivot table analysis of retail performance

**Your Task:**
1. Create a pivot table showing sales by category and region
2. Add calculated columns for:
   - Sales per transaction (sales_amount / quantity_sold)
   - Regional market share percentages
   - Month-over-month growth rates
3. Identify the top 3 performing region-category combinations
4. Create supporting visualizations for your findings
5. Write a 200-word executive summary with actionable recommendations

**Expected Deliverables:**
- Pivot table with calculations
- 3-4 supporting charts
- Executive summary with insights

## 🎯 Assignment 2: Visual Storytelling Portfolio
**Objective:** Create three distinct visualizations that tell a compelling data story

**Your Task:**
Choose one dataset (retail, student, or medical) and create:
1. **Exploratory Chart:** A chart that reveals an unexpected pattern or trend
2. **Comparative Analysis:** A visualization comparing multiple groups or categories
3. **Interactive Dashboard:** A multi-panel view showing different aspects of the data

**Requirements:**
- Each chart must have a clear title and purpose
- Use appropriate color schemes and styling
- Include annotations highlighting key insights
- Provide 2-3 sentence interpretations for each visualization

## 🎯 Assignment 3: Statistical Consulting Report
**Objective:** Use descriptive statistics and outlier detection to solve a business problem

**Your Task:**
You're a data consultant hired to analyze patient health metrics. Create a comprehensive report that includes:

1. **Executive Summary:** Key findings in 100 words
2. **Statistical Overview:** Descriptive statistics for all health metrics
3. **Outlier Analysis:** Identification and investigation of anomalies
4. **Department Comparison:** Statistical comparison across departments
5. **Recommendations:** Data-driven suggestions for healthcare management

**Format Requirements:**
- Professional report structure
- Include statistical tables and visualizations
- Provide interpretation for all statistics
- Make specific, actionable recommendations

---

# ❓ Recap Quiz

## 🧠 Test Your Knowledge

### Part A: Grouping & Aggregation
1. **What does `groupby()` return before applying an aggregation function?**
   - A) A DataFrame
   - B) A Series
   - C) A GroupBy object
   - D) A list

2. **Which method allows you to apply multiple aggregation functions at once?**
   - A) `.apply()`
   - B) `.agg()`
   - C) `.transform()`
   - D) `.aggregate_all()`

3. **In a pivot table, what determines the columns?**
   - A) The index parameter
   - B) The values parameter
   - C) The columns parameter
   - D) The aggfunc parameter

### Part B: Visualization
4. **Which seaborn function is best for showing correlation between variables?**
   - A) `sns.countplot()`
   - B) `sns.boxplot()`
   - C) `sns.heatmap()`
   - D) `sns.barplot()`

5. **What's the main advantage of using seaborn over matplotlib?**
   - A) Faster performance
   - B) Better statistical visualizations with less code
   - C) More customization options
   - D) Built-in data cleaning

6. **Which chart type is best for showing the distribution of a single numerical variable?**
   - A) Bar chart
   - B) Scatter plot
   - C) Histogram
   - D) Line plot

### Part C: Statistics & Outliers
7. **The IQR method defines outliers as values that are:**
   - A) More than 2 standard deviations from the mean
   - B) Below Q1 - 1.5×IQR or above Q3 + 1.5×IQR
   - C) In the top or bottom 5% of values
   - D) More than 3 standard deviations from the mean

8. **Which measure of central tendency is most resistant to outliers?**
   - A) Mean
   - B) Median
   - C) Mode
   - D) Standard deviation

9. **What does a correlation coefficient of -0.85 indicate?**
   - A) Strong positive relationship
   - B) Weak negative relationship
   - C) Strong negative relationship
   - D) No relationship

10. **When would you use `value_counts()` instead of `mean()`?**
    - A) For numerical data analysis
    - B) For categorical data frequency analysis
    - C) For outlier detection
    - D) For correlation analysis

---

# 💡 Pro Tips & Fun Facts

## 🎯 Visualization Wisdom
- **"A picture is worth a thousand data points"** - Well-designed visualizations can reveal patterns that would take hours to find in raw data
- **The 80/20 Rule:** 80% of insights come from 20% of your visualizations - focus on the charts that tell the most important stories
- **Color Psychology:** Blue suggests trust and stability (great for financial data), green implies growth and health, red indicates urgency or danger

## 📊 Statistical Insights
- **The Normal Distribution:** Approximately 68% of data falls within 1 standard deviation of the mean, 95% within 2 standard deviations
- **Outlier Impact:** A single extreme outlier can increase the standard deviation by 40% or more in small datasets
- **Correlation vs. Causation:** A correlation of 0.95 between ice cream sales and drowning incidents doesn't mean ice cream causes drowning - both increase with temperature!

## 🚀 Performance Boosters
- **Groupby Optimization:** Use `.agg()` with multiple functions instead of multiple separate groupby operations
- **Memory Management:** For large datasets, use `pd.cut()` to bin continuous variables before grouping
- **Visualization Speed:** Use `plt.ioff()` to turn off interactive mode for faster batch chart creation

## 🔍 Advanced Techniques Preview
- **Rolling Statistics:** Calculate moving averages with `df.rolling(window=7).mean()`
- **Conditional Formatting:** Highlight outliers in pandas with `df.style.applymap()`
- **Interactive Plots:** Combine with plotly for web-ready, interactive visualizations

---

# 🎓 Module Completion

## 🌟 Congratulations!

You've mastered the core skills of data analysis! You can now:
- ✅ Transform raw data into meaningful insights through grouping
- ✅ Create compelling visualizations that tell data stories
- ✅ Apply statistical methods to understand your data deeply
- ✅ Detect and investigate outliers like a professional analyst

## 🚀 Next Steps

**Ready to Level Up?**
1. **Advanced Analytics:** Explore time series analysis and forecasting
2. **Machine Learning:** Use your statistical foundation to build predictive models
3. **Interactive Dashboards:** Create web-based dashboards with Plotly Dash
4. **Big Data:** Apply these skills to larger datasets with tools like Dask

## 📚 Additional Resources

**Books:**
- "Python for Data Analysis" by Wes McKinney
- "Storytelling with Data" by Cole Nussbaumer Knaflic

**Online Practice:**
- Kaggle Learn courses
- DataCamp projects
- Real datasets from government open data portals

**Communities:**
- r/LearnPython
- Stack Overflow
- Local Python meetups

---

## 🎉 Final Challenge

**The Ultimate Data Analysis Challenge:**
Create a complete analysis of a real-world dataset that includes:
1. Data cleaning and preparation
2. Grouping and aggregation insights
3. Three different visualization types
4. Statistical analysis with outlier detection
5. A professional presentation of your findings

**Recommended Datasets:**
- Titanic passenger data
- World happiness report
- COVID-19 vaccination data
- Stock market historical data

Remember: The best data analysts don't just run analyses - they tell stories that drive decisions. Your journey in data analysis is just beginning!

---

*"In God we trust. All others must bring data." - W. Edwards Deming*

**Happy Analyzing! 📊✨**
