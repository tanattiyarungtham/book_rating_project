<div align="center">
  
# Project: Book Rating Prediction Model  
**Diavila Rostaing Engandzi, Tanattiya Rungtham, Yohan Walter Jothipala**  
**Data ScienceTech Institute (DSTI), School of Engineering, Paris & Nice, France**

</div>

## Book Rating Prediction: Exploratory Data Analysis and Modeling
## Project Overview

This project focuses on predicting book ratings using a dataset from Goodreads. The goal is to build a machine learning model that can predict book ratings based on features like average rating, number of pages, ratings count, text reviews count, and others. The project consists of two main components:

1. **Exploratory Data Analysis (EDA)**: Understanding and preprocessing the dataset to prepare for modeling.
2. **Modeling**: Training and evaluating machine learning models to predict book ratings.

## Files in this Repository

- `Exploratory_Data_Analysis.ipynb`: Contains data preprocessing, visualization, and feature engineering. It uses libraries such as `pandas`, `seaborn`, `missingno`, and `smogn`.
- `Models.ipynb`: Implements various machine learning models (e.g., Decision Tree, Random Forest, Gradient Boosting) and compares their performances based on different evaluation metrics.
- `Project_Report.pdf`: Provides an in-depth overview of the project, including architecture, workflow, data engineering, and detailed analysis of the models.

## Data

The dataset contains reviews and book details from the Goodreads platform, including:

- **Title**: The name of the book.
- **Authors**: The book's author(s).
- **Average Rating**: The average user rating for the book.
- **Number of Pages**: The number of pages in the book.
- **Ratings Count**: The total number of ratings a book has received.
- **Text Reviews Count**: The number of text reviews for the book.

### Data Preprocessing (Exploratory_Data_Analysis.ipynb)

- **Loading Data**: The dataset (`books.csv`) is loaded and cleaned, including the removal of unnecessary columns and handling of missing values.
- **Feature Engineering**: Features such as the publication year and author statistics are generated.
- **Visualizations**: Various plots are used to explore relationships between the features and the target variable (book rating).

### Machine Learning Models (Models.ipynb)

Several machine learning algorithms were used, including:

- **Linear Regression, Lasso, Ridge, and ElasticNet**: Linear models with regularization.
- **Tree-Based Models**: Decision Trees, Random Forest, Bagging Regressor.
- **Boosting Techniques**: Gradient Boosting, LightGBM, XGBoost.

### Best Model

After comparing various models, the **Bagging Regressor** was identified as the best model with the highest R² score (0.9967) and low MSE. The best-performing model was saved using the `joblib` library for future predictions.

```python
import joblib
joblib.dump(best_model, 'best_model.pkl')
```

## Project Report (Project_Report.pdf)

This report provides a comprehensive analysis of the book rating prediction project, including detailed explanations of the data used, the models evaluated, and the results obtained. Below is a summary of key sections:

- **Abstract**: The project aims to predict book ratings using the Goodreads dataset. Features such as average rating, number of pages, ratings count, and text reviews count were used. Multiple machine learning models were applied and compared to identify the best-performing model.
  
- **Introduction**: Goodreads, the world's largest site for book recommendations, provided the dataset for this project. The dataset contains 200 million reviews for 45,000 unique books, including various features such as book title, author, and ratings.

- **Architecture**: The architecture follows a typical machine learning workflow: data loading, exploratory data analysis, feature engineering, model building, evaluation, and tuning.

- **Data Engineering**: Steps include cleaning the dataset, handling missing values, and generating new features (e.g., author average ratings, number of occurrences of each book). 

- **Machine Learning Models**: Several models, including tree-based and ensemble methods, were evaluated. The **Bagging Regressor** achieved the highest accuracy with an R² score of 0.9967, making it the best model for this task.

- **Evaluation**: The models were evaluated using metrics such as MAE, MSE, RMSE, and R² score. The **Bagging Regressor** was selected based on its superior performance.

### Key Results:

- **Top Features**: Features like `rate_occurrence`, `number_occurrence`, and `author_average_rating` were identified as the most important for predicting book ratings.
- **Best Model**: The **Bagging Regressor** achieved the best results, with an R² score of 0.9967 and low MSE, making it suitable for deployment.

For a detailed explanation of the project, refer to the full **Project_Report.pdf**.

## Python Environment

This project was developed using **Python 3.12.3**. A virtual environment was created using `conda` to ensure compatibility with all required dependencies. 

To replicate the environment:

1. Install **conda** and create an isolated environment:
   ```bash
   conda create --name book_rating_env python=3.12.3
   conda activate book_rating_env
   ```
2. Install the necessary packages:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/book-rating-prediction.git
   cd book-rating-prediction
   ```
2. Install the required dependencies:
   ```bash
   conda activate book_rating_env
   pip install -r requirements.txt
   ```
3. Run the notebooks:
   - Open `Exploratory_Data_Analysis.ipynb` for data preprocessing and feature engineering.
   - Open `Models.ipynb` to train and evaluate the models.

## Conclusion

This project demonstrates a complete machine learning pipeline, from data preprocessing and feature engineering to model evaluation and selection. The **Bagging Regressor** proved to be the best model for predicting book ratings, achieving an R² score of 0.9967. The model is saved using `joblib` for future deployment.

## Future Work

- Experiment with deep learning models for further improvement.
- Apply more advanced hyperparameter tuning methods (e.g., Bayesian Optimization).
- Explore the use of textual data (reviews) to enhance the predictive power of the model.
