# H&M Personalized Fashion Recommendations

This project aims to build a recommendation system for H&M's online store, predicting which products customers are most likely to purchase in the near future based on their previous purchase history and product metadata.


## Project Overview
H&M Group seeks to enhance the shopping experience by providing personalized product recommendations. This project use machine learning techniques to analyze customer behavior, product attributes, and historical transaction data to generate accurate and relevant recommendations.

## Dataset
The dataset provided by H&M includes:

- Customer purchase history
- Product metadata
- Customer metadata
- Product images

Key files: not included but can be found here: [data link](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)

- images/: Folder containing product images
- articles.csv: Detailed metadata for each article
- customers.csv: Metadata for each customer
- transactions_train.csv: Training data with customer purchases

## Objective
The main goal is to predict which articles each customer will purchase in the 7-day period following the training data timeframe.

## Methodology
1. Exploratory Data Analysis
2. Data Preprocessing
	- Clean and organize transaction data
	- Process customer and product metadata
	- Handle image data if utilized
3. Feature Engineering
	- Create relevant features from transaction history
	- Extract useful information from product and customer metadata
	- Potentially develop image-based features
4. Model Development
	- Experiment with various recommendation algorithms
	- Potential approaches:
		- Collaborative Filtering
		- Content-Based Filtering
		- Hybrid Methods
		- Deep Learning Models (for text and image data)
5. Model Training and Validation
	- Train models on historical data
	- Validate performance using ...
6. Prediction Generation
	- Generate top 12 product recommendations for each customer
