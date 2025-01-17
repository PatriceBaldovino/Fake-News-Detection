# Fake News Classification Project  

## **Overview**  
The Fake News Classification Project is designed to tackle the growing problem of misinformation by determining whether a news article is real or fake. Using advanced machine learning techniques and natural language processing (NLP), this project provides a robust and reliable solution to identify and mitigate the impact of fake news. The project includes a user-friendly web interface, built using Flask, where users can input a news title and instantly view the classification results.  

---

## **Motivation**  
In today’s digital world, fake news spreads rapidly, often influencing public perception and decision-making. This manipulation can significantly impact society, as beliefs and behaviors are shaped by false narratives. The motivation for this project stems from the need to combat misinformation and promote media literacy. By leveraging AI, this project empowers users to verify the authenticity of news articles and make informed decisions.  

---

## **Key Features**  
- **Machine Learning Model**: A Logistic Regression model trained with TF-IDF vectorization, achieving high accuracy and reliability.  
- **Preprocessing Pipeline**: Includes lemmatization, stopword removal, and n-gram-based TF-IDF to capture linguistic patterns effectively.  
- **Performance Metrics**:  
  - Training Accuracy: 99%  
  - Testing Accuracy: 97%  
  - Balanced Precision, Recall, and F1-Score: 97% for both classes.  
- **Web Interface**: A Flask-based webpage where users can input a news title and view the classification result and confidence level.  

---

## **Technical Details**  
1. **Data Preprocessing**:  
   - Lemmatization to normalize words.  
   - Removal of stopwords to reduce noise.  
   - TF-IDF vectorization with n-grams to capture meaningful patterns in text.  

2. **Machine Learning Model**:  
   - Logistic Regression was chosen for its simplicity and effectiveness in binary classification tasks.  
   - The model was trained and tested on a labeled dataset, achieving robust performance metrics.  

3. **Web Application**:  
   - Built using Flask for a lightweight and responsive user experience.  
   - Allows users to enter a news title and displays the classification result in real time.  

