<h1>Sentiment Analysis </h1>

<h2>Description</h2>
This repository contains code for performing sentiment analysis on text data using the Naive Bayes classifier. The goal is to classify text into three categories: negative, neutral, and positive sentiment. The code demonstrates the entire process, including text preprocessing, exploratory data analysis (EDA), feature engineering using bag-of-words encoding, model training and evaluation, and visualization of accuracy scores.
<br />


<h2>Dependencies</h2>

<br> The following Python libraries are required to run the code:</b>

- <b>re<b/>
- <b>pandas<b/>
- <b>seaborn<b/>
- <b>nltk<b/>
- <b>sklearn<b/>
- <b>matplotlib<b/>

<h2>IDE Used </h2>

- <b>VS Code</b> 

<h2>Program Walk-through</h2>

<p align="center">
Text Preprocessing Function: <br/>
<img src="https://i.imgur.com/ZetngNR.png" height="80%" width="80%"/>
<br />
<br />
Bag of Words and Naive Bayes Classifier:  <br/>
<img src="https://i.imgur.com/K5p1kLx.png" height="80%" width="80%"/>
<br />
<br />
Plotting the Accuracy Score: <br/>
<img src="https://i.imgur.com/XmEBsXi.png" height="80%" width="80%"/>
<br />
<br />

<h2>Method</h2>
<h3>Text Preprocessing</h3>
<b>The preprocess_text function is used to preprocess the text data. It performs the following steps:<b/>
  
- <b>Removal of special characters and numbers.<b/>
- <b>Conversion of text to lowercase.<b/>
- <b>Tokenization of the text into individual words.<b/>
- <b>Removal of stopwords (commonly occurring words with little semantic value).<b/>
- <b>Lemmatization of the words to their base form.<b/>
<br />
<br />

<h3>Model Training and Evaluation</h3>
The code uses the Naive Bayes classifier (MultinomialNB) from the scikit-learn library for sentiment classification. The classifier is trained using the bag-of-words (BoW) encoding of the preprocessed text. The dataset is split into training and testing sets using a 80:20 ratio. The accuracy of the classifier is computed using the accuracy_score function and a confusion matrix is generated using the confusion_matrix function.
<br />
<br />

<h3>Accuracy Scores Visualization</h3>
The code also includes a visualization of the accuracy scores of the Naive Bayes classifier across multiple epochs. It demonstrates how the accuracy of the classifier evolves over time. The accuracy scores are plotted against the number of epochs using matplotlib.
<br />
<br />

</p>

<!--
 ```diff
- text in red
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold)@@
```
--!>
