# cs505-homework-4-solved
**TO GET THIS SOLUTION VISIT:** [CS505 Homework 4 Solved](https://www.ankitcodinghub.com/product/cs505-in-this-assignment-you-will-learn-about-text-classification-and-language-modeling-using-rnn-and-lstm-and-use-python-framework-for-deep-learning-such-as-pytorch-which-are-popular-in-nlp-you/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;115941&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CS505 Homework 4 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
– In this assignment, you will learn about text classification and language modeling using RNN and LSTM, and use python framework for deep learning such as pytorch, which are popular in NLP. You have 2 weeks to finish this particular assignment.

–Please indicate names of those you collaborate with.

–Every late day will reduce your score by 20

Submit your (1) code/Jupyter notebook and (2) write up in one zip file.

When necessary, you must show how you derive your answer

Problem 1. Neural Models (32 pts)

1. (2 pts) In a news framing classification task, where you have 5 frames and your model predicts each of the frames with equal probability for an article, what is the cross entropy loss of the article in this case?

2. (2 pts) Suppose during training of your neural model you realize that your training loss remains high. Mention some of the ways you can reduce this underfitting of your neural network.

3. (2 pts) After you do many changes to your neural network, you now realize that your training loss is much lower than your validation loss. Mention some of the ways you can reduce this overfitting of your neural network.

4. (2 pts) What is good about setting a large batch size for training? How about a small batch size?

5. (3 pts) How can an RNN be used for detecting toxic spans (spans of words containing toxic language) in a social media comment? Specifically, what should be the input to the RNN at each time step t? How many outputs (i.e., yˆ) are produced given a comment containing n words? What is each yˆ(t) a probability distribution over?

6. (3 pts) How about using RNNs for language modeling? Given a start word token as input at time step 1, what should be the input to the RNN at each time step t &gt; 1? How many outputs are produced? What is each yˆ(t) a probability distribution over?

7. (3 pts) How about using RNNs for frame classification? Given an article containing n words as input, what should be the input to the RNN at each time step t? How many outputs are produced? What is each yˆ(t) a probability distribution over?

8. (2 pts) What is the main advantage of using RNNs for frame classification over feed forward neural network?

9. (3 pts) What is the disadvantage of RNN when used to classify the sentiment of a very long tweet like this? “I am not sure I want this phone. It’s too big to fit in my back pocket. I put it in and accidentally sat on it and now it’s bent. I’m very disappointed. I’m now the proud owner of bendy iPhone6. Very proud.” What is the appropriate sentiment for this tweet? And what would the RNN classify it as?

10. How about LSTM? Given this formulation of LSTM: ft = σ(Wfxt + Ufht−1 + bf) (forget gate), it = σ(Wixt +

Uiht−1 + bi) and Cˆt = tanh(WCxt + UCht−1 + bC) (input gate), Ct = ft ∗ Ct−1 + it ∗ Cˆt (update gate), and ot = σ(Woxt + Uoht−1 + bo) and ht = ot ∗ tanh(Ct):

(a) (4 pts) derive the formulation of for two time steps t and t −1 in terms of , and .

(b) (2 pts) which part of reduces the effect of the vanishing gradient problem in RNNs?

(c) (2 pts) How does this help classify the correct sentiment of the tweet above?

(d) (2 pts) Instead of using the last hidden state of LSTM to classify the tweet, what other ways we can do to improvethe performance of this sentiment classification?

1

Problem 2. LSTM for language modeling (36 points)

2. (5 pts) Compute and report the perplexity of the saved model on test 1.txt file. Note that the test files are already preprocessed.

2.1?

4. (1 pts) Compute and report the perplexity of this saved model on test 1.txt file.

5. (1 pts) Use the better language model (the one with the lower perplexity on test 1.txt) to compute and report the perplexity on test 2.txt. Note that the test files are already pre-processed.

6. (5 pts) Train the better language model as before but start with pre-trained Glove6B 100d embeddings (see here on how to incorporate pretrained embeddings in your LSTM model). This time, use all your words, even those occurring only once in the corpus. Only assign UNK token to words that are not in Glove vocabulary and initialize random vectors in the embedding matrix for the UNK, &lt;s&gt;, &lt;/s&gt;, and PAD tokens. Save your trained model. Generate 10 examples of text from it, starting from ’&lt;s&gt;’ token and ending at ’&lt;/s&gt;’ token. Are there differences from the generated examples from before?

7. (1 pts) Compute and report the perplexity of this saved model on test 1.txt file.

8. (2 pts) Train a language model with input sequence lengths of 5 as before (Question 2.1) on texts from tweet.txt. Note that this file is already pre-processed. Save your trained model. Generate 10 examples of text from it, starting from ’&lt;s&gt;’ token and ending at ’&lt;/s&gt;’ token.

9. (1 pts) Compute and report the perplexity of this saved model on test 2.txt file.

10. (2 pts) Train a language model with input sequence lengths of 15 on texts from tweet.txt. Save your trained model. Generate 10 examples of text from it, starting from ’&lt;s&gt;’ token and ending at ’&lt;/s&gt;’ token. Are there differences from the generated examples from 2.8?

11. (1 pts) Compute and report the perplexity of this saved model on test 2.txt file.

12. (1 pts) Use the better language model (the one with the lower perplexity on test 2.txt) to compute and report the perplexity on test 1.txt.

13. (2 pts) Train the better language model on tweet.txt but starting from pre-trained Glove6B 100d embeddings like in 2.6. Save your trained model. Generate 10 examples of text from it, starting from ’&lt;s&gt;’ token and ending at ’&lt;/s&gt;’ token.

14. (1 pts) Compute and report the perplexity of this saved model on test 2.txt file.

15. (2 pts) Train the better language model on tweet.txt but starting from pre-trained GloveTwitter 100d like in 2.6. Save your trained model. Generate 10 examples of text from it, starting from ’&lt;s&gt;’ token and ending at ’&lt;/s&gt;’ token.

16. (1 pts) Compute and report the perplexity of this saved model on test 2.txt file.

Problem 3. LSTM for classification (32 points, BONUS: 10 pts)

1. (5 pts) Follow the tutorial here on how to build LSTM model for sentiment classification. Modify the tutorial to train on your tweet sentiment data (sentiment-train.csv) and test on test data (sentiment-test.csv) from HW3 (modify the tutorial so that the train data is not split into train and validation). Compute and report the accuracy on the test data.

2

2. (2 pts) Modify the model from 3.1 to use GRU. Compute and report the accuracy on the test data.

3. (5 pts) Modify the model from 3.1 to use bidirectional LSTM. Compute and report the accuracy on the test data.

4. (2 pts) Modify the model from 3.1 to use bidirectional GRU. Compute and report the accuracy on the test data.

5. (5 pts) Pick the best model so far and train the model starting from pretrained GloveTwitter 100d. Compute and report the accuracy on the test data.

6. (10 pts) Using your best model so far, conduct a 5-fold (stratified) cross validation on your training data and a grid search to pick the best hidden size (try 128 or 512) and embedding size (try 100 or 400). Compute and report the average accuracies for each of the choice combination.

7. (3 pts) Train the model on all your training data using the best combination of hyperparameters you find in 2.6. Compute and report the accuracy on the test data.

8. (BONUS: 10 pts) Train your best model using the hyperparameter from 2.6 on all the sentiment140 data. Compute and report the accuracy on the test data from HW3 (i.e., sentiment-test.csv) .

3
