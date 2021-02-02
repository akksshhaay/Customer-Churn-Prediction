# Customer-Churn-Prediction
Customer Churn Prediction using Neural Network

Dataset Contains: 
1) Customer Id – A unique identity for customer
2) Surname
3) Credit Score – Only applicable if customer has credit card (col 10.)
4) Geography – Region where customer resides
5) Gender
6) Age
7) Tenure – Time customer is been with the bank
8) Balance
9) Number of Products
10) Has credit card
11) Is Active member
12) Estimated Salary
13) Exit (Churn) – Whether customer has exited the bank or not

Experiments:

1. Artificial Neural Network
Artificial neural networks (ANNs) are a family of machine learning models inspired by biological neural networks.

Experiment – 1.1:
• In this Experiment we use 2 Hidden Layers of 10 and 6 Neurons each. The Activation function of First hidden Layer is ‘tanh’ and second hidden layer is ReLU.
• The Activation function of Output Layer is ‘sigmoid’
• We used ‘Adam’ optimizer of Keras. We trained it for 30 epochs.
• We got 86.5% of accuracy of churn Prediction.

Experiment – 1.2:
• In this Experiment we use 2 Hidden Layers of 10 Neurons each. The Activation function of First hidden Layer is ‘tanh’ and second hidden layer is ReLU.
• The Activation function of Output Layer is ‘sigmoid’
• We used ‘Adam’ optimizer of Keras. We trained it for 300 epochs.
• We got 85.5% of accuracy of churn Prediction.
