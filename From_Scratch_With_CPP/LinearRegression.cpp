#include "LinearRegression.h"
#include <iostream>
#include <cmath>
#include <vector>
using namespace std;


// Mean Squared Error. One of the loss functions
double LossFunction::MSE(vector<double> pred, vector<double> actual){

    // m stores number of datapoints
    double m = actual.size();

    // Initializing the loss
    double loss = 0.0;

    // Iterating m times
    for(int i=0; i<m; i++){
        // Adding the squared errors
        loss = pow(loss+(actual[i]-pred[i]),2);
    }

    double FinalLoss = loss/(m);

    // Returning the calculated loss
    return FinalLoss; 

}

/*
Gradient Descent Function.
Used to minimize error and get to the best weights and bias.
Takes in x, y, number of datapoints, number of features and the learning rate and the number of epochs
Formula for Gradient Descent for Linear Regression
*/
pair<vector<double>,double> Gradient_Descent(vector<vector<double>> x, vector<double> y, double alpha, int epochs){

    // m stores number of DataPoints
    int m = x.size();

    // n stores number of features
    int n = x[0].size();

    // Setting the main Bias and Weight to 0
    double bias = 0;

    // weights is a list where there are n number of 0s 
    vector<double> weights(n, 0.0);
    

    // for loop for number of epochs
    for (int _=0;_<epochs; _++) {

        //  Setting the updated weight and updated bias to 0 after every iteration
        double updated_bias = 0;
        vector<double> updated_weights(n, 0.0);
        
        // Loop for adding the derivated error for all the datapoints
        for (int i=0; i<m; i++){

            // Calculating error
            double prediction = bias;
            for (int k=0; k<n; k++){
                prediction += weights[k] * x[i][k];
            }
            double error = y[i] - prediction;

            // Derivating
            for (int k=0; k<n; k++){
                updated_weights[k] += -(2.0/m)*(x[i][k])*error;
            }
            updated_bias = updated_bias + (-(2.0/m)*error);
        }

        // Updating the weights and biases
        for (int i=0; i<n; i++){
            weights[i] = weights[i] - updated_weights[i]*alpha;
        }
        bias = bias - updated_bias*alpha;

    }

    // weights is a vector of doubles and bias just a double
    return make_pair(weights, bias);

}




// Takes in the learning rate and number of epochs
// Default epochs = 1000
// Default learning_rate = 0.0001
LinearRegressor::LinearRegressor(double learning_rate,int epoch){

    // Assigning the learning rate and epoch to the class public variables
    this->alpha = learning_rate;
    this->epoch = epoch;
}

// Used to train the model and set the weights and bias
// It uses the function Gradient_Descent
void LinearRegressor::train(vector<vector<double>> x, vector<double> y){

    int sz = x.size();

    int features = x[0].size(); 

    // Using the function
    pair<vector<double>,double> WeightsAndBias = Gradient_Descent(x,y,alpha,epoch);
    
    // Obtaining the optimized weights and bias
    this->weights = WeightsAndBias.first;
    this->bias = WeightsAndBias.second;

    // Storing number of features
    this->features = features;

}

// This is of course used to predict the set of x values
// As mentioned earlier it uses the function gettingValues 
vector<double> LinearRegressor::predict(vector<vector<double>> x, vector<double> pred){

    int pred_size = x.size();

    // Loop for getting all the predicted value individually
    for (int i=0; i<pred_size; i++){

        // Adding the predicted values to the vector
        pred.push_back(gettingValues(x[i]));
    }
    
    return pred;

}

// Used to predict a single y value for x
// Used in predict function
double LinearRegressor::gettingValues(vector<double> x){

    // Initialized y as bias
    double y = bias;

    // Adding mᵢ*xᵢ till i=number of features
    for (int i = 0; i < features; i++) {
        y += weights[i] * x[i];
    }

    return y;
}
