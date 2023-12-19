#include "LinearRegression.h"
#include <iostream>
#include <cmath>
#include <vector>
using namespace std;


// Mean Squared Error. One of the loss functions
double LossFunction::MSE(int sz,vector<double> pred, double actual[]){

    // m stores number od datapoints
    double m = (double) sz;

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
Takes in x, y and the learning rate and the number of epochs
Formula for Gradient Descent for Linear Regression
*/
pair<float,float> Gradient_Descent(double x[], double y[], int sz, float alpha, int epochs){

    // m stores number of DataPoints
    int m = sz;

    // Setting the main Bias and Weight to 0
    double weights = 0;
    double bias = 0;

    // for loop for number of epochs
    for (int _=0;_<epochs; _++) {

        //  Setting the updated weight and updated bias to 0 after every iteration
        double updated_weights = 0 ; double updated_bias = 0;

        // Loop for adding the derivated error for all the datapoints
        for (int i=0; i<m; i++){

            updated_weights =+ -(2.0/m)*(x[i])*((y[i])-(weights*x[i]+bias));
            updated_bias = updated_bias + (-(2.0/m)*(y[i]-(weights*x[i]+bias)));
        }

        // Updating the weights and biases
        weights = weights - updated_weights*alpha;
        bias = bias - updated_bias*alpha;

    }

    return make_pair(weights, bias);

}




// Takes in the learning rate and number of epochs
// Default epochs = 300
// Default learning_rate = 0.0001
LinearRegressor::LinearRegressor(double learning_rate,int epoch){

    // Assigning the learning rate and epoch to the class public variables
    this->alpha = learning_rate;
    this->epoch = epoch;
}

// Used to train the model and set the weights and bias
// It uses the function Gradient_Descent
void LinearRegressor::train(int sz, double x[], double y[]){

    // Using the function
    pair<float,float> WeightsAndBias = Gradient_Descent(x,y,sz,alpha,epoch);
    
    // Obtaining the optimized weights and bias
    this->weights = WeightsAndBias.first;
    this->bias = WeightsAndBias.second;

}

// This is of course used to predict the set of x values
// As mentioned earlier it uses the function gettingValues 
vector<double> LinearRegressor::predict(int sz, double x[]){

    // Creating a vector to store predicted values
    vector<double> PREDICTED_VALUES;

    int pred_size = sz;

    // Loop for getting all the predicted value individually
    for (int i=0; i<pred_size; i++){


        // Adding the predicted values to the vector
        PREDICTED_VALUES.push_back(gettingValues(x[i]));
    }

    return PREDICTED_VALUES;

}

// Used to predict a single y value for x
// Used in predict function
double LinearRegressor::gettingValues(double x){
    double y = weights*x+bias;
    return y;
}
