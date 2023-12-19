#include <iostream>
#include <vector>

// Including my custom LinearRegression model 
#include "LinearRegression.h"

using namespace std;

// Main Function
int main(){

    // Making an object of Linear Regression model
    LinearRegressor model(0.0001,300);

    // Making an object of the Loss Function model
    LossFunction LF;

    // Making dummy data for testing.
    // Data is linear here and 
    // thus optimum bias is 0 and weight is 1.
    double x_train[6] = {25, 30, 35, 40, 45, 60};
    double y_train[6] = {25, 30, 35, 40, 45, 60};
    double x_test[7] = {24, 55, 78, 67, 89, 90, 100};
    double y_test[7] = {24, 55, 78, 67, 89, 90, 100};

    // Training the model with .train() method
    model.train(6, x_train, y_train);

    // Predicting using .predict() method
    vector<double> pred = model.predict(7, x_test);

    // Displaying the model's bias and and weights
    cout<<"Bias: "<<model.bias<<endl<<"Weights: "<<model.weights<<endl<<endl;

    // Printing the predicted value alongside actual value for comparison 
    cout<<"Predicted Value\t\t"<<"Actual Value"<<endl;
    for(int i = 0; i<7; i++){
        cout<<pred.at(i)<<"\t\t\t";
        cout<<(y_test[i])<<endl;
    }

    // Calculating Loss
    double loss = LF.MSE(7, pred, y_test);

    // Printing Loss
    cout<<"Loss is: "<<loss<<endl;

    return 0;
}
