#include <iostream>
#include <vector>
#include <iomanip>

// Including my custom LinearRegression model 
#include "LinearRegression.h"

using namespace std;

// Main Function
int main(){

    // Making an object of Linear Regression model
    LinearRegressor model(0.0001,1000);

    // Making an object of the Loss Function model
    LossFunction LF;

    // Making dummy data for testing.
    // Data is linear here
    vector<vector<double>> x_train {{25, 25}, {50,50}};
    vector<double> y_train {25,50};
    vector<vector<double>> x_test {{40,40},{50,50}};
    vector<double> y_test {40,50};

    // Training the model with .train() method
    model.train(x_train, y_train);

    // Predicting using .predict() method
    vector<double> y_pred;
    vector<double> pred = model.predict(x_test, y_pred);

    // Displaying the model's bias and and weights
    cout<<"Bias: "<<model.bias<<endl<<"Weights: "<<model.weights[0]<<"\t"<<model.weights[1]<<endl<<endl;

    // Printing the predicted value alongside actual value for comparison 
    cout<<"Predicted Value\t\t"<<"Actual Value"<<endl;
    for(int i = 0; i<y_test.size(); i++){
        cout<<pred.at(i)<<"\t\t\t";
        cout<<(y_test.at(i))<<endl;
    }

    // Calculating Loss
    double loss = LF.MSE(pred, y_test);

    // Printing Loss
    cout << fixed << setprecision(9);
    cout<<endl<<"Loss is: "<<loss<<endl;

    return 0;
}
