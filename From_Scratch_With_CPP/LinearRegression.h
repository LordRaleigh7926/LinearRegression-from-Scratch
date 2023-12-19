#include <vector>
#include <iostream>
using namespace std;

// A class for Loss functions
class LossFunction{
    public:
        double MSE(int sz,vector<double> pred, double actual[]);
};

pair<float,float> Gradient_Descent(double x[], double y[], int sz, float alpha=0.0001, int epochs=300);

// The Linear Regressor
class LinearRegressor {  
    public:
        int epoch = 0; 
        double alpha = 0;
        double weights = 0, bias = 0;

        LinearRegressor(double learning_rate=0.0001,int epoch=300);
        void train(int sz, double x[], double y[]);
        vector<double> predict(int sz, double x[]);

    private:
        double gettingValues(double x);
};

