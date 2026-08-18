#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include<algorithm>
#include<cmath>
#include<cstdlib>

using namespace std;

class MultinomialLogisticReg
{
private:
    double lr;
    vector<vector<double>> wt;      //  4 x 3  (4 features, 3 classes)
    vector<vector<double>> b;       //  3 x 1
    int ep;
    int n;                          // number of training rows
    vector<vector<double>> z;
    vector<vector<double>> y_decod;
    vector<int> output;
public:
    MultinomialLogisticReg(double learning_rate, vector<vector<double>> weight, vector<vector<double>> bias, int epochs, int num_samples)
    {
        lr=learning_rate;
        n=num_samples;
        wt.resize(4, vector<double>(3, 0.0));
        b.resize(3, vector<double>(1, 0.0));
        z.resize(n, vector<double>(3, 0.0));
        y_decod.resize(n, vector<double>(3, 0.0));
        output.resize(n, 0);

        for(int i=0;i<4;i++)
        {
            for(int j=0;j<3;j++)
            {
                wt[i][j]=weight[i][j];
            }
        }
        for(int i=0;i<3;i++)
        {
            for(int j=0;j<1;j++)
            {
                b[i][j]=bias[i][j];
            }
        }
        ep=epochs;
    }

    void softmax()
    {
        double sum;
        for(int i=0;i<n;i++)
        {
            double maxVal=z[i][0];
            for(int j=1;j<3;j++)
            {
                if(z[i][j]>maxVal)
                {
                    maxVal=z[i][j];//finding max in row
                }
            }

            sum=0;
            for(int j=0;j<3;j++)
            {
                z[i][j]=exp(z[i][j]-maxVal);//subtracting so that the number wont be to big to compute and also no net effect
                sum += z[i][j];
            }
            for(int j=0;j<3;j++)
            {
                z[i][j]=z[i][j]/sum;
            }
        }
    }

    void Y_OHE(vector<double> &y_train)
    {
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<3;j++)
            {
                if(y_train[i]==j)
                {
                    y_decod[i][j]=1;
                }
                else
                {
                    y_decod[i][j]=0;
                }
            }
        }
    }

    void y_pred()
    {
        for(int i=0;i<n;i++)
        {
            int maxIndex=0;
            for(int j=1;j<3;j++)
            {
                if(z[i][j]>z[i][maxIndex])
                {
                    maxIndex=j;
                }
            }
            output[i]=maxIndex;   // output is now a plain vector<int>, one label per row
        }
    }

    void operation(vector<vector<double>> &X_train)
    {

        for(int i=0;i<n;i++)
        {
            for(int j=0;j<3;j++)
            {
                z[i][j]=0;
            }
        }

        for(int i=0;i<n;i++)
        {
            for(int j=0;j<3;j++)
            {
                for(int k=0;k<4;k++)
                {
                    z[i][j] += X_train[i][k]*wt[k][j];
                }
            }
        }
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<3;j++)
            {
                z[i][j] += b[j][0];
            }
        }
    }

    double crossEntropyLoss()
    {
        double loss=0;
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<3;j++)
            {
                if(y_decod[i][j]==1)
                {
                    loss += -log(z[i][j]+1e-9);   //to avoid log(0)
                }
            }
        }
        return loss/n;
    }

    void backward(vector<vector<double>> &X_train)
    {
        vector<vector<double>> dz(n, vector<double>(3, 0.0));
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<3;j++)
            {
                dz[i][j]=z[i][j]-y_decod[i][j];   // softmax + cross entropy gradient simplifies to this
            }
        }

        vector<vector<double>> dwt(4, vector<double>(3, 0.0));
        for(int k=0;k<4;k++)
        {
            for(int j=0;j<3;j++)
            {
                double s=0;
                for(int i=0;i<n;i++)
                {
                    s += X_train[i][k]*dz[i][j];
                }
                dwt[k][j]=s/n;
            }
        }

        vector<double> db(3, 0.0);
        for(int j=0;j<3;j++)
        {
            double s=0;
            for(int i=0;i<n;i++)
            {
                s += dz[i][j];
            }
            db[j]=s/n;
        }

        for(int k=0;k<4;k++)
        {
            for(int j=0;j<3;j++)
            {
                wt[k][j] -= lr*dwt[k][j];
            }
        }
        for(int j=0;j<3;j++)
        {
            b[j][0] -= lr*db[j];
        }
    }

    void train(vector<vector<double>> &X_train, vector<double> &y_train)
    {
        Y_OHE(y_train);

        for(int e=0;e<ep;e++)
        {
            operation(X_train);
            softmax();
            backward(X_train);

            if(e%200==0)
            {
                cout << "epoch " << e << " -> loss: " << crossEntropyLoss() << endl;
            }
        }

        operation(X_train);
        softmax();
        y_pred();
    }

    // runs the trained weights on data the model never saw during training
    vector<int> predict(vector<vector<double>> &X_test)
    {
        int m=X_test.size();
        vector<vector<double>> z_test(m, vector<double>(3, 0.0));

        for(int i=0;i<m;i++)
        {
            for(int j=0;j<3;j++)
            {
                for(int k=0;k<4;k++)
                {
                    z_test[i][j] += X_test[i][k]*wt[k][j];
                }
                z_test[i][j] += b[j][0];
            }
        }

        for(int i=0;i<m;i++)
        {
            double maxVal=z_test[i][0];
            for(int j=1;j<3;j++)
            {
                if(z_test[i][j]>maxVal)
                {
                    maxVal=z_test[i][j];
                }
            }
            double sum=0;
            for(int j=0;j<3;j++)
            {
                z_test[i][j]=exp(z_test[i][j]-maxVal);
                sum += z_test[i][j];
            }
            for(int j=0;j<3;j++)
            {
                z_test[i][j]=z_test[i][j]/sum;
            }
        }

        vector<int> preds(m);
        for(int i=0;i<m;i++)
        {
            int maxIndex=0;
            for(int j=1;j<3;j++)
            {
                if(z_test[i][j]>z_test[i][maxIndex])
                {
                    maxIndex=j;
                }
            }
            preds[i]=maxIndex;
        }
        return preds;
    }

    vector<int> getOutput()
    {
        return output;
    }
};

bool loadCSV(const string &filename, vector<vector<double>> &X, vector<double> &y)
{
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Error: Could not open file '" << filename << "'!" << endl;
        cerr << "Make sure " << filename << " is in your Code::Blocks project folder." << endl;
        return false;
    }

    string line;

    while (getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        if (line.empty()) continue;

        stringstream ss(line);
        string cell;
        vector<double> row;
        bool isHeader = false;

        while (getline(ss, cell, ',')) {
            if (cell.empty()) continue;

            try {
                row.push_back(stod(cell));
            } catch (const invalid_argument&) {
                isHeader = true;
                break;
            }
        }

        if (isHeader || row.empty()) {
            continue;
        }

        y.push_back(row.back());
        row.pop_back();
        X.push_back(row);
    }

    file.close();

    if (X.empty()) {
        cerr << "Error: No numeric data rows were loaded from " << filename << "!" << endl;
        return false;
    }

    return true;
}

double accuracy(vector<int> &preds, vector<double> &y_true)
{
    int correct=0;
    for(int i=0;i<(int)preds.size();i++)
    {
        if(preds[i]==(int)y_true[i])
        {
            correct++;
        }
    }
    return (double)correct/preds.size()*100.0;
}

int main() {
    vector<vector<double>> X_train, X_test;
    vector<double> y_train, y_test;

    if (!loadCSV("iris_train.csv", X_train, y_train)) return 1;
    if (!loadCSV("iris_test.csv", X_test, y_test)) return 1;


    cout << "Loaded " << X_train.size() << " training rows successfully!" << endl;
    cout << "Loaded " << X_test.size() << " testing rows successfully!" << endl;
    cout << "Each row has " << X_train[0].size() << " features." << endl;
    cout << "First row features: " << X_train[0][0] << ", " << X_train[0][1] << endl;
    cout << "First row target label: " << y_train[0] << endl;
    cout << endl;

    // small random starting weights instead of zeros, zeros work here too since
    // the classes are symmetric but small random values is closer to how you'd
    // normally set this up
    srand(42);
    vector<vector<double>> weight(4, vector<double>(3));
    for(int i=0;i<4;i++)
    {
        for(int j=0;j<3;j++)
        {
            weight[i][j] = ((double)rand()/RAND_MAX - 0.5) * 0.1;
        }
    }
    vector<vector<double>> bias(3, vector<double>(1, 0.0));

    double learning_rate = 0.5;
    int epochs = 8300;

    MultinomialLogisticReg model(learning_rate, weight, bias, epochs, X_train.size());

    cout << "Training..." << endl;
    model.train(X_train, y_train);

    vector<int> train_preds = model.getOutput();
    double train_acc = accuracy(train_preds, y_train);
    cout << "\nTraining Accuracy: " << train_acc << " %" << endl;

    vector<int> test_preds = model.predict(X_test);
    double test_acc = accuracy(test_preds, y_test);
    cout << "Testing Accuracy: " << test_acc << " %" << endl;

    return 0;
}
