#include <bits/stdc++.h>
#include "Mat.hpp"

using namespace std;

float _sigf(float x) { return 1.f / (1.f + exp(-x)); }

Mat _sigf(Mat m) {
    Mat res(m.m_Mat.size(), m.m_Mat[0].size());
    for(size_t i = 0; i < m.m_Mat.size(); i++) {
        for(size_t j = 0; j < m.m_Mat[0].size(); j++) {
            res[i][j] = _sigf(m[i][j]);
        }
    }
    return res;
}

vector<vector<float>> inputs = {
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1}
};

vector<float> outputsVec = {0.f, 1.f, 1.f, 0.f};

const size_t N = outputsVec.size();

void buildNN(
    vector<int> &arch, 
    vector<Mat> &weights, 
    vector<Mat> &biases, 
    vector<Mat> &outputs, 
    bool randomize = true) 
{
    int n = arch.size();
    weights.resize(n-1);
    biases.resize(n-1);
    outputs.resize(n-1);

    for(size_t i = 0; i < n - 1; i++) {
        weights[i] = Mat(arch[i], arch[i+1]);
        biases[i]  = Mat(1, arch[i+1]);
        outputs[i] = Mat(1, arch[i+1]);

        if (randomize) {
            weights[i].Randomize();
            biases[i].Randomize();
        }
    }
}

void PrintNN(vector<Mat> &w, vector<Mat> &b, vector<Mat> &o) {
    cout << "NN:\n";
    for(size_t i = 0; i < w.size(); i++) {
        cout << "w: "; w[i].Print();
        cout << "b: "; b[i].Print();
        cout << "O: "; o[i].Print();
        cout << "_________________\n";
    }
    cout << "_______xoxo_______\n";
}

void forward(vector<float> in, vector<Mat> &w, vector<Mat> &b, vector<Mat> &o) {
    Mat inMat = Mat(0,0).VecToRowMat(in);
    for(int i = 0; i < w.size(); i++) {
        o[i] = _sigf(inMat * w[i] + b[i]);
        inMat = o[i];
    }
}

float computeError(vector<Mat> &w, vector<Mat> &b) {
    float Err = 0.f;
    for(size_t i = 0; i < N; i++) {
        vector<float> inVec = inputs[i];
        Mat in = Mat(0,0).VecToRowMat(inVec);
        Mat out;
        for(size_t lyr = 0; lyr < w.size(); lyr++) {
            out = _sigf(in * w[lyr] + b[lyr]);
            in = out;
        }
        float e = outputsVec[i] - out[0][0];
        Err += (e * e);
    }
    Err /= N;
    return Err;
}

void computeAndApplyGradient(vector<int> &arch, vector<Mat> &w, vector<Mat> &b, vector<Mat> &o) {
    vector<Mat> dw, db, _o;
    buildNN(arch, dw, db, _o, false);

    for(size_t i = 0; i < N; i++) {
        vector<float> in = inputs[i];
        forward(in, w, b, o);
        Mat I = Mat(0,0).VecToRowMat(inputs[i]);
        vector<float> out = {outputsVec[i]};
        Mat Yi = Mat(0,0).VecToRowMat(out);
        Mat E = Yi - o[o.size()-1];
        Mat I_T;
        for(int lyr = w.size()-1; lyr >= 0; lyr--) {
            if (lyr == 0) I_T = Mat(0,0).Transpose(I);
            else I_T = Mat(0,0).Transpose(o[lyr-1]);

            Mat yi = o[lyr];
            Mat sigf_prime = (Mat(0,0).HadaMardProduct(yi, (1 - yi)));

            dw[lyr] = dw[lyr] + (I_T * (2 * Mat(0,0).HadaMardProduct((E), (-1 * sigf_prime))));
            db[lyr] = db[lyr] + (1   * (2 * Mat(0,0).HadaMardProduct((E), (-1 * sigf_prime))));

            E = (E * Mat(0,0).Transpose(w[lyr]));
        }
    }

    float lr = 1;
    for(size_t i = 0; i < w.size(); i++) {
        w[i] = w[i] - (lr * dw[i]);
        b[i] = b[i] - (lr * db[i]);
    }
}

int main() {
    srand(time(0)); rand();
    vector<Mat> weights;
    vector<Mat> biases;
    vector<Mat> outputs;

    vector<int> arch = {2,4,3,1};
    buildNN(arch, weights, biases, outputs);

    cout << "Error Before: " << computeError(weights, biases) << endl;

    for (size_t i = 0; i < 1000; i++) 
        computeAndApplyGradient(arch, weights, biases, outputs);

    cout << "Error After: " << computeError(weights, biases) << endl;

    cout << "------------testing------------\n";
    for(size_t i = 0; i < N; i++) {
        forward(inputs[i], weights, biases, outputs);
        float out = outputs[outputs.size()-1][0][0];
        printf("%f | %f = %f\n", inputs[i][0], inputs[i][1], out);
    }
}