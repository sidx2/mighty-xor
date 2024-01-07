#pragma once
#include <vector>
#include <string>

class Mat {
    private:

    public:
    std::vector<std::vector<float>> m_Mat;
    Mat();
    Mat(const size_t rows, const size_t cols);
    Mat(std::initializer_list<std::initializer_list<float>> values);

    ~Mat();

    std::vector<float>& operator [] (const size_t i);
    const std::vector<float>& operator [] (const size_t i) const;
    Mat operator*(const Mat& other) const;
    Mat operator-(const Mat& other) const;
    Mat operator-(int n) const;
    // Mat operator-(int n);
    Mat operator+(const Mat& other) const;

    void Randomize();
    void Print();
    void Print() const;

    void Activate(const std::string& activationFunction);
    void Sqr();
    Mat DSig(const Mat& m);
    Mat HadaMardProduct(const Mat& a, const Mat& b);

    Mat Transpose(const Mat& m);
    Mat VecToRowMat(const std::vector<float> &v);

    float sigf(float);
};

Mat operator / (Mat m, float f);
Mat operator * (float f, Mat m);
Mat operator - (float f, Mat m);
