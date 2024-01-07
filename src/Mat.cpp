#include "Mat.hpp"
#include <iostream>
#include <assert.h>
#include <math.h>

Mat operator / (Mat m, float f) {
    Mat res(m.m_Mat.size(), m.m_Mat[0].size());
    for(size_t i = 0; i < m.m_Mat.size(); i++) {
        for(size_t j = 0; j < m.m_Mat[0].size(); j++) {
            res[i][j] = m[i][j] / f;
        }
    }
    return res;
}

Mat operator * (float f, Mat m) {
    Mat res(m.m_Mat.size(), m.m_Mat[0].size());
    for(size_t i = 0; i < m.m_Mat.size(); i++) {
        for(size_t j = 0; j < m.m_Mat[0].size(); j++) {
            res[i][j] = f * m[i][j];
        }
    }
    return res;
}

Mat operator - (float f, Mat m) {
    Mat res(m.m_Mat.size(), m.m_Mat[0].size());
    for(size_t i = 0; i < m.m_Mat.size(); i++) {
        for(size_t j = 0; j < m.m_Mat[0].size(); j++) {
            res[i][j] = f - m[i][j];
        }
    }
    return res;
}

// 
Mat::Mat() {
    for(size_t i = 0; i < 0; i++) 
        m_Mat.push_back(std::vector<float>(0, 0.0f));
}

Mat::Mat(const size_t rows, const size_t cols) {
    for(size_t i = 0; i < rows; i++) 
        m_Mat.push_back(std::vector<float>(cols, 0.0f));
}

Mat::Mat(std::initializer_list<std::initializer_list<float>> values) {
    for (const auto& row : values) {
        m_Mat.emplace_back(row);
    }
}

Mat::~Mat() {}

std::vector<float>& Mat::operator[] (const size_t i) {
    return m_Mat[i];
}

const std::vector<float>& Mat::operator[] (const size_t i) const {
    return m_Mat[i];
}

Mat Mat::operator * (const Mat& other) const {
    const size_t aRows = m_Mat.size();
    const size_t aCols = m_Mat[0].size();
    const size_t bRows = other.m_Mat.size();
    const size_t bCols = other.m_Mat[0].size();

    if (aCols != bRows) {
        std::cout << "ERROR: Cannot Multiply: \n";
        this -> Print();
        std::cout << "With: \n";
        other.Print();
        std::cout << "ERROR: Invalid Dimenstions for Matrix dot product\n";
    }

    assert(aCols == bRows && "Matrix dimension are not compatible for dot product\n");
    Mat result(aRows, bCols);

    for (size_t i = 0; i < aRows; i++) {
        for (size_t j = 0; j < bCols; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < aCols; k++) {
                sum += m_Mat[i][k] * other.m_Mat[k][j];
            }
            result[i][j] = sum;
        }
    }

    return result;
}

Mat Mat::operator-(const Mat& other) const {
    size_t rows = m_Mat.size();
    size_t cols = m_Mat[0].size();

    assert(rows == other.m_Mat.size() && cols == other.m_Mat[0].size());

    Mat result(rows, cols);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[i][j] = m_Mat[i][j] - other[i][j];
        }
    }

    return result;
}

Mat Mat::operator-(int n) const {
    size_t rows = m_Mat.size();
    size_t cols = m_Mat[0].size();
    Mat result(rows, cols);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            std::cout << "m_Mat[i][j] - n: " << m_Mat[i][j] << " " "-" " " << n << "\n";
            result[i][j] = m_Mat[i][j] - n;
        }
    }

    return result;
}

Mat Mat::operator+(const Mat& other) const {
    size_t rows = m_Mat.size();
    size_t cols = m_Mat[0].size();

    assert(rows == other.m_Mat.size() && cols == other.m_Mat[0].size());

    Mat result(rows, cols);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[i][j] = m_Mat[i][j] + other[i][j];
        }
    }

    return result;
}

void Mat::Randomize() {
    for(int i = 0; i < m_Mat.size(); i++) {
        for(int j = 0; j < m_Mat[0].size(); j++) {
            m_Mat[i][j] = (float) rand() / RAND_MAX;
        }
    }
}

void Mat::Print() {
    std::cout << "[";
    for(size_t i = 0; i < m_Mat.size(); i++) {
        if (i == 0) std::cout << "[";
        else std::cout << " [";
        for (size_t j = 0; j < m_Mat[0].size(); j++) {
            std::cout << m_Mat[i][j];
            if (j == m_Mat[i].size() - 1) std::cout <<  "";
            else std::cout << ", ";
        }
        if (i == m_Mat.size() - 1) std::cout << "]";
        else std::cout << "]\n";
    }
    std::cout << "]\n";
}

void Mat::Print() const {
    std::cout << "[";
    for(int i = 0; i < m_Mat.size(); i++) {
        std::cout << " ";
        for(int j = 0; j < m_Mat[i].size(); j++) {
            std::cout << m_Mat[i][j] << " ";
        }
        std::cout<<"]\n";
    }
    std::cout << "]\n";
}

float Mat::sigf(float x) {
    return 1.0f / (1.0f + (exp(-x)));
}

void Mat::Activate(const std::string& activationFunction) {
    if(activationFunction == "sigmoid") {
        for(size_t i = 0; i < m_Mat.size(); i++) {
            for(size_t j = 0; j < m_Mat[0].size(); j++) {
                m_Mat[i][j] = sigf(m_Mat[i][j]);
            }
        }
    } 
}

void Mat::Sqr() {
    for(size_t i = 0; i < m_Mat.size(); i++) {
        for(size_t j = 0; j < m_Mat[0].size(); j++) {
            m_Mat[i][j] = m_Mat[i][j] * m_Mat[i][j];
        }
    }
}

Mat Mat::DSig(const Mat& m) {
    Mat result(m.m_Mat.size(), m.m_Mat[0].size());

    for(size_t i = 0; i < m.m_Mat.size(); i++) {
        for(size_t j = 0; j < m.m_Mat[0].size(); j++) {
            // std::cout << "m.m_Mat[i][j]: " << m.m_Mat[i][j] << "\n";
            result.m_Mat[i][j] = m.m_Mat[i][j] * (1 - m.m_Mat[i][j]);
        }
    }

    return result;
}

Mat Mat::HadaMardProduct(const Mat& a, const Mat& b) {
    const size_t aRows = a.m_Mat.size();
    const size_t aCols = a.m_Mat[0].size();
    const size_t bRows = b.m_Mat.size();
    const size_t bCols = b.m_Mat[0].size();

    if (aRows != bRows || aCols != bCols) {
        std::cout << "ERROR: Invalid Matrices for Hadamard Product\n";
        std::cout << "Take Hadamard Product of: \n";
        a.Print();
        std::cout << "With: \n";
        b.Print();
        std::cout << "ERROR: Can't take Hadamard Product\n";
    }
    
    Mat result(aRows, aCols);

    for (size_t i = 0; i < aRows; i++) {
        for(size_t j = 0; j < aCols; j++) {
            result.m_Mat[i][j] = a.m_Mat[i][j] * b.m_Mat[i][j];
        }
    }

    return result;
}

Mat Mat::Transpose(const Mat& m) {
    size_t rows = m.m_Mat.size();
    size_t cols = m.m_Mat[0].size();

    Mat result(cols, rows);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[j][i] = m.m_Mat[i][j];
        }
    }

    return result;
}

Mat Mat::VecToRowMat(const std::vector<float> &v) {
    const size_t n = v.size();
    Mat res(1, n);

    for(size_t i = 0; i < n; i++) res[0][i] = v[i];

    return res;
}