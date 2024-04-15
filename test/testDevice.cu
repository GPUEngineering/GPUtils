#include <gtest/gtest.h>
#include "../include/device.cuh"


class DeviceTest : public testing::Test {
protected:
    Context m_context;  ///< Create one context only
    DeviceTest() {}

    virtual ~DeviceTest() {}
};

template<typename T>
void VectorCapacity(Context &context) {
    DeviceVector<T> four(context, 4);
    EXPECT_EQ(4, four.capacity());
    DeviceVector<T> five(context, 0);
    five.allocateOnDevice(5);
    EXPECT_EQ(5, five.capacity());
}

TEST_F(DeviceTest, VectorCapacity) {
    VectorCapacity<float>(m_context);
    VectorCapacity<double>(m_context);
}

template<typename T>
void MatrixDimensions(Context &context) {
    DeviceMatrix<T> fourByThree(context, 4, 3);
    EXPECT_EQ(4, fourByThree.numRows());
    EXPECT_EQ(3, fourByThree.numCols());
}

TEST_F(DeviceTest, MatrixDimensions) {
    MatrixDimensions<float>(m_context);
    MatrixDimensions<double>(m_context);
}

template<typename T>
void Transfer(Context &context) {
    std::vector<T> data = {1, 2, 3, 4, 5, 6};
    std::vector<T> result(data.size());

    DeviceVector<T> vec(context, data.size());
    vec.upload(data);
    vec.download(result);
    EXPECT_EQ(data, result);

    DeviceMatrix<T> mat(context, 3, 2);
    mat.upload(data, MatrixStorageMode::rowMajor);
    mat.asVector().download(result);
    EXPECT_EQ(data, result);
}

TEST_F(DeviceTest, Transfer) {
    Transfer<float>(m_context);
    Transfer<double>(m_context);
}

template<typename T>
void MatrixVectorOperatorAsterisk(Context &context) {
    size_t rows = 4;
    std::vector<T> mat = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<T> vec = {1, 2, 3};
    std::vector<T> result(rows);

    DeviceMatrix<T> d_mat(context, rows, mat, MatrixStorageMode::rowMajor);
    DeviceVector<T> d_vec(context, vec);
    auto d_result = d_mat * d_vec;
    d_result.download(result);
    std::vector<T> expected = {14, 32, 50, 68};
    EXPECT_EQ(expected, result);
}

TEST_F(DeviceTest, MatrixVectorOperatorAsterisk) {
    MatrixVectorOperatorAsterisk<float>(m_context);
    MatrixVectorOperatorAsterisk<double>(m_context);
}

template<typename T>
requires std::floating_point<T>
void singularValues(Context &context) {
    size_t k = 8;
    std::vector<T> bData{1.0, 2.0, 3.0,
                         6.0, 7.0, 8.0,
                         6.0, 7.0, 8.0,
                         6.0, 7.0, 8.0,
                         6.0, 7.0, 8.0,
                         6.0, 7.0, 8.0,
                         6.0, 7.0, 8.0,
                         6.0, 7.0, 8.0,};
    DeviceMatrix<T> B(context, k, bData, rowMajor);
    SvdFactoriser<T> svdEngine(context, B, true, false);
    EXPECT_EQ(0, svdEngine.factorise());
    auto S = svdEngine.singularValues();
    unsigned int r = svdEngine.rank();
    EXPECT_EQ(2, r);

}

TEST_F(DeviceTest, singularValues) {
    singularValues<float>(m_context);
    singularValues<double>(m_context);
}