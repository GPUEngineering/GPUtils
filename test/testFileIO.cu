#include <gtest/gtest.h>
#include "tensor.cuh"
#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>

namespace {
class TemporaryIODirectory {
    std::filesystem::path m_path;
public:
    TemporaryIODirectory() {
        static std::atomic<unsigned long> counter{0};
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        for (int attempt = 0; attempt < 100; ++attempt) {
            auto path = std::filesystem::temp_directory_path()
                / ("gputils-io-" + std::to_string(stamp) + "-" + std::to_string(counter++));
            if (std::filesystem::create_directory(path)) { m_path = path; return; }
        }
        throw std::runtime_error("cannot create unique test directory");
    }
    ~TemporaryIODirectory() {
        std::error_code error;
        std::filesystem::remove_all(m_path, error);
    }
    std::filesystem::path file(const std::string &name) const { return m_path / name; }
};

template<typename T>
void appendNative(std::vector<char> &bytes, T value) {
    const auto *begin = reinterpret_cast<const char *>(&value);
    bytes.insert(bytes.end(), begin, begin + sizeof(T));
}

void writeBytes(const std::filesystem::path &path, const std::vector<char> &bytes) {
    std::ofstream output(path, std::ios::binary);
    ASSERT_TRUE(output.is_open());
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    output.close();
    ASSERT_TRUE(output.good());
}

template<typename T>
void roundTripShapes() {
    TemporaryIODirectory directory;
    const std::array<std::array<size_t, 3>, 5> shapes{{{1,1,1}, {5,1,1}, {1,5,1}, {2,3,1}, {2,3,2}}};
    for (const auto &shape : shapes) {
        std::vector<T> expected(shape[0] * shape[1] * shape[2]);
        for (size_t i = 0; i < expected.size(); ++i) expected[i] = T(int(i) - 5);
        DTensor<T> original(expected, shape[0], shape[1], shape[2]);
        for (const auto *name : {"tensor.bt", "tensor.dtensor"}) {
            SCOPED_TRACE(name);
            const auto path = directory.file(name).string();
            original.saveToFile(path);
            auto actual = DTensor<T>::parseFromFile(path);
            EXPECT_EQ(actual.numRows(), shape[0]);
            EXPECT_EQ(actual.numCols(), shape[1]);
            EXPECT_EQ(actual.numMats(), shape[2]);
            std::vector<T> values;
            actual.download(values);
            EXPECT_EQ(values, expected);
        }
    }
}

template<typename T>
void textPreservesPrecision() {
    TemporaryIODirectory directory;
    const auto path = directory.file("precision.dtensor").string();
    const std::vector<T> expected{T(0), T(-1), T(1) / T(3), std::nextafter(T(1), T(2)),
        std::numeric_limits<T>::min(), -std::numeric_limits<T>::min(),
        std::numeric_limits<T>::max(), -std::numeric_limits<T>::max()};
    DTensor<T> tensor(expected, 2, 4);
    tensor.saveToFile(path);
    auto actual = DTensor<T>::parseFromFile(path);
    std::vector<T> values;
    actual.download(values);
    EXPECT_EQ(values, expected);
}

template<typename T>
void binaryMatchesIndependentLayout() {
    TemporaryIODirectory directory;
    const auto path = directory.file("layout.bt");
    std::vector<char> expected;
    appendNative<uint64_t>(expected, 2);
    appendNative<uint64_t>(expected, 3);
    appendNative<uint64_t>(expected, 2);
    // A pair of matrices [[1,2,3],[4,5,6]] and its negative, stored column-major.
    const std::vector<T> values{1,4,2,5,3,6,-1,-4,-2,-5,-3,-6};
    for (T value : values) appendNative(expected, value);
    DTensor<T> tensor(values, 2, 3, 2);
    tensor.saveToFile(path.string());
    std::ifstream input(path, std::ios::binary);
    const std::vector<char> actual{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    EXPECT_EQ(actual, expected);
    // The reader is also tested directly against bytes independent of the writer.
    writeBytes(path, expected);
    auto parsed = DTensor<T>::parseFromFile(path.string());
    EXPECT_EQ(parsed(0, 1, 0), T(2));
    EXPECT_EQ(parsed(1, 2, 1), T(-6));
    std::vector<T> downloaded;
    parsed.download(downloaded);
    EXPECT_EQ(downloaded, values);
}
} // namespace

TEST(FileIOTest, shapesRoundTripFloat) { roundTripShapes<float>(); }
TEST(FileIOTest, shapesRoundTripDouble) { roundTripShapes<double>(); }
TEST(FileIOTest, shapesRoundTripInt) { roundTripShapes<int>(); }
TEST(FileIOTest, textPreservesFloatPrecision) { textPreservesPrecision<float>(); }
TEST(FileIOTest, textPreservesDoublePrecision) { textPreservesPrecision<double>(); }
TEST(FileIOTest, binaryLayoutFloat) { binaryMatchesIndependentLayout<float>(); }
TEST(FileIOTest, binaryLayoutDouble) { binaryMatchesIndependentLayout<double>(); }

TEST(FileIOTest, textPreservesIntegerLimits) {
    TemporaryIODirectory directory;
    const auto path = directory.file("integers.dtensor").string();
    const std::vector<int> expected{std::numeric_limits<int>::min(), -12345, 0, std::numeric_limits<int>::max()};
    DTensor<int> tensor(expected, 2, 2);
    tensor.saveToFile(path);
    auto parsed = DTensor<int>::parseFromFile(path);
    std::vector<int> actual;
    parsed.download(actual);
    EXPECT_EQ(actual, expected);
}

TEST(FileIOTest, rejectsEveryTruncatedBinaryHeaderLength) {
    TemporaryIODirectory directory;
    const auto path = directory.file("header.bt");
    std::vector<char> header;
    for (uint64_t dimension : {2, 3, 1}) appendNative(header, dimension);
    for (size_t length = 0; length < header.size(); ++length) {
        SCOPED_TRACE(length);
        writeBytes(path, std::vector<char>(header.begin(), header.begin() + length));
        EXPECT_THROW(DTensor<double>::parseFromFile(path.string()), std::invalid_argument);
    }
}

TEST(FileIOTest, rejectsInvalidBinaryDimensionsAndExtraBytes) {
    TemporaryIODirectory directory;
    const auto path = directory.file("invalid.bt");
    const std::array<std::array<uint64_t, 3>, 5> shapes{{{0,1,1}, {2,0,1}, {2,3,0},
        {uint64_t(INT_MAX) + 1,1,1}, {UINT64_MAX,UINT64_MAX,2}}};
    const size_t before = Session::getInstance().totalAllocatedBytes();
    for (const auto &shape : shapes) {
        std::vector<char> bytes;
        for (auto dimension : shape) appendNative(bytes, dimension);
        writeBytes(path, bytes);
        EXPECT_THROW(DTensor<double>::parseFromFile(path.string()), std::invalid_argument);
    }
    std::vector<char> bytes;
    for (uint64_t dimension : {2, 3, 1}) appendNative(bytes, dimension);
    for (double value : {1.,4.,2.,5.,3.,6.}) appendNative(bytes, value);
    bytes.push_back('\xff');
    writeBytes(path, bytes);
    EXPECT_THROW(DTensor<double>::parseFromFile(path.string()), std::invalid_argument);
    EXPECT_EQ(Session::getInstance().totalAllocatedBytes(), before);
}

TEST(FileIOTest, reportsOutputPathFailures) {
    TemporaryIODirectory directory;
    DTensor<double> tensor(std::vector<double>{1,2,3,4}, 2, 2);
    for (const auto *name : {"tensor.bt", "tensor.dtensor"}) {
        const auto missingParent = directory.file("missing") / name;
        EXPECT_THROW(tensor.saveToFile(missingParent.string()), std::runtime_error);
        const auto folder = directory.file(name);
        ASSERT_TRUE(std::filesystem::create_directory(folder));
        EXPECT_THROW(tensor.saveToFile(folder.string()), std::runtime_error);
    }
}

TEST(FileIOTest, overwriteTruncatesOldPayload) {
    TemporaryIODirectory directory;
    for (const auto *name : {"tensor.bt", "tensor.dtensor"}) {
        const auto path = directory.file(name).string();
        DTensor<double> large(std::vector<double>(100, 7), 10, 10);
        large.saveToFile(path);
        DTensor<double> small(std::vector<double>{1,2}, 2);
        small.saveToFile(path);
        auto parsed = DTensor<double>::parseFromFile(path);
        EXPECT_EQ(parsed.numRows(), 2u);
        EXPECT_EQ(parsed.numCols(), 1u);
        EXPECT_EQ(parsed.numMats(), 1u);
        std::vector<double> actual;
        parsed.download(actual);
        EXPECT_EQ(actual, (std::vector<double>{1,2}));
    }
}
