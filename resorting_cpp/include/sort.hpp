#pragma once

#include <Eigen/Dense>
#include <vector>
#ifndef COMPILED_AS_EXECUTABLE
#include <pybind11/pybind11.h>
#include "pybind11/eigen.h"
#endif
#include "spdlog/spdlog.h"
#include <stdexcept>
#include <sstream>

#include "config.hpp"

#ifdef _MSC_VER
#include <intrin.h>
#define popcount_archspec __popcnt
#else
#define popcount_archspec std::__popcount
#endif

#define NUM_THREADS 8

double inline costPerSubMove(double dist)
{
    return dist > DOUBLE_EQUIVALENCE_THRESHOLD ? (Config::getInstance().moveCostOffsetSubmove + 
        Config::getInstance().moveCostScalingLinear * dist + 
        (Config::getInstance().moveCostScalingSqrt != 0 ? Config::getInstance().moveCostScalingSqrt * sqrt(dist) : 0)) : 0;
}

bool inline isInCompZone(int row, int col, size_t compZoneRowStart, 
    size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd)
{
    return row >= (int)compZoneRowStart && row < (int)compZoneRowEnd && 
        col >= (int)compZoneColStart && col < (int)compZoneColEnd;
}

size_t inline roundCoordDown(double coord)
{
    return (size_t)(coord + 0.25);
}

typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> StrideDyn;

class ArrayAccessor
{
public:
    virtual ~ArrayAccessor() = default;
    virtual std::unique_ptr<ArrayAccessor> copy() const = 0;
    virtual void operator=(const ArrayAccessor& other) = 0;
    virtual const bool& operator()(size_t row, size_t col) const = 0;
    virtual bool& operator()(size_t row, size_t col) = 0;
    virtual size_t rows() const = 0;
    virtual size_t cols() const = 0;
};

class RowBitMask
{
public:
    std::vector<size_t> indices;
    size_t count;
private:
    uint64_t *bitMask;
public:
    RowBitMask() : indices(), count(0), bitMask(nullptr) {};
    RowBitMask(size_t count, std::optional<size_t> index = std::nullopt) : indices(), 
        count(count), bitMask(new uint64_t[this->count / 64 + 1]()) 
    {
        if(index.has_value())
        {
            this->indices.push_back(index.value());
        }
    }
    RowBitMask(std::vector<size_t> indices, size_t count, uint64_t *bitMask) : indices(indices), 
        count(count), bitMask(bitMask) {}
    RowBitMask(RowBitMask&& other) : indices(std::exchange(other.indices, std::vector<size_t>())), 
        count(std::exchange(other.count, 0)), 
        bitMask(std::exchange(other.bitMask, nullptr)) {}
    RowBitMask(const RowBitMask& other) : indices(other.indices), count(other.count), 
        bitMask(new uint64_t[this->count / 64 + 1]()) 
    {
        for(size_t i = 0; i <= this->count / 64; i++)
        {
            this->bitMask[i] = other.bitMask[i];
        }
    }
    ~RowBitMask()
    {
        delete[] this->bitMask;
    }
    static RowBitMask fromOr(const RowBitMask& l, const RowBitMask& r)
    {
        size_t c = l.count < r.count ? l.count : r.count;
        uint64_t *bitMask = new uint64_t[c / 64 + 1];
        for(size_t i = 0; i <= c / 64; i++)
        {
            bitMask[i] = l.bitMask[i] | r.bitMask[i];
        }
        std::vector<size_t> indices;
        indices.insert(indices.end(), l.indices.begin(), l.indices.end());
        indices.insert(indices.end(), r.indices.begin(), r.indices.end());
        return RowBitMask(indices, c, bitMask);
    }
    static RowBitMask fromAnd(const RowBitMask& l, const RowBitMask& r)
    {
        size_t c = l.count < r.count ? l.count : r.count;
        uint64_t *bitMask = new uint64_t[c / 64 + 1];
        for(size_t i = 0; i <= c / 64; i++)
        {
            bitMask[i] = l.bitMask[i] & r.bitMask[i];
        }
        std::vector<size_t> indices;
        indices.insert(indices.end(), l.indices.begin(), l.indices.end());
        indices.insert(indices.end(), r.indices.begin(), r.indices.end());
        return RowBitMask(indices, c, bitMask);
    }
    void set(size_t index, bool value)
    {
        if(index < this->count)
        {
            if(value)
            {
                this->bitMask[index / 64] |= UINT64_C(1) << (index % 64);
            }
            else
            {
                this->bitMask[index / 64] &= ~(UINT64_C(1) << (index % 64));
            }
        }
        else
        {
            std::stringstream msg;
            msg << "BitMask index " << index << " out of bounds (size: " << this->count << ") in set";
            throw std::invalid_argument(msg.str());
        }
    }
    bool operator[](size_t index)
    {
        if(index < this->count)
        {
            return this->bitMask[index / 64] & (UINT64_C(1) << (index % 64));
        }
        else
        {
            std::stringstream msg;
            msg << "BitMask index " << index << " out of bounds (size: " << this->count << ") in operator[]";
            throw std::invalid_argument(msg.str());
        }
    }
    const bool operator[](size_t index) const
    {
        if(index < this->count)
        {
            return this->bitMask[index / 64] & (UINT64_C(1) << (index % 64));
        }
        else
        {
            std::stringstream msg;
            msg << "BitMask index " << index << " out of bounds (size: " << this->count << ") in operator[]";
            throw std::invalid_argument(msg.str());
        }
    }
    RowBitMask& operator|=(const RowBitMask& other)
    {
        size_t c = count < other.count ? count : other.count;
        for(size_t i = 0; i <= c / 64; i++)
        {
            this->bitMask[i] = this->bitMask[i] | other.bitMask[i];
        }
        this->indices.insert(this->indices.end(), other.indices.begin(), other.indices.end());
        return *this;
    }
    RowBitMask& operator&=(const RowBitMask& other)
    {
        size_t c = count < other.count ? count : other.count;
        for(size_t i = 0; i <= c / 64; i++)
        {
            this->bitMask[i] = this->bitMask[i] & other.bitMask[i];
        }
        this->indices.insert(this->indices.end(), other.indices.begin(), other.indices.end());
        return *this;
    }
    RowBitMask operator=(const RowBitMask&) = delete;
    RowBitMask& operator=(RowBitMask&& other)
    {
        delete[] this->bitMask;
        this->bitMask = std::exchange(other.bitMask, nullptr);
        this->count = std::exchange(other.count, 0);
        this->indices = std::exchange(other.indices, std::vector<size_t>());

        return *this;
    };
    unsigned int bitsSet() const
    {
        unsigned int popCount = 0;
        for(size_t i = 0; i <= this->count / 64; i++)
        {
            popCount += popcount_archspec(this->bitMask[i]);
        }
        return popCount;
    }
};

class EigenArrayAccessor : public ArrayAccessor
{
private:
    Eigen::Ref<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> arrayData;
    bool *rawData;
    size_t rowStride, colStride;
public:
    EigenArrayAccessor(Eigen::Ref<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> array) : 
        arrayData(array), rawData(array.data()), rowStride(array.outerStride()), colStride(array.innerStride()) {}
    std::unique_ptr<ArrayAccessor> copy() const
    {
        Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> arrayCopy = arrayData;
        return std::unique_ptr<EigenArrayAccessor>(new EigenArrayAccessor(arrayCopy));
    }
    void operator=(const ArrayAccessor& other)
    {
        if(this->rows() != other.rows() || this->cols() != other.cols())
        {
            throw std::invalid_argument("Array cannot be assigned due to different sizes");
        }
        else
        {
            for(size_t r = 0; r < this->rows(); r++)
            {
                for(size_t c = 0; c < this->cols(); c++)
                {
                    this->rawData[r * this->rowStride + c * this->colStride] = other(r,c);
                }
            }
        }
    }
    const bool& operator()(size_t row, size_t col) const
    {
        return this->rawData[row * this->rowStride + col * this->colStride];
    }
    bool& operator()(size_t row, size_t col)
    {
        return this->rawData[row * this->rowStride + col * this->colStride];
    }
    size_t rows() const
    {
        return this->arrayData.rows();
    }
    size_t cols() const
    {
        return this->arrayData.cols();
    }
};

#ifndef COMPILED_AS_EXECUTABLE
namespace py = pybind11;

class PyEigenArrayAccessor : public ArrayAccessor
{
private:
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> arrayData;
    bool *rawData;
    size_t rowStride, colStride;
public:
    PyEigenArrayAccessor(py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> array) : 
        arrayData(array), rawData(array.data()), rowStride(array.outerStride()), colStride(array.innerStride()) {}
    std::unique_ptr<ArrayAccessor> copy() const
    {
        Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> arrayCopy = arrayData;
        return std::unique_ptr<PyEigenArrayAccessor>(new PyEigenArrayAccessor(arrayCopy));
    }
    void operator=(const ArrayAccessor& other)
    {
        if(this->rows() != other.rows() || this->cols() != other.cols())
        {
            throw std::invalid_argument("Array cannot be assigned due to different sizes");
        }
        else
        {
            for(size_t r = 0; r < this->rows(); r++)
            {
                for(size_t c = 0; c < this->cols(); c++)
                {
                    this->rawData[r * this->rowStride + c * this->colStride] = other(r,c);
                }
            }
        }
    }
    const bool& operator()(size_t row, size_t col) const
    {
        return this->rawData[row * this->rowStride + col * this->colStride];
    }
    bool& operator()(size_t row, size_t col)
    {
        return this->rawData[row * this->rowStride + col * this->colStride];
    }
    size_t rows() const
    {
        return this->arrayData.rows();
    }
    size_t cols() const
    {
        return this->arrayData.cols();
    }
};
#endif

class CStyle2DArrayAccessor : public ArrayAccessor
{
private:
    const bool dataOwned;
    bool **arrayData;
    size_t rowCount, colCount;
    CStyle2DArrayAccessor(const CStyle2DArrayAccessor& other) : dataOwned(true), arrayData(nullptr), 
        rowCount(other.rowCount), colCount(other.colCount)
    {
        arrayData = new bool*[rowCount];
        for(size_t i = 0; i < rowCount; i++)
        {
            arrayData[i] = new bool[colCount];
        }
    }
public:
    CStyle2DArrayAccessor(bool **array, size_t rows, size_t cols) : dataOwned(false), 
        arrayData(array), rowCount(rows), colCount(cols) {}
    std::unique_ptr<ArrayAccessor> copy() const
    {
        CStyle2DArrayAccessor *accessorCopy = new CStyle2DArrayAccessor(*this);
        return std::unique_ptr<CStyle2DArrayAccessor>(accessorCopy);
    }
    ~CStyle2DArrayAccessor()
    {
        if(dataOwned)
        {
            for(size_t i = 0; i < rowCount; i++)
            {
                delete[] arrayData[i];
            }
            delete[] arrayData;
        }
    }
    void operator=(const ArrayAccessor& other)
    {
        if(this->rowCount != other.rows() || this->colCount != other.cols())
        {
            throw std::invalid_argument("Array cannot be assigned due to different sizes");
        }
        else
        {
            for(size_t r = 0; r < this->rows(); r++)
            {
                for(size_t c = 0; c < this->cols(); c++)
                {
                    this->arrayData[r][c] = other(r,c);
                }
            }
        }
    }
    const bool& operator()(size_t row, size_t col) const
    {
        return this->arrayData[row][col];
    }
    bool& operator()(size_t row, size_t col)
    {
        return this->arrayData[row][col];
    }
    size_t rows() const
    {
        return this->rowCount;
    }
    size_t cols() const
    {
        return this->colCount;
    }
};

class CStyle1DArrayAccessor : public ArrayAccessor
{
private:
    const bool dataOwned;
    bool *arrayData;
    size_t rowCount, colCount;
    CStyle1DArrayAccessor(const CStyle1DArrayAccessor& other) : dataOwned(true), arrayData(nullptr), rowCount(other.rowCount), colCount(other.colCount)
    {
        arrayData = new bool[rowCount * colCount];
    }
public:
    CStyle1DArrayAccessor(bool *array, size_t rows, size_t cols) : dataOwned(false), arrayData(array), rowCount(rows), colCount(cols) {}
    std::unique_ptr<ArrayAccessor> copy() const
    {
        CStyle1DArrayAccessor *accessorCopy = new CStyle1DArrayAccessor(*this);
        return std::unique_ptr<CStyle1DArrayAccessor>(accessorCopy);
    }
    ~CStyle1DArrayAccessor()
    {
        if(dataOwned)
        {
            delete[] arrayData;
        }
    }
    void operator=(const ArrayAccessor& other)
    {
        if(this->rowCount != other.rows() || this->colCount != other.cols())
        {
            throw std::invalid_argument("Array cannot be assigned due to different sizes");
        }
        else
        {
            for(size_t r = 0; r < this->rows(); r++)
            {
                for(size_t c = 0; c < this->cols(); c++)
                {
                    this->arrayData[r * this->colCount + c] = other(r,c);
                }
            }
        }
    }
    const bool& operator()(size_t row, size_t col) const
    {
        return this->arrayData[row * this->colCount + col];
    }
    bool& operator()(size_t row, size_t col)
    {
        return this->arrayData[row * this->colCount + col];
    }
    size_t rows() const
    {
        return this->rowCount;
    }
    size_t cols() const
    {
        return this->colCount;
    }
};