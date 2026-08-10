#pragma once

#include "sort.hpp"

typedef struct SequentialMove {
    std::vector<std::pair<double,double>> sites_list;
    double distance;
} SequentialMove;

#ifndef COMPILED_AS_EXECUTABLE
std::optional<std::vector<SequentialMove>> sortSequentiallyByRow(
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& stateArray, 
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd);
#endif

bool sortSequentiallyByRowC1D(std::vector<SequentialMove>& ml, size_t rows, size_t cols, bool* stateArray, 
    size_t compZone[4], std::shared_ptr<spdlog::logger> logger);
bool sortSequentiallyByRowC2D(std::vector<SequentialMove>& ml, size_t rows, size_t cols, bool** stateArray, 
    size_t compZone[4], std::shared_ptr<spdlog::logger> logger);