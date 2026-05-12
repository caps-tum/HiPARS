#pragma once

#define _USE_MATH_DEFINES
#include <cmath>

#include "sort.hpp"

#define MAX_MULTI_ITER_COUNT 1000
#define CUTOFF_OVERLAP_FRACTION 0.7
#define MAX_INVESTIGATED_OVERLAP_SELECTIONS 10
#define BENEFIT_FRACTION_TO_ALSO_EXECUTE 1

#define M_4TH_ROOT_2 1.1892071150027210667
#define M_4TH_ROOT_1_2 1 / M_4TH_ROOT_2
/*#define HALF_STEP_COST (MOVE_COST_OFFSET_SUBMOVE + MOVE_COST_SCALING_SQRT * M_SQRT1_2 + MOVE_COST_SCALING_LINEAR / 2)
#define HALF_DIAG_STEP_COST (MOVE_COST_OFFSET_SUBMOVE + MOVE_COST_SCALING_SQRT * M_4TH_ROOT_1_2 + MOVE_COST_SCALING_LINEAR * M_SQRT1_2)
#define DIAG_STEP_COST (MOVE_COST_OFFSET_SUBMOVE + MOVE_COST_SCALING_SQRT * M_4TH_ROOT_2 + MOVE_COST_SCALING_LINEAR * M_SQRT2)*/

class ParallelMove
{
public:
    struct Step
    {
        std::vector<double> colSelection;
        std::vector<double> rowSelection;
    };
    std::vector<Step> steps;
    ParallelMove() : steps() {};
    static ParallelMove fromStartAndEnd(
        ArrayAccessor& stateArray, ParallelMove::Step start, ParallelMove::Step end, std::shared_ptr<spdlog::logger> logger);
    double cost() const;
    bool execute(ArrayAccessor& stateArray, std::shared_ptr<spdlog::logger> logger,
        std::optional<py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>> alreadyMoved = std::nullopt, 
        double minDist = DOUBLE_EQUIVALENCE_THRESHOLD) const;
    bool extendToUseAllTones(unsigned int stateArrayRows, unsigned int stateArrayCols, std::shared_ptr<spdlog::logger> logger, bool considerSpacing,
        std::optional<std::vector<int>> bufferRows = std::nullopt, std::optional<std::vector<int>> bufferCols = std::nullopt);
};

std::optional<std::vector<ParallelMove>> sortParallel(
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& stateArray, 
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd, 
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &targetGeometry);