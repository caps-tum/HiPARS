#include <Eigen/Dense>
#include <iostream>
#include <chrono>
#include <random>
#include <fstream>
#include "../include/sortParallel.hpp"
#include "../include/config.hpp"

const int iterations = 100;
const float sourceFactor = 1.5f;

void test_sortParallel_random()
{
    std::mt19937 rng(12345);
    std::bernoulli_distribution dist(0.5);

    Config::getInstance().readConfig("../../scripts/sortingConfig.cfg");

    std::ofstream outputFile("sorting_results.dat", std::ios_base::openmode::_S_out);
    outputFile << "Qubit\tCTime\tSRate\tAMC\tMCStd\tAMD\tMDStd\tAMT13\tMTStd13\tAMT55\tMTStd55" << std::endl;

    std::vector<size_t> subSizes = {7,11,13,17,19,23,27,31,35,40,45,50,55};
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> stateArrays[iterations];
    std::optional<std::vector<ParallelMove>> sortingResults[iterations];
    for(size_t subSize : subSizes)
    {
        std::cout << "Testing subSize: " << subSize;
        size_t totalSize = (size_t)((float)subSize * sourceFactor + 0.5); // Round up to the nearest integer
        for(int iter = 0; iter < iterations; ++iter)
        {
            stateArrays[iter] = Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Constant(totalSize, totalSize, false);
            for (int i = 0; i < totalSize; ++i)
            {
                for (int j = 0; j < totalSize; ++j)
                {
                    stateArrays[iter](i, j) = dist(rng);
                }
            }
        }

        Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> targetGeometry = 
            Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Constant(subSize, subSize, true);
        size_t firstBorder = (totalSize - subSize) / 2;

        auto startTime = std::chrono::steady_clock::now();
        for(int iter = 0; iter < iterations; ++iter)
        {
            sortingResults[iter] = sortParallelCpp(stateArrays[iter], firstBorder, firstBorder + subSize, firstBorder, firstBorder + subSize, targetGeometry);
        }
        auto endTime = std::chrono::steady_clock::now();
        std::chrono::microseconds elapsedMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        int successfulCount = 0;
        int moveCountSum = 0;
        double moveDistanceSum = 0;
        double moveTime1Sum = 0;
        double moveTime2Sum = 0;
        double moveDistances[iterations], moveTimes1[iterations], moveTimes2[iterations];
        for(int iter = 0; iter < iterations; ++iter)
        {
            moveDistances[iter] = 0;
            moveTimes1[iter] = 0;
            moveTimes2[iter] = 0;
            if(sortingResults[iter].has_value())
            {
                successfulCount++;
                moveCountSum += sortingResults[iter]->size();

                for(const auto& move : *sortingResults[iter])
                {
                    double moveTime0_13 = 120;
                    double moveTime0_55 = 120;
                    for(int stepIndex = 0; stepIndex < move.steps.size() - 1; ++stepIndex)
                    {
                        double maxRowDist = 0;
                        double maxColDist = 0;
                        for(int i = 0; i < move.steps[stepIndex + 1].rowSelection.size(); ++i)
                        {
                            double rowDist = std::abs(move.steps[stepIndex + 1].rowSelection[i] - move.steps[stepIndex].rowSelection[i]);
                            if(rowDist > maxRowDist) maxRowDist = rowDist;
                        }
                        for(int i = 0; i < move.steps[stepIndex + 1].colSelection.size(); ++i)
                        {
                            double colDist = std::abs(move.steps[stepIndex + 1].colSelection[i] - move.steps[stepIndex].colSelection[i]);
                            if(colDist > maxColDist) maxColDist = colDist;
                        }
                        double dist = sqrt(maxRowDist*maxRowDist + maxColDist*maxColDist);
                        moveTime0_13 += (4.49/0.13) * dist;
                        moveTime0_55 += (4.49/0.55) * dist;
                        moveDistances[iter] += dist;
                    }
                    moveTimes1[iter] += moveTime0_13;
                    moveTimes2[iter] += moveTime0_55;
                }

                moveDistanceSum += moveDistances[iter];
                moveTime1Sum += moveTimes1[iter];
                moveTime2Sum += moveTimes2[iter];
            }
        }
        double moveCountAvg = (double)moveCountSum / (double)successfulCount;
        double moveDistanceAvg = moveDistanceSum / (double)successfulCount;
        double moveTime1Avg = moveTime1Sum / (double)successfulCount;
        double moveTime2Avg = moveTime2Sum / (double)successfulCount;
        double moveCountStdev = 0;
        double moveDistanceStdev = 0;
        double moveTime1Stdev = 0;
        double moveTime2Stdev = 0;
        for(int iter = 0; iter < iterations; ++iter)
        {
            if(sortingResults[iter].has_value())
            {
                double moveCountDiff = sortingResults[iter]->size() - moveCountAvg;
                moveCountStdev += moveCountDiff * moveCountDiff;
                double moveDistanceDiff = moveDistances[iter] - moveDistanceAvg;
                moveDistanceStdev += moveDistanceDiff * moveDistanceDiff;
                double moveTime1Diff = moveTimes1[iter] - moveTime1Avg;
                moveTime1Stdev += moveTime1Diff * moveTime1Diff;
                double moveTime2Diff = moveTimes2[iter] - moveTime2Avg;
                moveTime2Stdev += moveTime2Diff * moveTime2Diff;
            }
        }
        moveCountStdev = sqrt(moveCountStdev / (double)successfulCount);
        moveDistanceStdev = sqrt(moveDistanceStdev / (double)successfulCount);
        moveTime1Stdev = sqrt(moveTime1Stdev / (double)successfulCount);
        moveTime2Stdev = sqrt(moveTime2Stdev / (double)successfulCount);

        outputFile << (subSize * subSize) << "\t" << 
            (elapsedMicroseconds.count() / (double)iterations) << "\t" << 
            ((double)successfulCount / (double)iterations) << "\t" <<
            moveCountAvg << "\t" << moveCountStdev << "\t" <<
            moveDistanceAvg << "\t" << moveDistanceStdev << "\t" <<
            moveTime1Avg << "\t" << moveTime1Stdev << "\t" <<
            moveTime2Avg << "\t" << moveTime2Stdev << std::endl;
        std::cout << " >> " << (elapsedMicroseconds.count() / (double)iterations) << std::endl;
    }
    outputFile.close();
}

int main()
{
    test_sortParallel_random();
    return 0;
}
