#include <Eigen/Dense>
#include <iostream>
#include <chrono>
#include <random>
#include <fstream>
#include "../include/sortParallel.hpp"
#include "../include/config.hpp"

const int iterations = 1000;

enum Pattern
{
    FULL, LIEB, CHECKERBOARD
};

void test_sortParallel_random()
{
    std::mt19937 rng(12345);
    Config::getInstance().readConfig("../../scripts/sortingConfig.cfg");
    std::string filename;
    std::vector<size_t> subSizes = {7,11,13,17,19,23,27,31,35,40,45,50,55,60};
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> stateArrays[iterations];
    ArrayAccessor *arrayAccessors[iterations];
    std::optional<std::vector<ParallelMove>> sortingResults[iterations];
    float sourceFactor = 1.5f;

    for(int testIteration = 0; testIteration < 12; testIteration++)
    {
        std::bernoulli_distribution dist(0.5);
        Pattern occupation = FULL;

        std::cout << "Test iteration " << testIteration << std::endl;
        switch(testIteration)
        {
            case 0:
                filename = "warmup.dat";
                Config::getInstance().moveCostOffset = 120;
                Config::getInstance().runTimeFocus = std::nullopt;
                break;
            case 1:
                Config::getInstance().moveCostScalingLinear = 4.49/0.13;
                filename = "sorting_results_0_13.dat";
                break;
            case 2:
                Config::getInstance().moveCostScalingLinear = 4.49/50.f;
                filename = "sorting_results_50.dat";
                break;
            case 3:
                Config::getInstance().moveCostScalingLinear = 4.49/0.55;
                filename = "sorting_results_0_55.dat";
                break;
            case 4:
                filename = "sorting_results_0_55_8x8.dat";
                Config::getInstance().aodRowLimit = 8;
                Config::getInstance().aodColLimit = 8;
                break;
            case 5:
                filename = "sorting_results_0_55_t16.dat";
                Config::getInstance().aodRowLimit = 16;
                Config::getInstance().aodColLimit = 16;
                Config::getInstance().aodTotalLimit = 16;
                break;
            case 6:
                filename = "sorting_results_0_55_RTF0.dat";
                Config::getInstance().aodTotalLimit = 256;
                Config::getInstance().runTimeFocus = 0;
                break;
            case 7:
                filename = "sorting_results_0_55_RTF10.dat";
                Config::getInstance().runTimeFocus = 10;
                break;
            case 8:
                filename = "sorting_results_0_55_f0_75.dat";
                Config::getInstance().runTimeFocus = std::nullopt;
                dist = std::bernoulli_distribution(0.75);
                break;
            case 9:
                filename = "sorting_results_0_55_r2.dat";
                sourceFactor = 2;
                break;
            case 10:
                // Lieb Lattice
                filename = "sorting_results_0_55_lieb.dat";
                occupation = LIEB;
                break;
            case 11:
                // Checkerboard Lattice
                filename = "sorting_results_0_55_checkerboard.dat";
                occupation = CHECKERBOARD;
                break;
        }

        std::ofstream outputFile(filename, std::ios_base::openmode::_S_out);
        outputFile << "Qubit\tCTime\tSRate\tAMC\tMCStd\tAMD\tMDStd\tAMT13\tMTStd13\tAMT55\tMTStd55" << std::endl;

        for(size_t subSize : subSizes)
        {
            std::cout << "Testing subSize: " << subSize;
            size_t totalSize = (size_t)((float)subSize * sourceFactor + 0.5); // Round up to the nearest integer
            for(int iter = 0; iter < iterations; ++iter)
            {
                stateArrays[iter] = Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Constant(totalSize, totalSize, false);
                for (size_t i = 0; i < totalSize; ++i)
                {
                    for (size_t j = 0; j < totalSize; ++j)
                    {
                        stateArrays[iter](i, j) = dist(rng);
                    }
                }
                arrayAccessors[iter] = new EigenArrayAccessor(stateArrays[iter]);
            }

            Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> targetGeometry = 
                Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Constant(subSize, subSize, true);
            if(occupation == Pattern::LIEB)
            {
                for (size_t i = 0; i < subSize; i += 2)
                {
                    for (size_t j = 0; j < subSize; j += 2)
                    {
                        targetGeometry(i, j) = false;
                    }
                }
            }
            else if(occupation == Pattern::CHECKERBOARD)
            {
                for (size_t i = 0; i < subSize; i++)
                {
                    for (size_t j = 0; j < subSize; j++)
                    {
                        targetGeometry(i, j) = ((i + j) % 2 == 0);
                    }
                }
            }
            EigenArrayAccessor targetGeometryAccessor(targetGeometry);
            size_t firstBorder = (totalSize - subSize) / 2;

            auto startTime = std::chrono::steady_clock::now();
            for(int iter = 0; iter < iterations; ++iter)
            {
                sortingResults[iter] = sortParallelCpp(*arrayAccessors[iter], firstBorder, 
                    firstBorder + subSize, firstBorder, firstBorder + subSize, targetGeometryAccessor);
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
                        for(size_t stepIndex = 0; stepIndex < move.steps.size() - 1; ++stepIndex)
                        {
                            double maxRowDist = 0;
                            double maxColDist = 0;
                            for(size_t i = 0; i < move.steps[stepIndex + 1].rowSelection.size(); ++i)
                            {
                                double rowDist = std::abs(move.steps[stepIndex + 1].rowSelection[i] - move.steps[stepIndex].rowSelection[i]);
                                if(rowDist > maxRowDist) maxRowDist = rowDist;
                            }
                            for(size_t i = 0; i < move.steps[stepIndex + 1].colSelection.size(); ++i)
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

            for(int iter = 0; iter < iterations; ++iter)
            {
                delete arrayAccessors[iter];
            }
        }
        outputFile.close();
    }
}

int main()
{
    test_sortParallel_random();
    return 0;
}
