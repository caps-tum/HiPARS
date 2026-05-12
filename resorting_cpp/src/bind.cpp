#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/attr.h>
#include "sortSequentiallyByRow.hpp"
#include "sortParallel.hpp"
#include "sortLattice.hpp"
#include "config.hpp"

namespace py = pybind11;

PYBIND11_MODULE(resorting_cpp, m) {
    m.doc() = "Underlying C++ library for sorting neutral atoms\n";

    py::class_<ParallelMove>(m, "ParallelMove", R"pbdoc(
    A class holding information about a parallel move.
)pbdoc")
    .def(py::init())
    .def_readwrite("steps", &ParallelMove::steps, R"pbdoc(
    :list[:class:`.ParallelMoveStep`]: Individual steps of the move
)pbdoc");

    py::class_<ParallelMove::Step>(m, "ParallelMoveStep", R"pbdoc(
    A class holding information about a parallel move step.
)pbdoc")
    .def(py::init())
    .def_readwrite("colSelection", &ParallelMove::Step::colSelection, R"pbdoc(
    :list[double]: List of column tones to activate at this step
)pbdoc")
    .def_readwrite("rowSelection", &ParallelMove::Step::rowSelection, R"pbdoc(
    :list[double]: List of column tones to activate at this step
)pbdoc");

    py::class_<SequentialMove>(m, "SequentialMove", R"pbdoc(
    A class holding information about a sequential move.
)pbdoc")
    .def(py::init())
    .def_readwrite("sites_list", &SequentialMove::sites_list, R"pbdoc(
    :list[(double,double)]: Coordinates of movable tweezer at each step
)pbdoc")
    .def_readwrite("distance", &SequentialMove::distance, R"pbdoc(
    :double: Total move length
)pbdoc");

#if defined(PYBIND11_HAS_NATIVE_ENUM)
#include <pybind11/native_enum.h>
    py::native_enum<TargetState>(m, "TargetState", "enum.Enum")
    .value("Empty", TargetState::EMPTY)
    .value("Occupied", TargetState::OCCUPIED)
    .value("Irrelevant", TargetState::IRRELEVANT)
    .export_values()
    .finalize();
#else
    py::enum_<TargetState>(m, "TargetState")
    .value("Empty", TargetState::EMPTY)
    .value("Occupied", TargetState::OCCUPIED)
    .value("Irrelevant", TargetState::IRRELEVANT)
    .export_values();
#endif

    py::class_<Config, std::unique_ptr<Config, py::nodelete>>(m, "Config", R"pbdoc(
    A class for configuring the sorting algorithm. At the moment, this is only used for configuring the logger.
)pbdoc")
    .def(py::init<>([]{return &Config::getInstance();}))
    .def("readConfig", &Config::readConfig, "A function that allows for reading in a provided config file", py::arg("filePath"), R"pbdoc(
    A function that allows for reading in a provided config file

    :param filePath: The path of the provided config file
    :type filePath: string
    :return: Whether reading was successful.
    :rtype: bool
)pbdoc")
    .def_readwrite("logFileName", &Config::logFileName, R"pbdoc(
    :string: Name of the log file
)pbdoc")
    .def_readwrite("sequentialLoggerName", &Config::sequentialLoggerName, R"pbdoc(
    :string: Name of the sequential logger
)pbdoc")
    .def_readwrite("parallelLoggerName", &Config::parallelLoggerName, R"pbdoc(
    :string: Name of the parallel logger
)pbdoc")
    .def_readwrite("greedyLatticeLoggerName", &Config::greedyLatticeLoggerName, R"pbdoc(
    :string: Name of the greedy-lattice logger
)pbdoc")
    .def_readwrite("latticeByRowLoggerName", &Config::latticeByRowLoggerName, R"pbdoc(
    :string: Name of the by-row lattice logger
)pbdoc")
    .def_readwrite("logLevel", &Config::logLevel, R"pbdoc(
    :string: String representation of log level ("trace", "debug", "info", "warning"/"warn", "error"/"err", "critical", "off"). Defaults to "off" if not recognised.
)pbdoc")
    .def_readwrite("rowSpacing", &Config::rowSpacing, R"pbdoc(
    :double: Physical spacing between rows. Only relevant for lattice algorithms
)pbdoc")
    .def_readwrite("columnSpacing", &Config::columnSpacing, R"pbdoc(
    :double: Physical spacing between columns. Only relevant for lattice algorithms
)pbdoc")
    .def_readwrite("minAodSpacing", &Config::minAodSpacing, R"pbdoc(
    :double: Minimal physical spacing between generated tones. Only relevant for sortLatticeByRowParallel
)pbdoc")
    .def_readwrite("allowMovingEmptyTrapOntoOccupied", &Config::allowMovingEmptyTrapOntoOccupied, R"pbdoc(
    :bool: Whether it is allowed to move empty traps onto occupied ones. Only relevant for sortParallel
)pbdoc")
    .def_readwrite("allowDiagonalMovement", &Config::allowDiagonalMovement, R"pbdoc(
    :bool: Whether diagonal movement is allowed. Only relevant for sortParallel
)pbdoc")
    .def_readwrite("allowMovesBetweenRows", &Config::allowMovesBetweenRows, R"pbdoc(
    :bool: Whether moves between rows are always allowed. Only relevant for sortParallel
)pbdoc")
    .def_readwrite("allowMovesBetweenCols", &Config::allowMovesBetweenCols, R"pbdoc(
    :bool: Whether moves between columns are always allowed. Only relevant for sortParallel
)pbdoc")
    .def_readwrite("allowMultipleMovesPerAtom", &Config::allowMultipleMovesPerAtom, R"pbdoc(
    :bool: Whether an atom may be moved multiple times. Only relevant for sortParallel and sortLatticeGreedyParallel
)pbdoc")
    .def_readwrite("alwaysGenerateAllAODTones", &Config::alwaysGenerateAllAODTones, R"pbdoc(
    :bool: Whether all allowed tones should always be generated. If true, moves are extended to always use all tones. 
    Only works if aodTotalLimit >= aodRowLimit * aodColLimit. 
    May add negative tones or ones greater than the array size for sortParallel. Only relevant for sortParallel and sortLatticeByRowParallel
)pbdoc")
    .def_readwrite("verticalSortingChannel", &Config::verticalSortingChannel, R"pbdoc(
    :optional[bool]: Whether the sorting channel should be vertical. If not set, it will be determined automatically. Only relevant for sortLatticeByRowParallel
)pbdoc")
    .def_readwrite("aodTotalLimit", &Config::aodTotalLimit, R"pbdoc(
    :unsigned int: How many movable traps may be generated in total. Only relevant for sortParallel, sortLatticeGreedyParallel, and sortLatticeByRowParallel
)pbdoc")
    .def_readwrite("aodRowLimit", &Config::aodRowLimit, R"pbdoc(
    :unsigned int: How many row tones may be fed to the AOD. Only relevant for sortParallel, sortLatticeGreedyParallel, and sortLatticeByRowParallel
)pbdoc")
    .def_readwrite("aodColLimit", &Config::aodColLimit, R"pbdoc(
    :unsigned int: How many column tones may be fed to the AOD. Only relevant for sortParallel, sortLatticeGreedyParallel, and sortLatticeByRowParallel
)pbdoc")
    .def_readwrite("moveCostOffset", &Config::moveCostOffset, R"pbdoc(
    :double: Constant time demand per move. Only relevant for sortParallel and sortLatticeGreedyParallel
)pbdoc")
    .def_readwrite("moveCostOffsetSubmove", &Config::moveCostOffsetSubmove, R"pbdoc(
    :double: Constant time demand per submove. Only relevant for sortParallel and sortLatticeGreedyParallel
)pbdoc")
    .def_readwrite("moveCostScalingSqrt", &Config::moveCostScalingSqrt, R"pbdoc(
    :double: Time demand depending on square root of submove distance. Only relevant for sortParallel and sortLatticeGreedyParallel
)pbdoc")
    .def_readwrite("moveCostScalingLinear", &Config::moveCostScalingLinear, R"pbdoc(
    :double: Time demand depending on submove distance linearly. Only relevant for sortParallel and sortLatticeGreedyParallel
)pbdoc")
    .def_readwrite("recommendedDistFromOccSites", &Config::recommendedDistFromOccSites, R"pbdoc(
    :double: Recommended distance from occupied sites. May be temporarily infringed depending on maxSubmoveDistInPenalizedArea. Only relevant for sortLatticeGreedyParallel
)pbdoc")
    .def_readwrite("recommendedDistFromEmptySites", &Config::recommendedDistFromEmptySites, R"pbdoc(
    :double: Recommended distance from empty sites. May be temporarily infringed depending on maxSubmoveDistInPenalizedArea. Only relevant for sortLatticeGreedyParallel
)pbdoc")
    .def_readwrite("minDistFromOccSites", &Config::minDistFromOccSites, R"pbdoc(
    :double: Minimum distance to keep from occupied sites at any time. Only relevant for sortLatticeGreedyParallel and sortLatticeByRowParallel
)pbdoc")
    .def_readwrite("maxSubmoveDistInPenalizedArea", &Config::maxSubmoveDistInPenalizedArea, R"pbdoc(
    :double: Maximum distance that one may travel in area that is to be avoided, e.g, to minimize heating. Only relevant for sortLatticeGreedyParallel and sortLatticeByRowParallel
)pbdoc")
    .def("flushLogs", &Config::flushLogs);

    m.def("sortSequentiallyByRow", &sortSequentiallyByRow, "A function that sorts an array of atoms row by row", py::arg("stateArray"), 
        py::arg("compZoneRowStart"), py::arg("compZoneRowEnd"), py::arg("compZoneColStart"), py::arg("compZoneColEnd"), R"pbdoc(
    A function that sorts an array of atoms sequentially row by row

    :param stateArray: The array of boolean values to be sorted
    :type stateArray: numpy.ndarray[bool[m, n], flags.writeable]
    :param compZoneRowStart: Start row of computational zone (inclusive)
    :type compZoneRowStart: int
    :param compZoneRowEnd: End row of computational zone (exclusive)
    :type compZoneRowEnd: int
    :param compZoneColStart: Start column of computational zone (inclusive)
    :type compZoneColStart: int
    :param compZoneColEnd: End column of computational zone (exclusive)
    :type compZoneColEnd: int
    :return: A list of moves to sort array or None if sorting has failed.
    :rtype: list[Move] | None
)pbdoc");

    m.def("sortParallel", &sortParallel, "A function that sorts an array of atoms in parallel", py::arg("stateArray"), 
        py::arg("compZoneRowStart"), py::arg("compZoneRowEnd"), py::arg("compZoneColStart"), py::arg("compZoneColEnd"), py::arg("targetGeometry"), 
        R"pbdoc(
    A function that sorts an array of atoms in parallel

    :param stateArray: The array of boolean values to be sorted
    :type stateArray: numpy.ndarray[bool[m, n], flags.writeable]
    :param compZoneRowStart: Start row of computational zone (inclusive)
    :type compZoneRowStart: int
    :param compZoneRowEnd: End row of computational zone (exclusive)
    :type compZoneRowEnd: int
    :param compZoneColStart: Start column of computational zone (inclusive)
    :type compZoneColStart: int
    :param compZoneColEnd: End column of computational zone (exclusive)
    :type compZoneColEnd: int
    :param targetGeometry: Array of boolean values of size (compZoneRowEnd - compZoneRowStart) x (compZoneColEnd - compZoneColStart) specifying target occupancy
    :type targetGeometry: numpy.ndarray[bool[m, n], flags.writeable]
    :return: A list of moves to sort array or None if sorting has failed.
    :rtype: list[ParallelMove] | None
)pbdoc");

    m.def("sortLatticeGreedyParallel", &sortLatticeGreedyParallel, "A function that sorts an array of atoms in parallel", py::arg("stateArray"), 
        py::arg("compZoneRowStart"), py::arg("compZoneRowEnd"), py::arg("compZoneColStart"), py::arg("compZoneColEnd"), py::arg("targetGeometry"), R"pbdoc(
    A function that sorts atoms in a lattice in parallel towards a given geometry

    :param stateArray: The array of boolean values to be sorted
    :type stateArray: numpy.ndarray[bool[m, n], flags.writeable]
    :param compZoneRowStart: Start row of computational zone (inclusive)
    :type compZoneRowStart: int
    :param compZoneRowEnd: End row of computational zone (exclusive)
    :type compZoneRowEnd: int
    :param compZoneColStart: Start column of computational zone (inclusive)
    :type compZoneColStart: int
    :param compZoneColEnd: End column of computational zone (exclusive)
    :type compZoneColEnd: int
    :param targetGeometry: Array of boolean values of size (compZoneRowEnd - compZoneRowStart) x (compZoneColEnd - compZoneColStart) specifying target occupancy
    :type targetGeometry: numpy.ndarray[bool[m, n], flags.writeable]
    :return: A list of moves to sort array or None if sorting has failed.
    :rtype: list[ParallelMove] | None
    )pbdoc");

    m.def("sortLatticeByRowParallel", &sortLatticeByRowParallel, "A function that sorts an array of atoms in a lattice row by row in parallel", 
        py::arg("stateArray"), py::arg("targetGeometry"), R"pbdoc(
    A function that sorts atoms in a lattice in parallel towards a given geometry

    :param stateArray: The array of boolean values to be sorted
    :type stateArray: numpy.ndarray[bool[m, n], flags.writeable]
    :param targetGeometry: Array of TargetState values of size equal to stateArray specifying target occupancy
    :type targetGeometry: numpy.ndarray[TargetState[m, n]]
    :return: A list of moves to sort array or None if sorting has failed.
    :rtype: list[ParallelMove] | None
    )pbdoc");

    m.def("fixLatticeByRowSortingDeficiencies", &fixLatticeByRowSortingDeficiencies, "A function that fixes deficiencies that arose while sorting using sortLatticeByRowParallel", 
        py::arg("stateArray"), py::arg("targetGeometry"), R"pbdoc(
    A function that fixes deficiencies that arose while sorting using sortLatticeByRowParallel

    :param stateArray: The array of boolean values to be sorted
    :type stateArray: numpy.ndarray[bool[m, n], flags.writeable]
    :param targetGeometry: Array of TargetState values of size equal to stateArray specifying target occupancy
    :type targetGeometry: numpy.ndarray[TargetState[m, n]]
    :return: A list of moves to sort array or None if sorting has failed.
    :rtype: list[ParallelMove] | None
    )pbdoc");
};