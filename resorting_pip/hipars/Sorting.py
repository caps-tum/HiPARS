import sys
import os
if sys.platform == "win32":
    os.add_dll_directory(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resorting_cpp
import numpy as np

class Sorting:
    """Python wrapper class for sorting arrays of neutral atoms
    """

    def __init__(self):
        """Constructor method
        """
        self.config = resorting_cpp.Config()
    
    def read_config_file(self, file_path):
        """Function to pass a config file for the underlying sorting library

        :param file_path: The path of the config file
        :type file_path: str
        :raises TypeError: file_path must be a str
        :return: A list of moves to sort array or None if sorting has failed. A move contains .distance, .init_dir, and .sites_list, which is a list of coordinate pairs to traverse
        :rtype: bool, optional
        """
        if not isinstance(file_path, str):
            raise TypeError("file_path must be a str")
        return self.config.readConfig(file_path)


    def sort_sequentially(self, state_array : np.ndarray, comp_zone_row_range : tuple[int,int], comp_zone_col_range : tuple[int,int]):
        """Function for sorting sequentially

        :param state_array: The array of boolean values to be sorted
        :type state_array: np.ndarray[bool]
        :param comp_zone_row_range: Tuple (start,end) of start(inclusive) and end(exclusive) of rows in computational zone
        :type comp_zone_row_range: tuple(int,int)
        :param comp_zone_col_range: Tuple (start,end) of start(inclusive) and end(exclusive) of columns in computational zone
        :type comp_zone_col_range: tuple(int,int)
        :raises TypeError: state_array must be numpy bool array
        :raises TypeError: state_array must be dtype bool
        :return: A list of moves to sort array or None if sorting has failed. A move contains .distance, .init_dir, and .sites_list, which is a list of coordinate pairs to traverse
        :rtype: list[SequentialMove], optional
        """
        if not isinstance(state_array, np.ndarray):
            raise TypeError("state_array must be numpy bool array")
        if not state_array.dtype == bool:
            raise TypeError("state_array must be dtype bool")
        return resorting_cpp.sortSequentiallyByRow(state_array, *comp_zone_row_range, *comp_zone_col_range)
    
    def sort_parallel(self, state_array : np.ndarray, comp_zone_row_range : tuple[int,int], comp_zone_col_range : tuple[int,int], 
                      target_geometry : np.ndarray | None = None):
        """Function for sorting in parallel
        
        :param state_array: The array of boolean values to be sorted
        :type state_array: np.ndarray[bool]
        :param comp_zone_row_range: Tuple (start,end) of start(inclusive) and end(exclusive) of rows in computational zone
        :type comp_zone_row_range: tuple(int,int)
        :param comp_zone_col_range: Tuple (start,end) of start(inclusive) and end(exclusive) of columns in computational zone
        :type comp_zone_col_range: tuple(int,int)
        :param target_geometry: The array of boolean values specifying the target geometry. Assume all true if None
        :type target_geometry: np.ndarray[bool] | None
        :raises TypeError: state_array must be numpy bool array
        :raises TypeError: state_array must be dtype bool
        :return: A list of moves to sort array or None if sorting has failed. A ParallelMove contains .steps, which is a list of ParallelMoveStep objects, each containing .colSelection and .rowSelection, which are lists of doubles
        :rtype: list[ParallelMove], optional
        """
        if not isinstance(state_array, np.ndarray):
            raise TypeError("state_array must be numpy bool array")
        if not state_array.dtype == bool:
            raise TypeError("state_array must be dtype bool")
        if target_geometry is None:
            target_geometry = np.ones((comp_zone_row_range[1] - comp_zone_row_range[0], comp_zone_col_range[1] - comp_zone_col_range[0]), dtype=bool)
        return resorting_cpp.sortParallel(state_array, *comp_zone_row_range, *comp_zone_col_range, target_geometry)
    
    def sort_parallel_lattice_greedy(self, state_array : np.ndarray, comp_zone_row_range : tuple[int,int], 
                                     comp_zone_col_range : tuple[int,int], target_geometry : np.ndarray | None):
        """Function for sorting in parallel
        
        :param state_array: The array of boolean values to be sorted
        :type state_array: np.ndarray[bool]
        :param comp_zone_row_range: Tuple (start,end) of start(inclusive) and end(exclusive) of rows in computational zone
        :type comp_zone_row_range: tuple(int,int)
        :param comp_zone_col_range: Tuple (start,end) of start(inclusive) and end(exclusive) of columns in computational zone
        :type comp_zone_col_range: tuple(int,int)
        :param target_geometry: The array of boolean values specifying the target geometry
        :type target_geometry: np.ndarray[bool]
        :raises TypeError: state_array must be numpy bool array
        :raises TypeError: state_array must be dtype bool
        :raises TypeError: target_geometry must be numpy bool array
        :raises TypeError: target_geometry must be dtype bool
        :return: A list of moves to sort array or None if sorting has failed. A ParallelMove contains .steps, which is a list of ParallelMoveStep objects, each containing .colSelection and .rowSelection, which are lists of doubles
        :rtype: list[ParallelMove], optional
        """
        if not isinstance(state_array, np.ndarray):
            raise TypeError("state_array must be numpy bool array")
        if not state_array.dtype == bool:
            raise TypeError("state_array must be dtype bool")
        if not isinstance(target_geometry, np.ndarray):
            raise TypeError("state_array must be numpy bool array")
        if not target_geometry.dtype == bool:
            raise TypeError("state_array must be dtype bool")
        if target_geometry is None:
            target_geometry = np.ones((comp_zone_row_range[1] - comp_zone_row_range[0], comp_zone_col_range[1] - comp_zone_col_range[0]), dtype=bool)
        return resorting_cpp.sortLatticeGreedyParallel(state_array, *comp_zone_row_range, *comp_zone_col_range, target_geometry)
    
    def sort_parallel_lattice_by_row(self, state_array : np.ndarray, target_geometry : np.ndarray[resorting_cpp.TargetState]):
        """Function for sorting lattice geometries row by row
        
        :param state_array: The array of boolean values to be sorted
        :type state_array: np.ndarray[bool]
        :param target_geometry: The array of target state values specifying the target geometry
        :type target_geometry: np.ndarray[resorting_cpp.TargetState]
        :return: A list of moves to sort array or None if sorting has failed. A ParallelMove contains .steps, which is a list of ParallelMoveStep objects, each containing .colSelection and .rowSelection, which are lists of doubles
        :rtype: list[ParallelMove], optional
        """
        return resorting_cpp.sortLatticeByRowParallel(state_array, target_geometry)
    
    def fix_lattice_by_row_sorting_deficiencies(self, state_array : np.ndarray, target_geometry : np.ndarray[resorting_cpp.TargetState]):
        """Function for fixing sorting deficiencies arising during sort_parallel_lattice_by_row
        
        :param state_array: The array of boolean values to be sorted
        :type state_array: np.ndarray[bool]
        :param target_geometry: The array of target state values specifying the target geometry
        :type target_geometry: np.ndarray[resorting_cpp.TargetState]
        :return: A list of moves to sort array or None if sorting has failed. A ParallelMove contains .steps, which is a list of ParallelMoveStep objects, each containing .colSelection and .rowSelection, which are lists of doubles
        :rtype: list[ParallelMove], optional
        """
        return resorting_cpp.fixLatticeByRowSortingDeficiencies(state_array, target_geometry)

    def flush_logs(self):
        """Function for flushing the logs
        """
        resorting_cpp.Config().flushLogs()