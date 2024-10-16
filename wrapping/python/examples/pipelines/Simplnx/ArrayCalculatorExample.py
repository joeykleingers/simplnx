import simplnx as nx

import itkimageprocessing as cxitk
import orientationanalysis as cxor
import simplnx_test_dirs as nxtest

import numpy as np

#Create a Data Structure
data_structure = nx.DataStructure()

# Filter 1
# Instantiate Filter
nx_filter = nx.CreateDataArrayFilter()
# Execute Filter with Parameters
result = nx_filter.execute(
    data_structure=data_structure,
    set_tuple_dimensions=True,
    component_count=1,
    data_format="",
    initialization_value_str="2",
    numeric_type_index=nx.NumericType.int32,
    output_array_path=nx.DataPath("TestArray"),
    tuple_dimensions=[[10.0]]
)
nxtest.check_filter_result(nx_filter, result)


# Filter 2
# Instantiate Filter
nx_filter = nx.CreateDataArrayFilter()
# Execute Filter with Parameters
result = nx_filter.execute(
    data_structure=data_structure,
    set_tuple_dimensions=True,
    component_count=1,
    data_format="",
    initialization_value_str="1.23878",
    numeric_type_index=nx.NumericType.float32,
    output_array_path=nx.DataPath("Confidence Index"),
    tuple_dimensions=[[10.0]]
)
nxtest.check_filter_result(nx_filter, result)

# Filter 3
# Instantiate Filter
nx_filter = nx.CreateDataArrayFilter()
# Execute Filter with Parameters
result = nx_filter.execute(
    data_structure=data_structure,
    set_tuple_dimensions=True,
    component_count=3,
    data_format="",
    initialization_value_str="1.23878",
    numeric_type_index=nx.NumericType.float32,
    output_array_path=nx.DataPath("EulerAngles"),
    tuple_dimensions=[[10.0]]
)
nxtest.check_filter_result(nx_filter, result)


# Filter 4
# Instantiate Filter
nx_filter = nx.ArrayCalculatorFilter()
calc_param = nx.CalculatorParameter.ValueType(nx.DataPath(""), "TestArray+TestArray", nx.CalculatorParameter.AngleUnits.Radians)
# Execute Filter with Parameters
result = nx_filter.execute(
    data_structure=data_structure,
    calculated_array_path=nx.DataPath("Caclulated_TestArray"),
    calculator_parameter=calc_param, 
    scalar_type_index=nx.NumericType.float32
)
nxtest.check_filter_result(nx_filter, result)

# Filter 5
# Instantiate Filter
nx_filter = nx.ArrayCalculatorFilter()
calc_param = nx.CalculatorParameter.ValueType(nx.DataPath(""), "Confidence Index*100", nx.CalculatorParameter.AngleUnits.Radians)
# Execute Filter with Parameters
result = nx_filter.execute(
    data_structure=data_structure,
    calculated_array_path=nx.DataPath("Caclulated_ConfidenceIndex"),
    calculator_parameter=calc_param, 
    scalar_type_index=nx.NumericType.float64
)
nxtest.check_filter_result(nx_filter, result)


# Filter 6
# Instantiate Filter
nx_filter = nx.ArrayCalculatorFilter()
calc_param = nx.CalculatorParameter.ValueType(nx.DataPath(""), "EulerAngles/2", nx.CalculatorParameter.AngleUnits.Radians)
# Execute Filter with Parameters
result = nx_filter.execute(
    data_structure=data_structure,
    calculated_array_path=nx.DataPath("Caclulated_EulerAngles"),
    calculator_parameter=calc_param, 
    scalar_type_index=nx.NumericType.float32
)
nxtest.check_filter_result(nx_filter, result)

# Filter 7
# Instantiate Filter
nx_filter = nx.ArrayCalculatorFilter()
calc_param = nx.CalculatorParameter.ValueType(nx.DataPath(""), "EulerAngles[0]+EulerAngles[1]", nx.CalculatorParameter.AngleUnits.Radians)
# Execute Filter with Parameters
result = nx_filter.execute(
    data_structure=data_structure,
    calculated_array_path=nx.DataPath("Caclulated_EulerAngles2"),
    calculator_parameter=calc_param, 
    scalar_type_index=nx.NumericType.float32
)
nxtest.check_filter_result(nx_filter, result)

# Filter 8
# Define output file path
output_file_path = nxtest.get_data_directory() / "Output/ArrayCalculatorExampleResults.dream3d"
# Instantiate Filter
nx_filter = nx.WriteDREAM3DFilter()
# Execute Filter with Parameters
result = nx_filter.execute(data_structure=data_structure, 
                       export_file_path=output_file_path, 
                       write_xdmf_file=True)
nxtest.check_filter_result(nx_filter, result)

# *****************************************************************************
# THIS SECTION IS ONLY HERE FOR CLEANING UP THE CI Machines
# If you are using this code, you should COMMENT out the next line
nxtest.cleanup_test_file(output_file_path)
# *****************************************************************************


print("===> Pipeline Complete")
