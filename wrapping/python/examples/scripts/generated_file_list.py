"""
Important Note
==============

This python file can be used as an example of how to execute a number of DREAM3D-NX
filters one after another, if you plan to use the codes below (and you are welcome to),
there are a few things that you, the developer, should take note of:

Import Statements
-----------------

You will most likely *NOT* need to include the following code:

   .. code:: python
      
      import simplnx_test_dirs as nxtest

Filter Error Detection
----------------------

In each section of code a filter is created and executed immediately. This may or
may *not* be what you want to do. You can also preflight the filter to verify the
correctness of the filters before executing the filter **although** this is done
for you when the filter is executed. As such, you will want to check the 'result'
variable to see if there are any errors or warnings. If there **are** any then
you, as the developer, should act appropriately on the errors or warnings. 
More specifically, this bit of code:

   .. code:: python

      nxtest.check_filter_result(nxor.ReadAngDataFilter, result)

is used by the simplnx unit testing framework and should be replaced by your own
error checking code. You are welcome to look up the function definition and use
that.

"""
import simplnx as nx

import itkimageprocessing as nxitk
import orientationanalysis as nxor
import simplnx_test_dirs as nxtest

import numpy as np

#------------------------------------------------------------------------------
# Print the various filesystem paths that are pre-generated for this machine.
#------------------------------------------------------------------------------
nxtest.print_all_paths()


# Create a Data Structure
data_structure = nx.DataStructure()


generated_file_list_value = nx.GeneratedFileListParameter.ValueType()
generated_file_list_value.input_path = str(nxtest.get_data_directory() / "Porosity_Image")
generated_file_list_value.ordering = nx.GeneratedFileListParameter.Ordering.LowToHigh

generated_file_list_value.file_prefix = "slice_"
generated_file_list_value.file_suffix = ""
generated_file_list_value.file_extension = ".tif"
generated_file_list_value.start_index = 11
generated_file_list_value.end_index = 174
generated_file_list_value.increment_index = 1
generated_file_list_value.padding_digits = 2

result = nxitk.ITKImportImageStackFilter.execute(data_structure=data_structure, 
                                   cell_attribute_matrix_name="Cell Data", 
                                   image_data_array_name="Image Data", 
                                   output_image_geometry_path=nx.DataPath(["Image Stack"]), 
                                   image_transform_index=0,
                                   input_file_list_object=generated_file_list_value,
                                   origin=[0., 0., 0.], 
                                   spacing=[1., 1.,1.])
nxtest.check_filter_result(nxitk.ITKImportImageStackFilter, result)

