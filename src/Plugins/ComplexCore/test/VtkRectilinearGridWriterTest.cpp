#include <catch2/catch.hpp>

#include "complex/Parameters/FileSystemPathParameter.hpp"
#include "complex/Parameters/MultiArraySelectionParameter.hpp"
#include "complex/UnitTest/UnitTestCommon.hpp"

#include "ComplexCore/ComplexCore_test_dirs.hpp"
#include "ComplexCore/Filters/VtkRectilinearGridWriterFilter.hpp"

#include <filesystem>
namespace fs = std::filesystem;

using namespace complex;
using namespace complex::Constants;

TEST_CASE("ComplexCore::VtkRectilinearGridWriterFilter: Valid Filter Execution", "[ComplexCore][VtkRectilinearGridWriterFilter]")
{
  // Read input DREAM3D File
  auto exemplarFilePath = fs::path(fmt::format("{}/Small_IN100.dream3d", unit_test::k_TestFilesDir));
  DataStructure dataStructure = UnitTest::LoadDataStructure(exemplarFilePath);

  // Instantiate the filter, and an Arguments Object
  VtkRectilinearGridWriterFilter filter;
  Arguments args;

  fs::path exemplarOutputPath = fs::path(fmt::format("{}/6_6_vtk_rectilinear_grid_writer/6_6_vtk_rectilinear_grid.vtk", unit_test::k_TestFilesDir));
  fs::path computedOutputPath(fmt::format("{}/NX_vtk_rectilinear_grid.vtk", unit_test::k_BinaryTestOutputDir));
  fs::path exemplarBinaryOutputPath = fs::path(fmt::format("{}/6_6_vtk_rectilinear_grid_writer/6_6_vtk_rectilinear_grid_binary.vtk", unit_test::k_TestFilesDir));
  fs::path computedBinaryOutputPath(fmt::format("{}/NX_vtk_rectilinear_grid_binary.vtk", unit_test::k_BinaryTestOutputDir));

  MultiArraySelectionParameter::ValueType selectedArrayPaths = {k_ConfidenceIndexArrayPath, k_EulersArrayPath, k_FitArrayPath, k_ImageQualityArrayPath, k_PhasesArrayPath, k_SEMSignalArrayPath};

  // write out ascii vtk file
  args.insertOrAssign(VtkRectilinearGridWriterFilter::k_OutputFile_Key, std::make_any<FileSystemPathParameter::ValueType>(computedOutputPath));
  args.insertOrAssign(VtkRectilinearGridWriterFilter::k_WriteBinaryFile_Key, std::make_any<bool>(false));
  args.insertOrAssign(VtkRectilinearGridWriterFilter::k_ImageGeometryPath_Key, std::make_any<DataPath>(k_DataContainerPath));
  args.insertOrAssign(VtkRectilinearGridWriterFilter::k_SelectedDataArrayPaths_Key, std::make_any<MultiArraySelectionParameter::ValueType>(selectedArrayPaths));

  auto preflightResult = filter.preflight(dataStructure, args);
  COMPLEX_RESULT_REQUIRE_VALID(preflightResult.outputActions)

  auto executeResult = filter.execute(dataStructure, args);
  COMPLEX_RESULT_REQUIRE_VALID(executeResult.result)

  // write out binary vtk file
  args.insertOrAssign(VtkRectilinearGridWriterFilter::k_OutputFile_Key, std::make_any<FileSystemPathParameter::ValueType>(computedBinaryOutputPath));
  args.insertOrAssign(VtkRectilinearGridWriterFilter::k_WriteBinaryFile_Key, std::make_any<bool>(true));

  preflightResult = filter.preflight(dataStructure, args);
  COMPLEX_RESULT_REQUIRE_VALID(preflightResult.outputActions)

  executeResult = filter.execute(dataStructure, args);
  COMPLEX_RESULT_REQUIRE_VALID(executeResult.result)

  std::vector<size_t> linesToSkip{1}; // skip the version line
  std::ifstream computedFile(computedOutputPath);
  std::ifstream exemplarFile(exemplarOutputPath);
  UnitTest::CompareAsciiFiles(computedFile, exemplarFile, linesToSkip);
  std::ifstream computedBinaryFile(computedBinaryOutputPath, std::ios::binary);
  std::ifstream exemplarBinaryFile(exemplarBinaryOutputPath, std::ios::binary);
  UnitTest::CompareAsciiFiles(computedBinaryFile, exemplarBinaryFile, linesToSkip);
}

TEST_CASE("ComplexCore::VtkRectilinearGridWriterFilter: InValid Filter Execution", "[ComplexCore][VtkRectilinearGridWriterFilter]")
{
  // Read input DREAM3D File
  auto exemplarFilePath = fs::path(fmt::format("{}/Small_IN100.dream3d", unit_test::k_TestFilesDir));
  DataStructure dataStructure = UnitTest::LoadDataStructure(exemplarFilePath);

  // Instantiate the filter, and an Arguments Object
  VtkRectilinearGridWriterFilter filter;
  Arguments args;

  fs::path computedOutputPath(fmt::format("{}/NX_vtk_rectilinear_grid.vtk", unit_test::k_BinaryTestOutputDir));

  args.insertOrAssign(VtkRectilinearGridWriterFilter::k_OutputFile_Key, std::make_any<FileSystemPathParameter::ValueType>(computedOutputPath));
  args.insertOrAssign(VtkRectilinearGridWriterFilter::k_WriteBinaryFile_Key, std::make_any<bool>(false));
  args.insertOrAssign(VtkRectilinearGridWriterFilter::k_ImageGeometryPath_Key, std::make_any<DataPath>(k_DataContainerPath));

  SECTION("No Selected Arrays")
  {
    args.insertOrAssign(VtkRectilinearGridWriterFilter::k_SelectedDataArrayPaths_Key, std::make_any<MultiArraySelectionParameter::ValueType>(MultiArraySelectionParameter::ValueType{}));
  }
  SECTION("Mismatching Tuples in Selected Arrays")
  {
    args.insertOrAssign(VtkRectilinearGridWriterFilter::k_SelectedDataArrayPaths_Key,
                        std::make_any<MultiArraySelectionParameter::ValueType>(MultiArraySelectionParameter::ValueType{
                            k_CrystalStructuresArrayPath, k_ConfidenceIndexArrayPath, k_EulersArrayPath, k_FitArrayPath, k_ImageQualityArrayPath, k_PhasesArrayPath, k_SEMSignalArrayPath}));
  }
  SECTION("Selected Arrays are not Cell Level Arrays")
  {
    args.insertOrAssign(VtkRectilinearGridWriterFilter::k_SelectedDataArrayPaths_Key,
                        std::make_any<MultiArraySelectionParameter::ValueType>(MultiArraySelectionParameter::ValueType{k_CrystalStructuresArrayPath}));
  }

  auto preflightResult = filter.preflight(dataStructure, args);
  COMPLEX_RESULT_REQUIRE_INVALID(preflightResult.outputActions)

  auto executeResult = filter.execute(dataStructure, args);
  COMPLEX_RESULT_REQUIRE_INVALID(executeResult.result)
}