#include "OrientationAnalysis/Filters/ComputeGBCDFilter.hpp"
#include "OrientationAnalysis/OrientationAnalysis_test_dirs.hpp"

#include "simplnx/Core/Application.hpp"
#include "simplnx/Parameters/ArrayCreationParameter.hpp"
#include "simplnx/Parameters/ArraySelectionParameter.hpp"
#include "simplnx/Parameters/DataObjectNameParameter.hpp"
#include "simplnx/Parameters/GeometrySelectionParameter.hpp"
#include "simplnx/Parameters/NumberParameter.hpp"
#include "simplnx/Pipeline/Pipeline.hpp"
#include "simplnx/Pipeline/PipelineFilter.hpp"
#include "simplnx/UnitTest/UnitTestCommon.hpp"

#include <catch2/catch.hpp>

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;
using namespace nx::core;
using namespace nx::core::UnitTest;

namespace
{
constexpr StringLiteral k_FaceEnsembleDataPath("FaceEnsembleData [NX]");

} // namespace

TEST_CASE("OrientationAnalysis::ComputeGBCD", "[OrientationAnalysis][ComputeGBCD]")
{
  UnitTest::LoadPlugins();

  const nx::core::UnitTest::TestFileSentinel testDataSentinel(nx::core::unit_test::k_TestFilesDir, "6_6_Small_IN100_GBCD.tar.gz", "6_6_Small_IN100_GBCD");

  // Read the Small IN100 Data set
  auto baseDataFilePath = fs::path(fmt::format("{}/6_6_Small_IN100_GBCD/6_6_Small_IN100_GBCD.dream3d", unit_test::k_TestFilesDir));
  DataStructure dataStructure = UnitTest::LoadDataStructure(baseDataFilePath);
  DataPath smallIn100Group({nx::core::Constants::k_SmallIN100});
  DataPath featureDataPath = smallIn100Group.createChildPath(Constants::k_Grain_Data);
  DataPath avgEulerAnglesPath = featureDataPath.createChildPath(Constants::k_AvgEulerAngles);
  DataPath featurePhasesPath = featureDataPath.createChildPath(Constants::k_Phases);

  DataPath ensembleDataPath = smallIn100Group.createChildPath(Constants::k_Phase_Data);
  DataPath crystalStructurePath = ensembleDataPath.createChildPath(Constants::k_CrystalStructures);

  DataPath triangleDataContainerPath({Constants::k_TriangleDataContainerName});
  DataPath faceDataGroup = triangleDataContainerPath.createChildPath(Constants::k_FaceData);
  DataPath faceEnsemblePath = triangleDataContainerPath.createChildPath(k_FaceEnsembleDataPath);

  DataPath faceLabels = faceDataGroup.createChildPath(Constants::k_FaceLabels);
  DataPath faceNormals = faceDataGroup.createChildPath(Constants::k_FaceNormals);
  DataPath faceAreas = faceDataGroup.createChildPath(Constants::k_FaceAreas);

  {
    // Instantiate the filter, a DataStructure object and an Arguments Object
    ComputeGBCDFilter filter;
    Arguments args;

    // Create default Parameters for the filter.
    args.insertOrAssign(ComputeGBCDFilter::k_GBCDRes_Key, std::make_any<Float32Parameter::ValueType>(9.0F));

    args.insertOrAssign(ComputeGBCDFilter::k_SelectedTriangleGeometryPath_Key, std::make_any<GeometrySelectionParameter::ValueType>(triangleDataContainerPath));

    args.insertOrAssign(ComputeGBCDFilter::k_SurfaceMeshFaceLabelsArrayPath_Key, std::make_any<ArraySelectionParameter::ValueType>(faceLabels));
    args.insertOrAssign(ComputeGBCDFilter::k_SurfaceMeshFaceNormalsArrayPath_Key, std::make_any<ArraySelectionParameter::ValueType>(faceNormals));
    args.insertOrAssign(ComputeGBCDFilter::k_SurfaceMeshFaceAreasArrayPath_Key, std::make_any<ArraySelectionParameter::ValueType>(faceAreas));
    args.insertOrAssign(ComputeGBCDFilter::k_FeatureEulerAnglesArrayPath_Key, std::make_any<ArraySelectionParameter::ValueType>(avgEulerAnglesPath));
    args.insertOrAssign(ComputeGBCDFilter::k_FeaturePhasesArrayPath_Key, std::make_any<ArraySelectionParameter::ValueType>(featurePhasesPath));
    args.insertOrAssign(ComputeGBCDFilter::k_CrystalStructuresArrayPath_Key, std::make_any<ArraySelectionParameter::ValueType>(crystalStructurePath));
    args.insertOrAssign(ComputeGBCDFilter::k_FaceEnsembleAttributeMatrixName_Key, std::make_any<DataObjectNameParameter::ValueType>(k_FaceEnsembleDataPath));
    args.insertOrAssign(ComputeGBCDFilter::k_GBCDArrayName_Key, std::make_any<DataObjectNameParameter::ValueType>(Constants::k_GBCD_Name));

    // Preflight the filter and check result
    auto preflightResult = filter.preflight(dataStructure, args);
    SIMPLNX_RESULT_REQUIRE_VALID(preflightResult.outputActions);

    // Execute the filter and check the result
    auto executeResult = filter.execute(dataStructure, args);
    SIMPLNX_RESULT_REQUIRE_VALID(executeResult.result);
  }

  // Compare the Output GBCD Data
  {
    const DataPath k_GeneratedDataPath = faceEnsemblePath.createChildPath(Constants::k_GBCD_Name);
    const DataPath k_ExemplarArrayPath = triangleDataContainerPath.createChildPath("FaceEnsembleData").createChildPath(Constants::k_GBCD_Name);

    UnitTest::CompareFloatArraysWithNans<float64>(dataStructure, k_ExemplarArrayPath, k_GeneratedDataPath);
  }

#ifdef SIMPLNX_WRITE_TEST_OUTPUT
  WriteTestDataStructure(dataStructure, fs::path(fmt::format("{}/find_gbcd.dream3d", unit_test::k_BinaryTestOutputDir)));
#endif

  UnitTest::CheckArraysInheritTupleDims(dataStructure);
}

TEST_CASE("OrientationAnalysis::ComputeGBCDFilter: Phase and Laue Index Bounds", "[OrientationAnalysis][ComputeGBCDFilter]")
{
  UnitTest::LoadPlugins();
  const UnitTest::PreferencesSentinel preferencesSentinel(DataStorageMode::ForceOutOfCore, 1);
  const UnitTest::TestFileSentinel testDataSentinel(unit_test::k_TestFilesDir, "6_6_Small_IN100_GBCD.tar.gz", "6_6_Small_IN100_GBCD");
  const fs::path inputFile = fs::path(unit_test::k_TestFilesDir.view()) / "6_6_Small_IN100_GBCD" / "6_6_Small_IN100_GBCD.dream3d";
  DataStructure dataStructure = UnitTest::LoadDataStructure(inputFile);

  const DataPath featureDataPath = DataPath({Constants::k_SmallIN100}).createChildPath(Constants::k_Grain_Data);
  const DataPath avgEulerAnglesPath = featureDataPath.createChildPath(Constants::k_AvgEulerAngles);
  const DataPath featurePhasesPath = featureDataPath.createChildPath(Constants::k_Phases);
  const DataPath crystalStructuresPath = DataPath({Constants::k_SmallIN100}).createChildPath(Constants::k_Phase_Data).createChildPath(Constants::k_CrystalStructures);
  const DataPath triangleGeometryPath({Constants::k_TriangleDataContainerName});
  const DataPath faceDataPath = triangleGeometryPath.createChildPath(Constants::k_FaceData);

  REQUIRE_NOTHROW(dataStructure.getDataRefAs<Int32Array>(featurePhasesPath));
  auto& featurePhasesArrayRef = dataStructure.getDataRefAs<Int32Array>(featurePhasesPath);
  REQUIRE_NOTHROW(dataStructure.getDataRefAs<UInt32Array>(crystalStructuresPath));
  auto& crystalStructuresArrayRef = dataStructure.getDataRefAs<UInt32Array>(crystalStructuresPath);

  ComputeGBCDFilter filter;
  Arguments args = filter.getDefaultArguments();
  args.insertOrAssign(ComputeGBCDFilter::k_GBCDRes_Key, std::make_any<Float32Parameter::ValueType>(9.0F));
  args.insertOrAssign(ComputeGBCDFilter::k_SelectedTriangleGeometryPath_Key, std::make_any<DataPath>(triangleGeometryPath));
  args.insertOrAssign(ComputeGBCDFilter::k_SurfaceMeshFaceLabelsArrayPath_Key, std::make_any<DataPath>(faceDataPath.createChildPath(Constants::k_FaceLabels)));
  args.insertOrAssign(ComputeGBCDFilter::k_SurfaceMeshFaceNormalsArrayPath_Key, std::make_any<DataPath>(faceDataPath.createChildPath(Constants::k_FaceNormals)));
  args.insertOrAssign(ComputeGBCDFilter::k_SurfaceMeshFaceAreasArrayPath_Key, std::make_any<DataPath>(faceDataPath.createChildPath(Constants::k_FaceAreas)));
  args.insertOrAssign(ComputeGBCDFilter::k_FeatureEulerAnglesArrayPath_Key, std::make_any<DataPath>(avgEulerAnglesPath));
  args.insertOrAssign(ComputeGBCDFilter::k_FeaturePhasesArrayPath_Key, std::make_any<DataPath>(featurePhasesPath));
  args.insertOrAssign(ComputeGBCDFilter::k_CrystalStructuresArrayPath_Key, std::make_any<DataPath>(crystalStructuresPath));
  args.insertOrAssign(ComputeGBCDFilter::k_FaceEnsembleAttributeMatrixName_Key, std::make_any<std::string>("Bounds Face Ensemble Data"));
  args.insertOrAssign(ComputeGBCDFilter::k_GBCDArrayName_Key, std::make_any<std::string>("Bounds GBCD"));

  SECTION("Participating Phase returns an error")
  {
    auto& featurePhasesStoreRef = featurePhasesArrayRef.getDataStoreRef();
    for(usize featureIdx = 1; featureIdx < featurePhasesStoreRef.getNumberOfTuples(); featureIdx++)
    {
      featurePhasesStoreRef[featureIdx] = static_cast<int32>(crystalStructuresArrayRef.getNumberOfTuples());
    }
    auto executeResult = filter.execute(dataStructure, args);
    SIMPLNX_RESULT_REQUIRE_INVALID(executeResult.result);
    REQUIRE(executeResult.result.errors()[0].code == -75000);
  }

  SECTION("Participating Laue index returns an error")
  {
    crystalStructuresArrayRef.getDataStoreRef()[1] = 999U;
    auto executeResult = filter.execute(dataStructure, args);
    SIMPLNX_RESULT_REQUIRE_INVALID(executeResult.result);
    REQUIRE(executeResult.result.errors()[0].code == -75001);
  }

  UnitTest::CheckArraysInheritTupleDims(dataStructure);
}

TEST_CASE("OrientationAnalysis::ComputeGBCDFilter: SIMPL Backwards Compatibility", "[OrientationAnalysis][ComputeGBCDFilter][BackwardsCompatibility]")
{
  auto app = Application::GetOrCreateInstance();
  UnitTest::LoadPlugins();
  auto filterList = app->getFilterList();

  const fs::path conversionDir = fs::path(nx::core::unit_test::k_SourceDir.view()) / "test" / "simpl_conversion";

  const std::vector<std::pair<std::string, fs::path>> fixtures = {
      {"SIMPL 6.5 (UUID)", conversionDir / "6_5" / "ComputeGBCDFilter.json"},
      {"SIMPL 6.4 (Filter_Name)", conversionDir / "6_4" / "ComputeGBCDFilter.json"},
  };

  for(const auto& [label, fixturePath] : fixtures)
  {
    DYNAMIC_SECTION(label)
    {
      auto pipelineResult = Pipeline::FromSIMPLFile(fixturePath, filterList);
      REQUIRE(pipelineResult.valid());

      auto& pipeline = pipelineResult.value();
      REQUIRE(pipeline.size() == 1);

      auto* pipelineFilter = dynamic_cast<PipelineFilter*>(pipeline.at(0));
      REQUIRE(pipelineFilter != nullptr);

      const IFilter* filter = pipelineFilter->getFilter();
      REQUIRE(filter != nullptr);
      REQUIRE(filter->uuid() == FilterTraits<ComputeGBCDFilter>::uuid);

      CHECK(pipelineFilter->getComments().empty());

      const Arguments args = pipelineFilter->getArguments();
      CHECK(args.value<float32>(ComputeGBCDFilter::k_GBCDRes_Key) == 2.5f);
      CHECK(args.value<DataPath>(ComputeGBCDFilter::k_SelectedTriangleGeometryPath_Key) == DataPath({"DataContainer"}));
      CHECK(args.value<DataPath>(ComputeGBCDFilter::k_SurfaceMeshFaceLabelsArrayPath_Key) == DataPath({"DataContainer", "CellData", "TestArray"}));
      CHECK(args.value<DataPath>(ComputeGBCDFilter::k_SurfaceMeshFaceNormalsArrayPath_Key) == DataPath({"DataContainer", "CellData", "TestArray"}));
      CHECK(args.value<DataPath>(ComputeGBCDFilter::k_SurfaceMeshFaceAreasArrayPath_Key) == DataPath({"DataContainer", "CellData", "TestArray"}));
      CHECK(args.value<DataPath>(ComputeGBCDFilter::k_FeatureEulerAnglesArrayPath_Key) == DataPath({"DataContainer", "CellData", "TestArray"}));
      CHECK(args.value<DataPath>(ComputeGBCDFilter::k_FeaturePhasesArrayPath_Key) == DataPath({"DataContainer", "CellData", "TestArray"}));
      CHECK(args.value<DataPath>(ComputeGBCDFilter::k_CrystalStructuresArrayPath_Key) == DataPath({"DataContainer", "CellData", "TestArray"}));
      CHECK(args.value<std::string>(ComputeGBCDFilter::k_FaceEnsembleAttributeMatrixName_Key) == "TestName");
      CHECK(args.value<std::string>(ComputeGBCDFilter::k_GBCDArrayName_Key) == "TestName");
    }
  }
}
