#include "ComplexCore/ComplexCore_test_dirs.hpp"
#include "ComplexCore/Filters/CreateDataArray.hpp"
#include "ComplexCore/Filters/CreateImageGeometry.hpp"
#include "ComplexCore/Filters/ExportDREAM3DFilter.hpp"
#include "ComplexCore/Filters/ImportDREAM3DFilter.hpp"
#include "ComplexCore/Filters/StlFileReaderFilter.hpp"
#include "ComplexCore/WindingNumber/UT_SolidAngle.h"

#include "complex/DataStructure/Geometry/TriangleGeom.hpp"
#include "complex/Parameters/Dream3dImportParameter.hpp"
#include "complex/Parameters/DynamicTableParameter.hpp"
#include "complex/Parameters/FileSystemPathParameter.hpp"
#include "complex/UnitTest/UnitTestCommon.hpp"
#include "complex/Utilities/SamplingUtils.hpp"

#include <Eigen/Core>
#include <tbb/parallel_for.h>

#include <catch2/catch.hpp>

namespace fs = std::filesystem;
using namespace complex;

namespace
{
const fs::path k_StlFilePath("/Volumes/ExData/Data/Spheres.stl");
const fs::path k_InputD3DFilePath("/private/tmp/SmallIN100_Mesh.dream3d");
const DataPath k_TriangleGeometryPath({"TriangleDataContainer"});
const std::string k_ImageGeometryName = "[Image Geometry]";
const std::string k_CellDataName = "Cell Data";
const std::string k_FeatureIdsName = "SampledFeatureIds";
const DataPath k_ImageGeometryPath({k_ImageGeometryName});
const DataPath k_FeatureIdsPath({k_ImageGeometryName, k_CellDataName, k_FeatureIdsName});
const DataPath k_FaceLabelsPath({k_TriangleGeometryPath.getTargetName(), "FaceData", "FaceLabels"});
const std::vector<uint64> k_ImageGeometryDims({21, 21, 6});
const std::vector<float32> k_ImageGeometrySpacing({1, 1, 1});
const std::vector<float32> k_ImageGeometryOrigin({0.0f, 0.0f, 0.0f});
const fs::path k_OutputD3DFilePath("/tmp/SmallIN100_Mesh_Output.dream3d");
constexpr double k_Epsilon = 0.5;

void importStlFile(const fs::path& stlFilePath, DataStructure& dataStructure)
{
  // Instantiate the filter, a DataStructure object and an Arguments Object
  Arguments args;
  StlFileReaderFilter filter;

  // Create default Parameters for the filter.
  args.insertOrAssign(StlFileReaderFilter::k_StlFilePath_Key, std::make_any<FileSystemPathParameter::ValueType>(stlFilePath));
  args.insertOrAssign(StlFileReaderFilter::k_TriangleGeometryName_Key, std::make_any<DataPath>(k_TriangleGeometryPath));

  // Execute the filter and check the result
  auto executeResult = filter.execute(dataStructure, args);
  COMPLEX_RESULT_REQUIRE_VALID(executeResult.result);
}

void importD3DFile(const fs::path& d3dFilePath, DataStructure& dataStructure)
{
  // Instantiate the filter, a DataStructure object and an Arguments Object
  Arguments args;
  ImportDREAM3DFilter filter;

  // Create default Parameters for the filter.
  args.insert(ImportDREAM3DFilter::k_ImportFileData, Dream3dImportParameter::ImportData{d3dFilePath});

  // Execute the filter and check the result
  auto executeResult = filter.execute(dataStructure, args);
  COMPLEX_RESULT_REQUIRE_VALID(executeResult.result);
}

void generateImageGeometry(DataStructure& dataStructure)
{
  {
    CreateImageGeometry filter;
    Arguments args;

    args.insertOrAssign(CreateImageGeometry::k_GeometryDataPath_Key, std::make_any<DataPath>(k_ImageGeometryPath));
    args.insertOrAssign(CreateImageGeometry::k_CellDataName_Key, std::make_any<std::string>(k_CellDataName));
    args.insertOrAssign(CreateImageGeometry::k_Dimensions_Key, k_ImageGeometryDims);
    args.insertOrAssign(CreateImageGeometry::k_Origin_Key, k_ImageGeometryOrigin);
    args.insertOrAssign(CreateImageGeometry::k_Spacing_Key, k_ImageGeometrySpacing);
    auto result = filter.execute(dataStructure, args);
    COMPLEX_RESULT_REQUIRE_VALID(result.result)
  }
  {
    CreateDataArray filter;
    Arguments args;

    usize numTuples = std::accumulate(k_ImageGeometryDims.begin(), k_ImageGeometryDims.end(), static_cast<usize>(1), std::multiplies<>());
    DynamicTableInfo::TableDataType tupleDims = {{static_cast<double>(numTuples)}};

    args.insert(CreateDataArray::k_NumericType_Key, std::make_any<NumericType>(NumericType::int32));
    args.insert(CreateDataArray::k_NumComps_Key, std::make_any<uint64>(1));
    args.insert(CreateDataArray::k_TupleDims_Key, std::make_any<DynamicTableParameter::ValueType>(tupleDims));
    args.insert(CreateDataArray::k_DataPath_Key, std::make_any<DataPath>(k_FeatureIdsPath));
    args.insert(CreateDataArray::k_InitilizationValue_Key, std::make_any<std::string>("0"));

    auto result = filter.execute(dataStructure, args);
    COMPLEX_RESULT_REQUIRE_VALID(result.result);
  }
}

std::vector<HDK_Sample::UT_Vector3T<float>> generateVerticesVector(DataArray<float32>& verticesArray)
{
  auto numVertices = verticesArray.getNumberOfTuples();
  std::vector<HDK_Sample::UT_Vector3T<float>> U(numVertices);

  for(int i = 0; i < numVertices; i++)
  {
    for(int j = 0; j < 3; j++)
    {
      auto val = verticesArray.getComponent(i, j);
      U[i][j] = val;
    }
  }

  return U;
}

Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> generateFacesMatrix(DataArray<IGeometry::MeshIndexType>& facesArray, std::vector<int32> faceIndices)
{
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> F(faceIndices.size(), facesArray.getNumberOfComponents());

  //  std::cout << "v1,v2,v3" << std::endl;
  for(int i = 0; i < F.rows(); i++)
  {
    for(int j = 0; j < F.cols(); j++)
    {
      auto val = facesArray.getComponent(faceIndices[i], j);
      F(i, j) = val;
    }
    //    std::cout << F(i, 0) << "," << F(i, 1) << "," << F(i, 2) << std::endl;
  }

  return F;
}

Eigen::MatrixXf generatePointsMatrix(const ImageGeom& imageGeom)
{
  Vec3<usize> dims = imageGeom.getDimensions();

  usize numOfPoints = std::accumulate(dims.begin(), dims.end(), static_cast<usize>(1), std::multiplies<>());
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> P(numOfPoints, 3);

  usize rowIndex = 0;
  //  std::cout << "x,y,z" << std::endl;
  for(usize z = 0; z < dims[2]; z++)
  {
    for(usize y = 0; y < dims[1]; y++)
    {
      for(usize x = 0; x < dims[0]; x++)
      {
        Vec3<float64> coord = imageGeom.getCoords(x, y, z);
        P(rowIndex, 0) = coord[0];
        P(rowIndex, 1) = coord[1];
        P(rowIndex, 2) = coord[2];
        rowIndex++;

        //        std::cout << coord[0] << "," << coord[1] << "," << coord[2] << std::endl;
      }
    }
  }

  return P;
}

void writeDream3dFile(const fs::path& d3dFilePath, DataStructure& dataStructure)
{
  // Instantiate the filter, a DataStructure object and an Arguments Object
  Arguments args;
  ExportDREAM3DFilter filter;

  // Create default Parameters for the filter.
  args.insertOrAssign(ExportDREAM3DFilter::k_ExportFilePath, std::make_any<FileSystemPathParameter::ValueType>(d3dFilePath));
  args.insertOrAssign(ExportDREAM3DFilter::k_WriteXdmf, std::make_any<BoolParameter::ValueType>(true));

  // Execute the filter and check the result
  auto executeResult = filter.execute(dataStructure, args);
  COMPLEX_RESULT_REQUIRE_VALID(executeResult.result);
}
} // namespace

TEST_CASE("WindingNumberTestPerFeature")
{
  DataStructure dataStructure;

  //  importStlFile(k_StlFilePath, dataStructure);
  importD3DFile(k_InputD3DFilePath, dataStructure);
  generateImageGeometry(dataStructure);

  REQUIRE_NOTHROW(dataStructure.getDataRefAs<TriangleGeom>(k_TriangleGeometryPath));
  auto& triangleGeom = dataStructure.getDataRefAs<TriangleGeom>(k_TriangleGeometryPath);
  auto& faceLabels = dataStructure.getDataRefAs<Int32Array>(k_FaceLabelsPath);
  auto& vertices = triangleGeom.getVerticesRef();
  auto& faces = triangleGeom.getFacesRef();

  REQUIRE(vertices.getNumberOfComponents() == 3);
  REQUIRE(faces.getNumberOfComponents() == 3);

  REQUIRE_NOTHROW(dataStructure.getDataRefAs<ImageGeom>(k_ImageGeometryPath));
  auto& imageGeom = dataStructure.getDataRefAs<ImageGeom>(k_ImageGeometryPath);

  REQUIRE_NOTHROW(dataStructure.getDataRefAs<Int32Array>(k_FeatureIdsPath));
  auto& featureIds = dataStructure.getDataRefAs<Int32Array>(k_FeatureIdsPath);
  featureIds.fill(0);

  complex::Sampling::FeatureFacesStatistics ffStats = complex::Sampling::CalculateFeatureFacesStatistics(faceLabels, triangleGeom, false, {});
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> P = generatePointsMatrix(imageGeom);
  std::vector<HDK_Sample::UT_Vector3T<float>> U = generateVerticesVector(vertices);
  HDK_Sample::UT_SolidAngle<float, float> solid_angle;
  int order = 2;
  double accuracy_scale = 2.0;

  auto initialTime = std::chrono::steady_clock::now();
  for(int32 featureId = 1; featureId < ffStats.numberOfFeatures; featureId++)
  {
    auto now = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(now - initialTime).count();
    if(diff > 1000 || featureId == ffStats.faceLists.size() - 1)
    {
      // Output every second, or on the last feature ID
      WARN(fmt::format("Sampling Feature ID {} ({}/{})...", featureId, featureId + 1, ffStats.numberOfFeatures));
      initialTime = std::chrono::steady_clock::now();
    }

    const std::vector<int32>& faceList = ffStats.faceLists[featureId];
    if(faceList.empty())
    {
      continue;
    }

    std::cout << fmt::format("{}: {}", featureId, ffStats.faceLists[featureId].size()) << std::endl;

    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> F = generateFacesMatrix(faces, ffStats.faceLists[featureId]);

    solid_angle.init(F.rows(), F.data(), U.size(), &U[0], order);

    //    tbb::parallel_for(tbb::blocked_range<int>(0, P.rows()), [&](const tbb::blocked_range<int>& range) {
    //      for(int p = range.begin(); p != range.end(); ++p)
    for(usize p = 0; p < P.rows(); ++p)
    {
      HDK_Sample::UT_Vector3T<float> Pp;
      Pp[0] = P(p, 0);
      Pp[1] = P(p, 1);
      Pp[2] = P(p, 2);
      float64 val = solid_angle.computeSolidAngle(Pp, accuracy_scale) / (4.0 * M_PI);
      std::cout << P(p, 0) << "," << P(p, 1) << "," << P(p, 2) << "," << val << std::endl;

      if(std::fabs(1.0f - val) <= k_Epsilon)
      {
        featureIds.setValue(p, featureId);
      }
    }
    //    });
  }

  // Write the DREAM.3D file
  writeDream3dFile(k_OutputD3DFilePath, dataStructure);
}
