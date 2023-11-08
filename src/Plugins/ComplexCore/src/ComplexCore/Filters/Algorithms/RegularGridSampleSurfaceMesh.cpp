#include "RegularGridSampleSurfaceMesh.hpp"

#include <Eigen/Core>

#include "ComplexCore/WindingNumber/UT_SolidAngle.h"

#include "complex/DataStructure/Geometry/ImageGeom.hpp"
#include "complex/DataStructure/Geometry/TriangleGeom.hpp"
#include "complex/Utilities/ParallelDataAlgorithm.hpp"
#include "complex/Utilities/SamplingUtils.hpp"

using namespace complex;

namespace
{
constexpr double k_Epsilon = 0.1;

void sendThreadSafeProgressMessage(usize numCompleted, usize totalFeatures, std::mutex& progressMessageMutex, usize& progressCounter, usize& lastProgressInt,
                                   std::chrono::steady_clock::time_point& initialTime, const IFilter::MessageHandler& msgHandler)
{
  std::lock_guard<std::mutex> lock(progressMessageMutex);

  progressCounter += numCompleted;
  auto now = std::chrono::steady_clock::now();
  auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(now - initialTime).count();
  if(diff > 1000)
  {
    std::string progMessage = fmt::format("Points Completed: {} of {}", progressCounter, totalFeatures);
    float inverseRate = static_cast<float>(diff) / static_cast<float>(progressCounter - lastProgressInt);
    auto remainMillis = std::chrono::milliseconds(static_cast<int64>(inverseRate * (totalFeatures - progressCounter)));
    auto secs = std::chrono::duration_cast<std::chrono::seconds>(remainMillis);
    remainMillis -= std::chrono::duration_cast<std::chrono::milliseconds>(secs);
    auto mins = std::chrono::duration_cast<std::chrono::minutes>(secs);
    secs -= std::chrono::duration_cast<std::chrono::seconds>(mins);
    auto hour = std::chrono::duration_cast<std::chrono::hours>(mins);
    mins -= std::chrono::duration_cast<std::chrono::minutes>(hour);
    progMessage += fmt::format(" || Est. Time Remain: {} hours {} minutes {} seconds", hour.count(), mins.count(), secs.count());
    msgHandler({IFilter::Message::Type::Info, progMessage});
    initialTime = std::chrono::steady_clock::now();
    lastProgressInt = progressCounter;
  }
}

class WindingNumberImpl
{
public:
  WindingNumberImpl(int32 featureId, const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& P, const HDK_Sample::UT_SolidAngle<float, float>& solid_angle, double accuracy_scale,
                    const Int32Array& faceLabels, Int32Array& featureIds)
  : m_FeatureId(featureId)
  , m_P(P)
  , m_SolidAngle(solid_angle)
  , m_AccuracyScale(accuracy_scale)
  , m_FaceLabels(faceLabels)
  , m_FeatureIds(featureIds)
  {
  }

  void operator()(const Range& range) const
  {
    for(size_t p = range.min(); p < range.max(); ++p)
    {
      //      m_PointsVisited++;

      HDK_Sample::UT_Vector3T<float> Pp;
      Pp[0] = m_P(p, 0);
      Pp[1] = m_P(p, 1);
      Pp[2] = m_P(p, 2);
      float64 val = m_SolidAngle.computeSolidAngle(Pp, m_AccuracyScale) / (4.0 * M_PI);

      usize comp0Idx = p * m_FaceLabels.getNumberOfComponents() + 0;
      usize comp1Idx = p * m_FaceLabels.getNumberOfComponents() + 1;
      int32 outsideValue = (m_FaceLabels[comp0Idx] != m_FeatureId) ? m_FaceLabels[comp0Idx] : m_FaceLabels[comp1Idx];
      int32 insideValue = m_FeatureId;

      m_FeatureIds.setValue(p, (std::fabs(1.0f - val) <= k_Epsilon) ? insideValue : outsideValue);

      //      // Send some feedback
      //      if(m_PointsVisited % 1000 == 0)
      //      {
      //        sendThreadSafeProgressMessage(1000, m_P.rows(), m_ProgressMessageMutex, m_ProgressCounter, m_LastProgressInt, m_InitialTime, m_MessageHandler);
      //      }
      // Check for the filter being cancelled.
      //      if(m_ShouldCancel)
      //      {
      //        return;
      //      }
    }
  }

private:
  int32 m_FeatureId;
  const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& m_P;
  const HDK_Sample::UT_SolidAngle<float, float>& m_SolidAngle;
  double m_AccuracyScale;
  const Int32Array& m_FaceLabels;
  Int32Array& m_FeatureIds;
};

std::vector<HDK_Sample::UT_Vector3T<float>> generateVerticesVector(DataArray<float32>& verticesArray)
{
  auto numVertices = verticesArray.getNumberOfTuples();
  std::vector<HDK_Sample::UT_Vector3T<float>> U(numVertices);

  for(int i = 0; i < numVertices; i++)
  {
    for(int j = 0; j < 3; j++)
    {
      U[i][j] = verticesArray.getComponent(i, j);
    }
  }

  return U;
}

Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> generateFacesMatrix(DataArray<IGeometry::MeshIndexType>& facesArray, std::vector<int32> faceIndices)
{
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> F(faceIndices.size(), facesArray.getNumberOfComponents());

  for(int i = 0; i < F.rows(); i++)
  {
    for(int j = 0; j < F.cols(); j++)
    {
      F(i, j) = facesArray.getComponent(faceIndices[i], j);
    }
  }

  return F;
}

Eigen::MatrixXf generatePointsMatrix(const ImageGeom& imageGeom)
{
  Vec3<usize> dims = imageGeom.getDimensions();

  usize numOfPoints = std::accumulate(dims.begin(), dims.end(), static_cast<usize>(1), std::multiplies<>());
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> P(numOfPoints, 3);

  usize rowIndex = 0;
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
      }
    }
  }

  return P;
}

} // namespace

// -----------------------------------------------------------------------------
RegularGridSampleSurfaceMesh::RegularGridSampleSurfaceMesh(DataStructure& dataStructure, const IFilter::MessageHandler& msgHandler, const std::atomic_bool& shouldCancel,
                                                           RegularGridSampleSurfaceMeshInputValues* inputValues)
: m_DataStructure(dataStructure)
, m_InputValues(inputValues)
, m_ShouldCancel(shouldCancel)
, m_MessageHandler(msgHandler)
{
}

// -----------------------------------------------------------------------------
RegularGridSampleSurfaceMesh::~RegularGridSampleSurfaceMesh() noexcept = default;

// -----------------------------------------------------------------------------
const std::atomic_bool& RegularGridSampleSurfaceMesh::getCancel()
{
  return m_ShouldCancel;
}

// -----------------------------------------------------------------------------
Result<> RegularGridSampleSurfaceMesh::operator()()
{
  auto& triangleGeom = m_DataStructure.getDataRefAs<TriangleGeom>(m_InputValues->TriangleGeometryPath);
  auto& faceLabels = m_DataStructure.getDataRefAs<Int32Array>(m_InputValues->SurfaceMeshFaceLabelsArrayPath);
  auto numVertices = triangleGeom.getNumberOfVertices();
  auto& vertices = triangleGeom.getVerticesRef();
  auto numFaces = triangleGeom.getNumberOfFaces();
  auto& faces = triangleGeom.getFacesRef();
  auto& imageGeom = m_DataStructure.getDataRefAs<ImageGeom>(m_InputValues->ImageGeometryPath);

  complex::Sampling::FeatureFacesStatistics ffStats = complex::Sampling::CalculateFeatureFacesStatistics(faceLabels, triangleGeom, m_ShouldCancel, m_MessageHandler);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> P = generatePointsMatrix(imageGeom);
  for(int32 featureId = 0; featureId < ffStats.faceLists.size(); featureId++)
  {
    std::vector<int32> firstFaceList = ffStats.faceLists[0];
    std::vector<usize> vertexIds(3);
    std::vector<float32> vertCoords(3);
    for(usize fIdx = 0; fIdx < firstFaceList.size(); fIdx++)
    {
      triangleGeom.getFacePointIds(firstFaceList[fIdx], vertexIds);
      for(usize vIdx = 0; vIdx < vertexIds.size(); vIdx++)
      {
        vertCoords[vIdx] = vertexIds[vIdx];
      }
    }

    m_MessageHandler({IFilter::Message::Type::Info, fmt::format("Sampling Feature ID {} ({}/{})...", featureId, featureId + 1, ffStats.faceLists.size())});

    HDK_Sample::UT_SolidAngle<float, float> solid_angle;
    int order = 2;
    double accuracy_scale = 2.0;

    std::vector<HDK_Sample::UT_Vector3T<float>> U = generateVerticesVector(vertices);
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> F = generateFacesMatrix(faces, ffStats.faceLists[featureId]);

    solid_angle.init(static_cast<int>(numFaces), F.data(), static_cast<int>(numVertices), &U[0], order);

    usize pointsVisited = 0;
    std::mutex progressMessageMutex;
    usize progressCounter = 0;
    usize lastProgressInt = 0;
    std::chrono::steady_clock::time_point initialTime = std::chrono::steady_clock::now();

    auto& featureIds = m_DataStructure.getDataRefAs<Int32Array>(m_InputValues->FeatureIdsArrayPath);

    ParallelDataAlgorithm parallelAlgorithm;
    parallelAlgorithm.setRange(0, P.rows());
    parallelAlgorithm.execute(::WindingNumberImpl(featureId, P, solid_angle, accuracy_scale, faceLabels, featureIds));
  }

  return {};

  //  SampleSurfaceMeshInputValues inputs;
  //  inputs.TriangleGeometryPath = m_InputValues->TriangleGeometryPath;
  //  inputs.SurfaceMeshFaceLabelsArrayPath = m_InputValues->SurfaceMeshFaceLabelsArrayPath;
  //  inputs.FeatureIdsArrayPath = m_InputValues->FeatureIdsArrayPath;
  //  return execute(inputs);
}
