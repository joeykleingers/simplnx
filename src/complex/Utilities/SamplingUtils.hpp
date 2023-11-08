#pragma once

#include "complex/DataStructure/DataGroup.hpp"
#include "complex/DataStructure/DataPath.hpp"
#include "complex/DataStructure/DataStructure.hpp"
#include "complex/DataStructure/Geometry/ImageGeom.hpp"
#include "complex/Filter/IFilter.hpp"
#include "complex/Utilities/DataGroupUtilities.hpp"
#include "complex/Utilities/Math/GeometryMath.hpp"

namespace complex
{
namespace Sampling
{
struct FeatureFacesStatistics
{
  std::vector<std::vector<int32>> faceLists;
  std::vector<BoundingBox3Df> faceBoundingBoxes;
  int32 maxFeatureId = 0;
  usize numberOfFeatures = 0;
};

inline Result<> RenumberFeatures(DataStructure& dataStructure, const DataPath& newGeomPath, const DataPath& destCellFeatAttributeMatrixPath, const DataPath& featureIdsArrayPath,
                                 const DataPath& destFeatureIdsArrayPath, const std::atomic_bool& shouldCancel = false)
{
  auto& destImageGeom = dataStructure.getDataRefAs<ImageGeom>(newGeomPath);
  // This just sanity checks to make sure there were existing features before the cropping
  auto& destCellFeatureAM = dataStructure.getDataRefAs<AttributeMatrix>(destCellFeatAttributeMatrixPath);

  usize totalPoints = destImageGeom.getNumberOfCells();

  auto& featureIdsArray = dataStructure.getDataRefAs<IDataArray>(featureIdsArrayPath);
  usize totalFeatures = destCellFeatureAM.getNumTuples();
  std::vector<bool> activeObjects(totalFeatures, false);
  if(0 == totalFeatures)
  {
    return MakeErrorResult(-600, "The number of Features is 0 and should be greater than 0");
  }

  auto& destFeatureIdsRef = dataStructure.getDataRefAs<Int32Array>(destFeatureIdsArrayPath);

  auto& featureIds = destFeatureIdsRef.getDataStoreRef();
  // Find the unique set of feature ids
  for(usize i = 0; i < totalPoints; ++i)
  {
    if(shouldCancel)
    {
      break;
    }

    int32 currentFeatureId = featureIds[i];
    if(currentFeatureId < 0)
    {
      std::string ss = fmt::format("FeatureIds values MUST be >= ZERO. Negative FeatureId found at index {} into the resampled feature ids array", i);
      return MakeErrorResult(-605, ss);
    }
    if(static_cast<usize>(currentFeatureId) < totalFeatures)
    {
      activeObjects[currentFeatureId] = true;
    }
    else
    {
      std::string ss = fmt::format("The total number of Features from {} is {}, but a value of {} was found in DataArray {}.", destFeatureIdsArrayPath.getTargetName(), totalFeatures, currentFeatureId,
                                   featureIdsArrayPath.toString());
      std::cout << ss;
      return MakeErrorResult(-602, ss);
    }
  }

  if(!RemoveInactiveObjects(dataStructure, destCellFeatAttributeMatrixPath, activeObjects, destFeatureIdsRef, totalFeatures))
  {
    std::string ss = fmt::format("An error occurred while trying to remove the inactive objects from Attribute Matrix '{}'", destCellFeatAttributeMatrixPath.toString());
    return MakeErrorResult(-606, ss);
  }
  return {};
}

inline FeatureFacesStatistics CalculateFeatureFacesStatistics(const Int32Array& faceLabelsArray, const TriangleGeom& triangleGeom, const std::atomic_bool& shouldCancel,
                                                              const IFilter::MessageHandler& msgHandler)
{
  FeatureFacesStatistics ffStats;

  msgHandler({IFilter::Message::Type::Info, "Counting number of Features..."});

  // pull down faces
  usize numFaces = faceLabelsArray.getNumberOfTuples();

  // walk through faces to see how many features there are
  for(usize i = 0; i < numFaces; i++)
  {
    int32 g1 = faceLabelsArray[2 * i];
    int32 g2 = faceLabelsArray[2 * i + 1];
    if(g1 > ffStats.maxFeatureId)
    {
      ffStats.maxFeatureId = g1;
    }
    if(g2 > ffStats.maxFeatureId)
    {
      ffStats.maxFeatureId = g2;
    }
  }

  // Check for user canceled flag.
  if(shouldCancel)
  {
    return {};
  }

  // add one to account for feature 0
  ffStats.numberOfFeatures = ffStats.maxFeatureId + 1;

  ffStats.faceLists = std::vector<std::vector<int32>>(ffStats.numberOfFeatures);
  msgHandler({IFilter::Message::Type::Info, "Counting number of triangle faces per feature ..."});

  // traverse data to determine number of faces belonging to each feature
  for(usize i = 0; i < numFaces; i++)
  {
    int32 g1 = faceLabelsArray[2 * i];
    int32 g2 = faceLabelsArray[2 * i + 1];
    if(g1 > 0)
    {
      ffStats.faceLists[g1].push_back(0);
    }
    if(g2 > 0)
    {
      ffStats.faceLists[g2].push_back(0);
    }
  }

  // Check for user canceled flag.
  if(shouldCancel)
  {
    return {};
  }

  msgHandler({IFilter::Message::Type::Info, "Allocating triangle faces per feature ..."});

  // fill out lists with number of references to cells
  std::vector<int32> linkLoc(numFaces, 0);

  // traverse data again to get the faces belonging to each feature
  for(int32 i = 0; i < numFaces; i++)
  {
    int32 g1 = faceLabelsArray[2 * i];
    int32 g2 = faceLabelsArray[2 * i + 1];
    if(g1 > 0)
    {
      ffStats.faceLists[g1][(linkLoc[g1])++] = i;
    }
    if(g2 > 0)
    {
      ffStats.faceLists[g2][(linkLoc[g2])++] = i;
    }
    // find bounding box for each face
    ffStats.faceBoundingBoxes.emplace_back(GeometryMath::FindBoundingBoxOfFace(triangleGeom, i));
  }

  return ffStats;
}
} // namespace Sampling
} // namespace complex
