#include "FlyingEdges3D.hpp"

#include "simplnx/Utilities/FilterUtilities.hpp"
#include "simplnx/Utilities/FlyingEdges.hpp"

using namespace nx::core;

namespace
{
struct ExecuteFlyingEdgesFunctor
{
  template <typename T>
  void operator()(const ImageGeom& image, const IDataArray* iDataArray, float64 isoVal, TriangleGeom& triangleGeom, Float32AbstractDataStore& normals, AttributeMatrix& normAM)
  {
    FlyingEdgesAlgorithm flyingEdges = FlyingEdgesAlgorithm<T>(image, iDataArray->template getIDataStoreRefAs<AbstractDataStore<T>>(), static_cast<T>(isoVal), triangleGeom, normals);
    flyingEdges.pass1();
    flyingEdges.pass2();
    flyingEdges.pass3();

    // pass 3 resized normals so be sure to resize parent AM
    normAM.resizeTuples(normals.getTupleShape());

    flyingEdges.pass4();
    triangleGeom.getFaceAttributeMatrix()->resizeTuples({triangleGeom.getNumberOfFaces()});
  }
};
} // namespace

// -----------------------------------------------------------------------------
FlyingEdges3D::FlyingEdges3D(DataStructure& dataStructure, const IFilter::MessageHandler& mesgHandler, const std::atomic_bool& shouldCancel, FlyingEdges3DInputValues* inputValues)
: m_DataStructure(dataStructure)
, m_InputValues(inputValues)
, m_ShouldCancel(shouldCancel)
, m_MessageHandler(mesgHandler)
{
}

// -----------------------------------------------------------------------------
FlyingEdges3D::~FlyingEdges3D() noexcept = default;

// -----------------------------------------------------------------------------
const std::atomic_bool& FlyingEdges3D::getCancel()
{
  return m_ShouldCancel;
}

// -----------------------------------------------------------------------------
Result<> FlyingEdges3D::operator()()
{
  const auto& image = m_DataStructure.getDataRefAs<ImageGeom>(m_InputValues->imageGeomPath);
  float64 isoVal = m_InputValues->isoVal;
  const auto* iDataArray = m_DataStructure.getDataAs<IDataArray>(m_InputValues->contouringArrayPath);
  auto triangleGeom = m_DataStructure.getDataRefAs<TriangleGeom>(m_InputValues->triangleGeomPath);
  auto& normalsStore = m_DataStructure.getDataAs<Float32Array>(m_InputValues->normalsArrayPath)->getDataStoreRef();

  // auto created so must have a parent
  DataPath normAMPath = m_InputValues->normalsArrayPath.getParent();

  auto& normAM = m_DataStructure.getDataRefAs<AttributeMatrix>(normAMPath);

  ExecuteNeighborFunction(ExecuteFlyingEdgesFunctor{}, iDataArray->getDataType(), image, iDataArray, isoVal, triangleGeom, normalsStore, normAM);

  return {};
}
