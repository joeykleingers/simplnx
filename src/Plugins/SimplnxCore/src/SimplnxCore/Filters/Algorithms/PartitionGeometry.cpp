#include "PartitionGeometry.hpp"

#include "simplnx/Common/Range.hpp"
#include "simplnx/DataStructure/Geometry/INodeGeometry3D.hpp"
#include "simplnx/DataStructure/Geometry/ImageGeom.hpp"
#include "simplnx/DataStructure/Geometry/RectGridGeom.hpp"
#include "simplnx/Utilities/ParallelData3DAlgorithm.hpp"
#include "simplnx/Utilities/ParallelDataAlgorithm.hpp"
#include "simplnx/Utilities/StringUtilities.hpp"

#include "SimplnxCore/Filters/PartitionGeometryFilter.hpp"

using namespace nx::core;

namespace
{
Result<> CheckDimensionality(const INodeGeometry0D& geometry)
{
  Result<bool> yzPlaneResult = geometry.isYZPlane();
  if(yzPlaneResult.valid() && yzPlaneResult.value())
  {
    return {MakeErrorResult(-3040, "Unable to create a partitioning scheme with a X dimension size of 0.  Vertices are in an YZ plane.  Use the Advanced or Bounding Box "
                                   "partitioning modes to manually create a partitioning scheme.")};
  }

  Result<bool> xzPlaneResult = geometry.isXZPlane();
  if(xzPlaneResult.valid() && xzPlaneResult.value())
  {
    return {MakeErrorResult(-3041, "Unable to create a partitioning scheme with a Y dimension size of 0.  Vertices are in an XZ plane.  Use the Advanced or Bounding Box "
                                   "partitioning modes to manually create a partitioning scheme.")};
  }

  Result<bool> xyPlaneResult = geometry.isXYPlane();
  if(xyPlaneResult.valid() && xyPlaneResult.value())
  {
    return {MakeErrorResult(-3042, "Unable to create a partitioning scheme with a Z dimension size of 0.  Vertices are in an XY plane.  Use the Advanced or Bounding Box "
                                   "partitioning modes to manually create a partitioning scheme.")};
  }

  return {};
}

/**
 * @brief   Walk a line segment through a 3D grid and pick the cell
 *          in which the segment spends the greatest physical length.
 *
 * Given:
 *   • geom  — an ImageGeom providing:
 *       – origin:   global coordinates of the grid corner [x0,y0,z0]
 *       – spacing:  cell size along each axis [dx,dy,dz]
 *       – dims:     number of cells [nx,ny,nz]
 *   • p0, p1 — two points in global space defining the ends of our segment
 *
 * We:
 *   1. Compute the total segment length L.
 *   2. Determine which cell contains p0 (the “starting cell”).
 *   3. March through the grid cell by cell (Amanatides–Woo algorithm),
 *      accumulating how much of the segment lies in each cell.
 *   4. Return the flattened index of the cell with the maximum accumulated length.
 *
 * @return  A single int32 index of the cell that contains the majority of the line segment.
 */
int32 CalculateMajorityPartitionId(const ImageGeom& geom, Point3D<float32> p0, Point3D<float32> p1)
{
  // ─── 1) Pull out grid metadata ──────────────────────────────────────────────
  auto origin = geom.getOrigin();   // [x0, y0, z0]
  auto spacing = geom.getSpacing(); // [dx, dy, dz]
  auto dims = geom.getDimensions(); // [nx, ny, nz]

  // ─── 2) Compute total physical length of the segment ────────────────────────
  //     L = ||p1 − p0||
  float32 L = std::sqrt((p1.getX() - p0.getX()) * (p1.getX() - p0.getX()) + (p1.getY() - p0.getY()) * (p1.getY() - p0.getY()) + (p1.getZ() - p0.getZ()) * (p1.getZ() - p0.getZ()));

  // ─── 3) Convert p0 into integer cell indices (i,j,k) ───────────────────────
  //     i = floor((x0 − origin_x) / dx), etc.
  int32 i = int32(std::floor((p0.getX() - origin[0]) / spacing[0]));
  int32 j = int32(std::floor((p0.getY() - origin[1]) / spacing[1]));
  int32 k = int32(std::floor((p0.getZ() - origin[2]) / spacing[2]));

  // ─── 4) Determine direction of travel in each axis ─────────────────────────
  //     +1 if p1>p0, −1 if p1<p0, 0 if no movement
  int32 stepX = (p1.getX() > p0.getX() ? 1 : (p1.getX() < p0.getX() ? -1 : 0));
  int32 stepY = (p1.getY() > p0.getY() ? 1 : (p1.getY() < p0.getY() ? -1 : 0));
  int32 stepZ = (p1.getZ() > p0.getZ() ? 1 : (p1.getZ() < p0.getZ() ? -1 : 0));

  // ─── 5) Set up parametric “t” from 0→1 along the segment ────────────────────
  float32 t = 0.0f; // current parametric position (0 at p0, 1 at p1)

  // Helper lambda: compute t at which we hit the next cell boundary on one axis
  auto makeBoundaryT = [&](float32 p0c, float32 dc, float32 ori, float32 sp, int idx, int step) {
    if(step == 0)
    {
      // No movement along this axis → never crosses
      return std::numeric_limits<float32>::infinity();
    }

    // Compute the global coordinate of the next face:
    //   if step>0 → face at (idx+1)*sp; if step<0 → face at idx*sp
    float32 boundary = ori + (idx + (step > 0 ? 1.0f : 0.0f)) * sp;

    // Solve for t in:  p0c + t*dc = boundary
    return (boundary - p0c) / dc;
  };

  // Vector from p0 to p1
  Point3D<float32> d{p1.getX() - p0.getX(), p1.getY() - p0.getY(), p1.getZ() - p0.getZ()};

  // ─── 6) Compute when (t) we exit the starting cell on each axis ────────────
  float32 tMaxX = makeBoundaryT(p0.getX(), d.getX(), origin[0], spacing[0], i, stepX);
  float32 tMaxY = makeBoundaryT(p0.getY(), d.getY(), origin[1], spacing[1], j, stepY);
  float32 tMaxZ = makeBoundaryT(p0.getZ(), d.getZ(), origin[2], spacing[2], k, stepZ);

  // ─── 7) Compute how much t advances to cross one full cell on each axis ───
  float32 tDeltaX = (stepX != 0 ? spacing[0] / std::abs(d.getX()) : std::numeric_limits<float32>::infinity());
  float32 tDeltaY = (stepY != 0 ? spacing[1] / std::abs(d.getY()) : std::numeric_limits<float32>::infinity());
  float32 tDeltaZ = (stepZ != 0 ? spacing[2] / std::abs(d.getZ()) : std::numeric_limits<float32>::infinity());

  // ─── 8) Traverse through cells, accumulating physical length per cell ───────
  std::unordered_map<int64, float32> lengthPerCell;

  while(t < 1.0f)
  {
    // Find the next t at which we either:
    //  • cross an X, Y, or Z face; or
    //  • reach the end of the segment (t=1).
    float32 tNext = std::min({tMaxX, tMaxY, tMaxZ, 1.0f});

    // Physical length inside the current cell = (tNext - t) * totalLength
    float32 segLen = (tNext - t) * L;

    // Flatten (i,j,k) → single index: i + j*nx + k*(nx*ny)
    int64 flat = int64(i) + int64(j) * dims[0] + int64(k) * dims[0] * dims[1];

    // Accumulate
    lengthPerCell[flat] += segLen;

    // Advance along the ray
    t = tNext;

    // Step into the neighboring cell on whichever face we crossed
    if(tNext == tMaxX)
    {
      i += stepX;
      tMaxX += tDeltaX;
    }
    else if(tNext == tMaxY)
    {
      j += stepY;
      tMaxY += tDeltaY;
    }
    else if(tNext == tMaxZ)
    {
      k += stepZ;
      tMaxZ += tDeltaZ;
    }
    else
    {
      // tNext == 1.0 → segment end reached
      break;
    }
  }

  // ─── 9) Pick the cell with the maximum accumulated length ─────────────────
  int32 bestFlat = -1;
  float32 bestLen = -1.0f;
  for(auto& [flat, len] : lengthPerCell)
  {
    if(len > bestLen)
    {
      bestLen = len;
      bestFlat = static_cast<int32>(flat);
    }
  }

  return bestFlat;
}

// -----------------------------------------------------------------------------
class PartitionCellBasedGeometryImpl
{
public:
  PartitionCellBasedGeometryImpl(const IGridGeometry& inputGeometry, Int32AbstractDataStore& partitionIdsStore, const ImageGeom& psImageGeom, int startingPartitionId, int outOfBoundsValue,
                                 const std::atomic_bool& shouldCancel)
  : m_InputGeometry(inputGeometry)
  , m_PartitionIdsStore(partitionIdsStore)
  , m_PSImageGeom(psImageGeom)
  , m_StartingPartitionId(startingPartitionId)
  , m_OutOfBoundsValue(outOfBoundsValue)
  , m_ShouldCancel(shouldCancel)
  {
  }

  // -----------------------------------------------------------------------------
  void compute(size_t xStart, size_t xEnd, size_t yStart, size_t yEnd, size_t zStart, size_t zEnd) const
  {
    SizeVec3 dims = m_InputGeometry.getDimensions();

    for(usize z = zStart; z < zEnd; z++)
    {
      for(usize y = yStart; y < yEnd; y++)
      {
        for(usize x = xStart; x < xEnd; x++)
        {
          if(m_ShouldCancel)
          {
            return;
          }

          const usize index = (z * dims[1] * dims[0]) + (y * dims[0]) + x;

          Point3D<float64> coord = m_InputGeometry.getCoords(x, y, z);
          auto partitionIndexResult = m_PSImageGeom.getIndex(coord[0], coord[1], coord[2]);
          if(partitionIndexResult.has_value())
          {
            m_PartitionIdsStore[index] = static_cast<int32>(*partitionIndexResult) + m_StartingPartitionId;
          }
          else
          {
            m_PartitionIdsStore[index] = m_OutOfBoundsValue;
          }
        }
      }
    }
  }

  void operator()(const Range3D& r) const
  {
    compute(r[0], r[1], r[2], r[3], r[4], r[5]);
  }

private:
  const IGridGeometry& m_InputGeometry;
  Int32AbstractDataStore& m_PartitionIdsStore;
  const ImageGeom& m_PSImageGeom;
  int m_StartingPartitionId;
  int m_OutOfBoundsValue;
  const std::atomic_bool& m_ShouldCancel;
};

// -----------------------------------------------------------------------------
class PartitionVerticesImpl
{
public:
  PartitionVerticesImpl(const PartitionGeometry::VertexStore& verticesStore, Int32AbstractDataStore& partitionIdsStore, const ImageGeom& psImageGeom, int32 startingPartitionId,
                        int32 defaultPartitionId, const std::optional<const BoolArray>& maskArrayOpt, const std::atomic_bool& shouldCancel)
  : m_VerticesStore(verticesStore)
  , m_PartitionIdsStore(partitionIdsStore)
  , m_PSImageGeom(psImageGeom)
  , m_StartingPartitionId(startingPartitionId)
  , m_DefaultPartitionID(defaultPartitionId)
  , m_MaskArrayOpt(maskArrayOpt)
  , m_ShouldCancel(shouldCancel)
  {
  }

  // -----------------------------------------------------------------------------
  void compute(size_t start, size_t end) const
  {
    for(usize idx = start; idx < end; idx++)
    {
      if(m_ShouldCancel)
      {
        return;
      }

      const float32 x = m_VerticesStore[idx * 3];
      const float32 y = m_VerticesStore[idx * 3 + 1];
      const float32 z = m_VerticesStore[idx * 3 + 2];

      auto partitionIndexResult = m_PSImageGeom.getIndex(x, y, z);
      if((m_MaskArrayOpt.has_value() && !(*m_MaskArrayOpt)[idx]) || !partitionIndexResult.has_value())
      {
        m_PartitionIdsStore[idx] = m_DefaultPartitionID;
      }
      else
      {
        m_PartitionIdsStore[idx] = static_cast<int32>(*partitionIndexResult) + m_StartingPartitionId;
      }
    }
  }

  void operator()(const Range& range) const
  {
    compute(range.min(), range.max());
  }

private:
  const PartitionGeometry::VertexStore& m_VerticesStore;
  Int32AbstractDataStore& m_PartitionIdsStore;
  const ImageGeom& m_PSImageGeom;
  int32 m_StartingPartitionId;
  int32 m_DefaultPartitionID;
  const std::optional<const BoolArray>& m_MaskArrayOpt;
  const std::atomic_bool& m_ShouldCancel;
};

// -----------------------------------------------------------------------------
class PartitionEdgesImpl
{
public:
  PartitionEdgesImpl(const PartitionGeometry::EdgesStore& edgesStore, const PartitionGeometry::VertexStore& verticesStore, Int32AbstractDataStore& partitionIdsStore, const ImageGeom& psImageGeom,
                     int32 startingPartitionId, int32 defaultPartitionID, const std::optional<const BoolArray>& maskArrayOpt,
                     PartitionGeometry::BoundaryIntersectionBehavior boundaryIntersectionBehavior, std::optional<PartitionGeometry::BoundaryIntersectionMetadata>& biMetadata,
                     const std::atomic_bool& shouldCancel)
  : m_EdgesStore(edgesStore)
  , m_VerticesStore(verticesStore)
  , m_PartitionIdsStore(partitionIdsStore)
  , m_PSImageGeom(psImageGeom)
  , m_StartingPartitionId(startingPartitionId)
  , m_DefaultPartitionID(defaultPartitionID)
  , m_MaskArrayOpt(maskArrayOpt)
  , m_BoundaryIntersectionBehavior(boundaryIntersectionBehavior)
  , m_BoundaryIntersectionMetadata(biMetadata)
  , m_ShouldCancel(shouldCancel)
  {
  }

  // -----------------------------------------------------------------------------
  void compute(size_t start, size_t end) const
  {
    for(usize edgeIdx = start; edgeIdx < end; edgeIdx++)
    {
      if(m_ShouldCancel || m_BoundaryIntersectionMetadata.has_value())
      {
        return;
      }

      const uint32 vertex1Idx = m_EdgesStore[edgeIdx * 2 + 0];
      const float32 x1 = m_VerticesStore[vertex1Idx * 3 + 0];
      const float32 y1 = m_VerticesStore[vertex1Idx * 3 + 1];
      const float32 z1 = m_VerticesStore[vertex1Idx * 3 + 2];
      auto v1PartitionValueResult = m_PSImageGeom.getIndex(x1, y1, z1);

      const uint32 vertex2Idx = m_EdgesStore[edgeIdx * 2 + 1];
      const float32 x2 = m_VerticesStore[vertex2Idx * 3 + 0];
      const float32 y2 = m_VerticesStore[vertex2Idx * 3 + 1];
      const float32 z2 = m_VerticesStore[vertex2Idx * 3 + 2];
      auto v2PartitionValueResult = m_PSImageGeom.getIndex(x2, y2, z2);

      bool edgeMaskedOut = m_MaskArrayOpt.has_value() && !(*m_MaskArrayOpt)[edgeIdx];
      bool v1OutOfBounds = !v1PartitionValueResult.has_value();
      bool v2OutOfBounds = !v2PartitionValueResult.has_value();
      if(edgeMaskedOut || v1OutOfBounds || v2OutOfBounds || m_BoundaryIntersectionBehavior == PartitionGeometry::BoundaryIntersectionBehavior::IgnoreEdge)
      {
        m_PartitionIdsStore[edgeIdx] = m_DefaultPartitionID;
        continue;
      }

      auto v1PartitionValue = static_cast<int32>(v1PartitionValueResult.value());
      auto v2PartitionValue = static_cast<int32>(v2PartitionValueResult.value());

      if(v1PartitionValue == v2PartitionValue)
      {
        // Edge endpoints are contained in the same partition
        m_PartitionIdsStore[edgeIdx] = v1PartitionValue + m_StartingPartitionId;
      }
      else
      {
        // Edge endpoints are NOT contained in the same partition
        if(m_BoundaryIntersectionBehavior == PartitionGeometry::BoundaryIntersectionBehavior::AssignToMajorityPartition)
        {
          // Assign the ID of the partition that contains the majority of the edge
          int32 majorityPartitionIdResult = CalculateMajorityPartitionId(m_PSImageGeom, Point3D<float32>{x1, y1, z1}, Point3D<float32>{x2, y2, z2});
          m_PartitionIdsStore[edgeIdx] = majorityPartitionIdResult + m_StartingPartitionId;
        }
        else
        {
          // Set the boundary intersection metadata so that we can cancel and return a filter error
          m_BoundaryIntersectionMetadata = PartitionGeometry::BoundaryIntersectionMetadata{v1PartitionValue, v2PartitionValue, edgeIdx, FloatVec3{x1, y1, z1}, FloatVec3{x2, y2, z2}};
          return;
        }
      }
    }
  }

  void operator()(const Range& range) const
  {
    compute(range.min(), range.max());
  }

private:
  const PartitionGeometry::EdgesStore& m_EdgesStore;
  const PartitionGeometry::VertexStore& m_VerticesStore;
  Int32AbstractDataStore& m_PartitionIdsStore;
  const ImageGeom& m_PSImageGeom;
  int32 m_StartingPartitionId;
  int32 m_DefaultPartitionID;
  const std::optional<const BoolArray>& m_MaskArrayOpt;
  PartitionGeometry::BoundaryIntersectionBehavior m_BoundaryIntersectionBehavior;
  std::optional<PartitionGeometry::BoundaryIntersectionMetadata>& m_BoundaryIntersectionMetadata;
  const std::atomic_bool& m_ShouldCancel;
};

template <typename G>
concept GridGeom = std::derived_from<G, IGridGeometry>;

template <typename G>
concept Node3DGeom = std::derived_from<G, INodeGeometry3D>;

template <typename G>
concept Node2DGeom = std::derived_from<G, INodeGeometry2D> && !Node3DGeom<G>;

template <typename G>
concept Node1DGeom = std::derived_from<G, INodeGeometry1D> && !Node2DGeom<G>;

template <typename G>
concept Node0DGeom = std::derived_from<G, INodeGeometry0D> && !Node1DGeom<G>;

/// Cells: ImageGeom & RectGridGeom both satisfy IGridGeometry
template <GridGeom Geom>
Result<> partitionGeometry(const Geom& geom, Int32AbstractDataStore& partitionIdsStore, const ImageGeom& partitionGridGeom, int32 startingFeatureId, int32 outOfBoundsID,
                           const std::atomic_bool& shouldCancel)
{
  SizeVec3 dims = geom.getDimensions();

  IParallelAlgorithm::AlgorithmStores algStores;
  algStores.push_back(&partitionIdsStore);

  ParallelData3DAlgorithm dataAlg;
  dataAlg.setRange(dims[0], dims[1], dims[2]);
  dataAlg.requireStoresInMemory(algStores);
  dataAlg.execute(PartitionCellBasedGeometryImpl(geom, partitionIdsStore, partitionGridGeom, startingFeatureId, outOfBoundsID, shouldCancel));

  return {};
}

/**
 * @brief Partitions a vertex geometry according to the partitioning scheme geometry provided,
 * and stores the assigned partition ids in the partitionIds array.
 *
 * If a given vertex is outside the partitioning scheme bounds and an out of bounds value
 * is provided, the vertex will be labeled with the out of bounds value.  Otherwise,
 * the function will return an invalid Result with an error message.
 *
 * @param vertexListStore The list of vertices from the node-based geometry
 * @param partitionIdsStore The partition ids array that stores the results.
 * @param psImageGeom The partitioning scheme image geometry that is used
 * to partition the vertex list.
 * @param defaultPartitionId Value that ignored and out-of-bounds vertices will be labeled with
 * @param maskArrayOpt Optional mask array
 * @return The result of the partitioning algorithm.  Valid if successful, invalid
 * if there was an error.
 */
template <Node0DGeom Geom>
Result<> partitionGeometry(const Geom& geom, Int32AbstractDataStore& partitionIdsStore, const ImageGeom& partitionGridGeom, int32 startingFeatureId, int32 defaultPartitionId,
                           const std::optional<const BoolArray>& maskArrayOpt, const std::atomic_bool& shouldCancel)
{
  auto dimRes = CheckDimensionality(geom);
  if(dimRes.invalid())
  {
    return dimRes;
  }

  auto const& verts = geom.getVerticesRef().getDataStoreRef();

  IParallelAlgorithm::AlgorithmStores algStores;
  algStores.push_back(&verts);
  algStores.push_back(&partitionIdsStore);

  IParallelAlgorithm::AlgorithmArrays algArrays;
  if(maskArrayOpt.has_value())
  {
    algArrays.push_back(&(maskArrayOpt.value()));
  }

  // Allow data-based parallelization
  ParallelDataAlgorithm dataAlg;
  dataAlg.setRange(0, verts.getNumberOfTuples());
  dataAlg.requireArraysInMemory(algArrays);
  dataAlg.requireStoresInMemory(algStores);
  dataAlg.execute(PartitionVerticesImpl(verts, partitionIdsStore, partitionGridGeom, startingFeatureId, defaultPartitionId, maskArrayOpt, shouldCancel));

  return {};
}

/**
 * @brief Partitions an edge geometry according to the partitioning scheme geometry provided,
 * and stores the assigned partition ids in the partitionIds array.
 *
 * If a given edge is completely outside the partitioning scheme bounds and an out
 * of bounds value is provided, the edge will be labeled with the out of bounds value.
 * If a given edge intersects the partitioning scheme bounds, the edge will be handled based
 * on the chosen boundary intersection behavior.  Otherwise, the function will return an
 * invalid Result with an error message.
 *
 * @param geom The edge geometry to partition
 * @param partitionIdsStore The partition ids array that stores the results.
 * @param partitionGridGeom The partitioning scheme image geometry that is used
 * to partition the edges list.
 * @param startingFeatureId The feature ID that will be used first when partitioning
 * @param defaultPartitionId Value that ignored and out-of-bounds edges will be labeled with
 * @param bbBehavior The boundary intersection behavior to handle cases where the edge
 * intersects a partition boundary
 * @param maskArrayOpt Optional mask array
 * @param shouldCancel Value used to cancel this method's processing
 * @return The result of the partitioning algorithm.  Valid if successful, invalid
 * if there was an error.
 */
template <Node1DGeom Geom>
Result<> partitionGeometry(const Geom& geom, Int32AbstractDataStore& partitionIdsStore, const ImageGeom& partitionGridGeom, int32 startingFeatureId, int32 defaultPartitionId,
                           PartitionGeometry::BoundaryIntersectionBehavior bbBehavior, const std::optional<const BoolArray>& maskArrayOpt, const std::atomic_bool& shouldCancel)
{
  // exactly the old Edge branch:
  auto dimRes = CheckDimensionality(geom);
  if(dimRes.invalid())
  {
    return dimRes;
  }

  auto const& verts = geom.getVerticesRef().getDataStoreRef();
  auto const& edges = geom.getEdgesRef().getDataStoreRef();

  IParallelAlgorithm::AlgorithmStores algStores;
  algStores.push_back(&edges);
  algStores.push_back(&verts);
  algStores.push_back(&partitionIdsStore);

  IParallelAlgorithm::AlgorithmArrays algArrays;
  if(maskArrayOpt.has_value())
  {
    algArrays.push_back(&(maskArrayOpt.value()));
  }

  // Allow data-based parallelization
  ParallelDataAlgorithm dataAlg;
  dataAlg.setRange(0, edges.getNumberOfTuples());
  dataAlg.requireArraysInMemory(algArrays);
  dataAlg.requireStoresInMemory(algStores);

  std::optional<PartitionGeometry::BoundaryIntersectionMetadata> biMetadata;
  dataAlg.execute(PartitionEdgesImpl(edges, verts, partitionIdsStore, partitionGridGeom, startingFeatureId, defaultPartitionId, maskArrayOpt, bbBehavior, biMetadata, shouldCancel));
  if(biMetadata.has_value())
  {
    PartitionGeometry::BoundaryIntersectionMetadata biMetadataVal = biMetadata.value();
    return MakeErrorResult(-34, fmt::format("Unable to partition edge geometry '{}': Detected an intersection between edge (index {}) and partition cells (IDs {} and {}).", geom.getName(),
                                            biMetadataVal.edgeId, biMetadataVal.partitionId1, biMetadataVal.partitionId2));
  }

  return {};
}
} // namespace

// -----------------------------------------------------------------------------
PartitionGeometry::PartitionGeometry(DataStructure& dataStructure, const IFilter::MessageHandler& mesgHandler, const std::atomic_bool& shouldCancel, PartitionGeometryInputValues* inputValues)
: m_DataStructure(dataStructure)
, m_InputValues(inputValues)
, m_ShouldCancel(shouldCancel)
, m_MessageHandler(mesgHandler)
{
}

// -----------------------------------------------------------------------------
PartitionGeometry::~PartitionGeometry() noexcept = default;

// -----------------------------------------------------------------------------
const std::atomic_bool& PartitionGeometry::getCancel()
{
  return m_ShouldCancel;
}

// -----------------------------------------------------------------------------
Result<> PartitionGeometry::operator()()
{
  auto partitioningMode = static_cast<PartitionGeometryFilter::PartitioningMode>(m_InputValues->PartitioningMode);
  auto boundaryIntersectionBehavior = static_cast<PartitionGeometry::BoundaryIntersectionBehavior>(m_InputValues->BoundaryIntersectionBehavior);

  DataPath partitionGridGeomPath;
  if(partitioningMode == PartitionGeometryFilter::PartitioningMode::ExistingPartitionGrid)
  {
    partitionGridGeomPath = m_InputValues->ExistingPartitionGridPath;
  }
  else
  {
    partitionGridGeomPath = m_InputValues->PartitionGridGeomPath;
    const DataPath partitionGridFeatureIdsPath =
        m_InputValues->PartitionGridGeomPath.createChildPath(m_InputValues->PartitionGridCellAMName).createChildPath(m_InputValues->PartitionGridFeatureIDsArrayName);
    auto& pgFeatureIdsStore = m_DataStructure.getDataAs<Int32Array>(partitionGridFeatureIdsPath)->getDataStoreRef();

    for(usize i = 0; i < pgFeatureIdsStore.getNumberOfTuples(); i++)
    {
      pgFeatureIdsStore[i] = static_cast<int32>(i) + m_InputValues->StartingFeatureID;
    }
  }

  const ImageGeom& partitionGridGeom = m_DataStructure.getDataRefAs<ImageGeom>({partitionGridGeomPath});

  std::optional<BoolArray> vertexMask = {};
  if(m_InputValues->UseVertexMask)
  {
    vertexMask = m_DataStructure.getDataRefAs<BoolArray>(m_InputValues->VertexMaskPath);
  }

  const DataPath partitionIdsPath = m_InputValues->InputGeomCellAMPath.createChildPath(m_InputValues->PartitionIdsArrayName);
  auto& partitionIdsStore = m_DataStructure.getDataAs<Int32Array>(partitionIdsPath)->getDataStoreRef();

  const IGeometry& iGeomToPartition = m_DataStructure.getDataRefAs<IGeometry>(m_InputValues->InputGeometryToPartition);
  Result<> result;
  switch(iGeomToPartition.getGeomType())
  {
  case IGeometry::Type::Image: {
    const ImageGeom& inputGeomToPartition = m_DataStructure.getDataRefAs<ImageGeom>({m_InputValues->InputGeometryToPartition});
    result = partitionGeometry(inputGeomToPartition, partitionIdsStore, partitionGridGeom, m_InputValues->StartingFeatureID, m_InputValues->DefaultFeatureID, m_ShouldCancel);
    break;
  }
  case IGeometry::Type::RectGrid: {
    const RectGridGeom& inputGeomToPartition = m_DataStructure.getDataRefAs<RectGridGeom>({m_InputValues->InputGeometryToPartition});
    result = partitionGeometry(inputGeomToPartition, partitionIdsStore, partitionGridGeom, m_InputValues->StartingFeatureID, m_InputValues->DefaultFeatureID, m_ShouldCancel);
    break;
  }
  case IGeometry::Type::Vertex: {
    const INodeGeometry0D& inputGeomToPartition = m_DataStructure.getDataRefAs<INodeGeometry0D>({m_InputValues->InputGeometryToPartition});
    const AbstractDataStore<IGeometry::SharedVertexList::value_type>& vertexListStore = inputGeomToPartition.getVertices()->getDataStoreRef();
    result = partitionGeometry(inputGeomToPartition, partitionIdsStore, partitionGridGeom, m_InputValues->StartingFeatureID, m_InputValues->DefaultFeatureID, vertexMask, m_ShouldCancel);
    break;
  }
  case IGeometry::Type::Edge: {
    const INodeGeometry1D& inputGeomToPartition = m_DataStructure.getDataRefAs<INodeGeometry1D>({m_InputValues->InputGeometryToPartition});
    const AbstractDataStore<IGeometry::SharedVertexList::value_type>& vertexListStore = inputGeomToPartition.getVertices()->getDataStoreRef();
    auto& edgesListStore = inputGeomToPartition.getEdges()->getDataStoreRef();
    result = partitionGeometry(inputGeomToPartition, partitionIdsStore, partitionGridGeom, m_InputValues->StartingFeatureID, m_InputValues->DefaultFeatureID, boundaryIntersectionBehavior, vertexMask,
                               m_ShouldCancel);
    break;
  }
  default: {
    return {MakeErrorResult(-3012, fmt::format("Unable to partition geometry - Geometry type '{}' not supported by this filter.", IGeometry::GeomTypeToString(iGeomToPartition.getGeomType())))};
  }
  }

  if(result.invalid())
  {
    return result;
  }

  return {};
}
