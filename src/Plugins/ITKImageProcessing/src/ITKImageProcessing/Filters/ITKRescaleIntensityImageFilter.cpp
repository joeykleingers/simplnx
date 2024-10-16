#include "ITKRescaleIntensityImageFilter.hpp"

#include "ITKImageProcessing/Common/ITKArrayHelper.hpp"
#include "ITKImageProcessing/Common/sitkCommon.hpp"

#include "simplnx/Parameters/ArraySelectionParameter.hpp"
#include "simplnx/Parameters/DataGroupSelectionParameter.hpp"
#include "simplnx/Parameters/DataObjectNameParameter.hpp"
#include "simplnx/Parameters/GeometrySelectionParameter.hpp"
#include "simplnx/Parameters/NumberParameter.hpp"
#include "simplnx/Parameters/StringParameter.hpp"

#include "simplnx/Utilities/SIMPLConversion.hpp"

#include <itkRescaleIntensityImageFilter.h>

using namespace nx::core;

namespace cxITKRescaleIntensityImageFilter
{
using ArrayOptionsType = ITK::ScalarPixelIdTypeList;
// VectorPixelIDTypeList;

struct ITKRescaleIntensityImageFunctor
{
  float64 outputMinimum = 0;
  float64 outputMaximum = 255;

  template <class InputImageT, class OutputImageT, uint32 Dimension>
  auto createFilter() const
  {
    using FilterType = itk::RescaleIntensityImageFilter<InputImageT, OutputImageT>;
    auto filter = FilterType::New();
    filter->SetOutputMinimum(outputMinimum);
    filter->SetOutputMaximum(outputMaximum);
    return filter;
  }
};
} // namespace cxITKRescaleIntensityImageFilter

namespace nx::core
{
//------------------------------------------------------------------------------
std::string ITKRescaleIntensityImageFilter::name() const
{
  return FilterTraits<ITKRescaleIntensityImageFilter>::name;
}

//------------------------------------------------------------------------------
std::string ITKRescaleIntensityImageFilter::className() const
{
  return FilterTraits<ITKRescaleIntensityImageFilter>::className;
}

//------------------------------------------------------------------------------
Uuid ITKRescaleIntensityImageFilter::uuid() const
{
  return FilterTraits<ITKRescaleIntensityImageFilter>::uuid;
}

//------------------------------------------------------------------------------
std::string ITKRescaleIntensityImageFilter::humanName() const
{
  return "ITK Rescale Intensity Image Filter";
}

//------------------------------------------------------------------------------
std::vector<std::string> ITKRescaleIntensityImageFilter::defaultTags() const
{
  return {className(), "ITKImageProcessing", "ITKRescaleIntensityImage", "ITKImageIntensity", "ImageIntensity"};
}

//------------------------------------------------------------------------------
Parameters ITKRescaleIntensityImageFilter::parameters() const
{
  Parameters params;
  params.insertSeparator(Parameters::Separator{"Input Parameter(s)"});
  params.insert(std::make_unique<Float64Parameter>(k_OutputMinimum_Key, "Output Minimum", "The minimum output value that is used.", 0));
  params.insert(std::make_unique<Float64Parameter>(k_OutputMaximum_Key, "Output Maximum", "The maximum output value that is used.", 255));

  params.insertSeparator(Parameters::Separator{"Input Cell Data"});
  params.insert(std::make_unique<GeometrySelectionParameter>(k_InputImageGeomPath_Key, "Image Geometry", "Select the Image Geometry Group from the DataStructure.", DataPath({"Image Geometry"}),
                                                             GeometrySelectionParameter::AllowedTypes{IGeometry::Type::Image}));
  params.insert(std::make_unique<ArraySelectionParameter>(k_InputImageDataPath_Key, "Input Cell Data", "The image data that will be processed by this filter.", DataPath{},
                                                          nx::core::ITK::GetScalarPixelAllowedTypes()));

  params.insertSeparator(Parameters::Separator{"Output Cell Data"});
  params.insert(std::make_unique<DataObjectNameParameter>(k_OutputImageArrayName_Key, "Output Cell Data",
                                                          "The result of the processing will be stored in this Data Array inside the same group as the input data.", "Output Image Data"));

  return params;
}

//------------------------------------------------------------------------------
IFilter::VersionType ITKRescaleIntensityImageFilter::parametersVersion() const
{
  return 1;
}

//------------------------------------------------------------------------------
IFilter::UniquePointer ITKRescaleIntensityImageFilter::clone() const
{
  return std::make_unique<ITKRescaleIntensityImageFilter>();
}

//------------------------------------------------------------------------------
IFilter::PreflightResult ITKRescaleIntensityImageFilter::preflightImpl(const DataStructure& dataStructure, const Arguments& filterArgs, const MessageHandler& messageHandler,
                                                                       const std::atomic_bool& shouldCancel) const
{
  auto imageGeomPath = filterArgs.value<DataPath>(k_InputImageGeomPath_Key);
  auto selectedInputArray = filterArgs.value<DataPath>(k_InputImageDataPath_Key);
  auto outputArrayName = filterArgs.value<DataObjectNameParameter::ValueType>(k_OutputImageArrayName_Key);
  auto outputMinimum = filterArgs.value<float64>(k_OutputMinimum_Key);
  auto outputMaximum = filterArgs.value<float64>(k_OutputMaximum_Key);
  const DataPath outputArrayPath = selectedInputArray.replaceName(outputArrayName);

  Result<OutputActions> resultOutputActions = ITK::DataCheck<cxITKRescaleIntensityImageFilter::ArrayOptionsType>(dataStructure, selectedInputArray, imageGeomPath, outputArrayPath);

  return {std::move(resultOutputActions)};
}

//------------------------------------------------------------------------------
Result<> ITKRescaleIntensityImageFilter::executeImpl(DataStructure& dataStructure, const Arguments& filterArgs, const PipelineFilter* pipelineNode, const MessageHandler& messageHandler,
                                                     const std::atomic_bool& shouldCancel) const
{
  auto imageGeomPath = filterArgs.value<DataPath>(k_InputImageGeomPath_Key);
  auto selectedInputArray = filterArgs.value<DataPath>(k_InputImageDataPath_Key);
  auto outputArrayName = filterArgs.value<DataObjectNameParameter::ValueType>(k_OutputImageArrayName_Key);
  const DataPath outputArrayPath = selectedInputArray.replaceName(outputArrayName);

  auto outputMinimum = filterArgs.value<float64>(k_OutputMinimum_Key);
  auto outputMaximum = filterArgs.value<float64>(k_OutputMaximum_Key);

  const cxITKRescaleIntensityImageFilter::ITKRescaleIntensityImageFunctor itkFunctor = {outputMinimum, outputMaximum};

  auto& imageGeom = dataStructure.getDataRefAs<ImageGeom>(imageGeomPath);

  return ITK::Execute<cxITKRescaleIntensityImageFilter::ArrayOptionsType>(dataStructure, selectedInputArray, imageGeomPath, outputArrayPath, itkFunctor, shouldCancel);
}

namespace
{
namespace SIMPL
{
constexpr StringLiteral k_OutputTypeKey = "OutputType";
constexpr StringLiteral k_OutputMinimumKey = "OutputMinimum";
constexpr StringLiteral k_OutputMaximumKey = "OutputMaximum";
constexpr StringLiteral k_SelectedCellArrayPathKey = "SelectedCellArrayPath";
constexpr StringLiteral k_NewCellArrayNameKey = "NewCellArrayName";
} // namespace SIMPL
} // namespace

Result<Arguments> ITKRescaleIntensityImageFilter::FromSIMPLJson(const nlohmann::json& json)
{
  Arguments args = ITKRescaleIntensityImageFilter().getDefaultArguments();

  std::vector<Result<>> results;

  // Output Type parameter is not applicable in NX
  results.push_back(SIMPLConversion::ConvertParameter<SIMPLConversion::DoubleFilterParameterConverter>(args, json, SIMPL::k_OutputMinimumKey, k_OutputMinimum_Key));
  results.push_back(SIMPLConversion::ConvertParameter<SIMPLConversion::DoubleFilterParameterConverter>(args, json, SIMPL::k_OutputMaximumKey, k_OutputMaximum_Key));
  results.push_back(SIMPLConversion::ConvertParameter<SIMPLConversion::DataContainerSelectionFilterParameterConverter>(args, json, SIMPL::k_SelectedCellArrayPathKey, k_InputImageGeomPath_Key));
  results.push_back(SIMPLConversion::ConvertParameter<SIMPLConversion::DataArraySelectionFilterParameterConverter>(args, json, SIMPL::k_SelectedCellArrayPathKey, k_InputImageDataPath_Key));
  results.push_back(SIMPLConversion::ConvertParameter<SIMPLConversion::StringFilterParameterConverter>(args, json, SIMPL::k_NewCellArrayNameKey, k_OutputImageArrayName_Key));

  Result<> conversionResult = MergeResults(std::move(results));

  return ConvertResultTo<Arguments>(std::move(conversionResult), std::move(args));
}
} // namespace nx::core
