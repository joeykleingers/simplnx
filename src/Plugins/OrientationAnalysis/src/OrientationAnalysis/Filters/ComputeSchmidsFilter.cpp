#include "ComputeSchmidsFilter.hpp"
#include "OrientationAnalysis/Filters/Algorithms/ComputeSchmids.hpp"

#include "simplnx/DataStructure/DataArray.hpp"
#include "simplnx/DataStructure/DataPath.hpp"
#include "simplnx/Filter/Actions/CreateArrayAction.hpp"
#include "simplnx/Parameters/ArraySelectionParameter.hpp"
#include "simplnx/Parameters/BoolParameter.hpp"
#include "simplnx/Parameters/DataObjectNameParameter.hpp"

#include "simplnx/Utilities/SIMPLConversion.hpp"

#include "simplnx/Parameters/VectorParameter.hpp"

using namespace nx::core;

namespace nx::core
{
//------------------------------------------------------------------------------
std::string ComputeSchmidsFilter::name() const
{
  return FilterTraits<ComputeSchmidsFilter>::name.str();
}

//------------------------------------------------------------------------------
std::string ComputeSchmidsFilter::className() const
{
  return FilterTraits<ComputeSchmidsFilter>::className;
}

//------------------------------------------------------------------------------
Uuid ComputeSchmidsFilter::uuid() const
{
  return FilterTraits<ComputeSchmidsFilter>::uuid;
}

//------------------------------------------------------------------------------
std::string ComputeSchmidsFilter::humanName() const
{
  return "Compute Schmid Factors";
}

//------------------------------------------------------------------------------
std::vector<std::string> ComputeSchmidsFilter::defaultTags() const
{
  return {className(), "Statistics", "Crystallography", "Find", "Generate", "Calculate", "Determine"};
}

//------------------------------------------------------------------------------
Parameters ComputeSchmidsFilter::parameters() const
{
  Parameters params;

  // Create the parameter descriptors that are needed for this filter
  params.insertSeparator(Parameters::Separator{"Input Parameter(s)"});

  params.insert(std::make_unique<VectorFloat32Parameter>(k_LoadingDirection_Key, "Loading Direction", "The loading axis for the sample", std::vector<float32>({1.0F, 1.0F, 1.0F}),
                                                         std::vector<std::string>({"X", "Y", "Z"})));
  params.insertLinkableParameter(
      std::make_unique<BoolParameter>(k_StoreAngleComponents_Key, "Store Angle Components of Schmid Factor", "Whether to store the angle components for each Feature", false));

  params.insertLinkableParameter(std::make_unique<BoolParameter>(k_OverrideSystem_Key, "Override Default Slip System", "Allows the user to manually input the slip plane and slip direction", false));
  params.insert(std::make_unique<VectorFloat32Parameter>(k_SlipPlane_Key, "Slip Plane", "Vector defining the slip plane normal.", std::vector<float32>({0.0F, 0.0F, 1.0F}),
                                                         std::vector<std::string>({"X", "Y", "Z"})));
  params.insert(std::make_unique<VectorFloat32Parameter>(k_SlipDirection_Key, "Slip Direction", "Vector defining the slip direction.", std::vector<float32>({1.0F, 0.0F, 0.0F}),
                                                         std::vector<std::string>({"X", "Y", "Z"})));

  params.insertSeparator(Parameters::Separator{"Input Feature Data"});
  params.insert(std::make_unique<ArraySelectionParameter>(k_FeaturePhasesArrayPath_Key, "Phases", "Specifies to which Ensemble each cell belongs", DataPath({"Cell Feature Data", "Phases"}),
                                                          ArraySelectionParameter::AllowedTypes{nx::core::DataType::int32}, ArraySelectionParameter::AllowedComponentShapes{{1}}));
  params.insert(std::make_unique<ArraySelectionParameter>(k_AvgQuatsArrayPath_Key, "Average Quaternions", "Specifies the average orienation of each Feature in quaternion representation",
                                                          DataPath({"Cell Feature Data", "AvgQuats"}), ArraySelectionParameter::AllowedTypes{nx::core::DataType::float32},
                                                          ArraySelectionParameter::AllowedComponentShapes{{4}}));
  params.insertSeparator(Parameters::Separator{"Input Ensemble Data"});
  params.insert(std::make_unique<ArraySelectionParameter>(k_CrystalStructuresArrayPath_Key, "Crystal Structures", "Enumeration representing the crystal structure for each Ensemble",
                                                          DataPath({"Ensemble Data", "CrystalStructures"}), ArraySelectionParameter::AllowedTypes{nx::core::DataType::uint32},
                                                          ArraySelectionParameter::AllowedComponentShapes{{1}}));

  params.insertSeparator(Parameters::Separator{"Output Feature Data"});
  params.insert(std::make_unique<DataObjectNameParameter>(
      k_SchmidsArrayName_Key, "Schmids", "The name of the array containing the value of the Schmid factor for the most favorably oriented slip system (i.e., the one with the highest Schmid factor)",
      "Schmids"));
  params.insert(std::make_unique<DataObjectNameParameter>(k_SlipSystemsArrayName_Key, "Slip Systems",
                                                          "The name of the array containing the enumeration of the slip system that has the highest Schmid factor", "SlipSystems"));
  params.insert(std::make_unique<DataObjectNameParameter>(k_PolesArrayName_Key, "Poles",
                                                          "The name of the array specifying the crystallographic pole that points along the user defined loading direction", "Poles"));
  params.insert(std::make_unique<DataObjectNameParameter>(k_PhisArrayName_Key, "Phis", "The name of the array containing the angle between tensile axis and slip plane normal. ", "Schmid_Phis"));
  params.insert(std::make_unique<DataObjectNameParameter>(k_LambdasArrayName_Key, "Lambdas", "The name of the array containing the angle between tensile axis and slip direction.", "Schmid_Lambdas"));
  // Associate the Linkable Parameter(s) to the children parameters that they control
  params.linkParameters(k_StoreAngleComponents_Key, k_PhisArrayName_Key, true);
  params.linkParameters(k_StoreAngleComponents_Key, k_LambdasArrayName_Key, true);
  params.linkParameters(k_OverrideSystem_Key, k_SlipPlane_Key, true);
  params.linkParameters(k_OverrideSystem_Key, k_SlipDirection_Key, true);

  return params;
}

//------------------------------------------------------------------------------
IFilter::VersionType ComputeSchmidsFilter::parametersVersion() const
{
  return 1;
}

//------------------------------------------------------------------------------
IFilter::UniquePointer ComputeSchmidsFilter::clone() const
{
  return std::make_unique<ComputeSchmidsFilter>();
}

//------------------------------------------------------------------------------
IFilter::PreflightResult ComputeSchmidsFilter::preflightImpl(const DataStructure& dataStructure, const Arguments& filterArgs, const MessageHandler& messageHandler,
                                                             const std::atomic_bool& shouldCancel) const
{
  auto pLoadingDirectionValue = filterArgs.value<VectorFloat32Parameter::ValueType>(k_LoadingDirection_Key);
  auto pStoreAngleComponentsValue = filterArgs.value<bool>(k_StoreAngleComponents_Key);
  auto pOverrideSystemValue = filterArgs.value<bool>(k_OverrideSystem_Key);
  auto pSlipPlaneValue = filterArgs.value<VectorFloat32Parameter::ValueType>(k_SlipPlane_Key);
  auto pSlipDirectionValue = filterArgs.value<VectorFloat32Parameter::ValueType>(k_SlipDirection_Key);
  auto pFeaturePhasesArrayPathValue = filterArgs.value<DataPath>(k_FeaturePhasesArrayPath_Key);
  auto pAvgQuatsArrayPathValue = filterArgs.value<DataPath>(k_AvgQuatsArrayPath_Key);
  auto pCrystalStructuresArrayPathValue = filterArgs.value<DataPath>(k_CrystalStructuresArrayPath_Key);
  DataPath cellFeatDataPath = pFeaturePhasesArrayPathValue.getParent();
  auto pSchmidsArrayNameValue = cellFeatDataPath.createChildPath(filterArgs.value<std::string>(k_SchmidsArrayName_Key));
  auto pSlipSystemsArrayNameValue = cellFeatDataPath.createChildPath(filterArgs.value<std::string>(k_SlipSystemsArrayName_Key));
  auto pPolesArrayNameValue = cellFeatDataPath.createChildPath(filterArgs.value<std::string>(k_PolesArrayName_Key));
  auto pPhisArrayNameValue = cellFeatDataPath.createChildPath(filterArgs.value<std::string>(k_PhisArrayName_Key));
  auto pLambdasArrayNameValue = cellFeatDataPath.createChildPath(filterArgs.value<std::string>(k_LambdasArrayName_Key));

  PreflightResult preflightResult;

  nx::core::Result<OutputActions> resultOutputActions;

  DataPath featureDataGroup = pFeaturePhasesArrayPathValue.getParent();

  const Int32Array& phases = dataStructure.getDataRefAs<Int32Array>(pFeaturePhasesArrayPathValue);
  auto tupleShape = phases.getIDataStore()->getTupleShape();

  // Create output Schmids Array
  {
    auto createArrayAction = std::make_unique<CreateArrayAction>(DataType::float32, tupleShape, std::vector<usize>{1}, pSchmidsArrayNameValue);
    resultOutputActions.value().appendAction(std::move(createArrayAction));
  }
  // Create output SlipSystems Array
  {
    auto createArrayAction = std::make_unique<CreateArrayAction>(DataType::int32, tupleShape, std::vector<usize>{1}, pSlipSystemsArrayNameValue);
    resultOutputActions.value().appendAction(std::move(createArrayAction));
  }
  // Create output SlipSystems Array
  {
    auto createArrayAction = std::make_unique<CreateArrayAction>(DataType::int32, tupleShape, std::vector<usize>{3}, pPolesArrayNameValue);
    resultOutputActions.value().appendAction(std::move(createArrayAction));
  }
  // Create output SlipSystems Array
  if(pStoreAngleComponentsValue)
  {
    auto createArrayAction = std::make_unique<CreateArrayAction>(DataType::float32, tupleShape, std::vector<usize>{1}, pPhisArrayNameValue);
    resultOutputActions.value().appendAction(std::move(createArrayAction));
  }
  // Create output Lambdas Array
  if(pStoreAngleComponentsValue)
  {
    auto createArrayAction = std::make_unique<CreateArrayAction>(DataType::float32, tupleShape, std::vector<usize>{1}, pLambdasArrayNameValue);
    resultOutputActions.value().appendAction(std::move(createArrayAction));
  }

  if(pOverrideSystemValue)
  {
    // make sure direction lies in plane
    float cosVec = pSlipPlaneValue[0] * pSlipDirectionValue[0] + pSlipPlaneValue[1] * pSlipDirectionValue[1] + pSlipPlaneValue[2] * pSlipDirectionValue[2];
    if(0.0F != cosVec)
    {
      return {MakeErrorResult<OutputActions>(-13500, "Slip Plane and Slip Direction must be normal")};
    }
  }

  std::vector<PreflightValue> preflightUpdatedValues;

  // If the filter needs to pass back some updated values via a key:value string:string set of values
  // you can declare and update that string here.
  // None found in this filter based on the filter parameters

  // Return both the resultOutputActions and the preflightUpdatedValues via std::move()
  return {std::move(resultOutputActions), std::move(preflightUpdatedValues)};
}

//------------------------------------------------------------------------------
Result<> ComputeSchmidsFilter::executeImpl(DataStructure& dataStructure, const Arguments& filterArgs, const PipelineFilter* pipelineNode, const MessageHandler& messageHandler,
                                           const std::atomic_bool& shouldCancel) const
{
  ComputeSchmidsInputValues inputValues;

  inputValues.LoadingDirection = filterArgs.value<VectorFloat32Parameter::ValueType>(k_LoadingDirection_Key);
  inputValues.StoreAngleComponents = filterArgs.value<bool>(k_StoreAngleComponents_Key);
  inputValues.OverrideSystem = filterArgs.value<bool>(k_OverrideSystem_Key);
  inputValues.SlipPlane = filterArgs.value<VectorFloat32Parameter::ValueType>(k_SlipPlane_Key);
  inputValues.SlipDirection = filterArgs.value<VectorFloat32Parameter::ValueType>(k_SlipDirection_Key);
  inputValues.FeaturePhasesArrayPath = filterArgs.value<DataPath>(k_FeaturePhasesArrayPath_Key);
  inputValues.AvgQuatsArrayPath = filterArgs.value<DataPath>(k_AvgQuatsArrayPath_Key);
  inputValues.CrystalStructuresArrayPath = filterArgs.value<DataPath>(k_CrystalStructuresArrayPath_Key);
  DataPath cellFeatDataPath = inputValues.FeaturePhasesArrayPath.getParent();
  inputValues.SchmidsArrayName = cellFeatDataPath.createChildPath(filterArgs.value<std::string>(k_SchmidsArrayName_Key));
  inputValues.SlipSystemsArrayName = cellFeatDataPath.createChildPath(filterArgs.value<std::string>(k_SlipSystemsArrayName_Key));
  inputValues.PolesArrayName = cellFeatDataPath.createChildPath(filterArgs.value<std::string>(k_PolesArrayName_Key));
  inputValues.PhisArrayName = cellFeatDataPath.createChildPath(filterArgs.value<std::string>(k_PhisArrayName_Key));
  inputValues.LambdasArrayName = cellFeatDataPath.createChildPath(filterArgs.value<std::string>(k_LambdasArrayName_Key));

  return ComputeSchmids(dataStructure, messageHandler, shouldCancel, &inputValues)();
}

namespace
{
namespace SIMPL
{
constexpr StringLiteral k_LoadingDirectionKey = "LoadingDirection";
constexpr StringLiteral k_StoreAngleComponentsKey = "StoreAngleComponents";
constexpr StringLiteral k_OverrideSystemKey = "OverrideSystem";
constexpr StringLiteral k_SlipPlaneKey = "SlipPlane";
constexpr StringLiteral k_SlipDirectionKey = "SlipDirection";
constexpr StringLiteral k_FeaturePhasesArrayPathKey = "FeaturePhasesArrayPath";
constexpr StringLiteral k_AvgQuatsArrayPathKey = "AvgQuatsArrayPath";
constexpr StringLiteral k_CrystalStructuresArrayPathKey = "CrystalStructuresArrayPath";
constexpr StringLiteral k_SchmidsArrayNameKey = "SchmidsArrayName";
constexpr StringLiteral k_SlipSystemsArrayNameKey = "SlipSystemsArrayName";
constexpr StringLiteral k_PolesArrayNameKey = "PolesArrayName";
constexpr StringLiteral k_PhisArrayNameKey = "PhisArrayName";
constexpr StringLiteral k_LambdasArrayNameKey = "LambdasArrayName";
} // namespace SIMPL
} // namespace

Result<Arguments> ComputeSchmidsFilter::FromSIMPLJson(const nlohmann::json& json)
{
  Arguments args = ComputeSchmidsFilter().getDefaultArguments();

  std::vector<Result<>> results;

  results.push_back(SIMPLConversion::ConvertParameter<SIMPLConversion::FloatVec3FilterParameterConverter>(args, json, SIMPL::k_LoadingDirectionKey, k_LoadingDirection_Key));
  results.push_back(SIMPLConversion::ConvertParameter<SIMPLConversion::LinkedBooleanFilterParameterConverter>(args, json, SIMPL::k_StoreAngleComponentsKey, k_StoreAngleComponents_Key));
  results.push_back(SIMPLConversion::ConvertParameter<SIMPLConversion::LinkedBooleanFilterParameterConverter>(args, json, SIMPL::k_OverrideSystemKey, k_OverrideSystem_Key));
  results.push_back(SIMPLConversion::ConvertParameter<SIMPLConversion::FloatVec3FilterParameterConverter>(args, json, SIMPL::k_SlipPlaneKey, k_SlipPlane_Key));
  results.push_back(SIMPLConversion::ConvertParameter<SIMPLConversion::FloatVec3FilterParameterConverter>(args, json, SIMPL::k_SlipDirectionKey, k_SlipDirection_Key));
  results.push_back(SIMPLConversion::ConvertParameter<SIMPLConversion::DataArraySelectionFilterParameterConverter>(args, json, SIMPL::k_FeaturePhasesArrayPathKey, k_FeaturePhasesArrayPath_Key));
  results.push_back(SIMPLConversion::ConvertParameter<SIMPLConversion::DataArraySelectionFilterParameterConverter>(args, json, SIMPL::k_AvgQuatsArrayPathKey, k_AvgQuatsArrayPath_Key));
  results.push_back(
      SIMPLConversion::ConvertParameter<SIMPLConversion::DataArraySelectionFilterParameterConverter>(args, json, SIMPL::k_CrystalStructuresArrayPathKey, k_CrystalStructuresArrayPath_Key));
  results.push_back(SIMPLConversion::ConvertParameter<SIMPLConversion::LinkedPathCreationFilterParameterConverter>(args, json, SIMPL::k_SchmidsArrayNameKey, k_SchmidsArrayName_Key));
  results.push_back(SIMPLConversion::ConvertParameter<SIMPLConversion::LinkedPathCreationFilterParameterConverter>(args, json, SIMPL::k_SlipSystemsArrayNameKey, k_SlipSystemsArrayName_Key));
  results.push_back(SIMPLConversion::ConvertParameter<SIMPLConversion::LinkedPathCreationFilterParameterConverter>(args, json, SIMPL::k_PolesArrayNameKey, k_PolesArrayName_Key));
  results.push_back(SIMPLConversion::ConvertParameter<SIMPLConversion::LinkedPathCreationFilterParameterConverter>(args, json, SIMPL::k_PhisArrayNameKey, k_PhisArrayName_Key));
  results.push_back(SIMPLConversion::ConvertParameter<SIMPLConversion::LinkedPathCreationFilterParameterConverter>(args, json, SIMPL::k_LambdasArrayNameKey, k_LambdasArrayName_Key));

  Result<> conversionResult = MergeResults(std::move(results));

  return ConvertResultTo<Arguments>(std::move(conversionResult), std::move(args));
}
} // namespace nx::core
