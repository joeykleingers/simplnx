#pragma once

#include "OrientationAnalysis/OrientationAnalysis_export.hpp"

#include "simplnx/Filter/FilterTraits.hpp"
#include "simplnx/Filter/IFilter.hpp"

namespace nx::core
{
/**
 * @class ConvertHexGridToSquareGridFilter
 * @brief This filter will...
 */
class ORIENTATIONANALYSIS_EXPORT ConvertHexGridToSquareGridFilter : public IFilter
{
public:
  ConvertHexGridToSquareGridFilter();
  ~ConvertHexGridToSquareGridFilter() noexcept override;

  ConvertHexGridToSquareGridFilter(const ConvertHexGridToSquareGridFilter&) = delete;
  ConvertHexGridToSquareGridFilter(ConvertHexGridToSquareGridFilter&&) noexcept = delete;

  ConvertHexGridToSquareGridFilter& operator=(const ConvertHexGridToSquareGridFilter&) = delete;
  ConvertHexGridToSquareGridFilter& operator=(ConvertHexGridToSquareGridFilter&&) noexcept = delete;

  // Parameter Keys
  static inline constexpr StringLiteral k_MultipleFiles_Key = "multiple_files";
  static inline constexpr StringLiteral k_OutputPrefix_Key = "output_prefix";
  static inline constexpr StringLiteral k_OutputPath_Key = "output_path";
  static inline constexpr StringLiteral k_InputPath_Key = "input_path";
  static inline constexpr StringLiteral k_Spacing_Key = "spacing";
  static inline constexpr StringLiteral k_GeneratedFileList_Key = "generated_file_list";

  /**
   * @brief Returns the name of the filter.
   * @return
   */
  std::string name() const override;

  /**
   * @brief Returns the C++ classname of this filter.
   * @return
   */
  std::string className() const override;

  /**
   * @brief Returns the uuid of the filter.
   * @return
   */
  Uuid uuid() const override;

  /**
   * @brief Returns the human readable name of the filter.
   * @return
   */
  std::string humanName() const override;

  /**
   * @brief Returns the default tags for this filter.
   * @return
   */
  std::vector<std::string> defaultTags() const override;

  /**
   * @brief Returns the parameters of the filter (i.e. its inputs)
   * @return
   */
  Parameters parameters() const override;

  /**
   * @brief Returns parameters version integer.
   * Initial version should always be 1.
   * Should be incremented everytime the parameters change.
   * @return VersionType
   */
  VersionType parametersVersion() const override;

  /**
   * @brief Returns a copy of the filter.
   * @return
   */
  UniquePointer clone() const override;

protected:
  /**
   * @brief Takes in a DataStructure and checks that the filter can be run on it with the given arguments.
   * Returns any warnings/errors. Also returns the changes that would be applied to the DataStructure.
   * Some parts of the actions may not be completely filled out if all the required information is not available at preflight time.
   * @param dataStructure The input DataStructure instance
   * @param filterArgs These are the input values for each parameter that is required for the filter
   * @param messageHandler The MessageHandler object
   * @return Returns a Result object with error or warning values if any of those occurred during execution of this function
   */
  PreflightResult preflightImpl(const DataStructure& dataStructure, const Arguments& filterArgs, const MessageHandler& messageHandler, const std::atomic_bool& shouldCancel) const override;

  /**
   * @brief Applies the filter's algorithm to the DataStructure with the given arguments. Returns any warnings/errors.
   * On failure, there is no guarantee that the DataStructure is in a correct state.
   * @param dataStructure The input DataStructure instance
   * @param filterArgs These are the input values for each parameter that is required for the filter
   * @param messageHandler The MessageHandler object
   * @return Returns a Result object with error or warning values if any of those occurred during execution of this function
   */
  Result<> executeImpl(DataStructure& dataStructure, const Arguments& filterArgs, const PipelineFilter* pipelineNode, const MessageHandler& messageHandler,
                       const std::atomic_bool& shouldCancel) const override;

private:
  int32 m_InstanceId;
};
} // namespace nx::core

SIMPLNX_DEF_FILTER_TRAITS(nx::core, ConvertHexGridToSquareGridFilter, "960b0e35-de9f-496a-9423-0f55133b39c7");
/* LEGACY UUID FOR THIS FILTER e1343abe-e5ad-5eb1-a89d-c209e620e4de */
