#pragma once

#include "simplnx/Filter/FilterTraits.hpp"
#include "simplnx/Filter/IFilter.hpp"

#include "TestOne/TestOne_export.hpp"

namespace nx::core
{
class TESTONE_EXPORT CreateOutOfCoreArray : public IFilter
{
public:
  CreateOutOfCoreArray() = default;
  ~CreateOutOfCoreArray() noexcept override = default;

  CreateOutOfCoreArray(const CreateOutOfCoreArray&) = delete;
  CreateOutOfCoreArray(CreateOutOfCoreArray&&) noexcept = delete;

  CreateOutOfCoreArray& operator=(const CreateOutOfCoreArray&) = delete;
  CreateOutOfCoreArray& operator=(CreateOutOfCoreArray&&) noexcept = delete;

  // Parameter Keys
  static inline constexpr StringLiteral k_NumericType_Key = "numeric_type_index";
  static inline constexpr StringLiteral k_NumComps_Key = "component_count";
  static inline constexpr StringLiteral k_TupleDims_Key = "tuple_dimensions";
  static inline constexpr StringLiteral k_DataPath_Key = "output_array_path";
  static inline constexpr StringLiteral k_InitilizationValue_Key = "initialization_value_str";

  /**
   * @brief
   * @return
   */
  std::string name() const override;

  /**
   * @brief Returns the C++ classname of this filter.
   * @return
   */
  std::string className() const override;

  /**
   * @brief
   * @return
   */
  Uuid uuid() const override;

  /**
   * @brief
   * @return
   */
  std::string humanName() const override;

  /**
   * @brief
   * @return
   */
  Parameters parameters() const override;

  /**
   * @brief
   * @return
   */
  UniquePointer clone() const override;

protected:
  /**
   * @brief
   * @param data
   * @param args
   * @param messageHandler
   * @return Result<OutputActions>
   */
  PreflightResult preflightImpl(const DataStructure& dataStructure, const Arguments& args, const MessageHandler& messageHandler, const std::atomic_bool& shouldCancel) const override;

  /**
   * @brief Applies the filter's algorithm to the DataStructure with the given arguments. Returns any warnings/errors.
   * On failure, there is no guarantee that the DataStructure is in a correct state.
   * @param dataStructure The input DataStructure instance
   * @param filterArgs These are the input values for each parameter that is required for the filter
   * @param pipelineNode The PipelineNode object that called this filter
   * @param messageHandler The MessageHandler object
   * @param shouldCancel The atomic boolean that holds if the filter should be canceled
   * @return Returns a Result object with error or warning values if any of those occurred during execution of this function
   */
  Result<> executeImpl(DataStructure& dataStructure, const Arguments& args, const PipelineFilter* pipelineNode, const MessageHandler& messageHandler,
                       const std::atomic_bool& shouldCancel) const override;
};
} // namespace nx::core

SIMPLNX_DEF_FILTER_TRAITS(nx::core, CreateOutOfCoreArray, "f1ac8cb2-ec9b-4764-a6f8-9211de3f8975");
