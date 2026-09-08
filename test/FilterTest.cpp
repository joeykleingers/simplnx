#include "simplnx/Common/StringLiteral.hpp"
#include "simplnx/DataStructure/DataStructure.hpp"
#include "simplnx/Filter/Arguments.hpp"
#include "simplnx/Filter/IFilter.hpp"
#include "simplnx/Utilities/StringUtilities.hpp"

#include <catch2/catch.hpp>

#include <stdexcept>
#include <string>

using namespace nx::core;

namespace
{
constexpr StringLiteral k_ThrowMessage = "boom";

/**
 * @brief A filter whose executeImpl always throws a std::runtime_error.
 *
 * IFilter::execute has to convert any exception that escapes a filter into an invalid result so a
 * pipeline reports the failure instead of terminating the process. This filter exercises that
 * backstop without depending on a plugin being loaded.
 */
class ThrowingFilter : public IFilter
{
public:
  ThrowingFilter() = default;
  ~ThrowingFilter() noexcept override = default;

  ThrowingFilter(const ThrowingFilter&) = delete;
  ThrowingFilter(ThrowingFilter&&) noexcept = delete;
  ThrowingFilter& operator=(const ThrowingFilter&) = delete;
  ThrowingFilter& operator=(ThrowingFilter&&) noexcept = delete;

  /**
   * @brief Returns the filter name used in the backstop error message.
   * @return Filter name.
   */
  std::string name() const override
  {
    return "ThrowingFilter";
  }

  /**
   * @brief Returns the C++ class name of this filter.
   * @return Class name.
   */
  std::string className() const override
  {
    return "ThrowingFilter";
  }

  /**
   * @brief Returns the identifier of this filter.
   * @return A fixed test-only uuid.
   */
  Uuid uuid() const override
  {
    return *Uuid::FromString("1f0d5f4a-9f7d-4c67-9d1e-2a3b4c5d6e7f");
  }

  /**
   * @brief Returns the human readable name of this filter.
   * @return Human name.
   */
  std::string humanName() const override
  {
    return "Throwing Filter";
  }

  /**
   * @brief Returns the parameters of this filter.
   * @return An empty parameter set because the backstop needs no inputs.
   */
  Parameters parameters() const override
  {
    return {};
  }

  /**
   * @brief Returns the version of this filter's parameter set.
   * @return The initial version.
   */
  VersionType parametersVersion() const override
  {
    return 1;
  }

  /**
   * @brief Returns a copy of this filter.
   * @return An owning pointer to the copy.
   */
  UniquePointer clone() const override
  {
    return std::make_unique<ThrowingFilter>();
  }

protected:
  /**
   * @brief Reports a valid preflight so execute() reaches executeImpl.
   * @return An empty preflight result.
   */
  PreflightResult preflightImpl(const DataStructure& dataStructure, const Arguments& filterArgs, const MessageHandler& messageHandler, const std::atomic_bool& shouldCancel,
                                const ExecutionContext& executionContext) const override
  {
    return {};
  }

  /**
   * @brief Throws so the caller can observe how execute() handles an escaped exception.
   * @throws std::runtime_error Always.
   */
  Result<> executeImpl(DataStructure& dataStructure, const Arguments& filterArgs, const PipelineFilter* pipelineNode, const MessageHandler& messageHandler, const std::atomic_bool& shouldCancel,
                       const ExecutionContext& executionContext) const override
  {
    throw std::runtime_error(k_ThrowMessage.str());
  }
};
} // namespace

TEST_CASE("IFilter: An exception thrown by executeImpl becomes an invalid result", "[simplnx][IFilter]")
{
  ThrowingFilter filter;
  DataStructure dataStructure;
  Arguments args;

  IFilter::ExecuteResult executeResult = filter.execute(dataStructure, args);

  REQUIRE(executeResult.result.invalid());
  REQUIRE(!executeResult.result.errors().empty());

  const std::string& message = executeResult.result.errors().front().message;
  REQUIRE(StringUtilities::contains(message, filter.name()));
  REQUIRE(StringUtilities::contains(message, k_ThrowMessage.view()));
}
