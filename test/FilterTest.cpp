#include "simplnx/Common/StringLiteral.hpp"
#include "simplnx/DataStructure/DataStructure.hpp"
#include "simplnx/Filter/Arguments.hpp"
#include "simplnx/Filter/IFilter.hpp"
#include "simplnx/Utilities/StringUtilities.hpp"

#include <catch2/catch.hpp>

#include <atomic>
#include <stdexcept>
#include <string>

using namespace nx::core;

namespace
{
constexpr StringLiteral k_ThrowMessage = "boom";
constexpr int32 k_ExecuteErrorCode = -9975;
constexpr StringLiteral k_ExecuteErrorMessage = "the store rejected the write";

/**
 * @struct NonStandardException
 * @brief An exception type that does not derive from std::exception.
 *
 * HDF5's C++ API, and several third party libraries simplnx links, throw types outside the
 * std::exception hierarchy. This type stands in for them.
 */
struct NonStandardException
{
};

/**
 * @class BackstopFilterBase
 * @brief Supplies the IFilter boilerplate the execute() backstop tests do not exercise.
 * @tparam DerivedT The concrete filter. It must declare k_FilterName and k_FilterUuid.
 *
 * Each test filter differs only in its executeImpl, so the identity and parameter methods
 * live here instead of being repeated in every filter.
 */
template <class DerivedT>
class BackstopFilterBase : public IFilter
{
public:
  BackstopFilterBase() = default;
  ~BackstopFilterBase() noexcept override = default;

  BackstopFilterBase(const BackstopFilterBase&) = delete;
  BackstopFilterBase(BackstopFilterBase&&) noexcept = delete;
  BackstopFilterBase& operator=(const BackstopFilterBase&) = delete;
  BackstopFilterBase& operator=(BackstopFilterBase&&) noexcept = delete;

  /**
   * @brief Returns the filter name used in the backstop error message.
   * @return Filter name.
   */
  std::string name() const override
  {
    return DerivedT::k_FilterName.str();
  }

  /**
   * @brief Returns the C++ class name of this filter.
   * @return Class name.
   */
  std::string className() const override
  {
    return DerivedT::k_FilterName.str();
  }

  /**
   * @brief Returns the identifier of this filter.
   * @return A fixed test-only uuid.
   */
  Uuid uuid() const override
  {
    return *Uuid::FromString(DerivedT::k_FilterUuid.view());
  }

  /**
   * @brief Returns the human readable name of this filter.
   * @return Human name.
   */
  std::string humanName() const override
  {
    return DerivedT::k_FilterName.str();
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
    return std::make_unique<DerivedT>();
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
};

/**
 * @class ThrowingFilter
 * @brief A filter whose executeImpl always throws a std::runtime_error.
 *
 * IFilter::execute has to convert any exception that escapes a filter into an invalid result so a
 * pipeline reports the failure instead of terminating the process. This filter exercises that
 * backstop without depending on a plugin being loaded.
 */
class ThrowingFilter : public BackstopFilterBase<ThrowingFilter>
{
public:
  static inline constexpr StringLiteral k_FilterName = "ThrowingFilter";
  static inline constexpr StringLiteral k_FilterUuid = "1f0d5f4a-9f7d-4c67-9d1e-2a3b4c5d6e7f";

protected:
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

/**
 * @class NonStandardThrowingFilter
 * @brief A filter whose executeImpl throws a type outside the std::exception hierarchy.
 *
 * A catch(const std::exception&) handler alone lets such a throw reach std::terminate, which
 * kills the application without telling the user what failed.
 */
class NonStandardThrowingFilter : public BackstopFilterBase<NonStandardThrowingFilter>
{
public:
  static inline constexpr StringLiteral k_FilterName = "NonStandardThrowingFilter";
  static inline constexpr StringLiteral k_FilterUuid = "2c9e7b31-5a44-4f18-8c33-6d1e9b0a7f52";

protected:
  /**
   * @brief Throws a non-std::exception type.
   * @throws NonStandardException Always.
   */
  Result<> executeImpl(DataStructure& dataStructure, const Arguments& filterArgs, const PipelineFilter* pipelineNode, const MessageHandler& messageHandler, const std::atomic_bool& shouldCancel,
                       const ExecutionContext& executionContext) const override
  {
    throw NonStandardException{};
  }
};

/**
 * @class CancelAfterErrorFilter
 * @brief A filter that raises the cancellation flag and then reports a storage error.
 *
 * This is the shape of a real failure: the user presses Cancel while a store write is already
 * failing. The reported error has to stay the storage error, because "Filter cancelled" would
 * hide the cause.
 */
class CancelAfterErrorFilter : public BackstopFilterBase<CancelAfterErrorFilter>
{
public:
  static inline constexpr StringLiteral k_FilterName = "CancelAfterErrorFilter";
  static inline constexpr StringLiteral k_FilterUuid = "84b1c0de-3f27-4a6b-9e15-0c7d2f38ab41";

  /**
   * @brief Selects the flag executeImpl raises.
   * @param cancelFlag Supplies the same flag the caller passes to execute(). The caller must keep
   * it alive for the filter lifetime.
   */
  void setCancelFlag(std::atomic_bool* cancelFlag)
  {
    m_CancelFlag = cancelFlag;
  }

protected:
  /**
   * @brief Raises the cancellation flag and reports a storage error.
   * @return An error carrying k_ExecuteErrorCode.
   */
  Result<> executeImpl(DataStructure& dataStructure, const Arguments& filterArgs, const PipelineFilter* pipelineNode, const MessageHandler& messageHandler, const std::atomic_bool& shouldCancel,
                       const ExecutionContext& executionContext) const override
  {
    if(m_CancelFlag != nullptr)
    {
      m_CancelFlag->store(true);
    }
    return MakeErrorResult(k_ExecuteErrorCode, k_ExecuteErrorMessage.str());
  }

private:
  std::atomic_bool* m_CancelFlag = nullptr;
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

TEST_CASE("IFilter: A non-standard exception thrown by executeImpl becomes an invalid result", "[simplnx][IFilter]")
{
  NonStandardThrowingFilter filter;
  DataStructure dataStructure;
  Arguments args;

  IFilter::ExecuteResult executeResult = filter.execute(dataStructure, args);

  REQUIRE(executeResult.result.invalid());
  REQUIRE(!executeResult.result.errors().empty());

  const Error& error = executeResult.result.errors().front();
  REQUIRE(error.code == -2);
  REQUIRE(StringUtilities::contains(error.message, filter.name()));
}

TEST_CASE("IFilter: A cancellation does not replace an execution error", "[simplnx][IFilter]")
{
  CancelAfterErrorFilter filter;
  DataStructure dataStructure;
  Arguments args;

  std::atomic_bool shouldCancel = false;
  filter.setCancelFlag(&shouldCancel);

  IFilter::ExecuteResult executeResult = filter.execute(dataStructure, args, nullptr, {}, shouldCancel);

  REQUIRE(shouldCancel.load());
  REQUIRE(executeResult.result.invalid());
  REQUIRE(!executeResult.result.errors().empty());

  const Error& error = executeResult.result.errors().front();
  REQUIRE(error.code == k_ExecuteErrorCode);
  REQUIRE(StringUtilities::contains(error.message, k_ExecuteErrorMessage.view()));
}
