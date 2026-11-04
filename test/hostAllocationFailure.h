#ifndef GPUTILS_TEST_HOST_ALLOCATION_FAILURE_H
#define GPUTILS_TEST_HOST_ALLOCATION_FAILURE_H

#include <cstddef>

namespace gputils_test {
using AllocationFailurePredicate = bool (*)(std::size_t) noexcept;

// Test-executable-only, thread-local, one-shot interception of host operator new.
// The predicate must not allocate. Production targets do not link this helper.
class FailHostAllocation {
public:
    explicit FailHostAllocation(AllocationFailurePredicate predicate) noexcept;
    ~FailHostAllocation();
    FailHostAllocation(const FailHostAllocation &) = delete;
    FailHostAllocation &operator=(const FailHostAllocation &) = delete;
private:
    AllocationFailurePredicate previous;
};
}

#endif
