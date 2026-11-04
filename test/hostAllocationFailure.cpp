#include "hostAllocationFailure.h"
#include <cstdlib>
#include <new>

namespace {
thread_local gputils_test::AllocationFailurePredicate failurePredicate = nullptr;
}

gputils_test::FailHostAllocation::FailHostAllocation(AllocationFailurePredicate predicate) noexcept
    : previous(failurePredicate) {
    failurePredicate = predicate;
}

gputils_test::FailHostAllocation::~FailHostAllocation() {
    failurePredicate = previous;
}

void *operator new(std::size_t bytes) {
    if (failurePredicate && failurePredicate(bytes)) {
        // Disarm before throwing so cleanup and exception reporting can allocate.
        failurePredicate = nullptr;
        throw std::bad_alloc();
    }
    for (;;) {
        if (void *memory = std::malloc(bytes == 0 ? 1 : bytes)) return memory;
        const auto handler = std::get_new_handler();
        if (!handler) throw std::bad_alloc();
        handler();
    }
}

void *operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete[](void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void *memory, std::size_t) noexcept { std::free(memory); }
