#include "sessionAcrossTranslationUnits.h"
#include "tensor.cuh"

void gputils_test::configureStreamsFromOtherTranslationUnit(std::size_t count) {
    Session::setStreams(count);
}

Session &gputils_test::sessionFromOtherTranslationUnit() {
    return Session::getInstance();
}

bool gputils_test::executeOnSecondStreamFromOtherTranslationUnit() {
    DTensor<double> value(std::vector<double>{3,4},2);
    value.setStreamIdx(1);
    value *= 2;
    std::vector<double> actual;
    value.download(actual);
    return actual == std::vector<double>{6,8};
}
