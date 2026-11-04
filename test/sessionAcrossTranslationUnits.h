#ifndef GPUTILS_TEST_SESSION_ACROSS_TRANSLATION_UNITS_H
#define GPUTILS_TEST_SESSION_ACROSS_TRANSLATION_UNITS_H

#include <cstddef>
class Session;

namespace gputils_test {
void configureStreamsFromOtherTranslationUnit(std::size_t count);
Session &sessionFromOtherTranslationUnit();
bool executeOnSecondStreamFromOtherTranslationUnit();
}

#endif
