#ifndef RESOURCE_COLLECTION_HPP
#define RESOURCE_COLLECTION_HPP

// C and C++ headers
#include <vector>
#include <string>

namespace cognition {

// Resources API
std::vector<std::string> getResourceXmlPaths();

std::vector<std::pair<std::string, std::string> > getResourceXmlExports();

// Plugins API
std::vector<std::string> getPluginXmlPaths(
  const std::string & origin_package);

std::vector<std::pair<std::string, std::string> > getPluginXmlExports(
  const std::string & origin_package);

// Base API For Internal
std::vector<std::string> getXmlPaths(
  const std::string & package,
  const std::string & attrib_name,
  bool force_recrawl);

std::vector<std::pair<std::string, std::string> > getXmlExports(
  const std::string & package,
  const std::string & attrib_name,
  bool force_recrawl);

} //namespace cognition

// Note: The implementation of the methods is in a separate file for clarity.
#include "./resource_collection_impl.hpp"

#endif // RESOURCE_COLLECTION_HPP
