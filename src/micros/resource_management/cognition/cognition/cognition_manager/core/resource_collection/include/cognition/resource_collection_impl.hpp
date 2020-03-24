#ifndef RESOURCE_COLLECTION_IMPL_HPP
#define RESOURCE_COLLECTION_IMPL_HPP

// C and C++ headers
#include <vector>
#include <string>

// ROS headers
#include <ros/package.h>

// Interface headers
#include "./resource_collection.hpp"

#ifdef _WIN32
// NOLINT
#else
// NOLINT
#endif

namespace cognition
{
// Resources API
std::vector<std::string> getResourceXmlPaths()
{
  return getXmlPaths("cognition_bus","resource",true);
}

std::vector<std::pair<std::string, std::string> > getResourceXmlExports()
{
  return getXmlExports("cognition_bus","resource",true);
}

// Plugins API
std::vector<std::string> getPluginXmlPaths(const std::string & origin_package)
{
  return getXmlPaths(origin_package,"plugin",true);
}

std::vector<std::pair<std::string, std::string> > getPluginXmlExports(const std::string & origin_package)
{
  return getXmlExports(origin_package,"plugin",true);
}

// Base API For Internal
std::vector<std::string> getXmlPaths(
  const std::string & package,
  const std::string & attrib_name,
  bool force_recrawl)
{
  // Pull possible files from manifests of packages which depend on this package and export class
  //api-1
  std::vector<std::string> paths;
  ros::package::getPlugins(package, attrib_name, paths, force_recrawl);
  return paths;
}

std::vector<std::pair<std::string, std::string> > getXmlExports(
    const std::string & package,
    const std::string & attrib_name,
    bool force_recrawl)
{
  //api-2
  std::vector<std::pair<std::string, std::string> > exports;
  ros::package::getPlugins(package, attrib_name, exports, force_recrawl);
  return exports;
}

} // namespace cognition

#endif // RESOURCE_COLLECTION_IMPL_HPP
