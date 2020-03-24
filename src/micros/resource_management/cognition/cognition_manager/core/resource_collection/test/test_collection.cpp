#include <cognition/resource_collection.hpp>
#include <vector>
#include <iostream>

using namespace cognition;
using namespace std;

int main(int argc, char **argv)
{
//TEST 1:
    //string package_ = "cognition_bus";
    //string attrib_name_ = "plugin";

    string package_ = "cognition_bus";
    string attrib_name_ = "resource";

    // 1-1
    std::vector<std::string> plugin_xml_paths_ = getXmlPaths(package_, attrib_name_, true);
    //std::map<std::string, ClassDesc> updated_classes = determineAvailableClasses(plugin_xml_paths_);
    for (auto it : plugin_xml_paths_) {
        cout << it << endl;
    }
    cout << "end~~" << endl;

    // 1-2
    std::vector<std::pair<std::string, std::string> > plugin_exports_ = getXmlExports(package_, attrib_name_, true);
    //std::map<std::string, ClassDesc> updated_classes = determineAvailableClasses(plugin_xml_paths_);
    for (int i=0; i<plugin_exports_.size(); ++i) {
        cout << plugin_exports_[i].first << endl;
        cout << plugin_exports_[i].second << endl;
    }
    cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;

    // 2-1
    std::vector<std::string> resource_xml_paths_ = cognition::getResourceXmlPaths();
    //std::map<std::string, ClassDesc> updated_classes = determineAvailableClasses(plugin_xml_paths_);
    for (auto it : resource_xml_paths_) {
        cout << it << endl;
    }
    cout << "end~~" << endl;

    // 2-2
    std::vector<std::pair<std::string, std::string> > resource_exports_ = cognition::getResourceXmlExports();
    //std::map<std::string, ClassDesc> updated_classes = determineAvailableClasses(plugin_xml_paths_);
    for (int i=0; i<resource_exports_.size(); ++i) {
        cout << resource_exports_[i].first << endl;
        cout << resource_exports_[i].second << endl;
    }
    cout << "end~~" << endl;

    return 0;
}
//rospack plugins --attrib=plugin cognition_bus
