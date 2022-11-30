#include "ground_extraction/GroundExtraction.h"

int main(int argc, char** argv){

  ros::init(argc, argv, "gpinsac_node");
  ros::NodeHandle node;
  ros::NodeHandle privateNode("~");
 
  GroundExtraction ground_extraction(node,privateNode);

  ros::spin();

  return 0;
}