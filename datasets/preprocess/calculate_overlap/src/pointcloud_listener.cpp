#include <iostream>
#include <thread>
#include <atomic>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <sstream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <yaml-cpp/yaml.h>
#include <mutex>

std::mutex coutMutex;
// Assuming df_env and overlapMatrix are already defined
std::atomic<size_t> operationsCompleted(0);

namespace fs = std::filesystem;

struct Location {
    std::string file;
    double northing;
    double easting;
};

struct Config {
    std::string base_path;
    std::string runs_folder;
    std::string filename;
    std::string dir_txt;
    std::vector<std::string> folder_list;
};

// Function to read config from YAML file
bool readConfig(const std::string& config_file, Config& config) {
    try {
        YAML::Node yaml_config = YAML::LoadFile(config_file);

        // Read scalar values
        config.base_path = yaml_config["base_path"].as<std::string>();
        config.runs_folder = yaml_config["runs_folder"].as<std::string>();
        config.filename = yaml_config["filename"].as<std::string>();
        config.dir_txt = yaml_config["dir_txt"].as<std::string>();

        // Read the folder_list as a sequence
        if (yaml_config["folder_list"]) {
            for (const auto& folder : yaml_config["folder_list"]) {
                config.folder_list.push_back(folder.as<std::string>());
            }
        }
    } catch (const YAML::Exception& e) {
        std::cerr << "Error reading YAML file: " << e.what() << std::endl;
        return false;
    }
    return true;
}

double calculateEuclideanDistance(const Eigen::Vector3f& point1, const Eigen::Vector3f& point2) {
    return (point1 - point2).norm();
}

double calc_symmetric_overlap(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud1,
                    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud2,
                    double dis_threshold) {
  int point_kip = 1;
  double match_num = 0;
  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZ>);
  kd_tree->setInputCloud(cloud2);
  std::vector<int> pointIdxNKNSearch(1);
  std::vector<float> pointNKNSquaredDistance(1);
  for (size_t i = 0; i < cloud1->size(); i += point_kip) {
    pcl::PointXYZ searchPoint = cloud1->points[i];
    if (kd_tree->nearestKSearch(searchPoint, 1, pointIdxNKNSearch,
                                pointNKNSquaredDistance) > 0) {
      if (pointNKNSquaredDistance[0] < dis_threshold * dis_threshold) {
        match_num++;
      }
    }
  }

  double overlap =
      2 * match_num * point_kip / (cloud1->size() + cloud2->size());
  if(overlap > 1.0)
    overlap = 1.0;
  return overlap;
}


pcl::PointCloud<pcl::PointXYZ>::Ptr load_pc(const std::string& file_path) {
    std::ifstream input(file_path, std::ios::binary);

    if (!input) {
        std::cerr << "Error: Could not open file " << file_path << std::endl;
        return nullptr; // Error handling
    }

    // Determine the file size
    input.seekg(0, std::ios::end);
    size_t fileSize = input.tellg();
    input.seekg(0, std::ios::beg);

    // Calculate the number of points
    size_t numPoints = fileSize / (3 * sizeof(float));

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // Reserve space in the point cloud to improve performance
    cloud->reserve(numPoints);

    float x, y, z;
    for (size_t i = 0; i < numPoints; ++i) {
        // Read each point (assuming the file structure is x, y, z for each point)
        input.read(reinterpret_cast<char*>(&x), sizeof(float));
        input.read(reinterpret_cast<char*>(&y), sizeof(float));
        input.read(reinterpret_cast<char*>(&z), sizeof(float));

        // Add the point to the cloud
        cloud->push_back(pcl::PointXYZ(x, y, z));
    }

    input.close();
    
    return cloud;
}

std::vector<Location> parse_csv(const std::string& folder_path, const std::string& filepath) {
    std::vector<Location> locations;
    std::ifstream file(filepath);
    std::string line;
    
    // Skipping CSV header
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<std::string> parsedRow;
        
        while(std::getline(lineStream, cell, ',')) {
            parsedRow.push_back(cell);
        }
        
        // Assuming the CSV has northing and easting as the second and third columns
        Location loc;
        
        loc.file = folder_path + "LiDAR_transformed/" + parsedRow[0] +  ".bin"; // Modify as needed
        loc.northing = std::stod(parsedRow[1]);
        loc.easting = std::stod(parsedRow[2]);
        locations.push_back(loc);
    }
    
    return locations;
}

void saveOverlapMatrix(const std::string& filename, const std::vector<std::vector<double>>& overlapMatrix) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    // check process
    int idx = 0;

    file.precision(5);
    for (const auto& row : overlapMatrix) {
        std::cout << idx << "/" << overlapMatrix.size() << std::endl;
        for (const auto& overlap : row) {
            file << overlap << " ";
        }
        file << "\n";
        idx++;
    }
    file.close();
}

std::string replaceString(std::string subject, const std::string& search, const std::string& replace) {
    size_t pos = 0;
    while ((pos = subject.find(search, pos)) != std::string::npos) {
        subject.replace(pos, search.length(), replace);
        pos += replace.length();
    }
    return subject;
}

void monitorProgress(size_t totalOperations) {
    while (operationsCompleted < totalOperations) {
        std::cout << "Progress: " << operationsCompleted << "/" << totalOperations << " operations completed." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1)); // Sleep for a bit before checking again
    }
    std::cout << "All operations completed!" << std::endl;
}


void processSubset(const std::vector<Location>& df_env, std::vector<std::vector<double>>& overlapMatrix, size_t startIdx, size_t endIdx) {
    for (size_t i = startIdx; i < endIdx; ++i) {
        auto cloud1 = load_pc(df_env[i].file);

        for (size_t j = i; j < df_env.size(); ++j) { // Start from i to process only the upper triangle

            if (i == j) {
                overlapMatrix[i][j] = 1.0; // Maximum overlap with itself
                overlapMatrix[j][i] = 1.0;
                continue;
            }

            if (calculateEuclideanDistance(Eigen::Vector3f(df_env[i].northing, df_env[i].easting, 0), Eigen::Vector3f(df_env[j].northing, df_env[j].easting, 0)) > 200) {
                overlapMatrix[i][j] = 0.0;
                overlapMatrix[j][i] = 0.0; // Symmetrically update the lower triangle
                continue;
            }

            auto cloud2 = load_pc(df_env[j].file);

            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1_temp(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2_temp(new pcl::PointCloud<pcl::PointXYZ>);

            pcl::VoxelGrid<pcl::PointXYZ> sor;
            sor.setInputCloud(cloud1);
            sor.setLeafSize(0.04f, 0.04f, 0.04f);
            sor.filter(*cloud1_temp);

            sor.setInputCloud(cloud2);
            sor.setLeafSize(0.04f, 0.04f, 0.04f);
            sor.filter(*cloud2_temp);

            double overlap = calc_symmetric_overlap(cloud1_temp, cloud2_temp, 0.06); // Example threshold
            double overlap2 = calc_symmetric_overlap(cloud2_temp, cloud1_temp, 0.06); // Example threshold
            double max_overlap = std::max(overlap, overlap2);


            {
                std::lock_guard<std::mutex> guard(coutMutex);
                overlapMatrix[i][j] = max_overlap;
                overlapMatrix[j][i] = max_overlap; // Symmetrically update the lower triangle
            }

        }
        ++operationsCompleted;
    }
}

// get arguments from command line. base_path, runs_folder, filename, dir_txt, folder_list
int main() {

    Config config;

    // Read the config file
    if (!readConfig("../config/config.yaml", config)) {
        return 1;
    }

    // Print the values to verify
    std::cout << "base_path: " << config.base_path << std::endl;
    std::cout << "runs_folder: " << config.runs_folder << std::endl;
    std::cout << "filename: " << config.filename << std::endl;
    std::cout << "dir_txt: " << config.dir_txt << std::endl;
    std::cout << "folder_list: ";
    for (const auto& folder : config.folder_list) {
        std::cout << folder << " ";
    }
    std::cout << std::endl;

    std::vector<std::string> folders;
    std::vector<Location> df_env;

    // Iterating over directories within runs_folder
    for (const auto& entry : fs::directory_iterator(config.base_path + config.runs_folder)) {
        if (entry.is_directory()) {
            folders.push_back(entry.path().filename());
        }
     }

    std::cout << "Number of runs: " << folders.size() << std::endl;
    
    // folders sort using alphabetical order
    std::sort(folders.begin(), folders.end());

    for (const auto& folder : folders) {
        //check not empty
        if (config.folder_list.size() > 0) {            
            if (std::find(config.folder_list.begin(), config.folder_list.end(), folder) == config.folder_list.end()) {
                continue;
            }
        }
    
        
        auto folder_path = config.base_path + config.runs_folder + folder + "/";
        auto csv_path = folder_path + config.filename;
        auto locations = parse_csv(folder_path, csv_path);

        df_env.insert(df_env.end(), locations.begin(), locations.end());
        std::cout << folder_path << " " << df_env.size() << std::endl;
    }
    
    std::cout << "Number of submaps: " << df_env.size() << std::endl;

    std::vector<std::vector<double>> overlapMatrix(df_env.size(), std::vector<double>(df_env.size(), -1.0));

    size_t numThreads = std::thread::hardware_concurrency();
    std::cout << numThreads << " concurrent threads are supported.\n";
    std::vector<std::thread> threads(numThreads);
    

    size_t totalPairs = (df_env.size() * (df_env.size() + 1)) / 2; // Total number of unique pairs
    size_t pairsPerThread = totalPairs / numThreads; // Average number of pairs each thread should process
    std::cout << pairsPerThread << std::endl;
    std::vector<std::pair<size_t, size_t>> taskRanges; // To store the start and end indices for each thread

    size_t startIdx = 0;
    size_t accumulatedPairs = 0;
    for (size_t threadId = 0; threadId < numThreads; ++threadId) {
        size_t pairsForThisThread = 0;
        while (startIdx < df_env.size()  && pairsForThisThread < pairsPerThread) {
            size_t remaining = df_env.size() - startIdx; // Remaining comparisons for this startIdx
            pairsForThisThread += remaining;
            if(threadId == numThreads - 1){
                // Too many pairs, this startIdx will be the first index for the next thread
                pairsForThisThread = (df_env.size() - startIdx) * (df_env.size() -startIdx+ 1) / 2;
                startIdx = df_env.size();
                
                std::cout << threadId << std::endl;
                break;
            }
            if (pairsForThisThread <= pairsPerThread || threadId == numThreads - 1) {
                // Acceptable number of pairs for this thread, or adjust for the last thread to take all remaining work
                accumulatedPairs += pairsForThisThread;
            } 
            else{
                pairsForThisThread -= remaining;
                break;
            }
            startIdx++;
        }
        std::cout << pairsForThisThread << std::endl;
        taskRanges.push_back(std::make_pair(threadId == 0 ? 0 : taskRanges.back().second, startIdx));
    }

    // Now launch threads based on these more balanced ranges
    for (size_t i = 0; i < numThreads; ++i) {
        std::cout << "Thread " << i << " handles from " << taskRanges[i].first << " to " << taskRanges[i].second << std::endl;
        threads[i] = std::thread(processSubset, std::ref(df_env), std::ref(overlapMatrix), taskRanges[i].first, taskRanges[i].second);
    }

    size_t totalOperations = df_env.size(); // Assuming each thread processes one chunk of df_env
    std::thread progressThread(monitorProgress, totalOperations);



    // Join the threads with the main thread
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
  
      // Wait for the progress monitoring thread to finish
    if (progressThread.joinable()) {
        progressThread.join();
    }

    // Save the overlap matrix to a text file
    saveOverlapMatrix(config.base_path + config.dir_txt, overlapMatrix);

    return 0;
}
