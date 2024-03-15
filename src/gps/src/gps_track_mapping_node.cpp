#include <ros/ros.h>
#include <cmath>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip> // Include for set precision
#include <tf/tf.h>
#include <sensor_msgs/NavSatFix.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>

using namespace std;

// Structure to represent a geographic point with latitude and longitude
struct Point
{
    double lat;
    double lon;
};
// Structure to represent a geographic point in UTM
struct utmPoint
{
    int zone;
    double easting;
    double northing;
};

// Constants for WGS84 ellipsoid
const double WGS84_A = 6378137.0;                              // Semi-major axis
const double WGS84_F = 1 / 298.257223563;                      // Flattening
const double WGS84_B = WGS84_A * (1 - WGS84_F);                // Semi-minor axis
const double WGS84_E_SQ = (2 * WGS84_F) - (WGS84_F * WGS84_F); // Eccentricity squared
int numPoints = 16;                                // Number of points for calculations
double MWD = 10;                                   // Initial distance in meters
double K = 1.112;                                  // Coefficient for distance calculation
vector<Point> trackMap_;                           // Track map data
Point current_gps_position;
Point previous_gps_Position;                            // Previous GPS position
vector<utmPoint> localizedNewPositionsUtmRelative; // Vector to store localized new positions in UTM

float q_current_heading = 0.0;
utmPoint q_current_position = { .zone = 1, .easting = 0.0, .northing = 0.0 };
ros::Publisher pose_pub;

// Read track map data from CSV file
vector<Point> readTrackMap(const string &filename)
{
    vector<Point> trackMap;
    ifstream file(filename);
    string line;

    while (getline(file, line))
    {
        stringstream ss(line);
        Point p;
        ss >> p.lat;
        ss.ignore(); // Ignore comma
        ss >> p.lon;
        trackMap.push_back(p);
    }

    return trackMap;
}

// Calculate distance between two points
double calculateDistance(const Point &p1, const Point &p2)
{
    const double earthRadius = 6371.0; // Earth radius in kilometers
    double dLat = (p2.lat - p1.lat) * (M_PI / 180.0);
    double dLon = (p2.lon - p1.lon) * (M_PI / 180.0);
    double a = sin(dLat / 2) * sin(dLat / 2) +
               cos(p1.lat * (M_PI / 180.0)) * cos(p2.lat * (M_PI / 180.0)) *
                   sin(dLon / 2) * sin(dLon / 2);
    double c = 2 * atan2(sqrt(a), sqrt(1 - a));
    double distance = earthRadius * c;
    return distance;
}

// Calculate orientation between two points
double calculateOrientation(const Point &p1, const Point &p2)
{
    double dLon = (p2.lon - p1.lon) * (M_PI / 180);
    double lat1 = p1.lat * (M_PI / 180);
    double lat2 = p2.lat * (M_PI / 180);

    double y = sin(dLon) * cos(lat2);
    double x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dLon);
    double orientation = atan2(y, x);
    fmod(orientation + 2 * M_PI, 2 * M_PI); // Normalize to [0, 2*pi]

    return orientation * (180 / M_PI); // Convert radians to degrees
}

// Find the two nearest points for interpolation
pair<Point, Point> findNearestPoints(const vector<Point> &trackMap, const Point &gpsPosition)
{
    double minDist1 = numeric_limits<double>::max();
    double minDist2 = numeric_limits<double>::max();
    Point nearest1, nearest2;

    for (const auto &point : trackMap)
    {
        double dist = calculateDistance(point, gpsPosition);
        if (dist < minDist1)
        {
            minDist2 = minDist1;
            nearest2 = nearest1;
            minDist1 = dist;
            nearest1 = point;
        }
        else if (dist < minDist2)
        {
            if (!(point.lat == nearest1.lat && point.lon == nearest1.lon))
            {
                minDist2 = dist;
                nearest2 = point;
            }
        }
    }

    if ((nearest1.lat == trackMap.back().lat && nearest1.lon == trackMap.back().lon) || (nearest2.lat == trackMap.back().lat && nearest2.lon == trackMap.back().lon))
    {
        nearest1 = trackMap.back();
        nearest2 = trackMap.back();
    }
    else if ((nearest1.lat == trackMap[1].lat && nearest1.lon == trackMap[1].lon) || (nearest2.lat == trackMap[1].lat && nearest2.lon == trackMap[1].lon))
    {
        nearest1 = trackMap[1];
        nearest2 = trackMap[1];
    }

    return make_pair(nearest1, nearest2);
}

// Linear interpolation between two points
Point interpolate(const Point &p1, const Point &p2, const Point &gpsPosition)
{
    Point interpolated;
    if (p1.lat == p2.lat && p1.lon == p2.lon)
    {
        interpolated.lat = p1.lat;
        interpolated.lon = p1.lon;
    }
    else
    {
        double totalDist = calculateDistance(p1, p2);
        double distToP1 = calculateDistance(gpsPosition, p1);
        double ratio = distToP1 / totalDist;
        interpolated.lat = p1.lat + ratio * (p2.lat - p1.lat);
        interpolated.lon = p1.lon + ratio * (p2.lon - p1.lon);
    }

    return interpolated;
}

// Main function to localize the position
Point localizePosition(const Point &gpsPosition)
{
    auto nearestPoints = findNearestPoints(trackMap_, gpsPosition);
    return interpolate(nearestPoints.first, nearestPoints.second, gpsPosition);
}

// Main function to determine orientation
double determineOrientation(const Point &prevGpsPosition, const Point &currGpsPosition)
{
    return calculateOrientation(prevGpsPosition, currGpsPosition);
}

// Calculate new GPS position based on orientation and distance
Point calculateNewPosition(const Point &gpsPosition, double orientation, double distance)
{
    const double earthRadius = 6371000;              // Earth radius in kilometers
    double angularDistance = distance / earthRadius; // Convert distance to radians
    double lat = gpsPosition.lat * (M_PI / 180);     // Convert latitude to radians
    double lon = gpsPosition.lon * (M_PI / 180);     // Convert longitude to radians
    double lat2 = asin(sin(lat) * cos(angularDistance) + cos(lat) * sin(angularDistance) * cos(orientation * (M_PI / 180.0)));
    double lon2 = lon + atan2(sin(orientation * (M_PI / 180.0)) * sin(angularDistance) * cos(lat),
                              cos(angularDistance) - sin(lat) * sin(lat2));
    // Convert back to degrees
    lat2 *= (180.0 / M_PI);
    lon2 *= (180.0 / M_PI);
    return {lat2, lon2};
}

// Converts degrees to radians
double toRadians(double degrees)
{
    return degrees * M_PI / 180.0;
}

// Converts radians to degrees
double toDegrees(double radians)
{
    return radians * 180.0 / M_PI;
}

// Convert latitude and longitude to UTM
utmPoint latLonToUTM(double lat, double lon)
{
    utmPoint utmPosition;

    // Constants for UTM
    const double k0 = 0.9996; // UTM scale factor

    // Convert latitude and longitude to radians
    double latRad = toRadians(lat);
    double lonRad = toRadians(lon);

    // Determine UTM zone
    utmPosition.zone = static_cast<int>((lon + 180.0) / 6) + 1;

    // Central meridian of the UTM zone
    double lon0 = toRadians(utmPosition.zone * 6 - 183);

    // Calculate parameters
    double N = WGS84_A / sqrt(1 - WGS84_E_SQ * sin(latRad) * sin(latRad));
    double T = tan(latRad) * tan(latRad);
    double C = WGS84_E_SQ * cos(latRad) * cos(latRad);
    double A = cos(latRad) * (lonRad - lon0);

    // Calculate M (meridional arc)
    double M = WGS84_A * ((1 - WGS84_E_SQ / 4 - 3 * pow(WGS84_E_SQ, 2) / 64 - 5 * pow(WGS84_E_SQ, 3) / 256) * latRad -
                          (3 * WGS84_E_SQ / 8 + 3 * pow(WGS84_E_SQ, 2) / 32 + 45 * pow(WGS84_E_SQ, 3) / 1024) * sin(2 * latRad) +
                          (15 * pow(WGS84_E_SQ, 2) / 256 + 45 * pow(WGS84_E_SQ, 3) / 1024) * sin(4 * latRad) -
                          (35 * pow(WGS84_E_SQ, 3) / 3072) * sin(6 * latRad));

    // Calculate UTM coordinates
    utmPosition.easting = k0 * N * (A + (1 - T + C) * pow(A, 3) / 6 + (5 - 18 * T + T * T + 72 * C - 58 * WGS84_E_SQ) * pow(A, 5) / 120) + 500000.0; // false easting
    utmPosition.northing = k0 * (M + N * tan(latRad) * (A * A / 2 + (5 - T + 9 * C + 4 * C * C) * pow(A, 4) / 24 + (61 - 58 * T + T * T + 600 * C - 330 * WGS84_E_SQ) * pow(A, 6) / 720));

    // UTM northing must be positive in the northern hemisphere
    if (lat < 0)
        utmPosition.northing += 10000000.0;

    return utmPosition;
}

// Calculate UTM position relative to different origin
utmPoint relativeUtmPosition(utmPoint p1, utmPoint p2, double angle)
{
    utmPoint relativePosition;

    relativePosition.zone = p2.zone;

    double dx = p2.easting - p1.easting;
    double dy = p2.northing - p1.northing;
    double angleRad = toRadians(angle);
    relativePosition.easting = dx * cos(angleRad) - dy * sin(angleRad);
    relativePosition.northing = dx * sin(angleRad) + dy * cos(angleRad);

    return relativePosition;
}




//Main Loop
void main_loop()
{
    while(true && ros::ok())
    {
        usleep(1000000);    // in MicroSeconds
        ros::spinOnce();
        ROS_INFO("GPS MAPPING Node MAIN LOOP");   
        geometry_msgs::PoseArray poses;
        poses.header.stamp = ros::Time::now();
        poses.header.frame_id = "neuvition";  

        // Initialize GPS positions
        // previous_gps_Position.lat = 42.1455742;
        // previous_gps_Position.lon = -80.0160358;
        // current_gps_position.lat = 42.1455836;
        // current_gps_position.lon = -80.0160176;
        // previous_gps_Position.lat = 42.16389981;
        // previous_gps_Position.lon = -79.96366773;
        // current_gps_position.lat = 42.16384375;
        // current_gps_position.lon = -79.96380311;
        // previous_gps_Position.lat = 42.1455646;
        // previous_gps_Position.lon = -80.0160627;
        // current_gps_position.lat = 42.1455913;
        // current_gps_position.lon = -80.01601215;

        // Localize GPS position and publish
        Point localized_position = localizePosition(current_gps_position);
        Point localizedPrevPosition = localized_position;
        cout << fixed << setprecision(8);
        cout << "Current GPS position: (" << localized_position.lat << ", " << localized_position.lon << ")" << endl;
        cout << "Previous GPS position: (" << previous_gps_Position.lat << ", " << previous_gps_Position.lon << ")" << endl;

        // Convert GPS current position to UTM        
        utmPoint localized_position_utm;                   // Localized GPS position converted to UTM variable
        localized_position_utm = latLonToUTM(localized_position.lat, localized_position.lon);

        double longDistance = 0.0;// Distance from origin to next point
        double longSpacing = 0.0;// Spacing from current to next point
        if (previous_gps_Position.lat != 0 && previous_gps_Position.lon != 0)
        {
            double orientation = determineOrientation(previous_gps_Position, current_gps_position);
            double firstOrientation = orientation;
            for (int n = 1; n <= numPoints; ++n)
            {
                // Calculate distance
                longDistance = MWD + K * longDistance;
                longSpacing = longDistance - longSpacing;
                // Add distance to same orientation and calculate next position
                Point newPosition = calculateNewPosition(localizedPrevPosition, orientation, longSpacing);
                // Localize this next position in the track map
                Point localizedNewPosition = localizePosition(newPosition);
                cout << fixed << setprecision(8);
                cout << "Position n" << n << ": (" << localizedNewPosition.lat << ", " << localizedNewPosition.lon << ")" << endl;
                //  Check if end of track map
                if ((localizedNewPosition.lat == trackMap_.back().lat && localizedNewPosition.lon == trackMap_.back().lon) 
                || (localizedNewPosition.lat == trackMap_[1].lat && localizedNewPosition.lon == trackMap_[1].lon))
                {
                    break;
                }
                // Convert GPS coordinates to UTM
                utmPoint localizedNewPositionUtm = latLonToUTM(localizedNewPosition.lat, localizedNewPosition.lon);
                // Calculate position relative to loco origin
                utmPoint localizedNewPositionUtmRelative = relativeUtmPosition(localized_position_utm, localizedNewPositionUtm, firstOrientation);

                localizedNewPositionsUtmRelative.push_back(localizedNewPositionUtmRelative); // Store converted next position
			    geometry_msgs::Pose pose;
                pose.position.x = localizedNewPositionUtmRelative.easting;
                pose.position.y = localizedNewPositionUtmRelative.northing;
			    poses.poses.push_back(pose);

                cout << fixed << setprecision(2);
                cout << "Position n" << n << ": (" << localizedNewPositionUtmRelative.easting << ", " << localizedNewPositionUtmRelative.northing << ")" << endl;
                // Determine next orientation
                orientation = determineOrientation(localizedPrevPosition, localizedNewPosition);
                // Store found distance and position as old for next loop
                localizedPrevPosition = localizedNewPosition;
                longSpacing = longDistance;
            }
            pose_pub.publish(poses);

        }
        previous_gps_Position = current_gps_position;
    }
}

void recvCurrentVehicleHeading(const geometry_msgs::PoseStamped& heading)
{
    q_current_heading = tf::getYaw(heading.pose.orientation);
    //cout<<"Vehicle_current_Heading = "<<Vehicle_current_Heading<<endl;
}
void recvCurrentVehicleLocation(const geometry_msgs::PoseStamped& location)
{	
    q_current_position.easting = location.pose.position.x;
    q_current_position.northing = location.pose.position.y;
}
void navfixCallback(const sensor_msgs::NavSatFixConstPtr& msg)
{
    current_gps_position.lat = msg->latitude;
    current_gps_position.lon = msg->longitude;
}

int main(int argc, char* argv[])
{    
    ros::init(argc, argv, "gps_track_mapping_node");
    ROS_INFO("GPS TRACK MAPPING Node STARTED!");
    // Create a ROS node handle
    ros::NodeHandle nh;
    ros::NodeHandle pn("~");
    std::string track_map_file;
    if (nh.getParam("track_map_file", track_map_file))
    {
        // Load track map data
        trackMap_ = readTrackMap(track_map_file);
        // trackMap_ = readTrackMap("/home/fev/wab_perception_02/src/gps_mapping/track_map.csv");
    }
    else
    {
        return -1;
    }
    
    ros::Subscriber sub_pose 	        = nh.subscribe("/current_pose", 0, &recvCurrentVehicleLocation);
    ros::Subscriber sub_orientation 	= nh.subscribe("/current_pose_fake_orientation", 1, &recvCurrentVehicleHeading);
    ros::Subscriber sub_Nav_fix         = nh.subscribe("/fix",0, &navfixCallback);
	pose_pub             = nh.advertise<geometry_msgs::PoseArray> ("/gps_track_map_points", 1);

    main_loop();

    return 0;
}
