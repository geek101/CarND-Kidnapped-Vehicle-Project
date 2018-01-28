/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
  //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  // Initialize the number of particles.
  num_particles = 101;

  default_random_engine gen;
  // This line creates a normal (Gaussian) distribution for x.
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i) {
    double sample_x, sample_y, sample_theta;

    sample_x = dist_x(gen);
    sample_y = dist_y(gen);
    sample_theta = dist_theta(gen);

    particles.emplace_back(Particle(i, sample_x, sample_y, sample_theta));
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  default_random_engine gen;
  normal_distribution<double> noise_x(0, std_pos[0]);
  normal_distribution<double> noise_y(0, std_pos[1]);
  normal_distribution<double> noise_t(0, std_pos[2]);


  for (int i = 0; i < (int)particles.size(); i++) {
    if (abs(yaw_rate) < 0.00001) {
      particles[i].x = particles[i].x + velocity * delta_t * cos(particles[i].theta) + noise_x(gen);
      particles[i].y = particles[i].y + velocity * delta_t * sin(particles[i].theta) + noise_y(gen);
    } else {
      particles[i].x = particles[i].x +
          (velocity/yaw_rate)*(sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta)) + noise_x(gen);
      particles[i].y = particles[i].y +
          (velocity/yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t)) + noise_y(gen);
      particles[i].theta = particles[i].theta + yaw_rate*delta_t + noise_t(gen);
    }
  }
}

vector<pair<int, int>> ParticleFilter::dataAssociation(const std::vector<LandmarkObs>& predicted,
                                                       const std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
  //   implement this method and use it as a helper during the updateWeights phase.

  // O(N^2) neigherst neighbors
  vector<pair<int, int>> result;
  for (int i = 0; i < (int)observations.size(); i++) {
    int min_mark = -1;
    double min_dist = 0.0;
    auto& o = observations[i];
    int j = 0;
    for (; j < (int)predicted.size(); j++) {
      auto& p = predicted[j];
      double d = dist(o.x, o.y, p.x, p.y);
      if (min_mark == -1) {
        min_mark = j;
        min_dist = d;
      } else if (d < min_dist) {
        min_mark = j;
        min_dist = d;
      }
    }

    result.emplace_back(make_pair(min_mark, i));
  }

  return result;
}

std::pair<double, double> ParticleFilter::translateRotateXY(const double xp, const double yp, const double xc,
                                                             const double yc, const double theta) {
  double xm = xp + xc * cos(theta) - yc*sin(theta);
  double ym = yp + xc * sin(theta) + yc*cos(theta);
  return make_pair(xm, ym);
}

double ParticleFilter::probXY(double x, double y, double ux, double uy, double sigx, double sigy) {
  double gauss_norm = (1 / (2 * M_PI * sigx * sigy));
  double exponent = (pow(x - ux, 2) / (2 * pow(sigx, 2)) + pow(y - uy, 2) / (2 * pow(sigy, 2)));
  double weight = gauss_norm * exp(-exponent);
  return weight;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  // transform observations to map co-ordinates
  for (Particle& p : particles) {
    std::vector<LandmarkObs> mapLandmarks;
    for (auto& m : map_landmarks.landmark_list) {
      if (dist(p.x, p.y, m.x_f, m.y_f) <= sensor_range) {
        mapLandmarks.emplace_back(LandmarkObs(m.id_i, m.x_f, m.y_f));
      }
    }
    std::vector<LandmarkObs> mapXYObservations;
    for (auto& o : observations) {
      std::pair<double, double> ret = translateRotateXY(p.x, p.y, o.x, o.y, p.theta);
      mapXYObservations.emplace_back(LandmarkObs(o.id, ret.first, ret.second));
    }

    // now get closest landmarks with what we have on the given map.
    vector<pair<int, int>> validMapLandmarks = dataAssociation(mapLandmarks, mapXYObservations);
    double prob_product = 1.0;
    for (auto& p : validMapLandmarks) {
      int i = p.second;  // observed
      int j = p.first;   // predicted
      if (j >= (int)mapLandmarks.size()) {
        cout << "j: " << j << " exceeded: " << mapLandmarks.size() << endl;
        exit(-1);
      }
      if (i >= (int)mapXYObservations.size()) {
        cout << "i: " << i << " exceeded: " << mapXYObservations.size() << endl;
        exit(-1);
      }
      prob_product *= probXY(mapXYObservations[i].x, mapXYObservations[i].y, mapLandmarks[j].x,
                             mapLandmarks[j].y, std_landmark[0], std_landmark[1]);
    }
    p.weight = prob_product;
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  std::random_device rd;
  std::mt19937 gen(rd());
  vector<double> particle_weights;
  for (int i = 0; i < (int)particles.size(); i++) {
    particle_weights.emplace_back(particles[i].weight);
  }
  std::discrete_distribution<> d(particle_weights.begin(), particle_weights.end());

  vector<Particle> resampled_particles;
  for (int i = 0; i < (int)particles.size(); i++) {
    resampled_particles.emplace_back(particles[d(gen)]);
  }

  particles = resampled_particles;
  cout << "Particles size: " << particles.size() << endl;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

