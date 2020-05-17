#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::min_element;
using std::max_element;
using std::cout;
using std::endl;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 50;  // Set the number of particles
  std::default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  for (int i=0; i<num_particles; i++) {
    
    Particle p = { i, dist_x(gen), dist_y(gen), dist_theta(gen), 1.0 };
    particles.push_back(p);
    weights.push_back(1.0);
  }
  is_initialized = true;
	
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  for (auto &p : particles) {
    if (fabs(yaw_rate) < 0.0001) {  // constant velocity
      p.x += velocity * delta_t * cos(p.theta);
      p.y += velocity * delta_t * sin(p.theta);
    } else {
      p.x += (velocity/yaw_rate)*(sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
      p.y += (velocity/yaw_rate)*(cos(p.theta) - cos(p.theta + yaw_rate*delta_t));
    }
    p.theta += yaw_rate*delta_t;
    normal_distribution<double> dist_x(p.x, std_pos[0]);
    normal_distribution<double> dist_y(p.y, std_pos[1]);
  	normal_distribution<double> dist_theta(p.theta, std_pos[2]);
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(LandmarkObs &predicted, 
                                     const vector<LandmarkObs> &map_landmarks) {
  /**
   * Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  vector<double> distances;
  float least_distance = INFINITY;
  for (auto &landmark : map_landmarks) {
    double distance = dist(predicted.x, predicted.y, landmark.x, landmark.y);
    if (distance < least_distance) {
      least_distance = distance;
      predicted.id = landmark.id;
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * Update the weights of each particle using a multi-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  weights.clear();
  for (auto &particle : particles) {
    vector<LandmarkObs> within_range_landmarks;
    for (auto &landmark : map_landmarks.landmark_list) {
      if (fabs(particle.x - landmark.x_f) <= sensor_range &&
          fabs(particle.y - landmark.y_f) <= sensor_range) {
        LandmarkObs within_range_landmark = { landmark.id_i, landmark.x_f, landmark.y_f };
        within_range_landmarks.push_back(within_range_landmark);
      }
    }
    particle.weight = 1.0;
    for (auto &observation : observations) {
      double x_obs = particle.x + (cos(particle.theta) * observation.x) - (sin(particle.theta) * observation.y);
      double y_obs = particle.y + (sin(particle.theta) * observation.x) + (cos(particle.theta) * observation.y);
      LandmarkObs predicted_obs = { -1, x_obs, y_obs };
      dataAssociation(predicted_obs, within_range_landmarks);
      for (auto &landmark : within_range_landmarks) {
        if (predicted_obs.id == landmark.id) {
          double mu_x = landmark.x;
          double mu_y = landmark.y;
          particle.weight *= multiv_prob(std_landmark[0], std_landmark[1], x_obs, y_obs, mu_x, mu_y);
        }
      }
    }
    weights.push_back(particle.weight);
  }
}

void ParticleFilter::resample() {
  /**
   * Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<Particle> sampled_particles;
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<> dist_int(0, num_particles - 1);
  std::uniform_real_distribution<> dist_real(0.0, 1.0);
  int index = dist_int(rng);
  double beta = 0.0;
  double mw = max_element(weights.begin(), weights.end())[0];
  for (int i=0; i<num_particles; i++) {
    beta += dist_real(rng) * 2.0 * mw;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    sampled_particles.push_back(particles[index]);
  }
  particles = sampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}