/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

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
using std::default_random_engine;
using std::discrete_distribution;
using std::numeric_limits;

#define EPS 0.00001 

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
    
    default_random_engine rand_eng;
    
    num_particles = 50;  // TODO: Set the number of particles
    
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);
    
    for (int i = 0; i < num_particles; ++i){
        Particle p;
        p.id = i;
        p.x = dist_x(rand_eng);
        p.y = dist_y(rand_eng);
        p.theta = dist_theta(rand_eng);
        p.weight = 1;
        
        particles.push_back(p);
        weights.push_back(p.weight);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    
    double theta_dt = delta_t * yaw_rate;
    double val = velocity / yaw_rate;
  
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);
    default_random_engine gen;
    
    // Update each particle
    for(int i = 0; i < num_particles; ++i){
        if (fabs(yaw_rate) < EPS){
            particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
            // In this case heading doesn't change
        }
        else{
            particles[i].x += val * (sin(particles[i].theta + theta_dt) - sin(particles[i].theta));
            particles[i].y += val * (cos(particles[i].theta) - cos(particles[i].theta + theta_dt));
            particles[i].theta += theta_dt;
        }
        
        // Adding noise
        particles[i].x += dist_x(gen);
        particles[i].y += dist_y(gen);
        particles[i].theta += dist_theta(gen);
    }
}


void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    for(auto &obs : observations){
        
        // Initialize minimum distance and index
        double min_dist = numeric_limits<double>::max();
        int landmark_idx = -1;
        
        for (auto &p : predicted){            
            double d = dist(obs.x, obs.y, p.x, p.y);
            
            if(d < min_dist){
                min_dist = d;
                landmark_idx = p.id;
            }
        }
        obs.id = landmark_idx;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
    
    // calculate normalization term
    double s_xx = std_landmark[0]*std_landmark[0];
    double s_yy = std_landmark[1]*std_landmark[1];
    double constant = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
    
    // For each particle
    for(auto &particle : particles){
        double x = particle.x;
        double y = particle.y;
        double theta = particle.theta;
                
        // Find landmarks within sensor range
        vector<LandmarkObs> predictions;
        for (int l = 0; l < map_landmarks.landmark_list.size(); ++l){
            float lx = map_landmarks.landmark_list[l].x_f;
            float ly = map_landmarks.landmark_list[l].y_f;
            int id = map_landmarks.landmark_list[l].id_i;
            
            if(dist(lx, ly, particle.x, particle.y) <= sensor_range){
                predictions.push_back(LandmarkObs{ id, lx, ly });
            }
        }
        
        // Transform each observation into map coordinate system
        vector<LandmarkObs> transformed_obs;
        for(int j = 0; j < observations.size(); ++j){
            double X = x + (cos(theta)*observations[j].x) - (sin(theta)*observations[j].y);
            double Y = y + (sin(theta)*observations[j].x) + (cos(theta)*observations[j].y);
            transformed_obs.push_back(LandmarkObs{ observations[j].id, X, Y });
        }
        
        // Associate observations to landmarks
        dataAssociation(predictions, transformed_obs);
        
        vector<int> associations;
        vector<double> sense_x;
        vector<double> sense_y;
        
        // Calculate particle weight
        particle.weight = 1.0;
        for(int j = 0; j < transformed_obs.size(); ++j){
            int associated_id = transformed_obs[j].id;
            
            double landmark_x, landmark_y;
            // get the x,y coordinates of the landmark
            for (auto &prediction : predictions) {
                if (prediction.id == associated_id) {
                    landmark_x = prediction.x;
                    landmark_y = prediction.y;
                }
            }
            
            // Calculating weight.
            double dx2 = (transformed_obs[j].x - landmark_x) * (transformed_obs[j].x - landmark_x);
            double dy2 = (transformed_obs[j].y - landmark_y) * (transformed_obs[j].y - landmark_y);
            
            double weight = constant * exp( -(dx2 / (2 * s_xx) + (dy2 / (2 * s_yy))));
            particle.weight *= weight;
 
            associations.push_back(associated_id);
            sense_x.push_back(transformed_obs[j].x);
            sense_y.push_back(transformed_obs[j].y);
        }
        
        // Visualization
        SetAssociations(particle, associations, sense_x, sense_y);
    }
}

void ParticleFilter::resample() {
    default_random_engine gen;

    // Update weights
    weights.clear();
    for(auto &particle : particles){
        weights.push_back(particle.weight);
    }
    
    // Resample particles
    discrete_distribution<int> particle_dist(weights.begin(),weights.end());
    vector<Particle> resampled_particles;
    resampled_particles.resize(num_particles);
  
    for(auto &resampled : resampled_particles){
      resampled = particles[particle_dist(gen)];
    }
    particles = resampled_particles;
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